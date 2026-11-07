"""Runs tool commands inside a bwrap sandbox built from a permission profile.

bwrap runs as root through sudo, so it can grant CAP_DAC_READ_SEARCH in the host user namespace.
setpriv then drops to the user's uid keeping only that capability, and confine.py adds Landlock and seccomp.
"""
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CONFINE = str(Path(__file__).with_name("confine.py"))
SCRATCH_MAX_AGE = 48 * 3600
SUDO_NOTICE = ("Artifice starts the model's sandbox with sudo so the model can read every file on this computer. "
	"The sandbox drops to your user with only read access before running anything.")


@dataclass(frozen=True)
class Profile:
	name: str
	timeout: int = 120  # default seconds per command
	max_timeout: int = 600
	memory_max: str = "8G"  # systemd MemoryMax for one command
	tasks_max: int = 512
	file_size_max: int = 1 << 30  # largest file a command may write to scratch
	output_max: int = 1 << 20  # bytes of output kept per command


READ_ONLY = Profile(name="read-only")


@dataclass
class Result:
	output: str
	exit_code: int | None  # None when killed by timeout
	timed_out: bool = False
	truncated_bytes: int = 0


class SandboxError(Exception):
	pass


class Sandbox:
	def __init__(self, profile: Profile, scratch: Path, cwd: str, notify: Callable[[str], None] | None = None):
		self.profile = profile
		self.scratch = scratch
		self.cwd = cwd
		self.notify = notify or (lambda text: print(text, file=sys.stderr))
		self.bwrap = shutil.which("bwrap") or "bwrap"
		self.setpriv = shutil.which("setpriv") or "setpriv"
		self._systemd = _systemd_scope_works()
		self._passwordless = False

	def command(self) -> list[str]:
		"""The sandbox command line. The command to run goes on stdin, which keeps it out of sudo's logs."""
		# Another instance's clean_scratch may have removed it while empty.
		self.scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
		home = os.path.expanduser("~")
		env = {
			"PATH": "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
			"HOME": home,
			"USER": os.environ.get("USER", ""),
			"LANG": os.environ.get("LANG", "C.UTF-8"),
			"TERM": "dumb",
			"NO_COLOR": "1",
			"PAGER": "cat",
			"GIT_PAGER": "cat",
			"GIT_OPTIONAL_LOCKS": "0",
			"TMPDIR": "/tmp",
			"XDG_CACHE_HOME": "/tmp/.cache",
			"PYTHONDONTWRITEBYTECODE": "1",
		}
		cmd: list[str] = []
		if self._systemd:
			cmd += [
				"systemd-run", "--user", "--scope", "--quiet", "--collect",
				"-p", f"MemoryMax={self.profile.memory_max}",
				"-p", "MemorySwapMax=0",
				"-p", f"TasksMax={self.profile.tasks_max}",
			]
		cmd += [
			"sudo", "-n", self.bwrap,
			"--ro-bind", "/", "/",
			"--dev", "/dev",
			# The host's /proc shows every process, but PIDs given to kill or ptrace resolve in the new PID namespace.
			"--ro-bind", "/proc", "/proc",
			"--bind", str(self.scratch), "/tmp",
			"--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
			"--die-with-parent",
			"--new-session",
			"--clearenv",
		]
		for key, value in env.items():
			cmd += ["--setenv", key, value]
		for cap in ("CAP_DAC_READ_SEARCH", "CAP_SETUID", "CAP_SETGID", "CAP_SETPCAP"):
			cmd += ["--cap-add", cap]
		cwd = self.cwd if os.path.isdir(self.cwd) else home
		groups = os.getgroups()
		cmd += ["--chdir", cwd, "--"]
		cmd += [
			self.setpriv, f"--reuid={os.getuid()}", f"--regid={os.getgid()}",
			f"--groups={','.join(map(str, groups))}" if groups else "--clear-groups",
			"--inh-caps=-all,+dac_read_search",
			"--ambient-caps=-all,+dac_read_search",
			"--bounding-set=-all,+dac_read_search",
			"--",
		]
		cmd += [sys.executable, "-I", "-S", CONFINE, "--write", "/tmp", "--write", "/dev"]
		cmd += ["--fsize", str(self.profile.file_size_max), "--argv-stdin"]
		return cmd

	def ensure_sudo(self) -> None:
		"""Makes sure sudo runs without a password, asking for it on the terminal if needed."""
		if self._passwordless:
			return
		if _quiet(["sudo", "-n", "-v"]):
			return
		if _quiet(["sudo", "-n", "true"]):
			self._passwordless = True
			return
		if not sys.stdin.isatty():
			raise SandboxError("Failed to use sudo: it needs a password and there is no terminal. Run `sudo -v` first.")
		self.notify(SUDO_NOTICE)
		if subprocess.run(["sudo", "-v"]).returncode != 0:
			raise SandboxError("Failed to authenticate with sudo.")

	def run(self, argv: list[str], timeout: int | None = None) -> Result:
		timeout = min(timeout or self.profile.timeout, self.profile.max_timeout)
		self.ensure_sudo()
		# No new session: sudo finds its cached login through this terminal.
		proc = subprocess.Popen(
			self.command(),
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
		)
		kept = bytearray()
		dropped = 0
		timed_out = False
		deadline = time.monotonic() + timeout
		sel = selectors.DefaultSelector()
		try:
			try:
				proc.stdin.write(json.dumps(argv).encode())
				proc.stdin.close()
			except BrokenPipeError:
				pass
			sel.register(proc.stdout, selectors.EVENT_READ)
			while True:
				remaining = deadline - time.monotonic()
				if remaining <= 0:
					timed_out = True
					break
				if not sel.select(remaining):
					continue
				chunk = os.read(proc.stdout.fileno(), 65536)
				if not chunk:
					break
				room = self.profile.output_max - len(kept)
				kept += chunk[:room]
				dropped += max(0, len(chunk) - room)
		finally:
			sel.close()
			if proc.poll() is None:
				# Killing sudo kills bwrap through --die-with-parent, and with it the PID namespace.
				try:
					os.kill(proc.pid, signal.SIGKILL)
				except ProcessLookupError:
					pass
			proc.wait()
			proc.stdout.close()
		return Result(
			output=kept.decode("utf-8", errors="replace"),
			exit_code=None if timed_out else proc.returncode,
			timed_out=timed_out,
			truncated_bytes=dropped,
		)

	def check(self) -> None:
		"""Raises SandboxError with a diagnosis when the sandbox cannot start."""
		for tool, package in (("bwrap", "bubblewrap"), ("setpriv", "util-linux"), ("sudo", "sudo")):
			if not shutil.which(tool):
				raise SandboxError(f"Failed to find {tool} (package: {package})")
		result = self.run(["true"], timeout=30)
		if result.exit_code != 0:
			raise SandboxError(f"Failed to start sandbox (exit {result.exit_code}): {result.output.strip()}")

	def scratch_bytes(self) -> int:
		total = 0
		for root, _dirs, files in os.walk(self.scratch):
			for name in files:
				try:
					total += os.lstat(os.path.join(root, name)).st_size
				except OSError:
					pass
		return total


def _quiet(argv: list[str]) -> bool:
	return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True).returncode == 0


def _systemd_scope_works() -> bool:
	if not shutil.which("systemd-run"):
		return False
	try:
		probe = subprocess.run(
			["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"],
			stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
		)
	except (OSError, subprocess.TimeoutExpired):
		return False
	return probe.returncode == 0


def clean_scratch(root: Path, max_age: int = SCRATCH_MAX_AGE) -> None:
	"""Deletes scratch files older than max_age, then empty directories. Never follows symlinks."""
	if not root.is_dir():
		return
	cutoff = time.time() - max_age
	for dirpath, dirnames, filenames in os.walk(root, topdown=False):
		for name in filenames:
			path = os.path.join(dirpath, name)
			try:
				if os.lstat(path).st_mtime < cutoff:
					os.unlink(path)
			except OSError:
				pass
		for name in dirnames:
			path = os.path.join(dirpath, name)
			try:
				if not os.path.islink(path):
					os.rmdir(path)
				elif os.lstat(path).st_mtime < cutoff:
					os.unlink(path)
			except OSError:
				pass
