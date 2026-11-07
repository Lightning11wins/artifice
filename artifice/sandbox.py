"""Runs tool commands inside a bwrap sandbox built from a permission profile."""
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CONFINE = str(Path(__file__).with_name("confine.py"))
SCRATCH_MAX_AGE = 48 * 3600


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
	def __init__(self, profile: Profile, scratch: Path, cwd: str):
		self.profile = profile
		self.scratch = scratch
		self.cwd = cwd
		self._systemd = _systemd_scope_works()

	def command(self, argv: list[str]) -> list[str]:
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
			"bwrap",
			"--ro-bind", "/", "/",
			"--dev", "/dev",
			"--proc", "/proc",
			"--bind", str(self.scratch), "/tmp",
			"--unshare-all",
			"--die-with-parent",
			"--new-session",
			"--clearenv",
		]
		for key, value in env.items():
			cmd += ["--setenv", key, value]
		cwd = self.cwd if os.path.isdir(self.cwd) else home
		cmd += ["--chdir", cwd, "--"]
		cmd += [sys.executable, "-I", "-S", CONFINE, "--write", "/tmp", "--write", "/dev"]
		cmd += ["--fsize", str(self.profile.file_size_max), "--", *argv]
		return cmd

	def run(self, argv: list[str], timeout: int | None = None) -> Result:
		timeout = min(timeout or self.profile.timeout, self.profile.max_timeout)
		proc = subprocess.Popen(
			self.command(argv),
			stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			start_new_session=True,
		)
		kept = bytearray()
		dropped = 0
		timed_out = False
		deadline = time.monotonic() + timeout
		sel = selectors.DefaultSelector()
		sel.register(proc.stdout, selectors.EVENT_READ)
		try:
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
				_kill(proc)
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
		if not shutil.which("bwrap"):
			raise SandboxError("Failed to find bwrap (package: bubblewrap)")
		result = self.run(["true"], timeout=30)
		if result.exit_code == 0:
			return
		message = result.output.strip()
		if "namespace" in message and _apparmor_restricts_userns():
			raise SandboxError(f"{message}\nAppArmor is blocking user namespaces for bwrap.")
		raise SandboxError(f"Failed to start sandbox (exit {result.exit_code}): {message}")

	def scratch_bytes(self) -> int:
		total = 0
		for root, _dirs, files in os.walk(self.scratch):
			for name in files:
				try:
					total += os.lstat(os.path.join(root, name)).st_size
				except OSError:
					pass
		return total


def _kill(proc: subprocess.Popen) -> None:
	try:
		os.killpg(proc.pid, signal.SIGKILL)
	except ProcessLookupError:
		pass


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


def _apparmor_restricts_userns() -> bool:
	try:
		return Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text().strip() == "1"
	except OSError:
		return False


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
