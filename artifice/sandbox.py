"""Runs tool commands inside a bwrap sandbox built from a permission profile.

bwrap runs as root through sudo, so it can grant CAP_DAC_READ_SEARCH in the host user namespace.
setpriv then drops to the user's uid keeping only that capability, and confine.py adds Landlock and seccomp.
"""
import errno
import fcntl
import json
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
import weakref
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
	scratch_max: int = 4 << 30  # bytes of disk scratch may use; a command past it is killed
	scratch_files_max: int = 100_000


READ_ONLY = Profile(name="read-only")


@dataclass
class Result:
	output: str
	exit_code: int | None  # None when killed by timeout
	timed_out: bool = False
	truncated_bytes: int = 0
	scratch_full: bool = False  # scratch is past the profile's limits; exit_code is None if that killed the command


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
		self._systemd = shutil.which("systemd-run") is not None and os.path.isdir("/run/systemd/system")
		self._passwordless = False
		self._lock = self._lock_scratch()
		weakref.finalize(self, os.close, self._lock)

	def _lock_scratch(self) -> int:
		"""Holds a shared lock on the scratch directory, so clean_scratch skips it while this chat is open."""
		for _ in range(3):
			self.scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
			fd = os.open(self.scratch, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
			fcntl.flock(fd, fcntl.LOCK_SH)
			try:
				if os.path.samestat(os.fstat(fd), os.lstat(self.scratch)):
					return fd
			except OSError:
				pass
			os.close(fd)  # clean_scratch removed it before the lock was taken
		raise SandboxError(f"Failed to lock scratch directory {self.scratch}")

	def command(self) -> list[str]:
		"""The sandbox command line. The command to run goes on stdin, which keeps it out of sudo's logs."""
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
		cmd = ["sudo", "-n"]
		if self._systemd:
			# A system scope, because sudo's PAM session can move it out of a user scope.
			cmd += [
				"systemd-run", "--scope", "--quiet", "--collect",
				"-p", f"MemoryMax={self.profile.memory_max}",
				"-p", "MemorySwapMax=0",
				"-p", f"TasksMax={self.profile.tasks_max}",
			]
		cmd += [
			self.bwrap,
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
		cwd = self.cwd if os.path.isdir(self.cwd) and not hidden_by_scratch(self.cwd) else home
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
		if self._starts_without_password():
			self._passwordless = True
			return
		if not sys.stdin.isatty():
			raise SandboxError("Failed to use sudo: it needs a password and there is no terminal. Run `sudo -v` first.")
		self.notify(SUDO_NOTICE)
		if subprocess.run(["sudo", "-v"]).returncode != 0:
			raise SandboxError("Failed to authenticate with sudo.")

	def _starts_without_password(self) -> bool:
		"""Whether sudo runs the sandbox command itself without a password, as a narrow NOPASSWD rule allows."""
		try:
			return subprocess.run(self.command(), input=b'["true"]', capture_output=True, timeout=30).returncode == 0
		except subprocess.TimeoutExpired:
			return False

	def run(self, argv: list[str], timeout: int | None = None) -> Result:
		timeout = max(1, min(timeout or self.profile.timeout, self.profile.max_timeout))
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
		killed_for_scratch = False
		deadline = time.monotonic() + timeout
		next_scratch_check = time.monotonic() + 1
		sel = selectors.DefaultSelector()
		try:
			try:
				proc.stdin.write(json.dumps(argv).encode())
				proc.stdin.close()
			except BrokenPipeError:
				pass
			sel.register(proc.stdout, selectors.EVENT_READ)
			while True:
				now = time.monotonic()
				if now >= deadline:
					timed_out = True
					break
				if now >= next_scratch_check:
					if self.scratch_full():
						killed_for_scratch = True
						break
					next_scratch_check = time.monotonic() + 1
				if not sel.select(min(deadline, next_scratch_check) - now):
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
			exit_code=None if timed_out or killed_for_scratch else proc.returncode,
			timed_out=timed_out,
			truncated_bytes=dropped,
			scratch_full=killed_for_scratch or self.scratch_full(),
		)

	def check(self) -> None:
		"""Raises SandboxError with a diagnosis when the sandbox cannot start."""
		for tool, package in (("bwrap", "bubblewrap"), ("setpriv", "util-linux"), ("sudo", "sudo")):
			if not shutil.which(tool):
				raise SandboxError(f"Failed to find {tool} (package: {package})")
		for path in (sys.executable, CONFINE):
			if hidden_by_scratch(path):
				raise SandboxError(f"Failed to start sandbox: {path} is under /tmp, which the sandbox replaces with scratch")
		if not self._systemd:
			self.notify("systemd is not running, so sandboxed commands have no memory or process limits.")
		result = self.run(["true"], timeout=30)
		if result.exit_code != 0:
			raise SandboxError(f"Failed to start sandbox (exit {result.exit_code}): {result.output.strip()}")

	def scratch_full(self) -> bool:
		"""Whether scratch is past the profile's limits, counting a directory that cannot be read as past them."""
		size, entries, unreadable = self.scratch_usage()
		return unreadable or size > self.profile.scratch_max or entries > self.profile.scratch_files_max

	def scratch_usage(self) -> tuple[int, int, bool]:
		"""Disk bytes and entries in scratch, and whether some directory could not be read."""
		size = entries = 0
		unreadable = False

		def error(e: OSError) -> None:
			nonlocal unreadable
			unreadable = unreadable or e.errno in (errno.EACCES, errno.EPERM)

		for root, dirs, files in os.walk(self.scratch, onerror=error):
			for name in files + dirs:
				entries += 1
				try:
					size += os.lstat(os.path.join(root, name)).st_blocks * 512
				except OSError:
					pass
		return size, entries, unreadable


def _quiet(argv: list[str]) -> bool:
	return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True).returncode == 0


def hidden_by_scratch(path: str) -> bool:
	"""Whether path is under the host's /tmp, which the sandbox replaces with scratch."""
	return any(p == "/tmp" or p.startswith("/tmp/") for p in (os.path.abspath(path), os.path.realpath(path)))


def clean_scratch(root: Path, max_age: int = SCRATCH_MAX_AGE) -> None:
	"""Deletes old files from the scratch directories of chats that are not open, then empty directories.

	A file's age counts from its last change in scratch, the later of mtime and ctime. Symlinks are never followed.
	"""
	try:
		root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
	except OSError:
		return
	cutoff = time.time() - max_age
	try:
		for name in os.listdir(root_fd):
			try:
				fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
			except OSError:
				continue
			try:
				fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
				_clean_dir(fd, cutoff)
				os.rmdir(name, dir_fd=root_fd)
			except OSError:
				pass
			finally:
				os.close(fd)
	finally:
		os.close(root_fd)


def _clean_dir(top_fd: int, cutoff: float) -> None:
	# Directories need owner write and search for their entries to be deleted.
	os.fchmod(top_fd, os.fstat(top_fd).st_mode | 0o700)
	for _path, dirnames, _files, dir_fd in os.fwalk(".", dir_fd=top_fd, follow_symlinks=False):
		for name in dirnames:
			try:
				st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
				if stat.S_ISDIR(st.st_mode) and st.st_mode & 0o700 != 0o700:
					os.chmod(name, st.st_mode | 0o700, dir_fd=dir_fd)
			except OSError:
				pass
	for _path, dirnames, filenames, dir_fd in os.fwalk(".", dir_fd=top_fd, topdown=False, follow_symlinks=False):
		for name in filenames + dirnames:
			try:
				st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
				if stat.S_ISDIR(st.st_mode):
					os.rmdir(name, dir_fd=dir_fd)
				elif max(st.st_mtime, st.st_ctime) < cutoff:
					os.unlink(name, dir_fd=dir_fd)
			except OSError:
				pass
