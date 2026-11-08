"""Runs tool commands inside a bwrap sandbox built from a permission profile.

bwrap runs as root through sudo, so it can grant CAP_DAC_READ_SEARCH in the host user namespace.
setpriv then drops to the user's uid keeping only that capability, and confine.py adds Landlock and seccomp.
"""
import errno
import fcntl
import json
import os
import resource
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
DEADLINE_GRACE = 10  # seconds past its timeout that a command survives Artifice being killed
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
	shm_max: int = 1 << 30  # bytes of /dev/shm, which lives in memory


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
	def __init__(self, profile: Profile, scratch: Path, cwd: str, notify: Callable[[str], None] | None = None,
			exclusive: bool = False):
		self.profile = profile
		self.scratch = scratch
		self.cwd = cwd
		self.notify = notify or (lambda text: print(text, file=sys.stderr))
		self.bwrap = shutil.which("bwrap") or "bwrap"
		self.setpriv = shutil.which("setpriv") or "setpriv"
		self.unshare = shutil.which("unshare") or "unshare"
		self.timeout = shutil.which("timeout") or "timeout"
		self._systemd = shutil.which("systemd-run") is not None and os.path.isdir("/run/systemd/system")
		self._passwordless = False
		self._lock = self._lock_scratch(exclusive)
		self.close = weakref.finalize(self, os.close, self._lock)  # releases the lock early; dropping the sandbox does too

	def _lock_scratch(self, exclusive: bool) -> int:
		"""Locks the scratch directory, so clean_scratch skips it while this chat is open.

		An exclusive lock also keeps a second Artifice session out of the chat.
		"""
		for _ in range(3):
			self.scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
			try:
				fd = _open_dir(self.scratch)  # the model may have chmodded its /tmp
			except FileNotFoundError:
				continue  # clean_scratch removed it after mkdir
			if exclusive:
				# clean_scratch locks the parent while it runs, and briefly locks each chat it cleans.
				parent = os.open(self.scratch.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
				try:
					fcntl.flock(parent, fcntl.LOCK_SH)
					fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
				except BlockingIOError:
					os.close(fd)
					raise SandboxError(f"Failed to open chat {self.scratch.name}: another Artifice session has it open.")
				finally:
					os.close(parent)
			else:
				fcntl.flock(fd, fcntl.LOCK_SH)
			try:
				if os.path.samestat(os.fstat(fd), os.lstat(self.scratch)):
					return fd
			except OSError:
				pass
			os.close(fd)  # clean_scratch removed it before the lock was taken
		raise SandboxError(f"Failed to lock scratch directory {self.scratch}")

	def command(self, deadline: int) -> list[str]:
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
		# timeout runs as root outside the sandbox, so it ends the sandbox even if Artifice was killed.
		# It dies with its parent, sudo or, under use_pty, sudo's monitor process, which kills it when sudo dies.
		# bwrap dies with it through --die-with-parent.
		# The second setpriv makes the rest of the chain die with timeout, before bwrap's --die-with-parent applies.
		cmd = ["sudo", "-n", self.setpriv, "--pdeathsig", "KILL", self.timeout, "--foreground", "-s", "KILL", str(deadline),
			self.setpriv, "--pdeathsig", "KILL"]
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
			# Writable shared memory, for multiprocessing and POSIX semaphores.
			"--perms", "1777", "--size", str(self.profile.shm_max), "--tmpfs", "/dev/shm",
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
		for cap in ("CAP_DAC_READ_SEARCH", "CAP_SETUID", "CAP_SETGID", "CAP_SETPCAP", "CAP_SYS_ADMIN"):
			cmd += ["--cap-add", cap]
		cwd = self.cwd if os.path.isdir(self.cwd) and not hidden_in_sandbox(self.cwd) else home
		groups = os.getgroups()
		# bwrap's mounts receive the host's new mounts, without the read-only flag. A private mount namespace does not.
		cmd += ["--chdir", "/", "--", self.unshare, "--mount", "--propagation", "private", "--"]
		cmd += [
			self.setpriv, f"--reuid={os.getuid()}", f"--regid={os.getgid()}",
			f"--groups={','.join(map(str, groups))}" if groups else "--clear-groups",
			"--inh-caps=-all,+dac_read_search",
			"--ambient-caps=-all,+dac_read_search",
			"--bounding-set=-all,+dac_read_search",
			"--",
		]
		# confine.py enters cwd as the user, who can enter FUSE mounts that root cannot.
		cmd += [sys.executable, "-I", "-S", CONFINE, "--chdir", cwd, "--write", "/tmp", "--write", "/dev"]
		cmd += ["--fsize", str(self.profile.file_size_max), "--argv-stdin"]
		return cmd

	def ensure_sudo(self) -> None:
		"""Makes sure sudo runs without a password, asking for it on the terminal if needed."""
		if self._passwordless:
			return
		if _quiet(["sudo", "-n", "-v"]):
			return
		if self._sudo_runs_sandbox():
			self._passwordless = True
			return
		if not sys.stdin.isatty():
			raise SandboxError("Failed to use sudo: it needs a password and there is no terminal. Run `sudo -v` first.")
		self.notify(SUDO_NOTICE)
		if subprocess.run(["sudo", "-v"]).returncode != 0:
			raise SandboxError("Failed to authenticate with sudo.")

	def _sudo_runs_sandbox(self) -> bool:
		"""Whether sudo runs the sandbox command itself without a password, as a narrow NOPASSWD rule allows.

		A sandbox that fails after sudo let it run still counts, so check() reports the real error.
		"""
		try:
			r = subprocess.run(self.command(30), input=b'["true"]', capture_output=True, timeout=30)
		except subprocess.TimeoutExpired:
			return False
		return r.returncode == 0 or not r.stderr.startswith(b"sudo:")

	def run(self, argv: list[str], timeout: int | None = None) -> Result:
		timeout = max(1, min(timeout or self.profile.timeout, self.profile.max_timeout))
		self.ensure_sudo()
		# Deleted files that the command keeps open use disk without showing in scratch, so free space is checked too.
		allowance = self.profile.scratch_max - self.scratch_usage()[0]
		free_at_start = self._disk_free()
		# No new session: sudo finds its cached login through this terminal.
		proc = subprocess.Popen(
			self.command(deadline=timeout + DEADLINE_GRACE),
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
					if self.scratch_full() or free_at_start - self._disk_free() > allowance:
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
				# Killing sudo kills timeout, directly or through sudo's monitor, then bwrap and its PID namespace.
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
		for tool, package in (("bwrap", "bubblewrap"), ("setpriv", "util-linux"), ("unshare", "util-linux"),
				("timeout", "coreutils"), ("sudo", "sudo")):
			if not shutil.which(tool):
				raise SandboxError(f"Failed to find {tool} (package: {package})")
		for path in (sys.executable, CONFINE):
			if hidden_in_sandbox(path):
				raise SandboxError(f"Failed to start sandbox: {path} is under /tmp or /dev, which the sandbox replaces")
		if not self._systemd:
			self.notify("systemd is not running, so sandboxed commands have no memory or process limits.")
		result = self.run(["true"], timeout=30)
		if result.exit_code != 0:
			hint = ""
			if os.path.isdir("/sys/kernel/security/apparmor"):
				hint = (" If AppArmor confines bwrap or unshare (check `sudo aa-status | grep -E 'bwrap|unshare'`), "
					"that profile may block it.")
			raise SandboxError(f"Failed to start sandbox (exit {result.exit_code}): {result.output.strip()}{hint}")

	def _disk_free(self) -> int:
		st = os.statvfs(self.scratch)
		return st.f_bavail * st.f_frsize

	def scratch_full(self) -> bool:
		"""Whether scratch is past the profile's limits, counting a directory that cannot be read as past them."""
		size, entries, unreadable = self.scratch_usage()
		return unreadable or size > self.profile.scratch_max or entries > self.profile.scratch_files_max

	def scratch_usage(self) -> tuple[int, int, bool]:
		"""Disk bytes and entries in scratch, and whether some entry could not be measured."""
		size = entries = 0
		unreadable = False
		flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
		# A running command can swap a directory for a symlink, so the walk opens each level without following one.
		stack = []
		try:
			top = os.open(self.scratch, flags)
			stack.append((top, iter(os.listdir(top))))
			while stack:
				dir_fd, names = stack[-1]
				name = next(names, None)
				if name is None:
					os.close(stack.pop()[0])
					continue
				entries += 1
				try:
					st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
					size += st.st_blocks * 512
					if stat.S_ISDIR(st.st_mode):
						fd = os.open(name, flags, dir_fd=dir_fd)
						try:
							stack.append((fd, iter(os.listdir(fd))))
						except OSError:
							os.close(fd)
							raise
				except OSError as e:
					unreadable = unreadable or e.errno not in (errno.ENOENT, errno.ELOOP, errno.ENOTDIR)
		except OSError as e:
			unreadable = unreadable or e.errno != errno.ENOENT
		finally:
			for fd, _names in stack:
				os.close(fd)
		return size, entries, unreadable


def _quiet(argv: list[str]) -> bool:
	return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True).returncode == 0


def hidden_in_sandbox(path: str) -> bool:
	"""Whether path is under the host's /tmp or /dev, which the sandbox replaces with scratch and a private /dev."""
	return any(p == top or p.startswith(top + "/")
		for p in (os.path.abspath(path), os.path.realpath(path)) for top in ("/tmp", "/dev"))


def clean_scratch(root: Path, max_age: int = SCRATCH_MAX_AGE) -> None:
	"""Deletes old files from the scratch directories of chats that are not open, then empty directories.

	A file's age counts from its last change in scratch, the later of mtime and ctime. Symlinks are never followed.
	"""
	try:
		root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
	except OSError:
		return
	try:
		fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # makes chats being opened wait; see _lock_scratch
	except BlockingIOError:
		os.close(root_fd)
		return  # another Artifice is cleaning or opening a chat
	# The walk holds a descriptor per directory level, and scratch trees can be deep.
	soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
	if hard == resource.RLIM_INFINITY or hard > soft:
		resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 20), hard))
	cutoff = time.time() - max_age
	try:
		for name in os.listdir(root_fd):
			try:
				fd = _open_dir(name, dir_fd=root_fd)
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


def _open_dir(name: str | Path, dir_fd: int | None = None) -> int:
	"""Opens a directory without following symlinks, first giving its owner the rwx needed to list and empty it.

	The chmod goes through an O_PATH descriptor, so a directory swapped for a symlink cannot redirect it.
	"""
	path_fd = os.open(name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
	try:
		mode = os.fstat(path_fd).st_mode
		if mode & 0o700 != 0o700:
			os.chmod(f"/proc/self/fd/{path_fd}", stat.S_IMODE(mode) | 0o700)
		return os.open(f"/proc/self/fd/{path_fd}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
	finally:
		os.close(path_fd)


def _clean_dir(top_fd: int, cutoff: float) -> None:
	"""Deletes old files and then empty directories under top_fd, without following symlinks.

	The walk keeps its own stack because scratch trees can be thousands of levels deep.
	"""
	stack = [(top_fd, iter(os.listdir(top_fd)), "")]
	try:
		while stack:
			dir_fd, names, dir_name = stack[-1]
			name = next(names, None)
			if name is None:
				stack.pop()
				if stack:
					os.close(dir_fd)
					try:
						os.rmdir(dir_name, dir_fd=stack[-1][0])
					except OSError:
						pass
				continue
			try:
				st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
				if stat.S_ISDIR(st.st_mode):
					fd = _open_dir(name, dir_fd=dir_fd)
					try:
						stack.append((fd, iter(os.listdir(fd)), name))
					except OSError:
						os.close(fd)
						raise
				elif max(st.st_mtime, st.st_ctime) < cutoff:
					os.unlink(name, dir_fd=dir_fd)
			except OSError:
				pass
	finally:
		for fd, _names, _name in stack[1:]:
			os.close(fd)
