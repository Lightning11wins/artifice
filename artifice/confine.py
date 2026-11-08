"""Confines this process with Landlock and seccomp, then execs a command.

Runs inside the bwrap sandbox, closing the gaps a read-only bind mount leaves open:
FIFOs on the host filesystem stay writable, unix sockets (dbus, ssh-agent, ...) stay reachable,
and socket families without network namespaces, like vsock and Bluetooth, can still talk to other machines.

Usage: confine.py [--write DIR]... [--fsize BYTES] (--argv-stdin | -- COMMAND [ARG]...)

--argv-stdin reads the command as a JSON list from stdin, then gives it /dev/null as stdin.
"""
import ctypes
import json
import os
import platform
import resource
import signal
import struct
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]

PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2

# Landlock syscall numbers are shared by x86_64 and aarch64.
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

# Landlock filesystem rights that modify something, by the ABI version that introduced them.
FS_WRITE_FILE = 1 << 1
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13
FS_TRUNCATE = 1 << 14
WRITE_RIGHTS_V1 = (FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_CHAR | FS_MAKE_DIR
	| FS_MAKE_REG | FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_BLOCK | FS_MAKE_SYM)

# Seccomp: per-architecture audit value and syscall numbers.
ARCHES = {
	"x86_64": {"audit": 0xC000003E, "socket": 41, "open_by_handle_at": 304, "add_key": 248, "request_key": 249,
		"keyctl": 250, "fallocate": 285, "ioctl": 16, "fcntl": 72, "notify_init": (253, 294, 300)},
	"aarch64": {"audit": 0xC00000B7, "socket": 198, "open_by_handle_at": 265, "add_key": 217, "request_key": 218,
		"keyctl": 219, "fallocate": 47, "ioctl": 29, "fcntl": 25, "notify_init": (26, 262)},
}
IO_URING_SYSCALLS = (425, 426, 427)  # setup, enter, register; io_uring can open sockets past seccomp
X32_SYSCALL_BIT = 0x40000000
ALLOWED_FAMILIES = (2, 10, 16)  # AF_INET, AF_INET6, AF_NETLINK; all confined by the network namespace
FALLOC_FL_KEEP_SIZE = 0x01
F_SETLEASE = 1024
F_LAST_KNOWN = 1038  # F_SET_FILE_RW_HINT; later commands, such as delegations, may conflict with host writers too
PREALLOCATE_IOCTLS = (0x40305828, 0x4030582A, 0x40305839)  # FS_IOC_RESVSP, FS_IOC_RESVSP64, FS_IOC_ZERO_RANGE

BPF_LD_W_ABS = 0x20
BPF_JEQ_K = 0x15
BPF_JGE_K = 0x35
BPF_JSET_K = 0x45
BPF_RET_K = 0x06
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EACCES = 13
ENOSYS = 38
EOPNOTSUPP = 95


class SockFprog(ctypes.Structure):
	_fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_char_p)]


def fail(message: str) -> None:
	print(f"artifice-confine: {message}", file=sys.stderr)
	sys.exit(125)


def landlock(write_dirs: list[str]) -> None:
	abi = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)
	if abi < 1:
		fail(f"Failed to enable Landlock (errno {ctypes.get_errno()}); it must be in the kernel's lsm= list")
	rights = WRITE_RIGHTS_V1
	if abi >= 2:
		rights |= FS_REFER
	if abi >= 3:
		rights |= FS_TRUNCATE

	# Only write rights are handled, so reads and execution stay unrestricted.
	attr = struct.pack("Q", rights)
	ruleset = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, attr, len(attr), 0)
	if ruleset < 0:
		fail(f"Failed to create Landlock ruleset (errno {ctypes.get_errno()})")
	for path in write_dirs:
		fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
		rule = struct.pack("=Qi", rights, fd)
		if libc.syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, rule, 0) != 0:
			fail(f"Failed to add Landlock rule for {path} (errno {ctypes.get_errno()})")
		os.close(fd)
	if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0) != 0:
		fail(f"Failed to apply Landlock ruleset (errno {ctypes.get_errno()})")
	os.close(ruleset)


def seccomp_program(arch: dict[str, int]) -> bytes:
	def stmt(code: int, k: int) -> bytes:
		return struct.pack("=HBBI", code, 0, 0, k)

	def jump(code: int, k: int, jt: int, jf: int) -> bytes:
		return struct.pack("=HBBI", code, jt, jf, k)

	errno = lambda e: SECCOMP_RET_ERRNO | e
	prog = [
		stmt(BPF_LD_W_ABS, 4),  # seccomp_data.arch
		jump(BPF_JEQ_K, arch["audit"], 1, 0),
		stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
		stmt(BPF_LD_W_ABS, 0),  # seccomp_data.nr
	]
	if arch["audit"] == ARCHES["x86_64"]["audit"]:
		prog += [jump(BPF_JGE_K, X32_SYSCALL_BIT, 0, 1), stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS)]
	# open_by_handle_at opens files outside the sandbox's mounts when CAP_DAC_READ_SEARCH is held.
	# inotify and fanotify count against per-user limits that the user's host programs share.
	denied = (*IO_URING_SYSCALLS, arch["open_by_handle_at"], arch["add_key"], arch["request_key"], arch["keyctl"],
		*arch["notify_init"])
	for nr in denied:
		prog += [jump(BPF_JEQ_K, nr, 0, 1), stmt(BPF_RET_K, errno(ENOSYS))]
	# The kernel skips RLIMIT_FSIZE for fallocate with KEEP_SIZE, which could fill the disk in one call.
	prog += [
		jump(BPF_JEQ_K, arch["fallocate"], 0, 4),
		stmt(BPF_LD_W_ABS, 24),  # low word of seccomp_data.args[1], the mode
		jump(BPF_JSET_K, FALLOC_FL_KEEP_SIZE, 0, 1),
		stmt(BPF_RET_K, errno(EOPNOTSUPP)),
		stmt(BPF_LD_W_ABS, 0),  # seccomp_data.nr again
	]
	# These ioctls call fallocate with KEEP_SIZE.
	prog += [
		jump(BPF_JEQ_K, arch["ioctl"], 0, 1 + 2 * len(PREALLOCATE_IOCTLS)),
		stmt(BPF_LD_W_ABS, 24),  # low word of seccomp_data.args[1], the request
	]
	for request in PREALLOCATE_IOCTLS:
		prog += [jump(BPF_JEQ_K, request, 0, 1), stmt(BPF_RET_K, errno(EOPNOTSUPP))]
	prog.append(stmt(BPF_LD_W_ABS, 0))
	# A lease on a host file the user owns would make host programs block in open().
	prog += [
		jump(BPF_JEQ_K, arch["fcntl"], 0, 5),
		stmt(BPF_LD_W_ABS, 24),  # low word of seccomp_data.args[1], the command
		jump(BPF_JEQ_K, F_SETLEASE, 1, 0),
		jump(BPF_JGE_K, F_LAST_KNOWN + 1, 0, 1),
		stmt(BPF_RET_K, errno(EACCES)),
		stmt(BPF_LD_W_ABS, 0),
	]
	families = len(ALLOWED_FAMILIES)
	prog += [
		jump(BPF_JEQ_K, arch["socket"], 0, families + 2),
		stmt(BPF_LD_W_ABS, 16),  # low word of seccomp_data.args[0], the address family
	]
	for i, family in enumerate(ALLOWED_FAMILIES):
		prog.append(jump(BPF_JEQ_K, family, families - i, 0))
	prog += [
		stmt(BPF_RET_K, errno(EACCES)),
		stmt(BPF_RET_K, SECCOMP_RET_ALLOW),
	]
	return b"".join(prog)


def seccomp() -> None:
	machine = platform.machine()
	if machine not in ARCHES:
		fail(f"Failed to build seccomp filter: unsupported architecture {machine}")
	code = seccomp_program(ARCHES[machine])
	prog = SockFprog(len(code) // 8, code)
	if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(prog), 0, 0) != 0:
		fail(f"Failed to apply seccomp filter (errno {ctypes.get_errno()})")


def main() -> None:
	args = sys.argv[1:]
	write_dirs: list[str] = []
	fsize = resource.RLIM_INFINITY
	argv_stdin = False
	cwd = None
	while args and args[0] != "--":
		if args[0] == "--argv-stdin":
			argv_stdin, args = True, args[1:]
			continue
		flag, value, args = args[0], args[1], args[2:]
		if flag == "--write":
			write_dirs.append(value)
		elif flag == "--fsize":
			fsize = int(value)
		elif flag == "--chdir":
			cwd = value
		else:
			fail(f"Failed to parse arguments: unknown flag {flag}")
	if argv_stdin:
		command = json.loads(sys.stdin.read())
		devnull = os.open(os.devnull, os.O_RDONLY)
		os.dup2(devnull, 0)
		os.close(devnull)
	else:
		command = args[1:]
	if not command or not all(isinstance(a, str) for a in command):
		fail(f"Failed to parse arguments: bad command {command!r}")

	if cwd is not None:
		try:
			os.chdir(cwd)
		except OSError as e:
			fail(f"Failed to enter {cwd}: {e.strerror}")
	resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
	if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
		fail(f"Failed to set no_new_privs (errno {ctypes.get_errno()})")
	landlock(write_dirs)
	seccomp()
	# Python ignores these at startup, and an exec would pass that on: `yes | head -1` would never get SIGPIPE.
	for signum in (signal.SIGPIPE, signal.SIGXFSZ):
		signal.signal(signum, signal.SIG_DFL)
	try:
		os.execvp(command[0], command)
	except OSError as e:
		print(f"Failed to run {command[0]}: {e.strerror}", file=sys.stderr)
		sys.exit(127)



if __name__ == "__main__":
	main()
