#!/usr/bin/python3 -IS
"""Starts a sandboxed command for the user who ran sudo. Runs as root.

bin/install-sandbox installs a root-owned copy, with confine.py beside it, and lets the user run it with sudo
and no password. So the request can only choose what the user could choose anyway: the command runs as the user,
the scratch directory is opened as the user, and the limits only constrain the command.

The request is JSON on stdin, which keeps the command out of sudo's logs:
{"argv": [...], "deadline": seconds, "scratch": dir, "cwd": dir, "lang": "C.UTF-8",
	"memory_max": "8G", "tasks_max": n, "file_size_max": bytes, "shm_max": bytes}
"""
import json
import os
import pwd
import re
import shutil
import sys

CONFINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confine.py")
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"  # sudo may keep the user's PATH
FIELDS = {"argv": list, "deadline": int, "scratch": str, "cwd": str, "lang": str, "memory_max": str,
	"tasks_max": int, "file_size_max": int, "shm_max": int}


def fail(message: str) -> None:
	print(f"artifice-launcher: {message}", file=sys.stderr)
	sys.exit(125)


def systemd_running() -> bool:
	return shutil.which("systemd-run", path=PATH) is not None and os.path.isdir("/run/systemd/system")


def which(name: str) -> str:
	path = shutil.which(name, path=PATH)
	if path is None:
		fail(f"Failed to find {name} in {PATH}")
	return path


def read_request() -> dict:
	try:
		request = json.loads(sys.stdin.buffer.read())
	except (ValueError, RecursionError) as e:
		fail(f"Failed to parse request: {e}")
	if not isinstance(request, dict) or set(request) != set(FIELDS):
		fail(f"Failed to parse request: want fields {sorted(FIELDS)}, got {request!r:.200}")
	for key, kind in FIELDS.items():
		value = request[key]
		if type(value) is not kind or (kind is int and value < 1) or (kind is str and "\0" in value):
			fail(f"Failed to parse request: bad {key} {value!r:.200}")
	argv = request["argv"]
	if not argv or not all(isinstance(a, str) for a in argv):
		fail(f"Failed to parse request: bad argv {argv!r:.200}")
	if not re.fullmatch(r"[1-9][0-9]*[KMGT]?", request["memory_max"]):
		fail(f"Failed to parse request: bad memory_max {request['memory_max']!r:.200}")
	return request


def open_as_user(path: str, uid: int, gid: int, groups: list[int]) -> int:
	"""Opens directory path with the user's permissions, so a path only root can reach fails."""
	os.setgroups(groups)
	os.setegid(gid)
	os.seteuid(uid)
	try:
		return os.open(path, os.O_PATH | os.O_DIRECTORY)
	except OSError as e:
		fail(f"Failed to open scratch directory {path}: {e.strerror}")
	finally:
		os.seteuid(0)
		os.setegid(0)
		os.setgroups([])


def main() -> None:
	if os.geteuid() != 0 or "SUDO_UID" not in os.environ:
		fail("Failed to start: run this through sudo")
	uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
	if uid == 0:
		fail("Failed to start: the sandbox runs commands as the user who ran sudo, and that is root")
	request = read_request()
	user = pwd.getpwuid(uid)
	groups = os.getgrouplist(user.pw_name, gid)
	scratch = open_as_user(request["scratch"], uid, gid, groups)
	os.set_inheritable(scratch, True)

	env = {
		"PATH": PATH,
		"HOME": user.pw_dir,
		"USER": user.pw_name,
		"LANG": request["lang"],
		"TERM": "dumb",
		"NO_COLOR": "1",
		"PAGER": "cat",
		"GIT_PAGER": "cat",
		"GIT_OPTIONAL_LOCKS": "0",
		"TMPDIR": "/tmp",
		"XDG_CACHE_HOME": "/tmp/.cache",
		"PYTHONDONTWRITEBYTECODE": "1",
	}
	setpriv = which("setpriv")
	# timeout runs as root outside the sandbox, so it ends the sandbox even if Artifice was killed.
	# It dies with its parent, sudo or, under use_pty, sudo's monitor process, which kills it when sudo dies.
	# bwrap dies with it through --die-with-parent.
	# The second setpriv makes the rest of the chain die with timeout, before bwrap's --die-with-parent applies.
	cmd = [setpriv, "--pdeathsig", "KILL", which("timeout"), "--foreground", "-s", "KILL", str(request["deadline"]),
		setpriv, "--pdeathsig", "KILL"]
	if systemd_running():
		# A system scope, because sudo's PAM session can move it out of a user scope.
		cmd += [
			which("systemd-run"), "--scope", "--quiet", "--collect",
			"-p", f"MemoryMax={request['memory_max']}",
			"-p", "MemorySwapMax=0",
			"-p", f"TasksMax={request['tasks_max']}",
		]
	cmd += [
		which("bwrap"),
		"--ro-bind", "/", "/",
		"--dev", "/dev",
		# Writable shared memory, for multiprocessing and POSIX semaphores.
		"--perms", "1777", "--size", str(request["shm_max"]), "--tmpfs", "/dev/shm",
		# The host's /proc shows every process, but PIDs given to kill or ptrace resolve in the new PID namespace.
		"--ro-bind", "/proc", "/proc",
		"--bind-fd", str(scratch), "/tmp",
		"--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
		"--die-with-parent",
		"--new-session",
		"--clearenv",
	]
	for key, value in env.items():
		cmd += ["--setenv", key, value]
	for cap in ("CAP_DAC_READ_SEARCH", "CAP_SETUID", "CAP_SETGID", "CAP_SETPCAP", "CAP_SYS_ADMIN"):
		cmd += ["--cap-add", cap]
	# bwrap's mounts receive the host's new mounts, without the read-only flag. A private mount namespace does not.
	cmd += ["--chdir", "/", "--", which("unshare"), "--mount", "--propagation", "private", "--"]
	cmd += [
		setpriv, f"--reuid={uid}", f"--regid={gid}",
		f"--groups={','.join(map(str, groups))}" if groups else "--clear-groups",
		"--inh-caps=-all,+dac_read_search",
		"--ambient-caps=-all,+dac_read_search",
		"--bounding-set=-all,+dac_read_search",
		"--",
	]
	# confine.py enters cwd as the user, who can enter FUSE mounts that root cannot.
	cmd += [sys.executable, "-I", "-S", CONFINE, "--chdir", request["cwd"], "--write", "/tmp", "--write", "/dev"]
	cmd += ["--fsize", str(request["file_size_max"]), "--argv-stdin"]

	argv = os.memfd_create("argv")
	os.write(argv, json.dumps(request["argv"]).encode())
	os.lseek(argv, 0, os.SEEK_SET)
	os.dup2(argv, 0)
	os.close(argv)
	os.execv(cmd[0], cmd)


if __name__ == "__main__":
	main()
