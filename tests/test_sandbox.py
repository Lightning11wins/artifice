"""Escape tests for the read-only sandbox. Every attempt to change the host must fail.

Run with: python3 -m unittest discover -s tests -v
Needs sudo without a password prompt (run `sudo -v` first), and cannot run inside another sandbox.
"""
import ctypes
import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import artifice.sandbox  # noqa: E402
from artifice.sandbox import READ_ONLY, Profile, Result, Sandbox, SandboxError, clean_scratch  # noqa: E402

# Host directory the sandbox can see but must not modify. Under $HOME, not /tmp, which the sandbox replaces.
HOST_DIR = Path.home() / ".cache" / "artifice-tests"
# The repo's launcher, which needs the sudo ticket from `sudo -v`, so tests need no reinstall.
LAUNCHER = str(Path(__file__).resolve().parent.parent / "artifice" / "launcher.py")
artifice.sandbox.LAUNCHER = LAUNCHER


def launcher_copy(path_prefix: str = "", text: str | None = None) -> str:
	"""A copy of the launcher with confine.py beside it, looking for tools in path_prefix's directories first."""
	d = Path(tempfile.mkdtemp(dir=HOST_DIR))
	text = text if text is not None else Path(LAUNCHER).read_text()
	(d / "launcher.py").write_text(text.replace('PATH = "', f'PATH = "{path_prefix}', 1))
	(d / "launcher.py").chmod(0o755)
	shutil.copy(Path(LAUNCHER).with_name("confine.py"), d / "confine.py")
	return str(d / "launcher.py")


class SandboxEscapeTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		HOST_DIR.mkdir(parents=True, exist_ok=True)
		cls.host = Path(tempfile.mkdtemp(dir=HOST_DIR))
		cls.scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		cls.sb = Sandbox(READ_ONLY, cls.scratch, str(cls.host))
		cls.sb.check()
		(cls.host / "file").write_text("original\n")
		cls.root_file = HOST_DIR / f"root-only-{os.getpid()}"
		subprocess.run(["sudo", "-n", "sh", "-c", f"umask 077; echo root-secret > {cls.root_file}"], check=True)

	@classmethod
	def tearDownClass(cls):
		subprocess.run(["sudo", "-n", "rm", "-f", str(cls.root_file)], check=True)
		shutil.rmtree(HOST_DIR)

	def sh(self, command: str, timeout: int = 20):
		return self.sb.run(["bash", "-c", command], timeout=timeout)

	def py(self, code: str, timeout: int = 20):
		return self.sb.run([sys.executable, "-I", "-c", code], timeout=timeout)

	def skipIfMissing(self, tool: str):
		if tool not in ("echo", "touch", "rm", "mv", "mkdir", "ln", "chmod", "truncate") and not shutil.which(tool):
			self.skipTest(f"{tool} is not installed")

	def assertUnchanged(self):
		self.assertEqual((self.host / "file").read_text(), "original\n")
		self.assertEqual(sorted(p.name for p in self.host.iterdir() if p.name != "fifo" and p.name != "sock"), ["file"])

	def test_reads_work(self):
		r = self.sh(f"cat {self.host}/file && head -c 10 /etc/os-release")
		self.assertEqual(r.exit_code, 0, r.output)
		self.assertIn("original", r.output)

	def test_reads_root_only_files(self):
		self.assertNotEqual(subprocess.run(["cat", str(self.root_file)], capture_output=True).returncode, 0)
		r = self.sh(f"cat {self.root_file}")
		self.assertEqual(r.output.strip(), "root-secret")
		r = self.sh(f"echo x > {self.root_file}")
		self.assertNotEqual(r.exit_code, 0, r.output)

	def test_runs_as_user_with_only_dac_read_search(self):
		r = self.sh("id -u; grep -E '^Cap(Inh|Prm|Eff|Bnd|Amb)' /proc/self/status")
		lines = r.output.split("\n")
		self.assertEqual(lines[0], str(os.getuid()), r.output)
		for line in lines[1:6]:
			self.assertEqual(int(line.split()[1], 16), 1 << 2, r.output)  # CAP_DAC_READ_SEARCH

	def test_open_by_handle_at_escape(self):
		# With CAP_DAC_READ_SEARCH, a handle opened through the writable /tmp mount would bypass the read-only one.
		r = self.py(f"""
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
class Handle(ctypes.Structure):
	_fields_ = [("size", ctypes.c_uint), ("type", ctypes.c_int), ("data", ctypes.c_ubyte * 128)]
h = Handle(128)
mount_id = ctypes.c_int()
print("name", libc.name_to_handle_at(-100, b"{self.host}/file", ctypes.byref(h), ctypes.byref(mount_id), 0))
mount_fd = os.open("/tmp", os.O_RDONLY | os.O_DIRECTORY)
fd = libc.open_by_handle_at(mount_fd, ctypes.byref(h), os.O_RDWR)
print("open", fd, ctypes.get_errno())
if fd >= 0:
	os.write(fd, b"pwned")
""")
		self.assertIn("name 0", r.output)
		self.assertRegex(r.output, r"(?m)^open -1 38$")  # ENOSYS from seccomp
		self.assertUnchanged()

	def test_linkat_empty_path_escape(self):
		# CAP_DAC_READ_SEARCH also allows linking an open file into /tmp, where Landlock allows writes.
		r = self.py(f"""
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
fd = os.open("{self.host}/file", os.O_RDONLY)
print("link", libc.linkat(fd, b"", -100, b"/tmp/linked", 0x1000), ctypes.get_errno())
open("/tmp/linked", "w").write("pwned")
""")
		self.assertIn("link -1", r.output)
		self.assertUnchanged()

	def test_write_create_delete_rename_fail(self):
		for cmd in [
			f"echo x > {self.host}/file",
			f"echo x >> {self.host}/file",
			f"touch {self.host}/new",
			f"rm -f {self.host}/file",
			f"mv {self.host}/file {self.host}/moved",
			f"mkdir {self.host}/dir",
			f"ln -s file {self.host}/link",
			f"chmod 777 {self.host}/file",
			f"touch -d 2000-01-01 {self.host}/file",
			f"truncate -s 0 {self.host}/file",
			f"mkfifo {self.host}/f2",
			f"setfattr -n user.x -v 1 {self.host}/file",
		]:
			with self.subTest(cmd=cmd):
				self.skipIfMissing(cmd.split()[0])
				r = self.sh(cmd)
				self.assertNotEqual(r.exit_code, 0, r.output)
		self.assertUnchanged()

	def test_scratch_is_writable_and_private(self):
		r = self.sh("echo hi > /tmp/a && mkdir -p /tmp/d && cat /tmp/a")
		self.assertEqual(r.exit_code, 0, r.output)
		self.assertEqual((self.scratch / "a").read_text(), "hi\n")
		with tempfile.NamedTemporaryFile(dir="/tmp", prefix="artifice-host-") as host_file:
			r = self.sh(f"ls {host_file.name}")
		self.assertNotEqual(r.exit_code, 0, r.output)

	def test_links_out_of_scratch_fail(self):
		r = self.sh(f"ln {self.host}/file /tmp/hard")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.sh(f"ln -sf {self.host}/file /tmp/sym && echo pwned > /tmp/sym")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.sh(f"echo x > /tmp/a && cp /tmp/a {self.host}/a")
		self.assertNotEqual(r.exit_code, 0, r.output)
		self.assertUnchanged()

	def test_unix_socket_on_host_fs(self):
		path = self.host / "sock"
		server = socket.socket(socket.AF_UNIX)
		server.bind(str(path))
		server.listen(1)
		server.settimeout(5)
		got: list = []

		def accept():
			try:
				conn, _ = server.accept()
				got.append(conn.recv(100))
			except OSError:
				pass

		t = threading.Thread(target=accept)
		t.start()
		r = self.py(f"import socket; s = socket.socket(socket.AF_UNIX); s.connect({str(path)!r}); s.send(b'x')")
		server.close()
		t.join()
		path.unlink()
		self.assertNotEqual(r.exit_code, 0, r.output)
		self.assertEqual(got, [])

	def test_abstract_socket(self):
		server = socket.socket(socket.AF_UNIX)
		server.bind("\0artifice-test")
		server.listen(1)
		r = self.py("import socket; s = socket.socket(socket.AF_UNIX); s.connect('\\0artifice-test')")
		server.close()
		self.assertNotEqual(r.exit_code, 0, r.output)

	def test_socket_families(self):
		# Only families confined by the network namespace are allowed. vsock and Bluetooth reach other machines.
		r = self.py("""
import socket
for family in (1, 40, 31, 38, 17):  # AF_UNIX, AF_VSOCK, AF_BLUETOOTH, AF_ALG, AF_PACKET
	try:
		socket.socket(family, socket.SOCK_STREAM if family != 17 else socket.SOCK_RAW).close()
		print(family, 0)
	except OSError as e:
		print(family, e.errno)
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_NETLINK):
	socket.socket(family, socket.SOCK_DGRAM).close()
print("ok")
""")
		self.assertEqual(r.output.split(), ["1", "13", "40", "13", "31", "13", "38", "13", "17", "13", "ok"], r.output)

	def test_fifo_on_host_fs(self):
		fifo = self.host / "fifo"
		os.mkfifo(fifo)
		got: list = []

		def reader():
			fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
			deadline = time.time() + 5
			while time.time() < deadline:
				try:
					data = os.read(fd, 100)
					if data:
						got.append(data)
				except BlockingIOError:
					pass
				time.sleep(0.05)
			os.close(fd)

		t = threading.Thread(target=reader)
		t.start()
		time.sleep(0.2)
		r = self.sh(f"echo pwned > {fifo}", timeout=5)
		t.join()
		fifo.unlink()
		self.assertNotEqual(r.exit_code, 0, r.output)
		self.assertEqual(got, [])

	def test_no_network(self):
		r = self.py("import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.py("import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'x', ('1.1.1.1', 53))")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.sh("cat /proc/net/dev")
		self.assertNotRegex(r.output, r"\n\s*(eth|en|wl|virbr)", r.output)

	def test_host_processes_visible_but_untouchable(self):
		victim = subprocess.Popen(["sleep", "60"])
		pid = victim.pid
		oom_before = Path(f"/proc/{pid}/oom_score_adj").read_text()
		try:
			r = self.sh(f"ps -p {pid} -o comm=")
			self.assertEqual(r.output.strip(), "sleep", r.output)
			for cmd in [
				f"kill -9 {pid}",
				f"renice -n 19 -p {pid}",
				f"echo 1000 > /proc/{pid}/oom_score_adj",
				f"python3 -c \"open('/proc/{pid}/mem', 'r+b')\"",
				f"cat /proc/{pid}/environ",
				f"timeout 5 gdb -p {pid} -batch -ex 'call (int)kill(0, 9)'",
			]:
				with self.subTest(cmd=cmd):
					if "gdb" in cmd:
						self.skipIfMissing("gdb")
					self.assertNotEqual(self.sh(cmd).exit_code, 0)
			time.sleep(0.2)
			self.assertIsNone(victim.poll())
			self.assertEqual(Path(f"/proc/{pid}/oom_score_adj").read_text(), oom_before)
		finally:
			victim.kill()
			victim.wait()

	def test_no_privilege_escalation(self):
		r = self.sh("sudo -n true")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.sh("grep NoNewPrivs /proc/self/status")
		self.assertIn("1", r.output)

	def test_mounts_outside_tmp_and_dev_are_read_only(self):
		r = self.sh("cat /proc/self/mountinfo")
		mounts = [line.split() for line in r.output.splitlines()]
		self.assertIn("/proc", [m[4] for m in mounts], r.output)
		for m in mounts:
			fstype = m[m.index("-") + 1]
			if fstype == "autofs":
				continue  # an automount trigger, which only the automount daemon can change
			if not any(m[4] == top or m[4].startswith(top + "/") for top in ("/tmp", "/dev")):
				self.assertIn("ro", m[5].split(","), " ".join(m))

	def test_kernel_interfaces_read_only(self):
		for cmd in ["echo 1 > /proc/sys/vm/drop_caches", "echo 1 > /sys/kernel/mm/ksm/run",
				"echo x > /proc/sys/kernel/hostname"]:
			with self.subTest(cmd=cmd):
				self.assertNotEqual(self.sh(cmd).exit_code, 0)

	def test_blocked_syscalls(self):
		code = ("import ctypes, platform; libc = ctypes.CDLL(None, use_errno=True); "
			"nr = {'x86_64': {'uring': 425, 'add_key': 248, 'request_key': 249, 'keyctl': 250, 'handle': 304, "
			"'inotify': 294, 'fanotify': 300}, "
			"'aarch64': {'uring': 425, 'add_key': 217, 'request_key': 218, 'keyctl': 219, 'handle': 265, "
			"'inotify': 26, 'fanotify': 262}}"
			"[platform.machine()]; "
			"buf = ctypes.create_string_buffer(128); "
			"calls = [(nr['uring'], 8, buf), (nr['add_key'], b'user', b'artifice', b'x', 1, -3), "
			"(nr['request_key'], b'user', b'artifice', None, -3), (nr['keyctl'], 0, -4, 0), "
			"(nr['handle'], -100, buf, 0), (nr['inotify'], 0), (nr['fanotify'], 0x200, 0)]; "
			"print(*[(ctypes.set_errno(0), f'{libc.syscall(*c)}/{ctypes.get_errno()}')[1] for c in calls])")
		r = self.py(code)
		self.assertEqual(r.output.split(), ["-1/38"] * 7, r.output)  # ENOSYS from seccomp
		if os.uname().machine == "x86_64":
			r = self.py("import ctypes; libc = ctypes.CDLL(None, use_errno=True); print(libc.syscall(253), ctypes.get_errno())")
			self.assertEqual(r.output.split(), ["-1", "38"], r.output)

	def test_posix_message_queues_cannot_be_created(self):
		# Queue bytes count against a per-user limit that host programs share; Landlock refuses the create.
		nr = {"x86_64": 240, "aarch64": 180}[os.uname().machine]
		r = self.py(f"import ctypes; libc = ctypes.CDLL(None, use_errno=True); "
			f"print(libc.syscall({nr}, b'artifice', 0o100 | 2, 0o600, None), ctypes.get_errno())")
		self.assertEqual(r.output.split(), ["-1", "13"], r.output)  # EACCES

	def test_other_syscall_abis_are_killed(self):
		if os.uname().machine != "x86_64":
			self.skipTest("x86_64 only")
		code = (
			"import ctypes, mmap, sys\n"
			"libc = ctypes.CDLL(None, use_errno=True)\n"
			"if sys.argv[1] == 'x32':\n"
			"	print(libc.syscall(0x40000000 | 39), flush=True)\n"
			"else:\n"
			"	buf = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)\n"
			"	buf.write(b'\\xb8\\x14\\x00\\x00\\x00\\xcd\\x80\\xc3')  # mov eax, 20 (i386 getpid); int 0x80; ret\n"
			"	print(ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.addressof(ctypes.c_char.from_buffer(buf)))(), flush=True)\n"
		)
		for abi in ("x32", "i386"):
			with self.subTest(abi=abi):
				r = self.sh(f"python3 -I -c {shlex.quote(code)} {abi} 2>/dev/null; echo $?")
				self.assertEqual(r.output.split()[-1], "159", r.output)  # SIGSYS

	def test_memory_and_task_limits(self):
		sb = Sandbox(Profile(name="small", memory_max="64M", tasks_max=32), self.scratch, str(self.host))
		if not sb._systemd:
			self.skipTest("needs systemd")
		r = sb.run([sys.executable, "-I", "-c", "b = b'x' * (256 << 20); print('allocated')"], timeout=20)
		self.assertNotIn("allocated", r.output)
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = sb.run([sys.executable, "-I", "-c", """
import os, time
n = 0
try:
	for _ in range(100):
		if os.fork() == 0:
			os.close(1)
			os.close(2)
			time.sleep(30)
			os._exit(0)
		n += 1
except OSError as e:
	print("blocked", n, e.errno)
"""], timeout=20)
		self.assertRegex(r.output, r"^blocked \d+ 11", r.output)  # EAGAIN
		self.assertLess(int(r.output.split()[1]), 32)

	def test_scratch_limits(self):
		# Its own scratch, so a failure here cannot leave the shared one full.
		scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		self.addCleanup(subprocess.run, ["chmod", "-R", "u+rwx", str(scratch)])
		sb = Sandbox(Profile(name="small", scratch_max=20 << 20, scratch_files_max=500), scratch, str(self.host))
		r = sb.run(["bash", "-c", "for i in $(seq 30); do head -c 1M /dev/urandom > /tmp/big$i; sleep 0.1; done; echo done"])
		self.assertTrue(r.scratch_full, r.output)
		self.assertIsNone(r.exit_code)
		self.assertNotIn("done", r.output)
		r = sb.run(["bash", "-c", "mkdir -p /tmp/many && cd /tmp/many && for i in $(seq 2000); do : > $i; sleep 0.001; done; echo done"])
		self.assertTrue(r.scratch_full, r.output)
		self.assertNotIn("done", r.output)
		self.assertEqual(sb.run(["rm", "-rf", "/tmp/many"] + [f"/tmp/big{i}" for i in range(31)]).exit_code, 0)
		self.assertFalse(sb.run(["true"]).scratch_full)
		# A command shorter than the polling interval is caught when it exits.
		r = sb.run(["fallocate", "-l", "30M", "/tmp/fast"])
		self.assertTrue(r.scratch_full, r.output)
		sb.run(["rm", "/tmp/fast"])
		# A directory the host cannot read counts as full.
		r = sb.run(["bash", "-c", "mkdir /tmp/hidden && chmod 0 /tmp/hidden"])
		self.assertTrue(r.scratch_full, r.output)
		self.assertFalse(sb.run(["bash", "-c", "chmod 700 /tmp/hidden && rmdir /tmp/hidden"]).scratch_full)
		# So does a directory it can list but not search.
		r = sb.run(["bash", "-c", "mkdir /tmp/ro && head -c 30M /dev/zero > /tmp/ro/f && chmod 400 /tmp/ro"])
		self.assertTrue(r.scratch_full, r.output)
		self.assertFalse(sb.run(["bash", "-c", "chmod 700 /tmp/ro && rm -rf /tmp/ro"]).scratch_full)
		# And a tree nested past PATH_MAX.
		deep = "cd /tmp && n=$(printf 'd%.0s' $(seq 200)) && for i in $(seq 25); do mkdir $n && cd $n; done && head -c 30M /dev/zero > f"
		r = sb.run(["bash", "-c", deep])
		self.assertTrue(r.scratch_full, r.output)
		self.assertFalse(sb.run(["bash", "-c", "rm -rf /tmp/ddd*"]).scratch_full)

	def test_sigterm_and_sighup_kill_the_running_command(self):
		root = Path(__file__).resolve().parent.parent
		code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from pathlib import Path
from artifice.cli import exit_on_sigterm
from artifice.sandbox import READ_ONLY, Sandbox
import artifice.sandbox
artifice.sandbox.LAUNCHER = {LAUNCHER!r}
exit_on_sigterm()
Sandbox(READ_ONLY, Path({str(self.scratch)!r}), {str(self.host)!r}).run(["sleep", "4279"], timeout=60)
"""
		for signum in (signal.SIGTERM, signal.SIGHUP):
			with self.subTest(signal=signum.name):
				parent = subprocess.Popen([sys.executable, "-I", "-c", code])
				time.sleep(2)
				parent.send_signal(signum)
				self.assertEqual(parent.wait(10), 128 + signum)
				time.sleep(0.5)
				left = subprocess.run(["pgrep", "-f", "sleep 427[9]"], capture_output=True, text=True)
				self.assertEqual(left.stdout, "")

	def test_command_dies_when_artifice_is_killed(self):
		root = Path(__file__).resolve().parent.parent
		code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from pathlib import Path
from artifice.sandbox import READ_ONLY, Sandbox
import artifice.sandbox
artifice.sandbox.LAUNCHER = {LAUNCHER!r}
# A deadline kept inside the sandbox could be stopped like this.
Sandbox(READ_ONLY, Path({str(self.scratch)!r}), {str(self.host)!r}).run(
	["bash", "-c", "kill -STOP $PPID 2>/dev/null; exec sleep 4283"], timeout=2)
"""
		parent = subprocess.Popen([sys.executable, "-I", "-c", code])
		time.sleep(1.5)
		parent.kill()
		parent.wait()
		deadline = time.monotonic() + 2 + artifice.sandbox.DEADLINE_GRACE + 10
		while time.monotonic() < deadline:
			left = subprocess.run(["pgrep", "-f", "sleep 428[3]"], capture_output=True, text=True).stdout
			if not left:
				break
			time.sleep(0.5)
		self.assertEqual(left, "")

	def test_a_killed_command_is_gone_when_run_returns(self):
		# In a terminal, sudo's use_pty puts a monitor process between sudo and the sandbox.
		root = Path(__file__).resolve().parent.parent
		code = f"""
import os, sys
sys.path.insert(0, {str(root)!r})
from pathlib import Path
from artifice.sandbox import READ_ONLY, Sandbox
import artifice.sandbox
artifice.sandbox.LAUNCHER = {LAUNCHER!r}
Sandbox.scratch_full = lambda self: False  # so nothing delays the check below
Sandbox(READ_ONLY, Path({str(self.scratch)!r}), {str(self.host)!r}).run(["sleep", "4297"], timeout=1)
left = []
for pid in filter(str.isdigit, os.listdir("/proc")):
	try:
		if Path(f"/proc/{{pid}}/cmdline").read_bytes() == b"sleep\\x004297\\x00":
			left.append(pid)
	except OSError:
		pass
print("left:", left)
"""
		argv = [sys.executable, "-I", "-c", code]
		for terminal in (False, True):
			with self.subTest(terminal=terminal):
				r = subprocess.run(["script", "-qec", shlex.join(argv), "/dev/null"] if terminal else argv,
					stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
				self.assertIn("left: []", r.stdout, r.stdout + r.stderr)

	def test_a_command_cannot_write_to_the_users_terminal(self):
		# The sandbox keeps Artifice's session for sudo, so only bwrap's new session hides the terminal.
		root = Path(__file__).resolve().parent.parent
		code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from pathlib import Path
from artifice.sandbox import READ_ONLY, Sandbox
import artifice.sandbox
artifice.sandbox.LAUNCHER = {LAUNCHER!r}
r = Sandbox(READ_ONLY, Path({str(self.scratch)!r}), {str(self.host)!r}).run(
	["sh", "-c", "printf 'RAW%sTTY' - > /dev/tty && echo opened || echo refused"])
print("result:", r.output.strip().splitlines()[-1])
"""
		argv = [sys.executable, "-I", "-c", code]
		r = subprocess.run(["script", "-qec", shlex.join(argv), "/dev/null"],
			stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
		self.assertIn("result: refused", r.stdout, r.stdout + r.stderr)
		self.assertNotIn("RAW-TTY", r.stdout)

	def test_a_sandbox_interrupted_while_starting_does_not_outlive_it(self):
		# bwrap starting late leaves a window before --die-with-parent applies.
		wrapper = Path(tempfile.mkdtemp(dir=HOST_DIR)) / "bwrap"
		wrapper.write_text(f"#!/bin/sh\nsleep 2\nexec {shutil.which('bwrap')} \"$@\"\n")
		wrapper.chmod(0o755)
		sb = Sandbox(READ_ONLY, self.scratch, str(self.host))
		sb.launcher = launcher_copy(f"{wrapper.parent}:")
		sb.run(["sleep", "4303"], timeout=1)
		time.sleep(4)
		left = subprocess.run(["pgrep", "-x", "-f", "sleep 4303"], capture_output=True, text=True).stdout.split()
		for pid in left:
			os.kill(int(pid), signal.SIGKILL)
		self.assertEqual(left, [])

	def test_file_size_limit_and_default_signals(self):
		sb = Sandbox(Profile(name="small", file_size_max=1 << 20), self.scratch, str(self.host))
		r = sb.run(["bash", "-c", "head -c 2M /dev/zero > /tmp/big; echo $?; stat -c %s /tmp/big; rm /tmp/big; "
			"yes | head -1 > /dev/null; echo ${PIPESTATUS[0]}; "
			": > /tmp/kept; fallocate -n -l 10M /tmp/kept 2>/dev/null; echo $?; du -k /tmp/kept; rm /tmp/kept"])
		# fallocate --keep-size skips RLIMIT_FSIZE in the kernel, so seccomp refuses it.
		self.assertRegex(r.output, r"(?m)^153\n1048576\n141\n1\n0\t/tmp/kept$")

	def test_host_mounts_made_during_a_command_stay_out(self):
		# Landlock does not cover chmod, chown, utime or xattrs, so a writable host mount must not appear.
		base = Path(tempfile.mkdtemp(dir=HOST_DIR))
		mnt = base / "mnt"
		mnt.mkdir()
		results = []
		sb = Sandbox(READ_ONLY, self.scratch, str(base))
		worker = threading.Thread(target=lambda: results.append(sb.run(["bash", "-c",
			"sleep 3; chmod 600 mnt/f; touch -d 2000-01-01 mnt/f; ls mnt"])))
		worker.start()
		time.sleep(1)
		subprocess.run(["sudo", "-n", "mount", "-t", "tmpfs", "-o", "mode=777,size=1m", "artifice-test", str(mnt)], check=True)
		self.addCleanup(subprocess.run, ["sudo", "-n", "umount", str(mnt)])
		(mnt / "f").write_text("x")
		(mnt / "f").chmod(0o644)
		worker.join(30)
		st = (mnt / "f").stat()
		self.assertEqual(st.st_mode & 0o777, 0o644, results[0].output)
		self.assertGreater(st.st_mtime, time.time() - 60)

	def test_cwd_on_a_fuse_mount_root_cannot_enter(self):
		doc = f"/run/user/{os.getuid()}/doc"
		if not os.path.ismount(doc) or subprocess.run(["sudo", "-n", "test", "-x", doc]).returncode == 0:
			self.skipTest(f"{doc} is not a FUSE mount that root cannot enter")
		r = Sandbox(READ_ONLY, self.scratch, doc).run(["pwd"])
		self.assertEqual(r.output.strip(), doc)

	def test_signals_to_the_process_group_stay_in_the_command(self):
		code = ("import os, signal\n"
			"for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\n"
			"    signal.signal(s, signal.SIG_IGN)\n"
			"    os.kill(0, s)\n"
			"print('survived')")
		r = self.sb.run(["python3", "-c", code])
		self.assertEqual((r.output, r.exit_code), ("survived\n", 0))

	def test_negative_timeout_is_clamped(self):
		r = self.sh("sleep 0.2; echo done", timeout=-5)
		self.assertFalse(r.timed_out)
		self.assertIn("done", r.output)

	def test_python_under_host_tmp_is_refused(self):
		with mock.patch.object(sys, "executable", "/tmp/venv/bin/python3"):
			with self.assertRaisesRegex(SandboxError, "under /tmp"):
				self.sb.check()

	def test_a_missing_or_stale_launcher_is_refused(self):
		sb = Sandbox(READ_ONLY, self.scratch, str(self.host))
		sb.launcher = str(HOST_DIR / "missing" / "launcher.py")
		with self.assertRaisesRegex(SandboxError, "Failed to find .*install-sandbox"):
			sb.check()
		sb.launcher = launcher_copy(text=Path(LAUNCHER).read_text() + "\n")
		with self.assertRaisesRegex(SandboxError, "launcher.py differs from .*install-sandbox"):
			sb.check()
		sb.launcher = launcher_copy()
		Path(sb.launcher).with_name("confine.py").write_text("")
		with self.assertRaisesRegex(SandboxError, "confine.py differs from .*install-sandbox"):
			sb.check()

	def test_a_sudo_password_prompt_says_to_install(self):
		sb = Sandbox(READ_ONLY, self.scratch, str(self.host))
		with mock.patch.object(sb, "run", return_value=Result("sudo: a password is required\n", 1)):
			with self.assertRaisesRegex(SandboxError, "password is required.*install-sandbox"):
				sb.check()

	def test_launcher_rejects_bad_requests(self):
		good = self.sb.request(["true"], deadline=30)
		for change in ({"deadline": True}, {"deadline": 0}, {"tasks_max": "512"}, {"argv": []}, {"argv": [1]},
				{"memory_max": "8G -p Delegate=yes"}, {"memory_max": "8G\nDelegate=yes"}, {"cwd": "/\0"}, {"extra": 1}):
			with self.subTest(change=change):
				r = subprocess.run(["sudo", "-n", LAUNCHER], input=json.dumps(good | change), capture_output=True, text=True)
				self.assertEqual((r.returncode, r.stdout), (125, ""), r.stderr)
				self.assertIn("Failed to parse request", r.stderr)
		del good["lang"]
		r = subprocess.run(["sudo", "-n", LAUNCHER], input=json.dumps(good), capture_output=True, text=True)
		self.assertIn("Failed to parse request", r.stderr)
		r = subprocess.run([LAUNCHER], input=json.dumps(good), capture_output=True, text=True)
		self.assertIn("run this through sudo", r.stderr)

	def test_launcher_refuses_a_scratch_only_root_can_reach(self):
		# The scratch is writable in the sandbox, so a world-writable directory behind a root-only one must stay shut.
		locked = HOST_DIR / f"locked-{os.getpid()}"
		inner = locked / "inner"
		subprocess.run(["sudo", "-n", "sh", "-c", f"mkdir -m 700 {locked} && mkdir -m 777 {inner}"], check=True)
		self.addCleanup(subprocess.run, ["sudo", "-n", "rm", "-rf", str(locked)])
		request = self.sb.request(["sh", "-c", "echo pwned > /tmp/f"], deadline=30) | {"scratch": str(inner)}
		r = subprocess.run(["sudo", "-n", LAUNCHER], input=json.dumps(request), capture_output=True, text=True)
		self.assertNotEqual(r.returncode, 0)
		self.assertIn("Failed to open scratch directory", r.stderr)
		self.assertNotEqual(subprocess.run(["sudo", "-n", "test", "-e", str(inner / "f")]).returncode, 0)

	def test_the_command_inherits_only_stdio(self):
		r = self.py("""
import os
fds = {}
for fd in os.listdir("/proc/self/fd"):
	try:
		fds[fd] = os.readlink(f"/proc/self/fd/{fd}")
	except FileNotFoundError:
		pass  # the listing's own descriptor
print(sorted(fds), fds)
""")
		self.assertTrue(r.output.startswith("['0', '1', '2'] "), r.output)

	def test_preallocate_ioctls_are_refused(self):
		# These ioctls call fallocate with KEEP_SIZE, which skips RLIMIT_FSIZE.
		code = (
			"import fcntl, os, struct\n"
			"for request in (0x40305828, 0x4030582A, 0x40305839):\n"
			"	fd = os.open('/tmp/resv', os.O_RDWR | os.O_CREAT)\n"
			"	try:\n"
			"		fcntl.ioctl(fd, request, struct.pack('hhqqiI4i', 0, 0, 0, 50 << 20, 0, 0, 0, 0, 0, 0))\n"
			"	except OSError as e:\n"
			"		print(e.strerror)\n"
			"	print(os.stat('/tmp/resv').st_blocks)\n"
			"	os.close(fd)\n"
			"	os.unlink('/tmp/resv')\n"
		)
		sb = Sandbox(Profile(name="small", file_size_max=1 << 20), self.scratch, str(self.host))
		r = sb.run(["python3", "-c", code])
		self.assertEqual(r.output.split("\n"), ["Operation not supported", "0"] * 3 + [""], r.output)

	def test_leases_on_host_files_are_refused(self):
		# A read lease would make host programs that open the file for writing wait for the lease to break.
		r = self.py(f"""
import fcntl, os
fd = os.open({str(self.host / "file")!r}, os.O_RDONLY)
try:
	fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)
	print("leased")
except OSError as e:
	print(e.strerror)
try:
	fcntl.fcntl(fd, 1100)  # a command newer than the filter knows
except OSError as e:
	print(e.strerror)
""")
		self.assertEqual(r.output.split("\n"), ["Permission denied"] * 2 + [""], r.output)

	def test_host_sysv_ipc_is_hidden(self):
		libc = ctypes.CDLL(None, use_errno=True)
		key = 0x41520000 + os.getpid() % 0x10000
		queue = libc.msgget(key, 0o1000 | 0o600)  # IPC_CREAT
		self.assertGreaterEqual(queue, 0, os.strerror(ctypes.get_errno()))
		try:
			r = self.py(f"""
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
print(libc.msgget({key}, 0), ctypes.get_errno())
""")
		finally:
			libc.msgctl(queue, 0, None)  # IPC_RMID
		self.assertEqual(r.output.strip(), "-1 2", r.output)  # ENOENT

	def test_a_scratch_removed_after_mkdir_is_retried(self):
		scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		real = artifice.sandbox._open_dir
		calls = []

		def racy(name, dir_fd=None):
			if not calls:
				calls.append(name)
				os.rmdir(scratch)  # what clean_scratch does between mkdir and open
			return real(name, dir_fd=dir_fd)

		with mock.patch.object(artifice.sandbox, "_open_dir", racy):
			sb = Sandbox(READ_ONLY, scratch, str(self.host), exclusive=True)
		sb.close()
		self.assertEqual(len(calls), 1)

	def test_scratch_usage_does_not_follow_a_swapped_symlink(self):
		scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		outside = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="outside-"))
		for i in range(100):
			(outside / f"f{i}").write_text("x")
		(scratch / "d").mkdir()
		sb = Sandbox(READ_ONLY, scratch, str(self.host))
		real_stat, real_lstat = os.stat, os.lstat
		swapped = False

		def flipping(real):
			# A running command keeps the directory in place for each check and swaps in a symlink right after.
			def check(name, *args, **kwargs):
				nonlocal swapped
				if os.path.basename(os.fspath(name)) != "d":
					return real(name, *args, **kwargs)
				if swapped:
					os.unlink(scratch / "d")
					os.rename(scratch / "moved", scratch / "d")
				st = real(name, *args, **kwargs)
				os.rename(scratch / "d", scratch / "moved")
				os.symlink(outside, scratch / "d")
				swapped = True
				return st
			return check

		with mock.patch.object(os, "stat", flipping(real_stat)), mock.patch.object(os, "lstat", flipping(real_lstat)):
			size, entries, unreadable = sb.scratch_usage()
		self.assertLess(entries, 10)
		self.assertFalse(unreadable)

	def test_deleted_open_files_count_toward_scratch(self):
		sb = Sandbox(Profile(name="small", scratch_max=16 << 20, file_size_max=8 << 20), self.scratch, str(self.host))
		r = sb.run(["bash", "-c", "for i in 3 4 5 6 7 8; do eval \"exec $i>/tmp/held$i\"; rm /tmp/held$i; "
			"head -c 7M /dev/zero >&$i; done; sleep 20"], timeout=15)
		self.assertFalse(r.timed_out, r.output)
		self.assertIsNone(r.exit_code, r.output)
		self.assertTrue(r.scratch_full, r.output)

	def test_cwd_under_host_tmp_or_dev(self):
		for top in ("/tmp", "/dev/shm"):
			with self.subTest(top=top):
				tmp = tempfile.mkdtemp(dir=top, prefix="artifice-test-")
				try:
					r = Sandbox(READ_ONLY, self.scratch, tmp).run(["pwd"])
				finally:
					os.rmdir(tmp)
				self.assertEqual(r.output.strip(), str(Path.home()), r.output)

	def test_dev_shm_is_writable_and_private(self):
		r = self.sh("python3 -c 'import multiprocessing; multiprocessing.Lock(); print(\"ok\")' && echo x > /dev/shm/artifice-shm-test && ls -ld /dev/shm")
		self.assertIn("ok", r.output)
		self.assertRegex(r.output, r"drwxrwxrwt")
		self.assertFalse(os.path.exists("/dev/shm/artifice-shm-test"))

	def test_glob_patterns_are_relative_to_path(self):
		from artifice.tools import Tools
		base = Path(tempfile.mkdtemp(dir=HOST_DIR))
		tools = Tools(Sandbox(READ_ONLY, self.scratch, str(base)))
		if not tools.rg:
			self.skipTest("rg is not installed")
		project = base / "project"
		(project / "tests").mkdir(parents=True, exist_ok=True)
		(project / "tests" / "t.py").write_text("needle\n")
		for path in (str(project), "project", "project/"):
			with self.subTest(path=path):
				found = tools.call("glob", json.dumps({"pattern": "tests/*.py", "path": path})).text
				self.assertEqual(found, os.path.join(path, "tests/t.py"))
				found = tools.call("grep_search", json.dumps({"pattern": "needle", "path": path, "include": "tests/*.py"})).text
				self.assertEqual(found, os.path.join(path, "tests/t.py") + ":1:needle")
		self.assertEqual(tools.call("grep_search", json.dumps({"pattern": "needle", "path": "project/tests/t.py"})).text,
			"project/tests/t.py:1:needle")
		# Errors name files as rg would have without the cd. compact_memory is write-only, even for root.
		errors = tools.call("grep_search", json.dumps({"pattern": "x", "path": "/proc/sys/vm"})).text
		self.assertNotIn("rg: ./", errors)
		self.assertIn("rg: /proc/sys/vm/compact_memory", errors)

	def test_timeout_and_background_processes(self):
		start = time.monotonic()
		r = self.sh("(sleep 4243 &); sleep 4244", timeout=2)
		self.assertTrue(r.timed_out)
		self.assertLess(time.monotonic() - start, 10)
		r = self.sh("(sleep 4245 &) ; echo started", timeout=10)
		self.assertIn("started", r.output)
		self.assertLess(time.monotonic() - start, 20)
		time.sleep(0.5)
		left = subprocess.run(["pgrep", "-f", "sleep 424[345]"], capture_output=True, text=True)
		self.assertEqual(left.stdout, "")

	def test_output_cap(self):
		r = self.sh("yes | head -c 5000000")
		self.assertEqual(len(r.output), READ_ONLY.output_max)
		self.assertGreater(r.truncated_bytes, 0)

	def test_devices_are_private(self):
		r = self.sh("ls /dev")
		self.assertNotRegex(r.output, r"\b(sda|nvme0|vda|kvm|dri|input)\b", r.output)
		r = self.sh("echo x > /dev/shm/artifice-test")
		self.assertFalse(Path("/dev/shm/artifice-test").exists())

	def test_clean_scratch_does_not_follow_symlinks(self):
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		outside = self.host / "keep"
		outside.write_text("keep")
		chat = root / "chat"
		(chat / "sub").mkdir(parents=True)
		(chat / "link").symlink_to(self.host)
		(chat / "sub" / "link").symlink_to(outside)
		(chat / "file").write_text("x")
		clean_scratch(root, max_age=-60)  # everything counts as old
		self.assertTrue(outside.exists())
		self.assertFalse(chat.exists())
		outside.unlink()

	def test_clean_scratch_handles_deep_trees_with_a_low_fd_limit(self):
		# Deeper than Python's recursion limit.
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		subprocess.run(["bash", "-c", f"cd {root} && mkdir chat && cd chat && for i in $(seq 1500); do mkdir d && cd d; done && echo x > f"],
			check=True)
		# os.fwalk recurses once per level before Python 3.12.4, so the cleanup must not use it.
		code = f"import os, sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r}); " \
			f"os.fwalk = lambda *args, **kwargs: (_ for _ in ()).throw(RecursionError('os.fwalk')); " \
			f"from artifice.sandbox import clean_scratch; clean_scratch(__import__('pathlib').Path({str(root)!r}), max_age=-60)"
		subprocess.run(["bash", "-c", f"ulimit -Sn 64 && {sys.executable} -I -c \"{code}\""], check=True)
		self.assertFalse((root / "chat").exists())

	def test_a_chat_opens_in_one_session_at_a_time(self):
		scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		first = Sandbox(READ_ONLY, scratch, str(self.host), exclusive=True)
		with self.assertRaisesRegex(SandboxError, "another Artifice session"):
			Sandbox(READ_ONLY, scratch, str(self.host), exclusive=True)
		first.close()
		Sandbox(READ_ONLY, scratch, str(self.host), exclusive=True)

	def test_opening_a_chat_waits_for_cleanup(self):
		# clean_scratch holds the root's lock throughout, and a chat's lock while it cleans that chat.
		for lock_chat in (True, False):
			with self.subTest(lock_chat=lock_chat):
				root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
				scratch = root / "chat"
				scratch.mkdir()
				fds = [os.open(root, os.O_RDONLY)] + ([os.open(scratch, os.O_RDONLY)] if lock_chat else [])
				for fd in fds:
					fcntl.flock(fd, fcntl.LOCK_EX)
				threading.Timer(0.5, lambda: [os.close(fd) for fd in reversed(fds)]).start()  # chat first, as cleanup does
				start = time.monotonic()
				Sandbox(READ_ONLY, scratch, str(self.host), exclusive=True)
				self.assertGreater(time.monotonic() - start, 0.4)

	def test_clean_scratch_enters_read_only_directories(self):
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		self.addCleanup(subprocess.run, ["chmod", "-R", "u+rwx", str(root)])
		locked = root / "chat" / "ro" / "none"
		locked.mkdir(parents=True)
		(locked / "file").write_text("x")
		locked.chmod(0)
		locked.parent.chmod(0o555)
		(root / "chat").chmod(0o555)
		clean_scratch(root, max_age=-60)
		self.assertFalse((root / "chat").exists())

	def test_a_chat_whose_tmp_the_model_chmodded_reopens_and_cleans(self):
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		self.addCleanup(subprocess.run, ["chmod", "-R", "u+rwx", str(root)])
		sb = Sandbox(READ_ONLY, root / "chat", str(self.host), exclusive=True)
		sb.run(["bash", "-c", "touch /tmp/old; chmod 0 /tmp"])
		sb.close()
		Sandbox(READ_ONLY, root / "chat", str(self.host), exclusive=True).close()
		(root / "chat").chmod(0)
		clean_scratch(root, max_age=-60)
		self.assertFalse((root / "chat").exists())

	def test_clean_scratch_does_not_chmod_through_a_swapped_symlink(self):
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		outside = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="outside-"))
		self.addCleanup(subprocess.run, ["chmod", "-R", "u+rwx", str(root), str(outside)])
		(root / "chat" / "d").mkdir(parents=True)
		(root / "chat" / "d").chmod(0o500)
		outside.chmod(0o500)
		real_stat = os.stat

		def swap_after_stat(name, *args, **kwargs):
			# A command still running from a killed session swaps the directory for a symlink right after the check.
			st = real_stat(name, *args, **kwargs)
			if name == "d":
				os.rename(root / "chat" / "d", root / "chat" / "moved")
				os.symlink(outside, root / "chat" / "d")
			return st

		with mock.patch.object(artifice.sandbox.os, "stat", swap_after_stat):
			clean_scratch(root, max_age=-60)
		self.assertEqual(outside.stat().st_mode & 0o777, 0o500)

	def test_clean_scratch_skips_open_chats_and_counts_from_ctime(self):
		root = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="clean-"))
		Sandbox(READ_ONLY, root / "closed", str(self.host))  # dropped at once, which releases its lock
		open_chat = Sandbox(READ_ONLY, root / "open", str(self.host))
		for name in ("closed", "open"):
			(root / name / "file").write_text("x")
		clean_scratch(root, max_age=-60)
		self.assertFalse((root / "closed").exists())
		self.assertTrue((root / "open" / "file").exists())
		del open_chat
		# An extracted file keeps its old mtime, but its ctime is when it arrived in scratch.
		extracted = root / "new" / "extracted"
		extracted.parent.mkdir()
		extracted.write_text("x")
		old = time.time() - 3 * 24 * 3600
		os.utime(extracted, (old, old))
		clean_scratch(root)
		self.assertTrue(extracted.exists())


if __name__ == "__main__":
	unittest.main()
