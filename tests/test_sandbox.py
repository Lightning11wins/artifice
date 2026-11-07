"""Escape tests for the read-only sandbox. Every attempt to change the host must fail.

Run with: python3 -m unittest discover -s tests -v   (needs working bwrap, so not inside another sandbox)
"""
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifice.sandbox import READ_ONLY, Sandbox, clean_scratch  # noqa: E402

# Host directory the sandbox can see but must not modify. Under $HOME, not /tmp, which the sandbox replaces.
HOST_DIR = Path.home() / ".cache" / "artifice-tests"


class SandboxEscapeTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		HOST_DIR.mkdir(parents=True, exist_ok=True)
		cls.host = Path(tempfile.mkdtemp(dir=HOST_DIR))
		cls.scratch = Path(tempfile.mkdtemp(dir=HOST_DIR, prefix="scratch-"))
		cls.sb = Sandbox(READ_ONLY, cls.scratch, str(cls.host))
		cls.sb.check()
		(cls.host / "file").write_text("original\n")

	def sh(self, command: str, timeout: int = 20):
		return self.sb.run(["bash", "-c", command], timeout=timeout)

	def py(self, code: str, timeout: int = 20):
		return self.sb.run([sys.executable, "-I", "-c", code], timeout=timeout)

	def assertUnchanged(self):
		self.assertEqual((self.host / "file").read_text(), "original\n")
		self.assertEqual(sorted(p.name for p in self.host.iterdir() if p.name != "fifo" and p.name != "sock"), ["file"])

	def test_reads_work(self):
		r = self.sh(f"cat {self.host}/file && head -c 10 /etc/os-release")
		self.assertEqual(r.exit_code, 0, r.output)
		self.assertIn("original", r.output)

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
				r = self.sh(cmd)
				self.assertNotEqual(r.exit_code, 0, r.output)
		self.assertUnchanged()

	def test_scratch_is_writable_and_private(self):
		r = self.sh("echo hi > /tmp/a && mkdir -p /tmp/d && cat /tmp/a")
		self.assertEqual(r.exit_code, 0, r.output)
		self.assertEqual((self.scratch / "a").read_text(), "hi\n")

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

	def test_host_processes_invisible_and_unsignalable(self):
		victim = subprocess.Popen(["sleep", "60"])
		try:
			r = self.sh(f"ls /proc | grep -E '^[0-9]+$' | wc -l; kill -9 {victim.pid}")
			self.assertLess(int(r.output.split()[0]), 10, r.output)
			time.sleep(0.2)
			self.assertIsNone(victim.poll())
		finally:
			victim.kill()

	def test_no_privilege_escalation(self):
		r = self.sh("sudo -n true")
		self.assertNotEqual(r.exit_code, 0, r.output)
		r = self.sh("grep NoNewPrivs /proc/self/status")
		self.assertIn("1", r.output)

	def test_kernel_interfaces_read_only(self):
		for cmd in ["echo 1 > /proc/sys/vm/drop_caches", "echo 1 > /sys/kernel/mm/ksm/run",
				"echo x > /proc/sys/kernel/hostname"]:
			with self.subTest(cmd=cmd):
				self.assertNotEqual(self.sh(cmd).exit_code, 0)

	def test_blocked_syscalls(self):
		code = ("import ctypes, platform; libc = ctypes.CDLL(None, use_errno=True); "
			"nr = {'x86_64': {'uring': 425, 'keyctl': 250, 'handle': 304}, "
			"'aarch64': {'uring': 425, 'keyctl': 219, 'handle': 265}}[platform.machine()]; "
			"buf = ctypes.create_string_buffer(128); "
			"print(libc.syscall(nr['uring'], 8, buf), libc.syscall(nr['keyctl'], 0, -4, 0), "
			"libc.syscall(nr['handle'], -100, buf, 0))")
		r = self.py(code)
		self.assertEqual(r.output.split(), ["-1", "-1", "-1"], r.output)

	def test_timeout_and_background_processes(self):
		start = time.monotonic()
		r = self.sh("sleep 100", timeout=2)
		self.assertTrue(r.timed_out)
		self.assertLess(time.monotonic() - start, 10)
		r = self.sh("(sleep 100 &) ; echo started", timeout=10)
		self.assertIn("started", r.output)
		self.assertLess(time.monotonic() - start, 20)

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
		old = time.time() - 3 * 24 * 3600
		os.utime(outside, (old, old))
		chat = root / "chat"
		chat.mkdir()
		(chat / "link").symlink_to(self.host)
		(chat / "old").write_text("x")
		os.utime(chat / "old", (old, old))
		(chat / "new").write_text("x")
		clean_scratch(root)
		self.assertTrue(outside.exists())
		self.assertFalse((chat / "old").exists())
		self.assertTrue((chat / "new").exists())
		outside.unlink()


if __name__ == "__main__":
	unittest.main()
