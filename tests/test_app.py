"""Tests for the agent loop and the CLI, with a fake llm-server client and fake tools.

Run with: python3 -m unittest discover -s tests -v
"""
import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifice import cli, deps, helpers  # noqa: E402
from artifice.agent import Agent, Settings  # noqa: E402
from artifice.client import LLMError  # noqa: E402
from artifice.sandbox import READ_ONLY, Sandbox, SandboxError  # noqa: E402
from artifice.store import Chat  # noqa: E402
from artifice.tools import ToolResult  # noqa: E402


class FakeClient:
	"""Answers with tool_calls for the first `calls` turns, then text. Summary requests get notes."""

	def __init__(self, calls: int = 0, usage: dict | None = None, stop: str = ""):
		self.calls = calls
		self.usage = usage
		self.stop = stop
		self.summaries = 0
		self.max_tokens: list[int] = []

	def stream(self, body):
		if body["messages"][-1]["content"].startswith("Stop here."):
			self.summaries += 1
			yield {"token": "notes"}
			return
		self.max_tokens.append(body["max_tokens"])
		if self.usage is not None:
			yield {"usage": self.usage}
		if self.calls:
			self.calls -= 1
			call = {"id": f"c{self.calls}", "type": "function",
				"function": {"name": "run_shell_command", "arguments": '{"command": "x"}'}}
			yield {"tool_calls": [call]}
		else:
			yield {"token": "done"}
		if self.stop:
			yield {"stop_reason": self.stop}

	def load(self, model, ttl):
		return {"context_length": 1000}


class FakeTools:
	schemas: list = []

	def __init__(self, output: str = "y" * 30000, error: Exception | None = None):
		self.output = output
		self.error = error

	def call(self, name, arguments):
		if self.error:
			raise self.error
		return ToolResult(self.output)


class QuietUI:
	def __init__(self):
		self.notices: list[str] = []

	def notice(self, text):
		self.notices.append(text)

	def __getattr__(self, name):
		return lambda *args: None


class TempDataTest(unittest.TestCase):
	def setUp(self):
		self.data = tempfile.TemporaryDirectory()
		patcher = mock.patch.dict(os.environ, {"XDG_DATA_HOME": self.data.name})
		patcher.start()
		self.addCleanup(patcher.stop)
		self.addCleanup(self.data.cleanup)

	def chat(self, *history: tuple[str, str]) -> Chat:
		chat = Chat.new("m", "read-only", "/")
		chat.messages = [{"role": "system", "content": "sys"}]
		chat.messages += [{"role": role, "content": content} for role, content in history]
		return chat


class AgentTest(TempDataTest):
	def test_compaction_with_earlier_history_stays_within_context(self):
		client = FakeClient(calls=8)
		chat = self.chat(("user", "earlier"), ("assistant", "ok"))
		agent = Agent(client, chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.send("go")
		self.assertLessEqual(client.summaries, 1)
		self.assertLess(agent.estimate_tokens(), 32768)
		self.assertGreater(min(client.max_tokens), 256)

	def test_null_usage_values(self):
		client = FakeClient(usage={"prompt_tokens": None, "completion_tokens": None})
		agent = Agent(client, self.chat(), FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.send("hi")
		self.assertEqual(agent.messages[-1]["content"], "done")

	def test_failed_tool_call_is_answered_before_the_error_propagates(self):
		chat = self.chat()
		agent = Agent(FakeClient(calls=1), chat, FakeTools(error=RuntimeError("boom")), Settings(model="m"),
			QuietUI(), context_length=32768)
		with self.assertRaises(RuntimeError):
			agent.send("go")
		saved = Chat.load(chat.id).messages
		self.assertEqual(saved[-1]["role"], "tool")
		self.assertEqual(saved[-1]["tool_call_id"], saved[-2]["tool_calls"][0]["id"])


	def test_exit_answers_open_tool_calls(self):
		chat = self.chat()
		agent = Agent(FakeClient(calls=1), chat, FakeTools(error=SystemExit(143)), Settings(model="m"),
			QuietUI(), context_length=32768)
		with self.assertRaises(SystemExit):
			agent.send("go")
		self.assertEqual(Chat.load(chat.id).messages[-1]["role"], "tool")

	def test_max_tokens_notice_shows_the_limit_sent(self):
		client = FakeClient(usage={"prompt_tokens": 20000, "completion_tokens": 6000}, stop="length")
		ui = QuietUI()
		Agent(client, self.chat(), FakeTools(), Settings(model="m"), ui, context_length=32768).send("hi")
		self.assertIn(f"Response hit max_tokens ({client.max_tokens[0]})", " ".join(ui.notices))

	def test_tool_output_larger_than_the_context_is_elided(self):
		agent = Agent(FakeClient(calls=1), self.chat(), FakeTools(), Settings(model="m"), QuietUI(), context_length=8192)
		agent.send("go")
		self.assertLess(agent.estimate_tokens(), 8192)


class OpenChatTest(TempDataTest):
	def app(self) -> cli.App:
		args = argparse.Namespace(host="h", port=1, model=None, max_tokens=100, temperature=None, top_p=None,
			think=None, prompt="x")
		app = cli.App(args)
		app.client = FakeClient()
		app.settings.model = app.loaded = "good"
		app.context_length = 2000
		return app

	def test_failed_load_keeps_the_current_chat_settings(self):
		app = self.app()
		app.client.load = mock.Mock(side_effect=LLMError("Failed: HTTP 404"))
		chat = Chat.new("gone", "read-only", "/")
		with self.assertRaises(LLMError):
			app.open_chat(chat)
		self.assertEqual((app.settings.model, app.loaded, app.context_length), ("good", "good", 2000))

	def test_failed_sandbox_keeps_the_current_chat_settings(self):
		app = self.app()
		sandbox = mock.Mock()
		sandbox.return_value.check.side_effect = SandboxError("Failed to start sandbox")
		with mock.patch.object(cli, "Sandbox", sandbox):
			with self.assertRaises(SandboxError):
				app.open_chat(Chat.new("other", "read-only", "/"))
		self.assertEqual((app.settings.model, app.loaded, app.context_length), ("good", "good", 2000))


	def test_chat_open_elsewhere_does_not_load_its_model(self):
		app = self.app()
		app.client.load = mock.Mock()
		chat = Chat.new("other", "read-only", "/")
		held = Sandbox(READ_ONLY, cli.scratch_root() / chat.id, "/", exclusive=True)
		with self.assertRaisesRegex(SandboxError, "another Artifice session"):
			app.open_chat(chat)
		app.client.load.assert_not_called()
		held.close()

	def test_resuming_the_open_chat_keeps_its_sandbox(self):
		app = self.app()
		chat = Chat.new("good", "read-only", "/")
		with mock.patch.object(cli, "Sandbox") as sandbox:
			app.open_chat(chat)
			app.open_chat(chat)
		self.assertEqual(sandbox.call_count, 1)

	def test_new_chat_in_a_deleted_directory_starts_at_home(self):
		app = self.app()
		app.ui = QuietUI()
		gone = tempfile.mkdtemp()
		here = os.getcwd()
		self.addCleanup(os.chdir, here)
		os.chdir(gone)
		os.rmdir(gone)
		self.assertEqual(app.new_chat().cwd, str(Path.home()))


class CommandTest(TempDataTest):
	def test_paths_are_messages(self):
		self.assertTrue(cli.is_command("/resume 2026"))
		self.assertTrue(cli.is_command("/hlep"))
		self.assertFalse(cli.is_command("/var/log/syslog has errors, why?"))
		self.assertFalse(cli.is_command("/etc is huge, why?"))
		self.assertFalse(cli.is_command("why?"))

	def test_resume_by_path(self):
		chat = self.chat()
		chat.save()
		self.assertEqual(Chat.load(str(chat.path)).id, chat.id)
		with self.assertRaises(FileNotFoundError):
			Chat.load("/nonexistent/[x*.json")

	def test_multiline_input(self):
		with mock.patch("builtins.input", side_effect=['"""Here is the log:', "line 1", '"""']):
			self.assertEqual(cli.read_input(), "Here is the log:\nline 1")
		with mock.patch("builtins.input", side_effect=['"""one line"""']):
			self.assertEqual(cli.read_input(), "one line")

	def test_sampling_arguments_are_validated(self):
		for flag, value in (("--temperature", "0"), ("--top-p", "0"), ("--top-p", "1.5")):
			with self.subTest(flag=flag, value=value), mock.patch.object(sys, "argv", ["artifice", flag, value]), \
					mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
				cli.parse_args()


class GlobFallbackTest(unittest.TestCase):
	def test_matches_like_rg(self):
		with tempfile.TemporaryDirectory() as root:
			for name in ("a.py", "tests/b.py", "tests/deep/c.py", ".git/d.py", ".hidden/e.py", "tests/f.txt"):
				(Path(root) / name).parent.mkdir(parents=True, exist_ok=True)
				(Path(root) / name).write_text("")
			names = lambda pattern: [os.path.relpath(p, root) for p in helpers.glob_files(pattern, root).split("\n")]
			self.assertEqual(names("tests/*.py"), ["tests/b.py"])
			self.assertEqual(names("*.py"), [".hidden/e.py", "a.py", "tests/b.py", "tests/deep/c.py"])
			self.assertEqual(helpers.glob_files("*.rs", root), "No files matched.")
			self.assertEqual(names("**/*.{py,txt}"), [".hidden/e.py", "a.py", "tests/b.py", "tests/deep/c.py", "tests/f.txt"])
			self.assertEqual(names("{tests/deep,.hidden}/*.py"), [".hidden/e.py", "tests/deep/c.py"])
			self.assertRegex(helpers.glob_files("[z-a]", root), "^Error: bad glob pattern")
			(Path(root) / "tests" / "loop").symlink_to("..")
			self.assertEqual(names("**/b.py"), ["tests/b.py"])

	def test_schemas_without_rg_do_not_offer_gitignore(self):
		from artifice.tools import schemas
		text = str(schemas(False))
		self.assertNotIn("gitignore", text)
		self.assertNotIn("ripgrep", text)


class ReadFileTest(unittest.TestCase):
	def test_line_numbers_match_grep(self):
		with tempfile.NamedTemporaryFile("wb") as f:
			f.write(b"p 10%\rp 100%\nERROR\r\n")
			f.flush()
			self.assertIn("     2\tERROR\n", helpers.read_file(f.name) + "\n")


class DepsTest(TempDataTest):
	def test_failed_install_is_not_asked_again(self):
		state = Path(self.data.name) / "state.json"
		which = lambda command: None if command == "rg" else f"/usr/bin/{command}"
		with mock.patch.object(deps.shutil, "which", which), mock.patch.object(deps, "_ask", return_value=True), \
				mock.patch.object(deps.subprocess, "run") as run, mock.patch.object(deps.sys, "stderr"):
			self.assertTrue(deps.ensure(state))
			self.assertTrue(deps.ensure(state))
		self.assertEqual(run.call_count, 1)


class TerminalUITest(unittest.TestCase):
	def test_control_characters_are_escaped(self):
		out = mock.Mock()
		out.isatty.return_value = False
		ui = cli.TerminalUI(answer_out=out, trace_out=out)
		ui.token("a\x1b]52;c;cHduZWQ=\x07b\x9b2J\tc\n")
		ui.tool_result("x", ToolResult("\x1b[2Kfake"))
		written = "".join(c.args[0] for c in out.write.call_args_list)
		self.assertNotRegex(written, r"[\x1b\x07\x9b]")
		self.assertIn("a\\x1b]52;c;cHduZWQ=\\x07b\\x9b2J\tc\n", written)


if __name__ == "__main__":
	unittest.main()
