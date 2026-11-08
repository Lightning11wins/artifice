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

from artifice import cli  # noqa: E402
from artifice.agent import Agent, Settings  # noqa: E402
from artifice.client import LLMError  # noqa: E402
from artifice.sandbox import SandboxError  # noqa: E402
from artifice.store import Chat  # noqa: E402
from artifice.tools import ToolResult  # noqa: E402


class FakeClient:
	"""Answers with tool_calls for the first `calls` turns, then text. Summary requests get notes."""

	def __init__(self, calls: int = 0, usage: dict | None = None):
		self.calls = calls
		self.usage = usage
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
