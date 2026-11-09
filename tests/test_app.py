"""Tests for the agent loop and the CLI, with a fake llm-server client and fake tools.

Run with: python3 -m unittest discover -s tests -v
"""
import argparse
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifice import cli, deps, helpers, tools  # noqa: E402
from artifice.agent import Agent, Settings  # noqa: E402
from artifice.client import LLMError  # noqa: E402
from artifice.sandbox import READ_ONLY, Result, Sandbox, SandboxError  # noqa: E402
from artifice.store import Chat, scratch_root  # noqa: E402
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


class CountingClient:
	"""Fails a test that sends more than `cap` requests, so a retry loop fails rather than hangs."""

	def __init__(self, client, cap: int = 20):
		self.client = client
		self.cap = cap
		self.requests = 0

	def stream(self, body):
		self.requests += 1
		if self.requests > self.cap:
			raise AssertionError("the overflow retry did not stop")
		return self.client.stream(body)


class OverflowClient(FakeClient):
	"""Rejects requests over `limit` tokens, as a server does past its context. A digit is a token, as in Qwen."""

	def __init__(self, limit: int):
		super().__init__()
		self.limit = limit

	def stream(self, body):
		messages = body["messages"]
		last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
		text = "".join((m.get("content") or "") + json.dumps(m.get("tool_calls") or "")
			+ (m.get("reasoning_content") or "" if i > last_user else "") for i, m in enumerate(messages))
		digits = sum(c.isdigit() for c in text)
		if digits + (len(text) - digits) / 4 > self.limit:
			raise LLMError("Failed to generate: the request exceeds the available context size")
		yield from super().stream(body)


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
		return lambda *args, **kwargs: None


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

	def test_a_summary_that_calls_a_tool_is_not_kept(self):
		class PreambleClient(FakeClient):
			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					yield {"token": "Let me check the config first."}
					yield {"tool_calls": [{"id": "s", "type": "function",
						"function": {"name": "read_file", "arguments": '{"path": "x"}'}}]}
					return
				yield from super().stream(body)

		chat = self.chat(("user", "find the bug"), ("assistant", "The bug is at foo.c:42."))
		agent = Agent(PreambleClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.maybe_compact(force=True)
		self.assertIn("The bug is at foo.c:42.", [m.get("content") for m in agent.messages])

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

	def test_a_reply_cut_off_by_the_context_is_kept_and_the_next_message_compacts(self):
		history = [("user", "go"), ("assistant", "ok")] * 4 + [("user", "write a lot")]
		chat = self.chat(*history)
		client = FakeClient(usage={"prompt_tokens": 17000, "completion_tokens": 15768}, stop="length")
		ui = QuietUI()
		agent = Agent(CountingClient(client), chat, FakeTools(), Settings(model="m"), ui, context_length=32768)
		agent.send("x")
		self.assertEqual(chat.messages[-1]["content"], "done")
		self.assertIn("ran out of context", " ".join(ui.notices))
		self.assertNotIn("hit max_tokens", " ".join(ui.notices))
		self.assertEqual((client.summaries, agent.bytes_per_token), (0, 3.0))
		client.usage, client.stop = None, ""
		agent.send("continue")
		self.assertEqual(client.summaries, 1)  # the server's count of the long reply triggers compaction

	def test_estimate_counts_utf8_bytes(self):
		chat = self.chat(("user", "中文注释" * 3000))
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		self.assertGreaterEqual(agent.estimate_tokens(), 12000)

	def test_context_overflow_is_compacted_and_retried(self):
		# Digits take a token each, so the estimate is a third of the real count.
		chat = self.chat(("user", "go"))
		chat.messages += [
			{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
				"function": {"name": "read_file", "arguments": "{}"}}]},
			{"role": "tool", "tool_call_id": "c1", "content": "1" * 30000},
		]
		agent = Agent(OverflowClient(limit=16000), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("next")
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertNotIn("1" * 30000, [m["content"] for m in agent.messages])

	def test_context_overflow_from_the_current_turn_elides_it(self):
		for earlier in ("ok", "a" * 10000, "a" * 16000):
			with self.subTest(earlier=len(earlier)):
				chat = self.chat(("user", "earlier"), ("assistant", earlier))
				agent = Agent(OverflowClient(limit=16000), chat, FakeTools(output="1" * 20000), Settings(model="m"),
					QuietUI(), context_length=16384)
				agent.client.calls = 1
				agent.send("go")
				self.assertEqual(agent.messages[-1]["content"], "done")

	def test_estimate_skips_reasoning_before_the_last_user_message(self):
		chat = self.chat(("user", "earlier"))
		chat.messages.append({"role": "assistant", "content": "ok", "reasoning_content": "r" * 80000})
		chat.messages.append({"role": "user", "content": "next"})
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		self.assertLess(agent.estimate_tokens(), 100)

	def test_reasoning_the_server_counted_is_dropped_at_the_next_message(self):
		class ThinkingClient(FakeClient):
			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					yield from super().stream(body)
					return
				yield {"usage": {"prompt_tokens": 10000, "completion_tokens": 15000}}
				yield {"reasoning": "r" * 60000}
				yield {"token": "done"}

		client = ThinkingClient()
		agent = Agent(client, self.chat(), FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.send("hi")
		agent.send("next")
		self.assertEqual(client.summaries, 0)

	def test_a_resumed_chat_keeps_the_servers_token_count(self):
		chat = self.chat(("user", "z" * 2400), ("assistant", "ok"))
		live = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=1000)
		live.usage_tokens, live.usage_n = 300, len(chat.messages)
		chat.save()
		client = FakeClient()
		resumed = Agent(client, Chat.load(chat.id), FakeTools(), Settings(model="m"), QuietUI(), context_length=1000)
		self.assertEqual(resumed.estimate_tokens(), live.estimate_tokens())
		resumed.send("thanks")
		self.assertEqual(client.summaries, 0)

	def test_tool_calls_left_open_by_a_crash_are_answered_on_resume(self):
		chat = self.chat(("user", "go"))
		chat.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
			"function": {"name": "run_shell_command", "arguments": "{}"}}]})
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.send("hi")
		self.assertEqual([(m["role"], m.get("tool_call_id")) for m in agent.messages[2:5]],
			[("assistant", None), ("tool", "c1"), ("user", None)])

	def test_a_message_too_long_for_the_context_is_removed(self):
		chat = self.chat(("user", "earlier"), ("assistant", "ok"))
		agent = Agent(CountingClient(OverflowClient(limit=16000)), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		with self.assertRaisesRegex(LLMError, "too long for the context \\(80,000 characters\\)"):
			agent.send("x" * 80000)
		agent.send("hi")
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertNotIn("x" * 80000, [m["content"] for m in Chat.load(chat.id).messages])

	def test_an_earlier_message_too_long_for_the_context_is_replaced(self):
		# Left by Ctrl-C, a server error or an older version; digits are a token each.
		for earlier in ("x" * 80000, "1" * 30000):
			with self.subTest(earlier=earlier[:1]):
				chat = self.chat(("user", earlier), ("assistant", "ok"))
				agent = Agent(OverflowClient(limit=16000), chat, FakeTools(), Settings(model="m"), QuietUI(),
					context_length=16384)
				agent.send("hi")
				self.assertEqual(agent.messages[-1]["content"], "done")
				self.assertNotIn(earlier, [m["content"] for m in agent.messages])
				self.assertIn("hi", [m["content"] for m in agent.messages])

	def test_compaction_runs_again_after_a_replacement(self):
		chat = self.chat(("user", "1" * 30000), ("assistant", "ok"))
		for _ in range(40):
			chat.messages += [{"role": "user", "content": "2" * 120}, {"role": "assistant", "content": "3" * 120}]
		client = OverflowClient(limit=16000)
		agent = Agent(CountingClient(client), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("4" * 8000)
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertEqual(client.summaries, 1)

	def test_replacing_a_summary_does_not_loop(self):
		class LongNotesClient(OverflowClient):
			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					yield {"token": "n" * 1000}
					return
				yield from super().stream(body)

		client = LongNotesClient(limit=1500)
		client.calls = 200
		agent = Agent(CountingClient(client, cap=300), self.chat(("user", "earlier"), ("assistant", "ok")),
			FakeTools(output="ok"), Settings(model="m"), QuietUI(), context_length=16384)
		with self.assertRaisesRegex(LLMError, "context size"):
			agent.send("go")

	def test_a_replaced_summary_written_again_does_not_loop(self):
		class LongNotesClient(OverflowClient):
			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					self.summaries += 1
					yield {"token": "n" * 1000}
					return
				yield from super().stream(body)

		chat = self.chat(("user", "x" * 2000), ("assistant", "ok"))
		agent = Agent(CountingClient(LongNotesClient(limit=16000), cap=40), chat, FakeTools(), Settings(model="m"),
			QuietUI(), context_length=16384)
		with self.assertRaisesRegex(LLMError, "context size"):
			agent.send("1" * 15800)

	def test_the_reply_is_saved_when_stdout_is_closed(self):
		class ClosedPipe(io.StringIO):
			def write(self, text):
				raise BrokenPipeError

		chat = self.chat()
		ui = cli.TerminalUI(answer_out=ClosedPipe(), trace_out=io.StringIO())
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), ui, context_length=32768)
		with self.assertRaises(BrokenPipeError):
			agent.send("hi")
		self.assertEqual(Chat.load(chat.id).messages[-1]["role"], "assistant")

	def test_the_reply_is_shown_when_saving_it_fails(self):
		chat = self.chat()
		save = chat.save

		def full_disk():
			if chat.messages[-1]["role"] == "assistant":
				raise OSError(28, "No space left on device")
			save()

		chat.save = full_disk
		out = io.StringIO()
		ui = cli.TerminalUI(answer_out=out, trace_out=io.StringIO())
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), ui, context_length=32768)
		with self.assertRaises(OSError):
			agent.send("hi")
		self.assertEqual(out.getvalue(), "done\n")

	def test_a_partial_reply_is_shown_when_the_stream_fails(self):
		class BrokenClient(FakeClient):
			def stream(self, body):
				yield {"reasoning": "thinking"}
				yield {"token": "partial"}
				raise LLMError("Failed to generate: the stream ended without [DONE]")

		out = io.StringIO()
		ui = cli.TerminalUI(answer_out=out, trace_out=io.StringIO())
		agent = Agent(BrokenClient(), self.chat(), FakeTools(), Settings(model="m"), ui, context_length=32768)
		with self.assertRaises(LLMError):
			agent.send("hi")
		self.assertEqual(out.getvalue(), "partial\n")

	def test_undecodable_text_is_counted(self):
		chat = self.chat()
		chat.messages[0]["content"] += os.fsdecode(b" in /caf\xe9")
		agent = Agent(FakeClient(), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
		agent.send(os.fsdecode(b"explain caf\xe9"))
		self.assertEqual(chat.messages[-1]["role"], "assistant")

	def test_a_later_step_can_replace_again(self):
		class ScriptedClient(FakeClient):
			steps = 0

			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					raise LLMError("Failed to generate: HTTP 500")
				self.steps += 1
				if self.steps in (1, 2, 4, 5):
					raise LLMError("Failed to generate: the request exceeds the available context size")
				if self.steps == 3:
					yield {"tool_calls": [{"id": "w", "type": "function", "function": {"name": "run_shell_command",
						"arguments": json.dumps({"command": "z" * 5000})}}]}
					return
				yield {"token": "done"}

		chat = self.chat(("user", "x" * 30000), ("assistant", "ok"))
		agent = Agent(ScriptedClient(), chat, FakeTools(output="ok"), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("go")
		self.assertEqual(agent.messages[-1]["content"], "done")

	def test_an_overflow_compacts_even_when_the_server_reported_less(self):
		client = OverflowClient(limit=16000)
		client.calls, client.usage = 1, {"prompt_tokens": 7600, "completion_tokens": 10}
		chat = self.chat(("user", "hello"), ("assistant", "a" * 30000))
		agent = Agent(CountingClient(client), chat, FakeTools(output="1" * 9000), Settings(model="m"), QuietUI(),
			context_length=16384)
		agent.send("go")
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertEqual(client.summaries, 1)

	def test_the_largest_piece_by_bytes_freed_is_replaced_first(self):
		calls = [{"id": f"r{i}", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
			for i in range(8)]
		chat = self.chat(("user", "1" * 800), ("user", "1" * 800))
		chat.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
		chat.messages += [{"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in calls]
		chat.messages.append({"role": "assistant", "content": "ok"})
		ui = QuietUI()
		agent = Agent(OverflowClient(limit=1200), chat, FakeTools(), Settings(model="m"), ui, context_length=16384)
		agent.send("hi")
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertEqual([n.split(" with")[0] for n in ui.notices if n.startswith("Replaced")],
			["Replaced a long user message"])

	def test_the_current_message_is_replaced_last(self):
		chat = self.chat(("user", "go"))
		chat.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "d", "type": "function",
			"function": {"name": "run_shell_command", "arguments": json.dumps({"command": "1" * 12000})}}]})
		chat.messages.append({"role": "tool", "tool_call_id": "d", "content": "ok"})
		chat.messages.append({"role": "user", "content": "a" * 20000})
		agent = Agent(OverflowClient(limit=8000), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent._loop()
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertIn("a" * 20000, [m["content"] for m in agent.messages])

	def test_the_system_prompt_is_never_replaced(self):
		chat = self.chat()
		chat.messages[0]["content"] = "1" * 20000
		agent = Agent(CountingClient(OverflowClient(limit=16000)), chat, FakeTools(), Settings(model="m"), QuietUI(),
			context_length=16384)
		with self.assertRaisesRegex(LLMError, "context size"):
			agent.send("hi")
		self.assertEqual(agent.messages[0]["content"], "1" * 20000)

	def test_reasoning_too_long_for_the_context_is_replaced(self):
		class ThinkingClient(OverflowClient):
			def stream(self, body):
				if self.calls and not body["messages"][-1]["content"].startswith("Stop here."):
					yield {"reasoning": "1" * 30000}
				yield from super().stream(body)

		client = ThinkingClient(limit=16000)
		client.calls = 1
		agent = Agent(client, self.chat(), FakeTools(output="ok"), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("go")
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertGreater(client.max_tokens[-1], 256)

	def test_only_the_long_tool_call_arguments_are_replaced(self):
		chat = self.chat(("user", "write it"))
		chat.messages += [
			{"role": "assistant", "content": "", "tool_calls": [
				{"id": "r", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
				{"id": "w", "type": "function", "function": {"name": "run_shell_command",
					"arguments": json.dumps({"command": "x" * 80000})}}]},
			{"role": "tool", "tool_call_id": "w", "content": "ok"},
			{"role": "tool", "tool_call_id": "r", "content": "ok"},
			{"role": "user", "content": "hi"},
		]
		agent = Agent(OverflowClient(limit=16000), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent._loop()
		self.assertEqual(agent.messages[-1]["content"], "done")
		self.assertEqual(agent.messages[2]["tool_calls"][0]["function"]["arguments"], '{"path": "a"}')
		self.assertIn("Removed", agent.messages[2]["tool_calls"][1]["function"]["arguments"])

	def test_reasoning_the_server_no_longer_sees_is_not_replaced(self):
		chat = self.chat(("user", "earlier"))
		chat.messages.append({"role": "assistant", "content": "ok", "reasoning_content": "1" * 80000})
		chat.messages += [{"role": "user", "content": "1" * 30000}, {"role": "assistant", "content": "ok"}]
		agent = Agent(OverflowClient(limit=16000), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("hi")
		self.assertEqual(agent.messages[-1]["content"], "done")

	def test_earlier_tool_call_arguments_too_long_for_the_context_are_replaced(self):
		chat = self.chat(("user", "write it"))
		chat.messages += [
			{"role": "assistant", "content": "", "tool_calls": [{"id": "w", "type": "function",
				"function": {"name": "run_shell_command", "arguments": json.dumps({"command": "x" * 80000})}}]},
			{"role": "tool", "tool_call_id": "w", "content": "ok"},
			{"role": "assistant", "content": "ok"},
		]
		agent = Agent(OverflowClient(limit=16000), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		agent.send("hi")
		self.assertEqual(agent.messages[-1]["content"], "done")

	def test_an_overflow_nothing_can_shrink_stops(self):
		chat = self.chat(("user", "go"))
		chat.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
			"function": {"name": "glob", "arguments": "{}"}} for i in range(8)]})
		chat.messages += [{"role": "tool", "tool_call_id": f"c{i}", "content": "ok"} for i in range(8)]
		agent = Agent(CountingClient(OverflowClient(limit=100)), chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=16384)
		with self.assertRaisesRegex(LLMError, "context size"):
			agent.send("hi")

	def test_usage_from_summary_requests_is_not_counted_for_the_chat(self):
		class UsageClient(FakeClient):
			notes = "notes"

			def stream(self, body):
				if body["messages"][-1]["content"].startswith("Stop here."):
					yield {"usage": {"prompt_tokens": 50, "completion_tokens": 5}}
					yield {"token": self.notes}
					return
				yield from super().stream(body)

		for notes in ("notes", ""):
			with self.subTest(notes=notes):
				client = UsageClient()
				client.notes = notes
				chat = self.chat(("user", "earlier"), ("assistant", "ok"), ("user", "next"))
				agent = Agent(client, chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=32768)
				agent.usage_tokens, agent.usage_n = 30000, len(chat.messages)
				agent._summarize(keep_current=False)
				if notes:
					self.assertLess(agent.estimate_tokens(), 1000)
				else:
					self.assertGreaterEqual(agent.estimate_tokens(), 30000)

	def test_old_tool_outputs_are_elided_before_summarizing(self):
		chat = self.chat(("user", "earlier"))
		for i in range(8):
			chat.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"o{i}", "type": "function",
				"function": {"name": "read_file", "arguments": "{}"}}]})
			chat.messages.append({"role": "tool", "tool_call_id": f"o{i}", "content": "y" * 30000})
		chat.messages.append({"role": "assistant", "content": "ok"})
		client = FakeClient()
		agent = Agent(client, chat, FakeTools(), Settings(model="m"), QuietUI(), context_length=100000)
		agent.send("go")
		self.assertEqual(client.summaries, 0)
		self.assertEqual(sum(m["content"] == "y" * 30000 for m in agent.messages), 6)

	def test_one_shot_narration_before_a_tool_call_stays_off_stdout(self):
		class NarratingClient(FakeClient):
			def stream(self, body):
				if self.calls:
					yield {"token": "Checking."}
					yield {"reasoning": "r"}
				yield from super().stream(body)

		out, trace = io.StringIO(), io.StringIO()
		ui = cli.TerminalUI(answer_out=out, trace_out=trace)
		Agent(NarratingClient(calls=1), self.chat(), FakeTools(output="ok"), Settings(model="m"), ui,
			context_length=32768).send("go")
		self.assertEqual(out.getvalue(), "done\n")
		self.assertIn("Checking.\n", trace.getvalue())

	def test_tool_output_larger_than_the_context_is_elided(self):
		agent = Agent(FakeClient(calls=1), self.chat(), FakeTools(), Settings(model="m"), QuietUI(), context_length=8192)
		agent.send("go")
		self.assertLess(agent.estimate_tokens(), 8192)
		self.assertIn("Output too large for the context (30,000 characters)", agent.messages[-2]["content"])

	def test_newest_tool_output_that_fits_is_kept(self):
		# 10,000 estimated tokens: over 75% of the context, with room for a reply.
		agent = Agent(FakeClient(calls=1), self.chat(), FakeTools(), Settings(model="m"), QuietUI(), context_length=13000)
		agent.send("go")
		self.assertEqual(agent.messages[-2]["content"], "y" * 30000)
		self.assertGreater(min(agent.client.max_tokens), 256)

	def test_reasoning_from_earlier_steps_of_a_long_turn_is_dropped(self):
		class ThinkingClient(OverflowClient):
			def stream(self, body):
				if not body["messages"][-1]["content"].startswith("Stop here."):
					text = "".join(m.get("content") or "" for m in body["messages"])
					text += "".join(m.get("reasoning_content") or "" for m in body["messages"][2:])
					if len(text) / 4 > self.limit:
						raise LLMError("Failed to generate: the request exceeds the available context size")
					steps = [m for m in body["messages"][4:] if m["role"] == "assistant"]
					self.newest_kept.append(not steps or "reasoning_content" in steps[-1])
					yield {"reasoning": "r" * 6000}
				yield from FakeClient.stream(self, body)

		for usage in (None, {"prompt_tokens": 100, "completion_tokens": 100}):
			with self.subTest(usage=usage):
				client = ThinkingClient(limit=32768)
				client.calls, client.usage, client.newest_kept = 40, usage, []
				agent = Agent(client, self.chat(("user", "earlier"), ("assistant", "ok")), FakeTools(output="ok"),
					Settings(model="m"), QuietUI(), context_length=32768)
				agent.send("go")
				self.assertEqual(agent.messages[-1]["content"], "done")
				self.assertTrue(all(client.newest_kept))
				self.assertGreater(min(client.max_tokens), 256)


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


	def test_another_model_discards_the_saved_token_count(self):
		app = self.app()
		app.args.model = "good"
		chat = Chat.new("other", "read-only", "/")
		chat.usage_tokens, chat.usage_n, chat.bytes_per_token = 300, 1, 1.5
		with mock.patch.object(cli, "Sandbox"):
			app.open_chat(chat)
		self.assertEqual((chat.usage_tokens, chat.bytes_per_token), (None, None))

	def test_chat_open_elsewhere_does_not_load_its_model(self):
		app = self.app()
		app.client.load = mock.Mock()
		chat = Chat.new("other", "read-only", "/")
		held = Sandbox(READ_ONLY, cli.scratch_root() / chat.id, "/", exclusive=True)
		with self.assertRaisesRegex(SandboxError, "another Artifice session"):
			app.open_chat(chat)
		app.client.load.assert_not_called()
		held.close()

	def test_full_disk_during_a_turn_is_reported(self):
		app = self.app()
		app.ui = QuietUI()
		app.agent = mock.Mock()
		app.agent.send.side_effect = OSError(28, "No space left on device")
		app.turn("hi")
		self.assertIn("Failed to finish the turn", " ".join(app.ui.notices))

	def test_os_error_in_a_command_is_reported(self):
		app = self.app()
		app.ui = QuietUI()
		app.command = mock.Mock(side_effect=OSError(28, "No space left on device"))
		with mock.patch.object(cli, "read_input", side_effect=["/compact", None]), \
				mock.patch.object(app, "banner"), mock.patch("builtins.print"):
			app.repl()
		self.assertIn("Failed to run /compact", " ".join(app.ui.notices))

	def test_history_keeps_lines_another_session_saved(self):
		for tty in (True, False):
			with self.subTest(tty=tty):
				history = cli.data_dir() / "history"
				history.write_text("from before\n")
				history.chmod(0o644)
				app = self.app()
				app.turn = mock.Mock()

				def type_line():
					with open(history, "a") as f:
						f.write("from another session\n")  # saved while this session runs
					return "from this session"

				inputs = iter([type_line, lambda: None])
				with mock.patch.object(cli, "read_input", lambda: next(inputs)()), \
						mock.patch.object(cli.sys, "stdin", mock.Mock(**{"isatty.return_value": tty})), \
						mock.patch.object(app, "banner"), mock.patch("builtins.print"):
					app.repl()
				saved = ["from this session"] if tty else []  # piped input is not saved
				self.assertEqual(history.read_text().splitlines(), ["from before", "from another session"] + saved)
				self.assertEqual(history.stat().st_mode & 0o777, 0o600 if tty else 0o644)

	def test_resuming_the_open_chat_keeps_its_sandbox(self):
		app = self.app()
		chat = Chat.new("good", "read-only", "/")
		with mock.patch.object(cli, "Sandbox") as sandbox:
			app.open_chat(chat)
			app.open_chat(chat)
		self.assertEqual(sandbox.call_count, 1)
		sandbox.return_value.close.assert_not_called()  # closing would release the open chat's lock

	def test_switching_chats_locks_the_new_chat_and_releases_the_old(self):
		app = self.app()
		first, second = Chat.new("good", "read-only", "/"), Chat.new("good", "read-only", "/")
		with mock.patch.object(cli, "Sandbox", side_effect=lambda *a, **kw: mock.Mock(args=a, kwargs=kw)):
			app.open_chat(first)
			old = app.agent.tools.sandbox
			app.open_chat(second)
		new = app.agent.tools.sandbox
		self.assertEqual(new.args[1], cli.scratch_root() / second.id)
		self.assertTrue(new.kwargs["exclusive"])
		old.close.assert_called_once()
		new.close.assert_not_called()

	def test_resuming_a_chat_whose_directory_was_deleted_continues_at_home(self):
		app = self.app()
		app.ui = QuietUI()
		gone = tempfile.mkdtemp()
		chat = Chat.new("good", "read-only", gone)
		chat.messages = [{"role": "system", "content": f"cwd {gone}"}]
		os.rmdir(gone)
		with mock.patch.object(cli, "Sandbox") as sandbox:
			app.open_chat(chat)
		self.assertEqual(sandbox.call_args.args[2], str(Path.home()))
		self.assertEqual(chat.cwd, str(Path.home()))
		self.assertNotIn(gone, chat.messages[0]["content"])

	def test_new_chat_in_a_deleted_directory_starts_at_home(self):
		app = self.app()
		app.ui = QuietUI()
		gone = tempfile.mkdtemp()
		here = os.getcwd()
		self.addCleanup(os.chdir, here)
		os.chdir(gone)
		os.rmdir(gone)
		self.assertEqual(app.new_chat().cwd, str(Path.home()))


class ToolsTest(unittest.TestCase):
	def test_non_string_path_is_a_tool_error(self):
		with mock.patch.object(tools.shutil, "which", return_value="/usr/bin/rg"):
			t = tools.Tools(mock.Mock())
		for path in (5, ["a"], True):
			with self.subTest(path=path):
				result = t.call("glob", json.dumps({"pattern": "*.py", "path": path}))
				self.assertTrue(result.error)
				self.assertIn("path must be a string", result.text)

	def test_deeply_nested_arguments_are_an_error(self):
		arguments = "[" * 20000
		self.assertEqual(tools.describe_call("read_file", arguments), f"read_file({arguments})")
		t = tools.Tools(mock.Mock())
		result = t.call("read_file", arguments)
		self.assertTrue(result.error)
		self.assertIn("not a valid JSON object", result.text)
		t.sandbox.run.assert_not_called()

	def test_arguments_too_deep_to_show_are_shown_raw(self):
		with mock.patch.object(tools.json, "dumps", side_effect=RecursionError):
			self.assertEqual(tools.describe_call("read_file", '{"path": [[]]}'), 'read_file({"path": [[]]})')

	def test_sandbox_failures_are_tool_errors(self):
		t = tools.Tools(mock.Mock())
		t.sandbox.run.side_effect = SandboxError("Failed to authenticate with sudo.")
		result = t.call("run_shell_command", json.dumps({"command": "ls"}))
		self.assertEqual((result.text, result.error), ("Error: Failed to authenticate with sudo.", True))

	def test_a_full_scratch_is_reported(self):
		t = tools.Tools(mock.Mock())
		t.sandbox.scratch_usage.return_value = (0, 0)
		for exit_code, notice in ((None, f"[Killed: {tools.SCRATCH_FULL}]"), (0, "Commands are killed after a second")):
			with self.subTest(exit_code=exit_code):
				t.sandbox.run.return_value = Result(output="", exit_code=exit_code, scratch_full=True)
				self.assertIn(notice, t.call("run_shell_command", json.dumps({"command": "x"})).text)

	def test_nul_in_arguments_is_an_error(self):
		t = tools.Tools(mock.Mock())
		for name, args in (("read_file", {"path": "a\0b"}), ("run_shell_command", {"command": "echo a\0b"})):
			with self.subTest(name=name):
				result = t.call(name, json.dumps(args))
				self.assertTrue(result.error)
				self.assertIn("contains a NUL character", result.text)
		t.sandbox.run.assert_not_called()

	def test_boolean_arguments_must_be_booleans(self):
		with mock.patch.object(tools.shutil, "which", return_value="/usr/bin/rg"):
			t = tools.Tools(mock.Mock())
		result = t.call("grep_search", json.dumps({"pattern": "x", "ignore_case": "false"}))
		self.assertIn("ignore_case must be true or false", result.text)

	def test_null_optional_arguments_take_defaults(self):
		sandbox = mock.Mock()
		sandbox.run.return_value = Result(output="./a.py\n", exit_code=0, timed_out=False, truncated_bytes=0)
		with mock.patch.object(tools.shutil, "which", return_value="/usr/bin/rg"):
			t = tools.Tools(sandbox)
		self.assertFalse(t.call("glob", json.dumps({"pattern": "*.py", "path": None})).error)
		self.assertFalse(t.call("grep_search", json.dumps({"pattern": "x", "include": None, "context": None})).error)
		self.assertIn("pattern must be a string", t.call("glob", json.dumps({"pattern": None})).text)


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
		pasted = ['"""', "Module doc.", '"""', "def f():", '	"""', "	Return x.", '	"""', "	return x"]
		lines = ['"""Why does f fail?', *pasted, '"""']
		with mock.patch("builtins.input", side_effect=lambda prompt="": lines.pop(0)), \
				mock.patch.object(cli, "_input_waiting", lambda: bool(lines)):  # the paste arrives at once
			self.assertEqual(cli.read_input(), "\n".join(["Why does f fail?", *pasted]))
		# Bracketed pastes arrive as one input.
		with mock.patch("builtins.input", side_effect=['"""def f():\n	"""Return x."""', '"""']):
			self.assertEqual(cli.read_input(), 'def f():\n	"""Return x."""')
		with mock.patch("builtins.input", side_effect=['"""Module."""\nimport os']):
			self.assertEqual(cli.read_input(), '"""Module."""\nimport os')
		with mock.patch("builtins.input", side_effect=['"""Why?\nimport os\n"""']):
			self.assertEqual(cli.read_input(), "Why?\nimport os")
		with mock.patch("builtins.input", side_effect=['"""Doc.\n\nMore.\n"""\nimport os']):
			self.assertEqual(cli.read_input(), '"""Doc.\n\nMore.\n"""\nimport os')
		with mock.patch("builtins.input", side_effect=['"""Why?', 'code\nmore\n"""']):
			self.assertEqual(cli.read_input(), "Why?\ncode\nmore")
		with mock.patch("builtins.input", side_effect=['"""Why?', "if line.startswith('\"\"\"'):", '"""', "next"]):
			self.assertEqual(cli.read_input(), "Why?\nif line.startswith('\"\"\"'):")

	def test_nohup_is_kept(self):
		old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
		self.addCleanup(signal.signal, signal.SIGHUP, old)
		self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
		cli.exit_on_sigterm()
		self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

	def test_broken_pipe_in_one_shot_mode_reaches_main(self):
		args = argparse.Namespace(prompt="x", resume=None, cont=False)
		app = mock.Mock()
		app.agent.send.side_effect = BrokenPipeError()
		with mock.patch.object(cli, "parse_args", return_value=args), \
				mock.patch.object(cli.deps, "ensure", return_value=True), \
				mock.patch.object(cli, "clean_scratch"), mock.patch.object(cli, "App", return_value=app):
			with self.assertRaises(BrokenPipeError):
				cli._main()

	def test_bad_port_in_the_environment_is_an_argument_error(self):
		with mock.patch.dict(os.environ, {"LLM_PORT": "abc"}), mock.patch.object(sys, "argv", ["artifice"]), \
				mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
			cli.parse_args()

	def test_sampling_arguments_are_validated(self):
		for flag, value in (("--temperature", "0"), ("--top-p", "0"), ("--top-p", "1.5")):
			with self.subTest(flag=flag, value=value), mock.patch.object(sys, "argv", ["artifice", flag, value]), \
					mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
				cli.parse_args()


@unittest.skipUnless(shutil.which("rg"), "rg is not installed")
class SearchTest(unittest.TestCase):
	def test_unpaired_surrogates_are_rejected_before_the_sandbox(self):
		t = tools.Tools(mock.Mock())
		for name, args in (("grep_search", '{"pattern": "\\ud800"}'), ("glob", '{"pattern": "*", "path": "\\ud800"}'),
				("run_shell_command", '{"command": "echo \\udc7f"}')):
			with self.subTest(name=name):
				result = t.call(name, args)
				self.assertTrue(result.error)
				self.assertIn("unpaired surrogate", result.text)
		t.sandbox.run.assert_not_called()

	def test_leading_dot_slash_is_stripped_for_rg(self):
		with mock.patch.object(tools.shutil, "which", return_value="/usr/bin/rg"):
			t = tools.Tools(mock.Mock())
		t.sandbox.run.return_value = Result(output="", exit_code=1)
		t.call("glob", json.dumps({"pattern": "./src/**/*.py"}))
		t.call("grep_search", json.dumps({"pattern": "x", "include": "./*.py"}))
		globs = [argv[argv.index("-g") + 1] for (argv,), _ in t.sandbox.run.call_args_list]
		self.assertEqual(globs, ["src/**/*.py", "*.py"])

	def test_a_glob_filter_on_a_file_path_is_an_error(self):
		t = tools.Tools(mock.Mock())

		def run(argv, **kw):
			p = subprocess.run(argv, capture_output=True, text=True)
			return Result(output=p.stdout + p.stderr, exit_code=p.returncode)

		t.sandbox.run.side_effect = run
		with tempfile.TemporaryDirectory() as d:
			Path(d, "x.txt").write_text("needle\n")
			path = os.path.join(d, "x.txt")
			for args in ({"pattern": "*.txt", "path": path}, {"pattern": "*.rs", "path": path}):
				result = t.call("glob", json.dumps(args))
				self.assertTrue(result.error, result.text)
				self.assertIn("is not a directory", result.text)
			grep = {"pattern": "needle", "path": path}
			for include in ("*.txt", "*.py"):
				result = t.call("grep_search", json.dumps({**grep, "include": include}))
				self.assertTrue(result.error, result.text)
				self.assertIn("is not a directory", result.text)
			self.assertIn(f"{path}:1:needle", t.call("grep_search", json.dumps(grep)).text)

	def test_a_path_that_is_not_a_regular_file_or_is_named_dash(self):
		with tempfile.TemporaryDirectory() as d:
			def run(argv, **kw):
				p = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=30,
					cwd=d, env={"PATH": os.environ["PATH"], "LANG": "C.UTF-8"})
				return Result(output=p.stdout + p.stderr, exit_code=p.returncode)

			t = tools.Tools(mock.Mock())
			t.sandbox.run.side_effect = run
			os.mkfifo(Path(d, "data.log"))  # opening it would block with no writer
			for path in ("data.log", "/dev/null"):
				for include in ("*.py", "*.log", "null"):
					with self.subTest(path=path, include=include):
						grep = t.call("grep_search", json.dumps({"pattern": "x", "path": path, "include": include}))
						self.assertIn("is not a directory", grep.text)
						self.assertIn("is not a directory", t.call("glob", json.dumps({"pattern": include, "path": path})).text)
			Path(d, "-").write_text("needle\n")
			self.assertEqual(t.call("grep_search", json.dumps({"pattern": "needle", "path": "-"})).text, "./-:1:needle")
			os.remove(Path(d, "-"))
			Path(d, "-").mkdir()
			Path(d, "-", "a.txt").write_text("needle\n")
			self.assertEqual(t.call("glob", json.dumps({"pattern": "*.txt", "path": "-"})).text, "-/a.txt")
			self.assertEqual(t.call("grep_search", json.dumps({"pattern": "needle", "path": "-"})).text, "-/a.txt:1:needle")
			self.assertEqual(t.call("grep_search", json.dumps({"pattern": "needle", "path": "-/a.txt"})).text, "-/a.txt:1:needle")

	@unittest.skipUnless(sys.getfilesystemencoding() == "utf-8", "file names decode to surrogates only from UTF-8")
	def test_helpers_print_undecodable_names(self):
		with tempfile.TemporaryDirectory() as d:
			Path(d, os.fsdecode(b"n\xff")).mkdir()
			out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")  # strict, as in most UTF-8 locales
			argv = ["helpers.py", "list_directory", json.dumps({"path": d})]
			with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "stdout", out):
				helpers.main()
			out.flush()
		self.assertEqual(out.buffer.getvalue().decode(), "n\\udcff/\n")

	def test_search_options_reach_rg(self):
		def run(argv, **kw):
			p = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=30,
				env={"PATH": os.environ["PATH"], "LANG": "C.UTF-8"})  # like the sandbox's --clearenv
			return Result(output=p.stdout + p.stderr, exit_code=p.returncode)

		with tempfile.TemporaryDirectory() as d:
			Path(d, ".github").mkdir()
			Path(d, ".github", "ci.yml").write_text("--verbose\n")
			Path(d, ".git").mkdir()
			Path(d, ".git", "notes.yml").write_text("--verbose\n")
			t = tools.Tools(mock.Mock())
			t.sandbox.run.side_effect = run
			glob = t.call("glob", json.dumps({"pattern": "*.yml", "path": d}))
			self.assertEqual(glob.text, f"{d}/.github/ci.yml")
			self.assertFalse(glob.error, glob.text)
			grep = t.call("grep_search", json.dumps({"pattern": "--verbose", "path": d, "context": 10**30}))
			self.assertEqual(grep.text, f"{d}/.github/ci.yml:1:--verbose")
			self.assertEqual(t.call("grep_search", json.dumps({"pattern": "absent", "path": d})).text, "No matches.")
			self.assertTrue(t.call("grep_search", json.dumps({"pattern": "(", "path": d})).error)

	def test_a_timed_out_search_keeps_what_it_found(self):
		def run(argv, timeout=None, **kw):
			p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
			try:
				out, _ = p.communicate(timeout=3)
				return Result(output=out.decode(), exit_code=p.returncode)
			except subprocess.TimeoutExpired:
				os.killpg(p.pid, signal.SIGKILL)  # what Sandbox.run does at the timeout
				out, _ = p.communicate()
				return Result(output=out.decode(), exit_code=None, timed_out=True)

		with tempfile.TemporaryDirectory() as bin_dir, tempfile.TemporaryDirectory() as d:
			# A search that finds a match and then stalls, as on a slow mount or a blocking file.
			Path(d, "a.txt").write_text("needle\n")
			Path(d, "b.txt").write_text("needle\n")
			slow = Path(bin_dir, "slow")
			slow.write_text('#!/bin/bash\ncase "$1" in *b.txt) sleep 30;; esac; cat\n')
			stub = Path(bin_dir, "rg")
			stub.write_text(f'#!/bin/bash\nexec {shutil.which("rg")} --pre {slow} "$@"\n')
			listed = Path(d, "listed")  # rg --files opens .ignore files, and this one blocks
			listed.mkdir()
			Path(listed, "a.txt").write_text("")
			Path(listed, "z").mkdir()
			os.mkfifo(Path(listed, "z", ".ignore"))
			for script in (slow, stub):
				script.chmod(0o755)
			cases = [(str(stub), "grep_search", {"pattern": "needle", "path": d}, f"{d}/a.txt:1:needle"),
				(shutil.which("rg"), "glob", {"pattern": "*.txt", "path": str(listed)}, f"{listed}/a.txt")]
			for rg, tool, args, expected in cases:
				with self.subTest(rg=rg, tool=tool), mock.patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}):
					with mock.patch.object(tools.shutil, "which", return_value=rg):
						t = tools.Tools(mock.Mock())
					t.sandbox.run.side_effect = run
					self.assertIn(expected, t.call(tool, json.dumps(args)).text)

	def test_schemas_say_rg_globs_find_ignored_files(self):
		from artifice.tools import schemas
		glob, grep = (s["function"]["description"] for s in schemas() if s["function"]["name"] in ("glob", "grep_search"))
		self.assertIn("a matching pattern can include some; set include_ignored to list them all", glob)
		self.assertIn("a matching include glob can include some; set include_ignored to search them all", grep)


class ReadFileTest(unittest.TestCase):
	def test_line_numbers_match_grep(self):
		with tempfile.NamedTemporaryFile("wb") as f:
			f.write(b"p 10%\rp 100%\nERROR\r\n")
			f.flush()
			self.assertIn("     2\tERROR\n", helpers.read_file(f.name) + "\n")

	def test_output_fits_the_tool_cap_with_whole_lines(self):
		with tempfile.NamedTemporaryFile("w") as f:
			f.write(("x" * 1999 + "\n") * 20)
			f.flush()
			text = helpers.read_file(f.name)
		self.assertLessEqual(len(text), tools.OUTPUT_MAX_CHARS)
		last = int(text.split("Use offset=")[1].split(" ")[0]) - 1
		self.assertEqual(text.count("x" * 1999), last)


class MainTest(unittest.TestCase):
	def test_undecodable_input_does_not_crash(self):
		stdin = io.TextIOWrapper(io.BytesIO(b"hi\xff\n"), encoding="utf-8")
		with mock.patch.object(sys, "stdin", stdin), mock.patch.object(cli, "_main", lambda: print(input(), file=io.StringIO())):
			cli.main()

	def test_closed_standard_streams_do_not_crash(self):
		def use_streams():
			self.assertIsNone(cli.read_input())
			self.assertIsNone(deps._ask("Install?"))
			print("x")

		with mock.patch.object(sys, "stdin", None), mock.patch.object(sys, "stdout", None), \
				mock.patch.object(cli, "_main", use_streams):
			cli.main()


class ClientTest(unittest.TestCase):
	def stream(self, *lines: str, body: dict | None = None) -> list[dict]:
		import http.server
		import threading
		from artifice.client import LLMClient

		class Handler(http.server.BaseHTTPRequestHandler):
			def do_POST(self):
				received.append(self.rfile.read(int(self.headers["Content-Length"])))
				self.send_response(200)
				self.send_header("Content-Type", "text/event-stream")
				self.end_headers()
				self.wfile.write("".join(f"data: {line}\n\n" for line in lines).encode())

			def log_message(self, *args):
				pass

		received = self.received = []
		server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
		threading.Thread(target=server.serve_forever, daemon=True).start()
		self.addCleanup(server.server_close)
		self.addCleanup(server.shutdown)
		events = []
		try:
			for event in LLMClient("127.0.0.1", server.server_address[1]).stream(body or {}):
				events.append(event)
		finally:
			self.events = events
		return events

	def test_stream_yields_events_until_done(self):
		self.assertEqual(self.stream('{"token": "a"}', "[DONE]", '{"token": "b"}'), [{"token": "a"}])

	def test_undecodable_text_is_sent_as_valid_utf8(self):
		self.stream("[DONE]", body={"messages": [{"role": "user", "content": os.fsdecode(b"caf\xe9 caf\xc3\xa9")}]})
		self.assertEqual(json.loads(self.received[0].decode()), {"messages": [{"role": "user", "content": "caf? café"}]})

	def test_stream_without_done_is_an_error(self):
		with self.assertRaisesRegex(LLMError, "without \\[DONE\\]"):
			self.stream('{"token": "a"}')
		self.assertEqual(self.events, [{"token": "a"}])

	def test_error_event_is_an_error(self):
		with self.assertRaisesRegex(LLMError, "Failed to generate: boom"):
			self.stream('{"token": "a"}', '{"error": "boom"}', "[DONE]")

	def test_bad_host_is_an_llm_error(self):
		from artifice.client import LLMClient
		with self.assertRaisesRegex(LLMError, "Failed to reach llm-server"):
			LLMClient("local host", 8741).list()
		with self.assertRaisesRegex(LLMError, "Failed to reach llm-server"):
			LLMClient("foo..bar", 8741).list()


class StoreTest(TempDataTest):
	def test_saved_chats_are_private(self):
		self.addCleanup(os.umask, os.umask(0o022))
		os.makedirs(os.path.join(self.data.name, "artifice"), mode=0o755)
		os.chmod(os.path.join(self.data.name, "artifice"), 0o755)
		chat = self.chat(("user", "secret"))
		chat.save()
		self.assertEqual(chat.path.stat().st_mode & 0o777, 0o600)
		self.assertEqual(chat.path.parent.parent.stat().st_mode & 0o777, 0o700)
		self.assertEqual(chat.path.parent.stat().st_mode & 0o777, 0o700)
		self.assertEqual(scratch_root().stat().st_mode & 0o777, 0o700)


class DepsTest(unittest.TestCase):
	def test_rg_is_required(self):
		which = lambda command: None if command == "rg" else f"/usr/bin/{command}"
		with mock.patch.object(deps.shutil, "which", which), mock.patch.object(deps, "_manager", return_value="dnf"), \
				mock.patch.object(deps, "_ask", return_value=True), mock.patch.object(deps.subprocess, "run") as run, \
				mock.patch.object(deps.sys, "stderr"):
			self.assertFalse(deps.ensure())
		self.assertIn("ripgrep", run.call_args[0][0])


class TerminalUITest(unittest.TestCase):
	def test_one_shot_text_before_a_tool_call_goes_to_the_trace(self):
		out, trace = io.StringIO(), io.StringIO()
		ui = cli.TerminalUI(answer_out=out, trace_out=trace)
		ui.token("Let me check.")
		ui.end_stream(tool_calls=True)
		ui.reasoning("hmm")
		ui.token("6.12")
		ui.end_stream()
		self.assertEqual(out.getvalue(), "6.12\n")
		self.assertIn("Let me check.\n", trace.getvalue())

	def test_one_shot_text_followed_by_reasoning_is_kept(self):
		out, trace = io.StringIO(), io.StringIO()
		ui = cli.TerminalUI(answer_out=out, trace_out=trace)
		for text, calls in (("Let me check.", True), ("6.12", False)):
			ui.token(text)
			ui.reasoning("hmm")
			ui.end_stream(tool_calls=calls)
		self.assertEqual(out.getvalue(), "6.12\n")
		self.assertIn("Let me check.\n", trace.getvalue())

	def test_control_characters_are_escaped(self):
		out = mock.Mock()
		out.isatty.return_value = False
		ui = cli.TerminalUI(answer_out=out, trace_out=out)
		ui.token("a\x1b]52;c;cHduZWQ=\x07b\x9b2J\tc\n")
		ui.tool_result("x", ToolResult("\x1b[2Kfake"))
		written = "".join(c.args[0] for c in out.write.call_args_list)
		self.assertNotRegex(written, r"[\x1b\x07\x9b]")
		self.assertIn("a\\x1b]52;c;cHduZWQ=\\x07b\\x9b2J\tc\n", written)

	def test_every_trace_line_is_escaped(self):
		answer, trace = io.StringIO(), io.StringIO()
		ui = cli.TerminalUI(answer_out=answer, trace_out=trace)
		ui.reasoning("\x1b[2Jthinking")
		ui.token("\x1b[2Jheld")
		ui.end_stream()
		ui.tool_call("run_shell_command", json.dumps({"command": "\x1b[2Jls"}))
		ui.notice("\x1b[2Jnotice")
		self.assertNotIn("\x1b", answer.getvalue() + trace.getvalue())
		self.assertEqual((answer.getvalue() + trace.getvalue()).count("\\x1b[2J"), 4)

	def test_one_shot_errors_are_escaped(self):
		args = argparse.Namespace(prompt="x", resume=None, cont=False)
		error = LLMError("Failed to parse \x1b]52;c;cHduZWQ=\x07")
		for failing in ("open_chat", "agent.send"):
			with self.subTest(failing=failing):
				app = mock.Mock()
				app.configure_mock(**{f"{failing}.side_effect": error})
				with mock.patch.object(cli, "parse_args", return_value=args), \
						mock.patch.object(cli.deps, "ensure", return_value=True), \
						mock.patch.object(cli, "clean_scratch"), mock.patch.object(cli, "scratch_root"), \
						mock.patch.object(cli, "App", return_value=app), self.assertRaises(SystemExit) as exit:
					cli._main()
				self.assertEqual(exit.exception.code, "Failed to parse \\x1b]52;c;cHduZWQ=\\x07")


if __name__ == "__main__":
	unittest.main()
