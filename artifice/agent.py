"""The agent loop: generate, run tool calls, repeat, and compact the history when it grows too large."""
import json
from dataclasses import dataclass
from typing import Protocol

from .client import LLMClient, LLMError
from .prompt import ELIDED, SUMMARY_LABEL, SUMMARY_REQUEST
from .store import Chat
from .tools import ToolResult, Tools

CHARS_PER_TOKEN = 3.0  # conservative guess, used only until the server reports real usage
ELIDED_CHARS_PER_TOKEN = 6.0  # low guess at the tokens elision frees, so the estimate errs high
MIN_ELIDE_CHARS = 300
SUMMARY_MAX_TOKENS = 4096


@dataclass
class Settings:
	model: str
	ttl: int = 3600
	max_tokens: int = 16384  # thinking and answer together
	temperature: float | None = 0.6
	top_p: float | None = 0.95
	thinking: bool | None = None  # None leaves the model's default
	compact_at: float = 0.75  # fraction of the context that triggers compaction
	compact_to: float = 0.5  # fraction to elide old tool outputs down to
	keep_tool_results: int = 6  # newest tool outputs that are never elided


class UI(Protocol):
	def reasoning(self, text: str) -> None: ...
	def token(self, text: str) -> None: ...
	def end_stream(self) -> None: ...
	def tool_call(self, name: str, arguments: str) -> None: ...
	def tool_result(self, name: str, result: ToolResult) -> None: ...
	def notice(self, text: str) -> None: ...


class Agent:
	def __init__(self, client: LLMClient, chat: Chat, tools: Tools, settings: Settings, ui: UI, context_length: int):
		self.client = client
		self.chat = chat
		self.tools = tools
		self.settings = settings
		self.ui = ui
		self.context_length = context_length
		# Tokens in messages[:usage_n] per the server's last usage report.
		self.usage_tokens: int | None = None
		self.usage_n = 0

	@property
	def messages(self) -> list[dict]:
		return self.chat.messages

	def estimate_tokens(self) -> int:
		if self.usage_tokens is None:
			return int(sum(map(_chars, self.messages)) / CHARS_PER_TOKEN)
		return self.usage_tokens + int(sum(map(_chars, self.messages[self.usage_n:])) / CHARS_PER_TOKEN)

	def send(self, text: str) -> None:
		self.messages.append({"role": "user", "content": text})
		self.chat.save()
		try:
			self._loop()
		except KeyboardInterrupt:
			self._close_open_tool_calls("Interrupted by the user before this tool finished.")
			self.ui.notice("Interrupted.")
			raise
		except BaseException:
			self._close_open_tool_calls("Artifice stopped before this tool finished.")
			raise
		finally:
			self.chat.save()

	def _loop(self) -> None:
		while True:
			self.maybe_compact()
			message, stop_reason = self._generate(self.messages, self._template_args(), self._max_tokens())
			self.messages.append(message)
			self.chat.save()
			if stop_reason == "length":
				self.ui.notice(f"Response hit max_tokens ({self._max_tokens()}) and was cut off.")
			calls = message.get("tool_calls") or []
			if not calls:
				return
			for call in calls:
				fn = call.get("function", {})
				name, arguments = fn.get("name", ""), fn.get("arguments", "")
				self.ui.tool_call(name, arguments)
				result = self.tools.call(name, arguments)
				self.ui.tool_result(name, result)
				self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result.text})
			self.chat.save()

	def _template_args(self) -> dict:
		if self.settings.thinking is None:
			return {}
		return {"enable_thinking": self.settings.thinking}

	def _max_tokens(self) -> int:
		room = self.context_length - self.estimate_tokens() - 256
		return max(256, min(self.settings.max_tokens, room))

	def _generate(self, messages: list[dict], template_args: dict, max_tokens: int,
			display: bool = True) -> tuple[dict, str]:
		body: dict = {
			"model": self.settings.model,
			"messages": messages,
			"tools": self.tools.schemas,
			"autoload": True,
			"ttl": self.settings.ttl,
			"max_tokens": max_tokens,
		}
		if template_args:
			body["template_args"] = template_args
		if self.settings.temperature is not None:
			body["temperature"] = self.settings.temperature
		if self.settings.top_p is not None:
			body["top_p"] = self.settings.top_p

		content: list[str] = []
		reasoning: list[str] = []
		tool_calls: list[dict] = []
		stop_reason = ""
		try:
			for event in self.client.stream(body):
				if "token" in event:
					content.append(event["token"])
					if display:
						self.ui.token(event["token"])
				elif "reasoning" in event:
					reasoning.append(event["reasoning"])
					if display:
						self.ui.reasoning(event["reasoning"])
				elif "tool_calls" in event:
					tool_calls = event["tool_calls"]
				elif "stop_reason" in event:
					stop_reason = event["stop_reason"]
				elif "usage" in event and messages is self.messages:
					usage = event["usage"] or {}
					if usage.get("prompt_tokens") is not None:
						self.usage_tokens = usage["prompt_tokens"] + (usage.get("completion_tokens") or 0)
						self.usage_n = len(self.messages) + 1  # counts the reply about to be appended
					self.context_length = usage.get("context_length") or self.context_length
		finally:
			if display:
				self.ui.end_stream()

		message: dict = {"role": "assistant", "content": "".join(content)}
		if reasoning:
			message["reasoning_content"] = "".join(reasoning)
		if tool_calls:
			message["tool_calls"] = tool_calls
		return message, stop_reason

	def _close_open_tool_calls(self, text: str) -> None:
		"""Adds a result for every tool call in the last assistant message that has none."""
		for i in range(len(self.messages) - 1, -1, -1):
			m = self.messages[i]
			if m["role"] == "assistant":
				answered = {r.get("tool_call_id") for r in self.messages[i + 1:] if r["role"] == "tool"}
				for call in m.get("tool_calls") or []:
					if call.get("id", "") not in answered:
						self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": text})
				return
			if m["role"] == "user":
				return

	def maybe_compact(self, force: bool = False) -> None:
		"""Elides old tool outputs past compact_at, and summarizes when that is not enough.

		force summarizes the whole conversation, for use between turns."""
		limit = self.settings.compact_at * self.context_length
		target = self.settings.compact_to * self.context_length
		if not force and self.estimate_tokens() < limit:
			return
		if not force and self._elide(target, self.settings.keep_tool_results) < limit:
			return
		if self._summarize(keep_current=not force) and self.estimate_tokens() < limit:
			return
		if self._elide(target, keep=1) >= limit and self._elide(target, keep=0) >= limit:
			self.ui.notice("The conversation is still near the context limit. Start a new chat with /new.")

	def _elide(self, target: float, keep: int) -> int:
		estimate = self.estimate_tokens()
		candidates = [i for i, m in enumerate(self.messages)
			if m["role"] == "tool" and m["content"] != ELIDED and len(m["content"]) > MIN_ELIDE_CHARS]
		elided = 0
		for i in candidates[:max(0, len(candidates) - keep)]:
			if estimate <= target:
				break
			freed = int((len(self.messages[i]["content"]) - len(ELIDED)) / ELIDED_CHARS_PER_TOKEN)
			self.messages[i]["content"] = ELIDED
			elided += 1
			# Without a server count, the estimate is a character count, so count again.
			estimate = estimate - freed if self.usage_tokens is not None else self.estimate_tokens()
		if elided:
			if self.usage_tokens is not None:
				self.usage_tokens, self.usage_n = estimate, len(self.messages)
			self.chat.save()
			self.ui.notice(f"Elided {elided} old tool outputs (about {estimate:,} tokens now).")
		return estimate

	def _summarize(self, keep_current: bool) -> bool:
		"""Replaces the history with notes, keeping the current request and its tool calls when keep_current."""
		if keep_current:
			last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=0)
		else:
			last_user = len(self.messages)
		if all(m["role"] == "system" or _is_summary(m) for m in self.messages[:last_user]):
			if not keep_current:
				self.ui.notice("Nothing to summarize.")
			return False
		self.ui.notice("Summarizing earlier conversation...")
		request = self.messages[:last_user] + [{"role": "user", "content": SUMMARY_REQUEST}]
		try:
			reply, _ = self._generate(request, {"enable_thinking": False}, SUMMARY_MAX_TOKENS, display=False)
		except LLMError as e:
			self.ui.notice(f"Failed to summarize: {e}")
			return False
		notes = reply["content"].strip()
		if not notes:
			self.ui.notice("Failed to summarize: the model returned no notes.")
			return False
		system = [m for m in self.messages[:1] if m["role"] == "system"]
		self.chat.messages = [*system, {"role": "user", "content": SUMMARY_LABEL.format(notes=notes)},
			*self.messages[last_user:]]
		self.chat.compactions += 1
		self.usage_tokens = None
		self.chat.save()
		self.ui.notice(f"Compacted earlier conversation into notes (about {self.estimate_tokens():,} tokens now).")
		return True


def _is_summary(m: dict) -> bool:
	return m["role"] == "user" and (m.get("content") or "").startswith("[Compacted context]")


def _chars(m: dict) -> int:
	n = len(m.get("content") or "") + len(m.get("reasoning_content") or "")
	if m.get("tool_calls"):
		n += len(json.dumps(m["tool_calls"]))
	return n
