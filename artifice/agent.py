"""The agent loop: generate, run tool calls, repeat, and compact the history when it grows too large."""
import json
from dataclasses import dataclass
from typing import Protocol

from .client import LLMClient, LLMError
from .prompt import ELIDED, SUMMARY_LABEL, SUMMARY_REQUEST, TOO_LARGE, TOO_LONG
from .store import Chat
from .tools import ToolResult, Tools

BYTES_PER_TOKEN = 3.0  # UTF-8 bytes per token: a conservative guess used only until the server reports real usage
ELIDED_CHARS_PER_TOKEN = 6.0  # low guess at the tokens elision frees, so the estimate errs high
MIN_ELIDE_CHARS = 300
SUMMARY_MAX_TOKENS = 4096
CONTEXT_SLACK = 16  # a reply that ends this close to the context ran out of room, not out of max_tokens


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
	def end_stream(self, tool_calls: bool = False) -> None: ...
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

	@property
	def usage_tokens(self) -> int | None:
		"""Tokens in messages[:usage_n] per the server's last usage report."""
		return self.chat.usage_tokens

	@usage_tokens.setter
	def usage_tokens(self, value: int | None) -> None:
		self.chat.usage_tokens = value

	@property
	def usage_n(self) -> int:
		return self.chat.usage_n

	@usage_n.setter
	def usage_n(self, value: int) -> None:
		self.chat.usage_n = value

	@property
	def bytes_per_token(self) -> float:
		"""Lowered when the server rejects a request as too long, until it reports usage again."""
		return self.chat.bytes_per_token or BYTES_PER_TOKEN

	@bytes_per_token.setter
	def bytes_per_token(self, value: float) -> None:
		self.chat.bytes_per_token = value

	@property
	def messages(self) -> list[dict]:
		return self.chat.messages

	def estimate_tokens(self) -> int:
		if self.usage_tokens is None:
			return int(self._history_bytes() / self.bytes_per_token)
		return self.usage_tokens + int(self._history_bytes(self.usage_n) / self.bytes_per_token)

	def _history_bytes(self, start: int = 0) -> int:
		"""UTF-8 bytes in messages[start:]. Chat templates drop reasoning from turns before the last user message."""
		last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=0)
		return sum(_bytes(m, reasoning=i > last_user) for i, m in enumerate(self.messages) if i >= start)

	def send(self, text: str) -> None:
		self._close_open_tool_calls("Artifice stopped before this tool finished.")  # a chat saved mid-tool
		self._forget_dropped_reasoning()
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

	def _forget_dropped_reasoning(self) -> None:
		"""Takes reasoning the server counted out of usage_tokens, since chat templates drop it after a new user message.

		The low guess at its tokens keeps the estimate high."""
		if self.usage_tokens is None:
			return
		last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=0)
		dropped = sum(_size(m.get("reasoning_content") or "")
			for m in self.messages[last_user + 1:self.usage_n] if m["role"] == "assistant")
		self.usage_tokens -= int(dropped / ELIDED_CHARS_PER_TOKEN)

	def _loop(self) -> None:
		overflowed_at: int | None = None  # history bytes at the last overflow
		replaced_at = float("inf")  # history bytes after the last replacement, so replacements cannot cycle
		while True:
			self.maybe_compact()
			max_tokens = self._max_tokens()
			try:
				message, stop_reason = self._generate(self.messages, self._template_args(), max_tokens)
			except LLMError as e:
				if not _is_overflow(e):
					raise
				# The estimate was low. Count the history as full so compaction runs, and retry while it shrinks.
				size = self._history_bytes()
				self.bytes_per_token = max(0.5, min(self.bytes_per_token, size / self.context_length))
				self.usage_tokens, self.usage_n = self.context_length, len(self.messages)
				if overflowed_at is not None and size >= overflowed_at:
					if not self._replace_largest(e) or self._history_bytes() >= min(overflowed_at, replaced_at):
						raise
					replaced_at = self._history_bytes()
					self.usage_tokens = None  # count the shrunk history again, at the lowered bytes_per_token
				else:
					overflowed_at = size
				self.ui.notice("The conversation is over the context limit. Compacting.")
				continue
			overflowed_at, replaced_at = None, float("inf")
			self.messages.append(message)
			try:
				self.chat.save()
			finally:  # after the save, so a reply is kept even if the reader of stdout is gone
				self.ui.end_stream(tool_calls=bool(message.get("tool_calls")))
			if stop_reason == "length" and self._reply_ran_out_of_context():
				# The estimate was low. The server's usage report corrects it for the next message.
				self.ui.notice("The response ran out of context and was cut off. Send another message to continue.")
			elif stop_reason == "length":
				self.ui.notice(f"Response hit max_tokens ({max_tokens}) and was cut off.")
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

	def _replace_largest(self, error: LLMError) -> bool:
		"""Replaces the text whose note frees the most bytes, for an overflow compaction cannot fix. False if none is left.

		The current request goes last, and raises LLMError, since it cannot be answered without it."""
		last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=0)
		parts = []  # (is the current request, bytes freed, message index, key, tool call index)
		for i, m in enumerate(self.messages):
			if m["role"] == "system" or _is_summary(m):  # summaries fit by their token cap, and would be redone
				continue
			content = m.get("content") or ""
			parts.append((i == last_user, _size(content) - _size(self._note(m, len(content))), i, "content", 0))
			if i > last_user:
				parts.append((False, _size(m.get("reasoning_content") or ""), i, "reasoning_content", 0))
			for j, call in enumerate(m.get("tool_calls") or []):
				args = call.get("function", {}).get("arguments") or ""
				parts.append((False, _size(json.dumps(args)) - _size(json.dumps(_call_note(args))), i, "tool_calls", j))
		parts = sorted((p for p in parts if p[1] > MIN_ELIDE_CHARS), key=lambda p: (p[0], -p[1]))
		if not parts:
			return False
		request, freed, i, key, j = parts[0]
		m = self.messages[i]
		if key == "reasoning_content":
			del m[key]
		elif key == "tool_calls":
			fn = m["tool_calls"][j].setdefault("function", {})
			fn["arguments"] = _call_note(fn.get("arguments") or "")
		else:
			chars = len(m["content"])
			m["content"] = self._note(m, chars)
		self.chat.save()
		if request:
			raise LLMError(f"Failed to send: the message is too long for the context ({chars:,} characters).") from error
		self.ui.notice(f"Replaced a long {m['role']} message with a note to fit the context (about {freed:,} bytes freed).")
		return True

	@staticmethod
	def _note(m: dict, chars: int) -> str:
		return (TOO_LARGE if m["role"] == "tool" else TOO_LONG).format(chars=chars)

	def _template_args(self) -> dict:
		if self.settings.thinking is None:
			return {}
		return {"enable_thinking": self.settings.thinking}

	def _reply_ran_out_of_context(self) -> bool:
		reported = self.usage_tokens is not None and self.usage_n == len(self.messages)  # by the reply just appended
		return reported and self.usage_tokens >= self.context_length - CONTEXT_SLACK

	def _max_tokens(self) -> int:
		room = self.context_length - self.estimate_tokens() - 256
		return min(self.settings.max_tokens, max(256, room))

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
						self.bytes_per_token = BYTES_PER_TOKEN  # the count covers any dense text that lowered it
					self.context_length = usage.get("context_length") or self.context_length
		except BaseException:
			if display:
				self.ui.end_stream(tool_calls=bool(tool_calls))
			raise

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
		if self._drop_turn_reasoning() < limit or self._elide(target, keep=1) < limit:
			return
		# The newest output may be unseen, so it goes only when the reply would not fit beside it.
		reserve = min(self.settings.max_tokens, self.context_length // 8)
		if self.estimate_tokens() >= self.context_length - reserve and self._elide(target, keep=0) >= limit:
			self.ui.notice("The conversation is still near the context limit. Start a new chat with /new.")

	def _drop_turn_reasoning(self) -> int:
		"""Clears reasoning from the current turn's steps before the newest, which templates send until the next user message."""
		last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=0)
		steps = [i for i, m in enumerate(self.messages)
			if i > last_user and m["role"] == "assistant" and m.get("reasoning_content")]
		freed = 0
		for i in steps[:-1]:
			size = _size(self.messages[i].pop("reasoning_content"))
			freed += size if i < self.usage_n else 0
		estimate = self.estimate_tokens()
		if len(steps) > 1:
			if self.usage_tokens is not None:
				self.usage_tokens -= int(freed / ELIDED_CHARS_PER_TOKEN)
				estimate = self.estimate_tokens()
			self.chat.save()
			self.ui.notice(f"Dropped reasoning from {len(steps) - 1} earlier steps (about {estimate:,} tokens now).")
		return estimate

	def _elide(self, target: float, keep: int) -> int:
		estimate = self.estimate_tokens()
		candidates = [i for i, m in enumerate(self.messages)
			if m["role"] == "tool" and m["content"] != ELIDED and len(m["content"]) > MIN_ELIDE_CHARS]
		seen = max((i for i, m in enumerate(self.messages) if m["role"] == "assistant"), default=-1)
		elided = 0
		for i in candidates[:max(0, len(candidates) - keep)]:
			if estimate <= target:
				break
			size = len(self.messages[i]["content"])
			note = ELIDED if i < seen else TOO_LARGE.format(chars=size)
			freed = int((size - len(note)) / ELIDED_CHARS_PER_TOKEN)
			self.messages[i]["content"] = note
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
		if reply.get("tool_calls"):
			self.ui.notice("Failed to summarize: the model called a tool instead of writing notes.")
			return False
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


def _is_overflow(e: LLMError) -> bool:
	text = str(e).lower()
	return "context size" in text or "exceed_context" in text or "context length" in text


def _size(text: str) -> int:
	return len(text.encode(errors="surrogatepass"))


def _call_note(arguments: str) -> str:
	return json.dumps({"note": TOO_LONG.format(chars=len(arguments))})


def _bytes(m: dict, reasoning: bool = True) -> int:
	"""UTF-8 bytes, which track tokens better than characters for text like CJK."""
	n = _size(m.get("content") or "")
	if reasoning:
		n += _size(m.get("reasoning_content") or "")
	if m.get("tool_calls"):
		n += len(json.dumps(m["tool_calls"]))
	return n
