"""Agent Client Protocol server, so clients like T3 Code can run Artifice chats over stdin and stdout.

Requests run one at a time on the main thread, in order. A reader thread queues them and handles session/cancel
at once, by sending SIGUSR1 to the main thread, which raises KeyboardInterrupt in the running turn.
"""
import argparse
import json
import os
import secrets
import signal
import sys
import threading
import traceback
from queue import Queue
from urllib.parse import unquote, urlparse

from .agent import is_summary
from .cli import App
from .client import LLMError
from .sandbox import SandboxError
from .store import Chat
from .tools import ToolResult, describe_call

PROTOCOL_VERSION = 1
KINDS = {"read_file": "read", "list_directory": "read", "glob": "search", "grep_search": "search",
	"run_shell_command": "execute"}
COMMANDS = [{"name": "compact", "description": "Summarize the conversation so far into notes"}]
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
NOT_FOUND = -32002


class RpcError(Exception):
	def __init__(self, code: int, message: str, data: dict | None = None):
		super().__init__(message)
		self.code = code
		self.data = data


class AcpUI:
	"""Sends a session's output as session/update notifications. Notices outside a prompt turn go to stderr."""

	def __init__(self, server: "Server"):
		self.server = server
		self.session_id = ""
		self.in_turn = False
		self.call_id = ""

	def update(self, kind: str, /, **fields) -> None:
		self.server.notify("session/update", {"sessionId": self.session_id, "update": {"sessionUpdate": kind, **fields}})

	def reasoning(self, text: str) -> None:
		self.update("agent_thought_chunk", content=_text(text))

	def token(self, text: str) -> None:
		self.update("agent_message_chunk", content=_text(text))

	def end_stream(self, tool_calls: bool = False) -> None:
		pass

	def tool_call(self, name: str, arguments: str) -> None:
		self.call_id = f"call-{secrets.token_hex(8)}"
		self.update("tool_call", **_tool_call(self.call_id, name, arguments), status="in_progress")

	def tool_result(self, name: str, result: ToolResult) -> None:
		self.update("tool_call_update", toolCallId=self.call_id, status="failed" if result.error else "completed",
			content=_tool_output(result.text))

	def notice(self, text: str) -> None:
		if self.in_turn:
			self.update("agent_message_chunk", content=_text(f"\n\n[{text}]\n\n"))
		else:
			print(f"[{text}]", file=sys.stderr)

	def replay(self, messages: list[dict]) -> None:
		"""Sends a saved chat's history, as session/load requires."""
		call_ids: dict[str, str] = {}  # the model's call id -> its id in this replay, which must be unique
		for i, m in enumerate(messages):
			if m["role"] == "user":
				kind = "agent_thought_chunk" if is_summary(m) else "user_message_chunk"
				self.update(kind, content=_text(m["content"]))
			elif m["role"] == "assistant":
				if m.get("reasoning_content"):
					self.reasoning(m["reasoning_content"])
				if m.get("content"):
					self.token(m["content"])
				for call in m.get("tool_calls") or []:
					fn = call.get("function", {})
					call_ids[call.get("id", "")] = call_id = f"call-{i}-{call.get('id', '')}"
					self.update("tool_call", **_tool_call(call_id, fn.get("name", ""), fn.get("arguments", "")),
						status="completed")
			elif m["role"] == "tool" and m.get("tool_call_id", "") in call_ids:
				self.update("tool_call_update", toolCallId=call_ids[m["tool_call_id"]], content=_tool_output(m["content"]))


class Server:
	def __init__(self, args: argparse.Namespace, stdin, stdout):
		self.args = args
		self.stdin = stdin
		self.stdout = stdout
		self.apps: dict[str, App] = {}
		self.queue: Queue = Queue()
		self.write_lock = threading.Lock()
		self.lock = threading.Lock()  # guards the prompt counts below and running
		self.queued: dict[str, int] = {}  # session id -> prompts received
		self.cancelled: dict[str, int] = {}  # session id -> prompts received when it was last cancelled
		self.running: tuple[str, int] | None = None  # the session and number of the prompt running now
		self.interrupted = False  # whether the running prompt was interrupted, so cleanup is not interrupted again
		self.initialized = False

	def serve(self) -> None:
		signal.signal(signal.SIGUSR1, self._interrupt)
		threading.Thread(target=self._read, daemon=True).start()
		while (item := self.queue.get()) is not None:
			self._handle(*item)

	def send(self, message: dict) -> None:
		line = json.dumps(message) + "\n"
		with self.write_lock:
			try:
				signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})  # so a cancel cannot cut a message short
				self.stdout.write(line)
				self.stdout.flush()
			finally:
				signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})

	def notify(self, method: str, params: dict) -> None:
		self.send({"jsonrpc": "2.0", "method": method, "params": params})

	def _read(self) -> None:
		try:
			for line in self.stdin:
				if not line.strip():
					continue
				try:
					message = json.loads(line)
				except (ValueError, RecursionError) as e:
					self.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Failed to parse message: {e}"}})
					continue
				if isinstance(message, dict) and "method" not in message and ("result" in message or "error" in message):
					continue  # a response, and Artifice sends no requests
				if not isinstance(message, dict) or not isinstance(message.get("method"), str):
					request_id = message.get("id") if isinstance(message, dict) else None
					self.send({"jsonrpc": "2.0", "id": request_id if type(request_id) in (str, int) else None,
						"error": {"code": INVALID_REQUEST, "message": f"Failed to read message: want a request, got {line!r:.200}"}})
					continue
				params = message.get("params") if isinstance(message.get("params"), dict) else {}
				session_id = params.get("sessionId") if isinstance(params.get("sessionId"), str) else ""
				if message["method"] == "session/cancel":
					self._cancel([session_id])
					continue
				number = 0
				if message["method"] == "session/prompt":
					with self.lock:
						number = self.queued[session_id] = self.queued.get(session_id, 0) + 1
				self.queue.put((message, number))
		finally:
			self._cancel(list(self.queued))  # the client is gone
			self.queue.put(None)

	def _cancel(self, session_ids: list) -> None:
		with self.lock:
			for session_id in session_ids:
				self.cancelled[session_id] = self.queued.get(session_id, 0)
			if self._cancelling():
				signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)

	def _cancelling(self) -> bool:
		running = self.running
		return running is not None and running[1] <= self.cancelled.get(running[0], 0)

	def _interrupt(self, signum, frame) -> None:
		if self._cancelling() and not self.interrupted:
			self.interrupted = True
			raise KeyboardInterrupt

	def _handle(self, message: dict, number: int) -> None:
		method, request_id = message["method"], message.get("id")
		handlers = {
			"initialize": self.initialize,
			"session/new": self.new_session,
			"session/load": self.load_session,
			"session/resume": self.resume_session,
			"session/prompt": lambda params: self.prompt(params, number),
			"session/set_config_option": self.set_config_option,
		}
		try:
			params = message.get("params", {})
			if not isinstance(params, dict):
				raise RpcError(INVALID_PARAMS, f"Failed to read params: want an object, got {params!r:.200}")
			if method != "initialize" and not self.initialized:
				raise RpcError(INVALID_REQUEST, f"Failed to run {method}: send initialize first")
			if method not in handlers:
				raise RpcError(METHOD_NOT_FOUND, f"Failed to find method {method}")
			result = handlers[method](params)
		except RpcError as e:
			error = {"code": e.code, "message": str(e)} | ({"data": e.data} if e.data else {})
			if request_id is not None:
				self.send({"jsonrpc": "2.0", "id": request_id, "error": error})
			return
		except (LLMError, SandboxError, OSError, ValueError) as e:
			if request_id is not None:
				self.send({"jsonrpc": "2.0", "id": request_id, "error": {"code": INTERNAL_ERROR, "message": str(e)}})
			return
		except Exception as e:  # a bug fails one request, not every session
			traceback.print_exc()
			if request_id is not None:
				self.send({"jsonrpc": "2.0", "id": request_id, "error": {"code": INTERNAL_ERROR,
					"message": f"Failed to run {method}: {type(e).__name__}: {e}"}})
			return
		if request_id is not None:
			self.send({"jsonrpc": "2.0", "id": request_id, "result": result})
		if method in ("session/new", "session/load", "session/resume"):
			app = self.apps[result.get("sessionId") or params["sessionId"]]
			app.ui.update("available_commands_update", availableCommands=COMMANDS)
			self._report_usage(app)

	def initialize(self, params: dict) -> dict:
		self.initialized = True
		return {
			"protocolVersion": PROTOCOL_VERSION,
			"agentCapabilities": {
				"loadSession": True,
				"promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
				"sessionCapabilities": {"resume": {}},
			},
			"authMethods": [],
		}

	def new_session(self, params: dict) -> dict:
		cwd = _field(params, "cwd", str)
		if not os.path.isabs(cwd):
			raise RpcError(INVALID_PARAMS, f"Failed to start a session in {cwd!r}: want an absolute path")
		app = App(argparse.Namespace(**vars(self.args)), AcpUI(self))
		chat = app.new_chat(cwd)
		app.open_chat(chat, load=False)
		app.ui.session_id = chat.id
		self.apps[chat.id] = app
		return {"sessionId": chat.id, "configOptions": self._config_options(app)}

	def load_session(self, params: dict) -> dict:
		app = self._open(_field(params, "sessionId", str))
		app.ui.replay(app.agent.messages)
		return {"configOptions": self._config_options(app)}

	def resume_session(self, params: dict) -> dict:
		app = self._open(_field(params, "sessionId", str))
		return {"configOptions": self._config_options(app)}

	def _open(self, session_id: str) -> App:
		if session_id in self.apps:
			return self.apps[session_id]
		try:
			chat = Chat.load(session_id)
		except FileNotFoundError:
			chat = None
		if chat is None or chat.id != session_id:
			raise RpcError(NOT_FOUND, f"Failed to find session {session_id}", {"sessionId": session_id})
		app = App(argparse.Namespace(**vars(self.args)), AcpUI(self))
		app.open_chat(chat, load=False)
		app.ui.session_id = session_id
		self.apps[session_id] = app
		return app

	def _session(self, params: dict) -> App:
		session_id = _field(params, "sessionId", str)
		if session_id not in self.apps:
			raise RpcError(NOT_FOUND, f"Failed to find session {session_id}", {"sessionId": session_id})
		return self.apps[session_id]

	def prompt(self, params: dict, number: int) -> dict:
		app = self._session(params)
		text = _prompt_text(_field(params, "prompt", list))
		session_id = app.agent.chat.id
		with self.lock:
			if number <= self.cancelled.get(session_id, 0):
				return {"stopReason": "cancelled"}
			self.running, self.interrupted = (session_id, number), False
		try:
			try:
				app.ensure_model()
				app.ui.in_turn = True
				if text.strip() == "/compact":
					app.agent.maybe_compact(force=True)
				else:
					app.agent.send(text)
			finally:
				self.running = None
				app.ui.in_turn = False
		except KeyboardInterrupt:
			return {"stopReason": "cancelled"}
		self._report_usage(app)
		return {"stopReason": "end_turn"}

	def set_config_option(self, params: dict) -> dict:
		app = self._session(params)
		config_id, value = _field(params, "configId", str), _field(params, "value", str)
		if config_id != "model":
			raise RpcError(INVALID_PARAMS, f"Failed to set option {config_id}: the only option is model")
		if value not in _model_names(app):
			raise RpcError(INVALID_PARAMS, f"Failed to set model {value}: llm-server does not list it")
		previous, app.args.model = app.args.model, value
		try:
			app.open_chat(app.agent.chat, load=False)
		except BaseException:
			app.args.model = previous
			raise
		self._report_usage(app)
		return {"configOptions": self._config_options(app)}

	def _config_options(self, app: App) -> list[dict]:
		names = _model_names(app)
		if app.settings.model not in names:
			names.insert(0, app.settings.model)
		return [{"id": "model", "name": "Model", "category": "model", "type": "select",
			"currentValue": app.settings.model, "options": [{"value": n, "name": n} for n in names]}]

	def _report_usage(self, app: App) -> None:
		if not app.agent.context_length:
			return  # the model loads at the first prompt
		app.ui.update("usage_update", used=app.agent.estimate_tokens(), size=app.agent.context_length)


def serve(args: argparse.Namespace) -> None:
	"""Serves ACP on stdin and stdout. Anything else written to stdout goes to stderr, so it cannot corrupt the protocol."""
	stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w")
	sys.stdout.flush()
	os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
	Server(args, sys.stdin.buffer, stdout).serve()


def _field(params: dict, name: str, kind: type):
	value = params.get(name)
	if not isinstance(value, kind):
		raise RpcError(INVALID_PARAMS, f"Failed to read {name}: want a {kind.__name__}, got {value!r:.200}")
	return value


def _model_names(app: App) -> list[str]:
	return [m["name"] for m in app.client.list()]


def _text(text: str) -> dict:
	return {"type": "text", "text": text}


def _prompt_text(blocks: list) -> str:
	"""The prompt as one message. Linked and embedded files become their paths and contents."""
	parts = []
	for block in blocks:
		kind = block.get("type") if isinstance(block, dict) else None
		if kind == "text":
			parts.append(_field(block, "text", str))
		elif kind == "resource_link":
			parts.append(f"@{_path(_field(block, 'uri', str))}")
		elif kind == "resource":
			resource = _field(block, "resource", dict)
			path = _path(_field(resource, "uri", str))
			if "text" in resource:
				parts.append(f"\n\nContents of {path}:\n{_fenced(_field(resource, 'text', str))}\n")
			else:
				parts.append(f"@{path}")
		else:
			raise RpcError(INVALID_PARAMS, f"Failed to read prompt: unsupported content {block!r:.200}")
	return "".join(parts)


def _path(uri: str) -> str:
	parsed = urlparse(uri)
	return unquote(parsed.path) if parsed.scheme == "file" else uri


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
	try:
		args = json.loads(arguments or "{}")
	except (ValueError, RecursionError):
		args = arguments
	return {"toolCallId": call_id, "title": describe_call(name, arguments), "kind": KINDS.get(name, "other"),
		"rawInput": args}


def _tool_output(text: str) -> list[dict]:
	return [{"type": "content", "content": _text(_fenced(text))}]


def _fenced(text: str) -> str:
	"""text as a Markdown code block, with a fence longer than any backtick run inside it."""
	fence = "```"
	while fence in text:
		fence += "`"
	return f"{fence}\n{text}\n{fence}"
