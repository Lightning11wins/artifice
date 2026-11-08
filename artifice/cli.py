"""Terminal chat front end."""
import argparse
import os
import re
import readline
import signal
import sys
from pathlib import Path

from . import deps
from .agent import Agent, Settings
from .client import LLMClient, LLMError
from .prompt import system_prompt
from .sandbox import READ_ONLY, Sandbox, SandboxError, clean_scratch, hidden_in_sandbox
from .store import Chat, data_dir, list_chats, scratch_root
from .tools import ToolResult, Tools, describe_call

DEFAULT_MODEL = "qwen3.6-35b-a3b"
FALLBACK_CONTEXT = 32768
HELP = """Commands:
  /new              start a new chat
  /chats            list recent chats
  /resume ID        switch to a saved chat (an ID prefix is enough)
  /compact          summarize the conversation so far into notes
  /think on|off|default   turn model thinking on or off
  /reasoning        show or hide the model's thinking
  /usage            show context usage
  /quit             exit (or Ctrl-D)
Start a message with \"\"\" to type several lines; end it with \"\"\" on its own line."""


# Control characters other than tab and newline, which could move the cursor or retitle the terminal.
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _printable(text: str) -> str:
	return CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


class TerminalUI:
	def __init__(self, answer_out=sys.stdout, trace_out=sys.stdout, show_reasoning: bool = True):
		self.out = answer_out
		self.trace = trace_out
		self.show_reasoning = show_reasoning
		self.color = trace_out.isatty()
		self.state: str | None = None

	def _c(self, code: str, text: str) -> str:
		return f"\033[{code}m{text}\033[0m" if self.color else text

	def reasoning(self, text: str) -> None:
		if not self.show_reasoning:
			return
		self.state = "reasoning"
		self.trace.write(self._c("2", _printable(text)))
		self.trace.flush()

	def token(self, text: str) -> None:
		if self.state == "reasoning":
			self.trace.write("\n\n")
			self.trace.flush()
		self.state = "answer"
		self.out.write(_printable(text))
		self.out.flush()

	def end_stream(self) -> None:
		if self.state == "answer":
			self.out.write("\n")
			self.out.flush()
		elif self.state == "reasoning":
			self.trace.write("\n")
		self.state = None

	def tool_call(self, name: str, arguments: str) -> None:
		line = _printable(describe_call(name, arguments))
		if len(line) > 300:
			line = line[:300] + "..."
		self.trace.write(self._c("36", f"● {line}") + "\n")
		self.trace.flush()

	def tool_result(self, name: str, result: ToolResult) -> None:
		lines = result.text.splitlines() or [""]
		shown = [_printable(line[:160]) for line in lines[:4]]
		if len(lines) > 4:
			shown.append(f"... ({len(lines) - 4} more lines)")
		code = "31" if result.error else "2"
		self.trace.write("".join(self._c(code, f"  │ {line}") + "\n" for line in shown))
		self.trace.flush()

	def notice(self, text: str) -> None:
		self.trace.write(self._c("33", f"[{_printable(text)}]") + "\n")
		self.trace.flush()


def _positive(text: str) -> float:
	if not float(text) > 0:
		raise argparse.ArgumentTypeError(f"{text} is not greater than 0")
	return float(text)


def _probability(text: str) -> float:
	if not 0 < float(text) <= 1:
		raise argparse.ArgumentTypeError(f"{text} is not in (0, 1]")
	return float(text)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(prog="artifice", description="Chat with a local model that has read-only access to this computer.")
	p.add_argument("--model", help=f"model name from `llm list` (default {DEFAULT_MODEL})")
	p.add_argument("--host", default=os.environ.get("LLM_HOST", "localhost"))
	p.add_argument("--port", type=int, default=int(os.environ.get("LLM_PORT", "8741")))
	p.add_argument("-r", "--resume", metavar="ID", help="resume a saved chat")
	p.add_argument("-c", "--continue", dest="cont", action="store_true", help="resume the most recent chat")
	p.add_argument("-p", "--prompt", help="answer one prompt and exit; the answer goes to stdout, the trace to stderr")
	p.add_argument("--think", choices=["on", "off"], help="force model thinking on or off")
	p.add_argument("--max-tokens", type=int, default=16384, help="per-response limit, thinking included")
	p.add_argument("--temperature", type=_positive, default=0.6, help="greater than 0")
	p.add_argument("--top-p", type=_probability, default=0.95, help="greater than 0, at most 1")
	p.add_argument("--check-deps", action="store_true", help="ask again about optional tools you declined")
	return p.parse_args()


COMMANDS = {"quit", "exit", "q", "help", "new", "chats", "resume", "compact", "think", "reasoning", "usage"}


def is_command(line: str) -> bool:
	"""Whether line is a slash command. A message can start with a path like /boot or /var/log."""
	if not line.startswith("/"):
		return False
	word = line.split(maxsplit=1)[0]
	return word[1:] in COMMANDS or ("/" not in word[1:] and not os.path.exists(word))


def exit_on_sigterm() -> None:
	"""Turns SIGTERM and SIGHUP into SystemExit, so cleanup code runs and kills the sandbox's running command."""
	for signum in (signal.SIGTERM, signal.SIGHUP):
		signal.signal(signum, lambda signum, frame: sys.exit(128 + signum))


def read_input() -> str | None:
	try:
		line = input("\n\001\033[1;32m\002>\001\033[0m\002 " if sys.stdout.isatty() else "\n> ")
	except EOFError:
		return None
	if not line.lstrip().startswith('"""'):
		return line
	first = line.lstrip()[3:]
	if first.rstrip().endswith('"""'):
		return first.rstrip()[:-3]
	lines = [first] if first.strip() else []
	while True:
		try:
			more = input("")
		except EOFError:
			break
		if more.strip() == '"""':
			break
		lines.append(more)
	return "\n".join(lines)


class App:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.client = LLMClient(args.host, args.port)
		self.settings = Settings(
			model=args.model or DEFAULT_MODEL,
			max_tokens=args.max_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			thinking=None if args.think is None else args.think == "on",
		)
		one_shot = args.prompt is not None
		self.ui = TerminalUI(answer_out=sys.stdout, trace_out=sys.stderr if one_shot else sys.stdout)
		self.context_length = 0
		self.loaded: str | None = None
		self.agent: Agent | None = None

	def load_model(self, model: str) -> int:
		"""Loads model and returns its context length."""
		self.ui.notice(f"Loading {model} from {self.args.host}:{self.args.port}...")
		context_length = self.client.load(model, self.settings.ttl).get("context_length")
		if not context_length:
			self.ui.notice(f"The server did not report a context length. Assuming {FALLBACK_CONTEXT:,} tokens.")
		return context_length or FALLBACK_CONTEXT

	def open_chat(self, chat: Chat) -> None:
		"""Switches to chat. On failure, the current chat and its settings are unchanged."""
		model = self.args.model or chat.model
		current = self.agent
		if current and current.chat.id == chat.id:
			sandbox = current.tools.sandbox  # this session already holds the chat's lock
		else:
			sandbox = Sandbox(READ_ONLY, scratch_root() / chat.id, chat.cwd, notify=self.ui.notice, exclusive=True)
		context_length = self.context_length if model == self.loaded else self.load_model(model)
		sandbox.check()
		self.loaded, self.context_length = model, context_length
		self.settings.model = chat.model = model
		self.agent = Agent(self.client, chat, Tools(sandbox), self.settings, self.ui, self.context_length)
		if current and current.tools.sandbox is not sandbox:
			current.tools.sandbox.close()

	def new_chat(self) -> Chat:
		try:
			cwd = os.getcwd()
		except FileNotFoundError:
			cwd = str(Path.home())
			self.ui.notice(f"The current directory was deleted, so the chat starts in {cwd}.")
		if hidden_in_sandbox(cwd):
			self.ui.notice(f"The sandbox hides {cwd}, so the chat starts in {Path.home()} instead.")
			cwd = str(Path.home())
		chat = Chat.new(self.settings.model, READ_ONLY.name, cwd)
		chat.messages.append({"role": "system", "content": system_prompt(cwd, READ_ONLY.name)})
		return chat

	def banner(self) -> None:
		chat = self.agent.chat
		self.ui.notice(f"{self.settings.model} | {self.context_length:,} token context | profile {chat.profile} | "
			f"cwd {chat.cwd} | chat {chat.id}")

	def command(self, line: str) -> bool:
		"""Handles a slash command. Returns False to quit."""
		name, _, arg = line[1:].partition(" ")
		arg = arg.strip()
		if name in ("quit", "exit", "q"):
			return False
		if name == "help":
			print(HELP)
		elif name == "new":
			self.open_chat(self.new_chat())
			self.banner()
		elif name == "chats":
			for c in list_chats():
				print(f"  {c.id}  {c.updated}  {c.title}")
		elif name == "resume":
			try:
				self.open_chat(Chat.load(arg))
			except (FileNotFoundError, ValueError, LLMError) as e:
				self.ui.notice(str(e))
				return True
			self.banner()
			self.ui.notice(f"Resumed with {len(self.agent.messages)} messages.")
		elif name == "compact":
			self.agent.maybe_compact(force=True)
		elif name == "think":
			values = {"on": True, "off": False, "default": None}
			if arg not in values:
				self.ui.notice("Usage: /think on|off|default")
			else:
				self.settings.thinking = values[arg]
				self.ui.notice(f"Thinking: {arg}")
		elif name == "reasoning":
			self.ui.show_reasoning = not self.ui.show_reasoning
			self.ui.notice(f"Showing reasoning: {'on' if self.ui.show_reasoning else 'off'}")
		elif name == "usage":
			used = self.agent.estimate_tokens()
			self.ui.notice(f"About {used:,} of {self.agent.context_length:,} tokens "
				f"({100 * used // max(1, self.agent.context_length)}%), {self.agent.chat.compactions} compactions")
		else:
			self.ui.notice(f"Unknown command /{name}. Type /help.")
		return True

	def repl(self) -> None:
		history = data_dir() / "history"
		try:
			readline.read_history_file(history)
		except OSError:
			pass
		self.banner()
		print("Type /help for commands.")
		try:
			while True:
				try:
					line = read_input()
				except KeyboardInterrupt:
					print()
					continue
				if line is None:
					break
				if not line.strip():
					continue
				if is_command(line):
					try:
						if not self.command(line.strip()):
							break
					except (SandboxError, LLMError) as e:
						self.ui.notice(str(e))
					except KeyboardInterrupt:
						self.ui.notice("Interrupted.")
					continue
				self.turn(line)
		finally:
			readline.write_history_file(history)
			os.chmod(history, 0o600)

	def turn(self, text: str) -> None:
		try:
			self.agent.send(text)
		except LLMError as e:
			self.ui.notice(str(e))
			return
		except KeyboardInterrupt:
			return
		used = self.agent.estimate_tokens()
		self.ui.notice(f"{used:,} / {self.agent.context_length:,} tokens")


def main() -> None:
	for stream in (sys.stdout, sys.stderr):
		stream.reconfigure(errors="backslashreplace")
	exit_on_sigterm()
	try:
		_main()
	except KeyboardInterrupt:
		sys.exit(130)
	except BrokenPipeError:
		os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # so the exit flush does not fail again
		sys.exit(141)


def _main() -> None:
	args = parse_args()
	if not deps.ensure(data_dir() / "state.json", recheck=args.check_deps):
		sys.exit("Failed to start: bwrap is required to sandbox the model's tools.")
	clean_scratch(scratch_root())
	app = App(args)
	try:
		if args.resume or args.cont:
			chats = list_chats(1)
			if args.cont and not chats:
				sys.exit("Failed to continue: no saved chats.")
			chat = Chat.load(args.resume) if args.resume else chats[0]
		else:
			chat = app.new_chat()
		app.open_chat(chat)
	except (LLMError, SandboxError, FileNotFoundError, ValueError) as e:
		sys.exit(str(e))

	if args.prompt is not None:
		try:
			app.agent.send(args.prompt)
		except LLMError as e:
			sys.exit(str(e))
		return
	app.repl()


if __name__ == "__main__":
	main()
