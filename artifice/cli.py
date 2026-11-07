"""Terminal chat front end."""
import argparse
import os
import readline
import sys
from pathlib import Path

from . import deps
from .agent import Agent, Settings
from .client import LLMClient, LLMError
from .prompt import system_prompt
from .sandbox import READ_ONLY, Sandbox, SandboxError, clean_scratch
from .store import Chat, data_dir, list_chats, scratch_root
from .tools import ToolResult, Tools, describe_call

DEFAULT_MODEL = "qwen3.6-35b-a3b"
HELP = """Commands:
  /new              start a new chat
  /chats            list recent chats
  /resume ID        switch to a saved chat (an ID prefix is enough)
  /compact          summarize the conversation so far into notes
  /think on|off|default   turn model thinking on or off
  /reasoning        show or hide the model's thinking
  /usage            show context usage
  /quit             exit (or Ctrl-D)
Start a line with \"\"\" to type several lines; end with \"\"\" on its own line."""


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
		self.trace.write(self._c("2", text))
		self.trace.flush()

	def token(self, text: str) -> None:
		if self.state == "reasoning":
			self.trace.write("\n\n")
			self.trace.flush()
		self.state = "answer"
		self.out.write(text)
		self.out.flush()

	def end_stream(self) -> None:
		if self.state == "answer":
			self.out.write("\n")
			self.out.flush()
		elif self.state == "reasoning":
			self.trace.write("\n")
		self.state = None

	def tool_call(self, name: str, arguments: str) -> None:
		line = describe_call(name, arguments)
		if len(line) > 300:
			line = line[:300] + "..."
		self.trace.write(self._c("36", f"● {line}") + "\n")
		self.trace.flush()

	def tool_result(self, name: str, result: ToolResult) -> None:
		lines = result.text.splitlines() or [""]
		shown = [line[:160] for line in lines[:4]]
		if len(lines) > 4:
			shown.append(f"... ({len(lines) - 4} more lines)")
		code = "31" if result.error else "2"
		self.trace.write("".join(self._c(code, f"  │ {line}") + "\n" for line in shown))
		self.trace.flush()

	def notice(self, text: str) -> None:
		self.trace.write(self._c("33", f"[{text}]") + "\n")
		self.trace.flush()


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
	p.add_argument("--temperature", type=float, default=0.6)
	p.add_argument("--top-p", type=float, default=0.95)
	p.add_argument("--check-deps", action="store_true", help="ask again about optional tools you declined")
	return p.parse_args()


def read_input() -> str | None:
	try:
		line = input("\n\001\033[1;32m\002>\001\033[0m\002 " if sys.stdout.isatty() else "\n> ")
	except EOFError:
		return None
	if line.strip() != '"""':
		return line
	lines = []
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
		self.agent: Agent | None = None

	def load_model(self) -> None:
		self.ui.notice(f"Loading {self.settings.model} from {self.args.host}:{self.args.port}...")
		info = self.client.load(self.settings.model, self.settings.ttl)
		self.context_length = info["context_length"]

	def open_chat(self, chat: Chat) -> None:
		if self.args.model is None:
			self.settings.model = chat.model
		chat.model = self.settings.model
		sandbox = Sandbox(READ_ONLY, scratch_root() / chat.id, chat.cwd, notify=self.ui.notice)
		sandbox.check()
		self.agent = Agent(self.client, chat, Tools(sandbox), self.settings, self.ui, self.context_length)

	def new_chat(self) -> Chat:
		cwd = os.getcwd()
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
			except FileNotFoundError as e:
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
				if line.startswith("/"):
					try:
						if not self.command(line.strip()):
							break
					except SandboxError as e:
						self.ui.notice(str(e))
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
		used = self.agent.estimate_tokens()
		self.ui.notice(f"{used:,} / {self.agent.context_length:,} tokens")


def main() -> None:
	args = parse_args()
	if not deps.ensure(data_dir() / "state.json", recheck=args.check_deps):
		sys.exit("Failed to start: bwrap is required to sandbox the model's tools.")
	clean_scratch(scratch_root())
	app = App(args)
	try:
		app.load_model()
		if args.resume or args.cont:
			chats = list_chats(1)
			if args.cont and not chats:
				sys.exit("Failed to continue: no saved chats.")
			chat = Chat.load(args.resume) if args.resume else chats[0]
		else:
			chat = app.new_chat()
		app.open_chat(chat)
	except (LLMError, SandboxError, FileNotFoundError) as e:
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
