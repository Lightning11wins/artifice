"""Tool schemas for the model and their implementations, all run through the sandbox."""
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Result, Sandbox

HELPERS = str(Path(__file__).with_name("helpers.py"))
OUTPUT_MAX_CHARS = 30000
OUTPUT_TAIL_CHARS = 8000
SCRATCH_WARN_BYTES = 2 << 30


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
	return {"type": "function", "function": {
		"name": name,
		"description": description,
		"parameters": {"type": "object", "properties": properties, "required": required},
	}}


SCHEMAS = [
	_schema("read_file",
		"Read a text file. Returns numbered lines. Reads up to 2000 lines from offset; "
		"use offset and limit to page through large files.",
		{
			"path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
			"offset": {"type": "integer", "description": "First line to read, starting at 1"},
			"limit": {"type": "integer", "description": "Number of lines to read"},
		},
		["path"]),
	_schema("list_directory",
		"List a directory's entries with sizes. Directories end with /.",
		{"path": {"type": "string", "description": "Directory path; defaults to the working directory"}},
		[]),
	_schema("glob",
		"Find files by glob pattern, such as '**/*.py' or 'src/**/test_*.js'. "
		"Skips .git and files ignored by .gitignore unless include_ignored is true.",
		{
			"pattern": {"type": "string", "description": "Glob pattern, relative to path"},
			"path": {"type": "string", "description": "Directory to search; defaults to the working directory"},
			"include_ignored": {"type": "boolean", "description": "Also list files ignored by .gitignore"},
		},
		["pattern"]),
	_schema("grep_search",
		"Search file contents with a regular expression (ripgrep syntax). Returns path:line:text matches. "
		"Skips .git, binary files, and files ignored by .gitignore unless include_ignored is true.",
		{
			"pattern": {"type": "string", "description": "Regular expression to search for"},
			"path": {"type": "string", "description": "File or directory to search; defaults to the working directory"},
			"include": {"type": "string", "description": "Only search files matching this glob, such as '*.py'"},
			"ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
			"context": {"type": "integer", "description": "Lines of context to show around each match"},
			"include_ignored": {"type": "boolean", "description": "Also search files ignored by .gitignore"},
		},
		["pattern"]),
	_schema("run_shell_command",
		"Run a bash command and return its combined stdout and stderr. Each call starts a fresh shell "
		"in the working directory. The filesystem is read-only except /tmp, and there is no network.",
		{
			"command": {"type": "string", "description": "The bash command to run"},
			"timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120, max 600)"},
		},
		["command"]),
]


@dataclass
class ToolResult:
	text: str
	error: bool = False


class Tools:
	def __init__(self, sandbox: Sandbox):
		self.sandbox = sandbox
		self.rg = shutil.which("rg")
		self.schemas = SCHEMAS

	def call(self, name: str, arguments: str) -> ToolResult:
		try:
			args = json.loads(arguments or "{}")
			if not isinstance(args, dict):
				raise ValueError("arguments must be a JSON object")
		except ValueError as e:
			return ToolResult(f"Error: arguments for {name} are not a valid JSON object ({e}): {arguments}", True)
		method = getattr(self, f"_{name}", None)
		if name not in {s["function"]["name"] for s in self.schemas} or method is None:
			return ToolResult(f"Error: unknown tool {name}", True)
		try:
			result = method(**args)
		except (TypeError, ValueError) as e:
			return ToolResult(f"Error: bad arguments for {name}: {e}", True)
		result.text = _truncate(result.text)
		return result

	def _helper(self, tool: str, args: dict) -> ToolResult:
		r = self.sandbox.run([sys.executable, "-I", "-S", HELPERS, tool, json.dumps(args)], timeout=60)
		return ToolResult(_format(r, show_exit=False), r.exit_code != 0)

	def _read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
		return self._helper("read_file", {"path": path, "offset": int(offset), "limit": int(limit)})

	def _list_directory(self, path: str = ".") -> ToolResult:
		return self._helper("list_directory", {"path": path or "."})

	def _glob(self, pattern: str, path: str = ".", include_ignored: bool = False) -> ToolResult:
		path = path or "."
		if self.rg:
			# Later globs win in rg, so the .git exclusion goes last.
			argv = ["rg", "--files", "--hidden", "--sort", "path", "-g", pattern, "-g", "!.git"]
			if include_ignored:
				argv.append("--no-ignore")
			argv += ["--", path]
		else:
			name = pattern.rsplit("/", 1)[-1]
			argv = ["find", path, "-name", ".git", "-prune", "-o", "-type", "f", "-name", name, "-print"]
		r = self.sandbox.run(argv, timeout=60)
		if r.exit_code == 1 and not r.output.strip():
			return ToolResult("No files matched.")
		return ToolResult(_format(r, show_exit=r.exit_code not in (0, 1)), r.exit_code not in (0, 1))

	def _grep_search(self, pattern: str, path: str = ".", include: str = "", ignore_case: bool = False,
			context: int = 0, include_ignored: bool = False) -> ToolResult:
		path = path or "."
		if self.rg:
			argv = ["rg", "--line-number", "--with-filename", "--no-heading", "--color=never", "--hidden",
				"--sort", "path", "--max-columns", "400", "--max-columns-preview"]
			if include:
				argv += ["-g", include]
			argv += ["-g", "!.git"]
			if include_ignored:
				argv.append("--no-ignore")
		else:
			argv = ["grep", "-rnE", "--binary-files=without-match", "--exclude-dir=.git"]
			if include:
				argv.append(f"--include={include}")
		if ignore_case:
			argv.append("-i")
		if context:
			argv += ["-C", str(int(context))]
		argv += ["-e", pattern, "--", path]
		r = self.sandbox.run(argv, timeout=120)
		if r.exit_code == 1 and not r.output.strip():
			return ToolResult("No matches.")
		return ToolResult(_format(r, show_exit=r.exit_code not in (0, 1)), r.exit_code not in (0, 1))

	def _run_shell_command(self, command: str, timeout: int | None = None) -> ToolResult:
		r = self.sandbox.run(["bash", "-c", command], timeout=int(timeout) if timeout else None)
		text = _format(r, show_exit=True)
		if self.sandbox.scratch_bytes() > SCRATCH_WARN_BYTES:
			text += f"\n[Warning: /tmp holds over {SCRATCH_WARN_BYTES >> 30} GB. Delete files you no longer need.]"
		return ToolResult(text, r.exit_code != 0)


def _format(r: Result, show_exit: bool) -> str:
	text = r.output.rstrip("\n")
	if r.truncated_bytes:
		text += f"\n[{r.truncated_bytes} more bytes of output were discarded.]"
	if r.timed_out:
		text += "\n[Killed: timed out.]"
	elif show_exit and r.exit_code != 0:
		text += f"\n[Exit code {r.exit_code}]"
	return text or "(no output)"


def _truncate(text: str) -> str:
	if len(text) <= OUTPUT_MAX_CHARS:
		return text
	head = text[:OUTPUT_MAX_CHARS - OUTPUT_TAIL_CHARS]
	tail = text[-OUTPUT_TAIL_CHARS:]
	omitted = len(text) - len(head) - len(tail)
	return (f"{head}\n\n[... {omitted} characters omitted. Narrow the command or search, "
		f"or page with offset and limit ...]\n\n{tail}")


def describe_call(name: str, arguments: str) -> str:
	"""One-line summary of a tool call for display."""
	try:
		args = json.loads(arguments or "{}")
	except ValueError:
		return f"{name}({arguments})"
	if not isinstance(args, dict):
		return f"{name}({arguments})"
	if name == "run_shell_command" and "command" in args:
		return f"$ {args['command']}"
	parts = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
	return f"{name}({parts})"
