"""Tool schemas for the model and their implementations, all run through the sandbox."""
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .helpers import expand_braces, strip_dot_slash
from .sandbox import Result, Sandbox, SandboxError

HELPERS = str(Path(__file__).with_name("helpers.py"))
OUTPUT_MAX_CHARS = 30000
OUTPUT_TAIL_CHARS = 8000
SCRATCH_WARN_BYTES = 2 << 30
# Runs rg inside a directory given as $1, since rg matches globs relative to its working directory.
# Paths in its output and errors get $2 as a prefix, so they read as if rg had searched $1. A file in $1 is
# searched directly.
RG_IN_DIR = (
	'dir=$1 prefix=$2 filter=$3; shift 3; '
	# rg searches a file named on its command line even when the glob filter excludes it.
	'if [ ! -d "$dir" ]; then '
	'if [ -n "$filter" ] && [ -f "$dir" ]; then '
	'"$1" --files --max-depth 0 -g "$filter" -- . >/dev/null; [ $? -eq 2 ] && exit 2; '  # reports a bad glob
	'(cd -- "$(dirname -- "$dir")" && "$1" --files -L --hidden --no-ignore --max-depth 1 -g "$filter" -- . 2>/dev/null) '
	'| grep -qxF -- "./$(basename -- "$dir")" || exit 1; '
	'fi; exec "$@" -- "$dir"; fi; '
	'cd -- "$dir" || exit 2; set -o pipefail; '
	# Flushing each line keeps the results found so far when a timeout kills the search.
	'"$@" -- . 2>&1 | prefix="$prefix" awk \''
	'sub(/^\\.\\//, "") { print ENVIRON["prefix"] $0; fflush(); next } '
	'sub(/^rg: \\.\\//, "") { print "rg: " ENVIRON["prefix"] $0; fflush(); next } '
	'{ print; fflush() }\''
)
SCRATCH_FULL = ("/tmp is past its size or file count limit, or holds a directory you made unreadable "
	"or nested too deeply to measure. Delete files you no longer need, or chmod u+rwx the directory.")


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
	return {"type": "function", "function": {
		"name": name,
		"description": description,
		"parameters": {"type": "object", "properties": properties, "required": required},
	}}


def schemas(rg: bool) -> list[dict]:
	"""Tool schemas. Without rg, the fallbacks do not read .gitignore and grep uses grep -E syntax."""
	# rg's -g globs override .gitignore, so only ignored directories stay skipped for files the glob matches.
	ignored = "unless include_ignored is true"

	def include_ignored(verb: str) -> dict:
		return {"include_ignored": {"type": "boolean", "description": f"Also {verb} files ignored by .gitignore"}} if rg else {}

	return [
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
			+ (f"Skips .git and directories ignored by .gitignore {ignored}." if rg else "Skips .git."),
			{
				"pattern": {"type": "string", "description": "Glob pattern, relative to path"},
				"path": {"type": "string", "description": "Directory to search; defaults to the working directory"},
				**include_ignored("list"),
			},
			["pattern"]),
		_schema("grep_search",
			f"Search file contents with a regular expression ({'ripgrep' if rg else 'grep -E'} syntax). "
			"Returns path:line:text matches. "
			+ (f"Skips .git, binary files, and files ignored by .gitignore {ignored}; "
				"with include, ignored files it matches are searched unless in an ignored directory." if rg
				else "Skips .git and binary files."),
			{
				"pattern": {"type": "string", "description": "Regular expression to search for"},
				"path": {"type": "string", "description": "File or directory to search; defaults to the working directory"},
				"include": {"type": "string", "description": "Only search files matching this glob, such as '*.py'"
					if rg else "Only search files whose names match this glob, such as '*.py'"},
				"ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
				"context": {"type": "integer", "description": "Lines of context to show around each match"},
				**include_ignored("search"),
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
		self.schemas = schemas(self.rg is not None)

	def call(self, name: str, arguments: str) -> ToolResult:
		try:
			args = json.loads(arguments or "{}")
			if not isinstance(args, dict):
				raise ValueError("arguments must be a JSON object")
		except (ValueError, RecursionError) as e:
			return ToolResult(f"Error: arguments for {name} are not a valid JSON object ({e}): {arguments}", True)
		method = getattr(self, f"_{name}", None)
		schema = next((s["function"] for s in self.schemas if s["function"]["name"] == name), None)
		if schema is None or method is None:
			return ToolResult(f"Error: unknown tool {name}", True)
		properties, required = schema["parameters"]["properties"], schema["parameters"]["required"]
		for key, value in args.items():
			if value is None and key not in required:
				continue  # optional arguments sent as null take their defaults
			kind = properties.get(key, {}).get("type")
			if kind == "string" and not isinstance(value, str):
				return ToolResult(f"Error: bad arguments for {name}: {key} must be a string, not {value!r}", True)
			if isinstance(value, str) and "\0" in value:
				return ToolResult(f"Error: bad arguments for {name}: {key} contains a NUL character: {value!r}", True)
			if kind == "boolean" and not isinstance(value, bool):
				return ToolResult(f"Error: bad arguments for {name}: {key} must be true or false, not {value!r}", True)
		try:
			result = method(**args)
		except (TypeError, ValueError, OverflowError) as e:
			return ToolResult(f"Error: bad arguments for {name}: {e}", True)
		except SandboxError as e:
			return ToolResult(f"Error: {e}", True)
		result.text = _truncate(result.text)
		return result

	def _helper(self, tool: str, args: dict) -> ToolResult:
		r = self.sandbox.run([sys.executable, "-I", "-S", HELPERS, tool, json.dumps(args)], timeout=60)
		return ToolResult(_format(r, show_exit=False), r.exit_code != 0)

	def _read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
		return self._helper("read_file", {"path": path, "offset": int(offset or 1), "limit": int(limit or 2000)})

	def _list_directory(self, path: str = ".") -> ToolResult:
		return self._helper("list_directory", {"path": path or "."})

	def _glob(self, pattern: str, path: str = ".", include_ignored: bool = False) -> ToolResult:
		path = path or "."
		if self.rg:
			# Later globs win in rg, so the .git exclusion goes last.
			argv = [self.rg, "--files", "--line-buffered", "--hidden", "--sort", "path", "-g", strip_dot_slash(pattern), "-g", "!.git"]
			if include_ignored:
				argv.append("--no-ignore")
		else:
			return self._helper("glob_files", {"pattern": pattern, "path": path})
		r = self.sandbox.run(_rg_in(path, argv, strip_dot_slash(pattern)), timeout=60)
		if r.exit_code == 1 and not r.output.strip():
			return ToolResult("No files matched.")
		return ToolResult(_format(r, show_exit=r.exit_code not in (0, 1)), r.exit_code not in (0, 1))

	def _grep_search(self, pattern: str, path: str = ".", include: str = "", ignore_case: bool = False,
			context: int = 0, include_ignored: bool = False) -> ToolResult:
		path = path or "."
		include = strip_dot_slash(include or "")
		if self.rg:
			argv = [self.rg, "--line-buffered", "--line-number", "--with-filename", "--no-heading", "--color=never", "--hidden",
				"--sort", "path", "--max-columns", "400", "--max-columns-preview"]
			if include:
				argv += ["-g", include]
			argv += ["-g", "!.git"]
			if include_ignored:
				argv.append("--no-ignore")
		else:
			argv = ["grep", "-rnHE", "--line-buffered", "--binary-files=without-match", "--exclude-dir=.git"]
			if include:
				# grep's --include matches file names and has no braces.
				names = [re.sub(r"^(\*\*/)+", "", p) for p in expand_braces(include)]
				if any("/" in name for name in names):
					return ToolResult(f"Error: include matches file names only without ripgrep: {include}", True)
				argv += [f"--include={name}" for name in names]
		if ignore_case:
			argv.append("-i")
		if context:
			argv += ["-C", str(int(context))]
		argv += ["-e", pattern]
		r = self.sandbox.run(_rg_in(path, argv, include) if self.rg else argv + ["--", path], timeout=120)
		if r.exit_code == 1 and not r.output.strip():
			return ToolResult("No matches.")
		return ToolResult(_format(r, show_exit=r.exit_code not in (0, 1)), r.exit_code not in (0, 1))

	def _run_shell_command(self, command: str, timeout: int | None = None) -> ToolResult:
		r = self.sandbox.run(["bash", "-c", command], timeout=int(timeout) if timeout else None)
		text = _format(r, show_exit=True)
		if not r.scratch_full and self.sandbox.scratch_usage()[0] > SCRATCH_WARN_BYTES:
			text += f"\n[Warning: /tmp holds over {SCRATCH_WARN_BYTES >> 30} GB. Delete files you no longer need.]"
		return ToolResult(text, r.exit_code != 0)


def _rg_in(path: str, argv: list[str], name_glob: str) -> list[str]:
	"""Runs rg in path and prefixes results with it; a file path is searched only if its name matches name_glob."""
	prefix = path.rstrip("/") + "/" if path.rstrip("/") else "/"
	return ["bash", "-c", RG_IN_DIR, "bash", path, prefix, name_glob, *argv]


def _format(r: Result, show_exit: bool) -> str:
	text = r.output.rstrip("\n")
	if r.truncated_bytes:
		text += f"\n[{r.truncated_bytes} more bytes of output were discarded.]"
	if r.timed_out:
		text += "\n[Killed: timed out.]"
	elif r.scratch_full and r.exit_code is None:
		text += f"\n[Killed: {SCRATCH_FULL}]"
	elif show_exit and r.exit_code != 0:
		text += f"\n[Exit code {r.exit_code}]"
	if r.scratch_full and r.exit_code is not None:
		text += f"\n[{SCRATCH_FULL} Commands are killed after a second until you fix it.]"
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
	except (ValueError, RecursionError):
		return f"{name}({arguments})"
	if not isinstance(args, dict):
		return f"{name}({arguments})"
	if name == "run_shell_command" and "command" in args:
		return f"$ {args['command']}"
	try:
		parts = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
	except RecursionError:
		return f"{name}({arguments})"
	return f"{name}({parts})"
