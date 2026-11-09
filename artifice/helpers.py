"""File tools that run inside the sandbox, so they see the same filesystem as shell commands.

Usage: helpers.py TOOL JSON_ARGS
"""
import itertools
import json
import math
import os
import re
import stat
import sys

MAX_LINE = 2000
MAX_CHARS = 29800  # under the tools' 30,000 character cap, with room for the footer
MAX_ENTRIES = 500
BINARY_PROBE = 8192


def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
	st = os.stat(path)
	if stat.S_ISDIR(st.st_mode):
		return f"Error: {path} is a directory. Use list_directory."
	if not stat.S_ISREG(st.st_mode):
		return f"Error: {path} is not a regular file ({_kind(st.st_mode)}). Use run_shell_command if you need it."
	with open(path, "rb") as f:
		probe = f.read(BINARY_PROBE)
	if b"\0" in probe:
		return (f"Error: {path} looks binary ({st.st_size} bytes). "
			"Use run_shell_command with file, strings, xxd or pdftotext.")
	offset = max(1, offset)
	limit = max(1, limit)
	out: list[str] = []
	chars = 0
	full = False
	last = offset - 1
	total = 0
	# Only \n ends a line, as in grep, rg and sed.
	with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
		for number, line in enumerate(f, 1):
			total = number
			if number < offset or number >= offset + limit or full:
				continue
			line = line.removesuffix("\n").removesuffix("\r")
			if len(line) > MAX_LINE:
				line = line[:MAX_LINE] + f" [... line truncated, {len(line)} chars]"
			numbered = f"{number:>6}\t{line}"
			if out and chars + len(numbered) + 1 > MAX_CHARS:
				full = True
				continue
			out.append(numbered)
			chars += len(numbered) + 1
			last = number
	if total == 0:
		return f"({path} is empty)"
	if offset > total:
		return f"Error: offset {offset} is past the end of {path} ({total} lines)."
	if last < total:
		out.append(f"[Showed lines {offset}-{last} of {total}. Use offset={last + 1} to read more.]")
	return "\n".join(out)


def list_directory(path: str) -> str:
	entries = []
	with os.scandir(path) as it:
		for entry in it:
			try:
				st = entry.stat(follow_symlinks=False)
			except OSError:
				entries.append((2, entry.name, f"? {entry.name}"))
				continue
			if stat.S_ISDIR(st.st_mode):
				entries.append((0, entry.name, f"{entry.name}/"))
			elif stat.S_ISLNK(st.st_mode):
				try:
					target = os.readlink(entry.path)
				except OSError:
					target = "?"
				entries.append((1, entry.name, f"{entry.name} -> {target}"))
			elif stat.S_ISREG(st.st_mode):
				entries.append((1, entry.name, f"{entry.name}  ({_size(st.st_size)})"))
			else:
				entries.append((1, entry.name, f"{entry.name}  ({_kind(st.st_mode)})"))
	if not entries:
		return f"({path} is empty)"
	entries.sort(key=lambda e: (e[0], e[1].lower()))
	lines = [e[2] for e in entries[:MAX_ENTRIES]]
	if len(entries) > MAX_ENTRIES:
		lines.append(f"[{len(entries) - MAX_ENTRIES} more entries not shown. Use glob or run_shell_command to filter.]")
	return "\n".join(lines)


def glob_files(pattern: str, path: str) -> str:
	"""Files under path matching pattern, skipping .git. The fallback for glob when rg is missing."""
	# Like rg, a pattern with a slash is anchored at path, and one without matches at any depth.
	full = strip_dot_slash(pattern)
	if full.startswith("/"):
		full = full[1:]
	elif "/" not in full:
		full = f"**/{full}"
	try:
		regex = _glob_regex(full)
	except re.error as e:
		return f"Error: bad glob pattern {pattern}: {e}"
	os.listdir(path)  # reports a missing or unreadable path, which os.walk would skip
	matches = []
	for root, dirs, files in os.walk(path):  # does not follow symlinked directories
		dirs[:] = [d for d in dirs if d != ".git"]
		for name in files:
			full = os.path.join(root, name)
			if regex.match(os.path.relpath(full, path)):
				matches.append(full)
	return "\n".join(sorted(matches)) or "No files matched."


def strip_dot_slash(pattern: str) -> str:
	"""A glob without leading ./, which rg would match literally."""
	while pattern.startswith("./"):
		pattern = pattern[2:]
	return pattern


def _glob_regex(pattern: str) -> re.Pattern:
	"""Compiles a glob where ** spans directories, *, ? and [...] stay within one, and {a,b} picks one."""
	return re.compile(_glob_body(pattern) + r"\Z")


def _glob_body(pattern: str, whole: bool = True) -> str:
	"""whole is whether pattern is the whole glob, not a {a,b} option."""
	out = []
	i = 0
	while i < len(pattern):
		# ** spans directories only as a whole path component, as in rg. A bare ** option is not one.
		after_slash = i > 0 and pattern[i - 1] == "/"
		if (i == 0 or after_slash) and pattern.startswith("**/", i):
			out.append("(?:.*/)?")
			i += 3
		elif (after_slash or i == 0 and whole) and pattern[i:] == "**":
			out.append(".*")
			i += 2
		elif pattern[i] == "*":
			out.append("[^/]*")
			i += 1
		elif pattern[i] == "?":
			out.append("[^/]")
			i += 1
		elif pattern[i] == "[" and (end := pattern.find("]", i + 2)) != -1:
			body = pattern[i + 1:end]
			negate = body[0] in "!^"
			body = "".join(c if c == "-" else re.escape(c) for c in body[negate:])
			out.append("[" + ("^" if negate else "") + body + "]")
			i = end + 1
		elif pattern[i] == "{" and (alternatives := _brace_alternatives(pattern, i)):
			options, i = alternatives
			out.append("(?:" + "|".join(_glob_body(option, False) for option in options) + ")")
		else:
			out.append(re.escape(pattern[i]))
			i += 1
	return "".join(out)


def expand_braces(pattern: str, limit: int = 256) -> list[str]:
	"""Expands each {a,b,...} in pattern into separate patterns, up to limit of them. Groups cannot nest, as in rg."""
	if len(pattern) > 1024:
		raise ValueError(f"Failed to expand braces: the pattern is {len(pattern)} characters, over 1024")
	pieces = re.split(r"\{([^{}]*)\}", pattern)
	literals, groups = pieces[0::2], [group.split(",") for group in pieces[1::2]]
	if any("{" in literal for literal in literals):
		raise ValueError(f"Failed to expand braces: nested or unclosed {{ in {pattern}")
	if math.prod(len(group) for group in groups) > limit:
		raise ValueError(f"Failed to expand braces: {pattern} has more than {limit} alternatives")
	return ["".join(p for pair in itertools.zip_longest(literals, options, fillvalue="") for p in pair)
		for options in itertools.product(*groups)]


def _brace_alternatives(pattern: str, start: int) -> tuple[list[str], int] | None:
	"""Splits {a,b,...} at start into its options and returns them with the index after }."""
	depth = 0
	options = []
	begin = start + 1
	for i in range(start, len(pattern)):
		if pattern[i] == "{":
			depth += 1
		elif pattern[i] == "}":
			depth -= 1
			if depth == 0:
				return options + [pattern[begin:i]], i + 1
		elif pattern[i] == "," and depth == 1:
			options.append(pattern[begin:i])
			begin = i + 1
	return None


def _kind(mode: int) -> str:
	for test, name in ((stat.S_ISFIFO, "FIFO"), (stat.S_ISSOCK, "socket"), (stat.S_ISCHR, "character device"),
			(stat.S_ISBLK, "block device")):
		if test(mode):
			return name
	return "special file"


def _size(n: int) -> str:
	for unit in ("B", "KB", "MB", "GB"):
		if n < 1024 or unit == "GB":
			return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
		n /= 1024
	return str(n)


def main() -> None:
	tool, args = sys.argv[1], json.loads(sys.argv[2])
	sys.stdout.reconfigure(errors="backslashreplace")  # file names need not be UTF-8
	try:
		print(globals()[tool](**args))
	except OSError as e:
		print(f"Error: {e.strerror}: {e.filename}")
		sys.exit(1)


if __name__ == "__main__":
	main()
