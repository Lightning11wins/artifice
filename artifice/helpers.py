"""File tools that run inside the sandbox, so they see the same filesystem as shell commands.

Usage: helpers.py TOOL JSON_ARGS
"""
import json
import os
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


def strip_dot_slash(pattern: str) -> str:
	"""A glob without leading ./, which rg would match literally."""
	while pattern.startswith("./"):
		pattern = pattern[2:]
	return pattern


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
