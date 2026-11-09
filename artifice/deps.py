"""Finds missing system tools and offers to install them with the system package manager."""
import shutil
import subprocess
import sys

# Command -> what it is needed for, and its package name per package manager.
REQUIRED = {
	"bwrap": ("the sandbox", {"apt": "bubblewrap", "dnf": "bubblewrap"}),
	"rg": ("glob and grep_search", {"apt": "ripgrep", "dnf": "ripgrep"}),
}


def _manager() -> str | None:
	for name, cmd in (("apt", "apt-get"), ("dnf", "dnf")):
		if shutil.which(cmd):
			return name
	return None


def _install_command(manager: str, packages: list[str]) -> list[str]:
	if manager == "apt":
		return ["sudo", "apt-get", "install", "-y", *packages]
	return ["sudo", "dnf", "install", "-y", *packages]


def _ask(question: str) -> bool | None:
	"""Asks a yes or no question on stderr. Returns None when there is no terminal to ask on."""
	if not (sys.stdin.isatty() and sys.stderr.isatty()):
		return None
	sys.stderr.write(f"{question} [y/N] ")
	sys.stderr.flush()
	return sys.stdin.readline().strip().lower() in ("y", "yes")


def ensure() -> bool:
	"""Offers to install missing tools. Returns False when one is still missing."""
	manager = _manager()
	ok = True
	for command, (purpose, packages) in REQUIRED.items():
		if shutil.which(command):
			continue
		if manager is None:
			print(f"Missing {command} (needed for {purpose}). Install it with your package manager.", file=sys.stderr)
			ok = False
			continue
		argv = _install_command(manager, [packages[manager]])
		answer = _ask(f"Missing {command} (needed for {purpose}). Run `{' '.join(argv)}`?")
		if answer:
			subprocess.run(argv, stdout=sys.stderr)
		if not shutil.which(command):
			ok = False
			if answer is None:
				print(f"Missing {command} (needed for {purpose}). Run `{' '.join(argv)}`.", file=sys.stderr)
	return ok
