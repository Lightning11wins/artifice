"""Finds missing system tools and offers to install them with the system package manager."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Command -> package name per package manager.
REQUIRED = {"bwrap": {"apt": "bubblewrap", "dnf": "bubblewrap"}}
OPTIONAL = {"rg": {"apt": "ripgrep", "dnf": "ripgrep"}}


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


def _load_state(path: Path) -> dict:
	try:
		return json.loads(path.read_text())
	except (OSError, ValueError):
		return {}


def ensure(state_path: Path, recheck: bool = False) -> bool:
	"""Offers to install missing tools. Returns False when a required one is still missing."""
	state = _load_state(state_path)
	declined = set() if recheck else set(state.get("declined", []))
	manager = _manager()
	ok = True
	for table, required in ((REQUIRED, True), (OPTIONAL, False)):
		for command, packages in table.items():
			if shutil.which(command) or (not required and command in declined):
				continue
			purpose = "required for the sandbox" if required else "faster, gitignore-aware search"
			if manager is None:
				print(f"Missing {command} ({purpose}). Install it with your package manager.", file=sys.stderr)
				ok = ok and not required
				continue
			argv = _install_command(manager, [packages[manager]])
			answer = _ask(f"Missing {command} ({purpose}). Run `{' '.join(argv)}`?")
			if answer:
				subprocess.run(argv, stdout=sys.stderr)
			if shutil.which(command):
				continue
			if required:
				ok = False
				if answer is None:
					print(f"Missing {command} ({purpose}). Run `{' '.join(argv)}`.", file=sys.stderr)
			elif answer is not None:
				declined.add(command)
				print(f"Using fallbacks without {command}. Run with --check-deps to be asked again.", file=sys.stderr)
	state["declined"] = sorted(declined)
	state_path.write_text(json.dumps(state))
	return ok

