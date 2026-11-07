"""Finds missing system tools and offers to install them with the system package manager."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Command -> package name per package manager.
REQUIRED = {"bwrap": {"apt": "bubblewrap", "dnf": "bubblewrap"}}
OPTIONAL = {"rg": {"apt": "ripgrep", "dnf": "ripgrep"}}

APPARMOR_PROFILE_PATH = "/etc/apparmor.d/artifice-bwrap"
APPARMOR_PROFILE = """abi <abi/4.0>,
include <tunables/global>

# Lets bwrap create the user namespaces artifice sandboxes tools in.
profile artifice-bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/artifice-bwrap>
}
"""


def _manager() -> str | None:
	for name, cmd in (("apt", "apt-get"), ("dnf", "dnf")):
		if shutil.which(cmd):
			return name
	return None


def _install_command(manager: str, packages: list[str]) -> list[str]:
	if manager == "apt":
		return ["sudo", "apt-get", "install", "-y", *packages]
	return ["sudo", "dnf", "install", "-y", *packages]


def _ask(question: str) -> bool:
	if not sys.stdin.isatty():
		return False
	try:
		return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
	except EOFError:
		return False


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
			if _ask(f"Missing {command} ({purpose}). Run `{' '.join(argv)}`?"):
				subprocess.run(argv)
			if shutil.which(command):
				continue
			if required:
				ok = False
			else:
				declined.add(command)
				print(f"Using fallbacks without {command}. Run with --check-deps to be asked again.", file=sys.stderr)
	state["declined"] = sorted(declined)
	state_path.write_text(json.dumps(state))
	return ok


def offer_apparmor_fix() -> bool:
	"""Offers to install an AppArmor profile that lets bwrap create user namespaces."""
	print("AppArmor restricts unprivileged user namespaces, which bwrap needs. This profile allows them for bwrap:\n")
	print(APPARMOR_PROFILE)
	if not _ask(f"Install it to {APPARMOR_PROFILE_PATH} with sudo and load it?"):
		return False
	tee = subprocess.run(["sudo", "tee", APPARMOR_PROFILE_PATH], input=APPARMOR_PROFILE.encode(),
		stdout=subprocess.DEVNULL)
	if tee.returncode != 0:
		return False
	return subprocess.run(["sudo", "apparmor_parser", "-r", APPARMOR_PROFILE_PATH]).returncode == 0
