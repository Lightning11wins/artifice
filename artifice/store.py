"""Chat persistence. Transcripts can hold anything the model read, so files are private to the user."""
import datetime
import glob
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path


def data_dir() -> Path:
	base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
	path = base / "artifice"
	path.mkdir(mode=0o700, parents=True, exist_ok=True)
	os.chmod(path, 0o700)
	return path


def chats_dir() -> Path:
	path = data_dir() / "chats"
	path.mkdir(mode=0o700, exist_ok=True)
	return path


def scratch_root() -> Path:
	path = data_dir() / "scratch"
	path.mkdir(mode=0o700, exist_ok=True)
	return path


def _now() -> str:
	return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class Chat:
	id: str
	model: str
	profile: str
	cwd: str
	created: str = field(default_factory=_now)
	updated: str = field(default_factory=_now)
	messages: list[dict] = field(default_factory=list)
	compactions: int = 0
	# Token accounting, saved so a resumed chat estimates its size as the live session did.
	usage_tokens: int | None = None
	usage_n: int = 0
	bytes_per_token: float | None = None

	@classmethod
	def new(cls, model: str, profile: str, cwd: str) -> "Chat":
		stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		return cls(id=f"{stamp}-{secrets.token_hex(2)}", model=model, profile=profile, cwd=cwd)

	@property
	def path(self) -> Path:
		return chats_dir() / f"{self.id}.json"

	@property
	def title(self) -> str:
		for m in self.messages:
			if m.get("role") == "user" and not m.get("content", "").startswith("[Compacted context]"):
				return " ".join(m["content"].split())[:70]
		return "(empty)"

	def save(self) -> None:
		self.updated = _now()
		tmp = self.path.with_suffix(".tmp")
		fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
		with os.fdopen(fd, "w") as f:
			json.dump(asdict(self), f, indent=1)
		os.replace(tmp, self.path)

	@classmethod
	def load(cls, chat_id: str) -> "Chat":
		chat_id = Path(chat_id).name.removesuffix(".json")  # also accept a chat file's path
		matches = sorted(chats_dir().glob(f"{glob.escape(chat_id)}*.json"))
		if not matches:
			raise FileNotFoundError(f"Failed to find chat {chat_id!r} in {chats_dir()}")
		if len(matches) > 1:
			names = ", ".join(m.stem for m in matches[:5])
			raise FileNotFoundError(f"Failed to pick a chat: {chat_id!r} matches {names}")
		try:
			with open(matches[0]) as f:
				chat = cls(**json.load(f))
		except (ValueError, TypeError) as e:
			raise ValueError(f"Failed to load chat {matches[0]}: {e}") from e
		if not _named_for(chat, matches[0]):
			raise ValueError(f"Failed to load chat {matches[0]}: its id {chat.id!r} is not its file name")
		return chat


def _named_for(chat: Chat, path: Path) -> bool:
	"""Whether the chat's id is its file name, since the id names its save file and scratch directory."""
	return chat.id == path.stem and not chat.id.startswith(".")


def list_chats(limit: int = 20) -> list[Chat]:
	paths = sorted(chats_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
	chats = []
	for p in paths:
		if len(chats) == limit:
			break
		try:
			with open(p) as f:
				chat = Chat(**json.load(f))
		except (OSError, ValueError, TypeError):
			continue
		if _named_for(chat, p):
			chats.append(chat)
	return chats
