"""Client for the llm-server HTTP API."""
import http.client
import json
from collections.abc import Iterator


class LLMError(Exception):
	pass


class LLMClient:
	def __init__(self, host: str, port: int):
		self.host = host
		self.port = port

	def _connect(self, timeout: float | None) -> http.client.HTTPConnection:
		try:
			return http.client.HTTPConnection(self.host, self.port, timeout=timeout)
		except (http.client.HTTPException, ValueError) as e:  # such as a host with a space in it
			raise LLMError(f"Failed to reach llm-server at {self.host}:{self.port}: {e!r}") from e

	def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = 30) -> object:
		conn = self._connect(timeout)
		try:
			conn.request(method, path, body=json.dumps(body) if body is not None else None,
				headers={"Content-Type": "application/json"})
			resp = conn.getresponse()
			data = resp.read()
		except (OSError, http.client.HTTPException, UnicodeError) as e:
			raise LLMError(f"Failed to reach llm-server at {self.host}:{self.port}: {e!r}") from e
		finally:
			conn.close()
		if resp.status != 200:
			raise LLMError(f"Failed {method} {path}: HTTP {resp.status}: {_error_text(data)}")
		try:
			return json.loads(data)
		except ValueError as e:
			raise LLMError(f"Failed to parse the reply to {method} {path}: {data[:200]!r}") from e

	def list(self) -> list[dict]:
		return self._request("GET", "/list?loaded=any")

	def load(self, model: str, ttl: int) -> dict:
		return self._request("POST", "/load", {"model": model, "ttl": ttl}, timeout=None)

	def stream(self, body: dict) -> Iterator[dict]:
		"""POSTs /run and yields each event dict. Raises LLMError on an error event."""
		conn = self._connect(timeout=900)
		try:
			try:
				conn.request("POST", "/run", body=json.dumps(body), headers={"Content-Type": "application/json"})
				resp = conn.getresponse()
			except (OSError, http.client.HTTPException, UnicodeError) as e:
				raise LLMError(f"Failed to reach llm-server at {self.host}:{self.port}: {e!r}") from e
			if resp.status != 200:
				try:
					data = resp.read()
				except (OSError, http.client.HTTPException) as e:
					raise LLMError(f"Failed to generate: HTTP {resp.status}, then {e!r}") from e
				raise LLMError(f"Failed to generate: HTTP {resp.status}: {_error_text(data)}")
			saw_done = False
			for payload in _events(resp, self.host, self.port):
				if payload == "[DONE]":
					saw_done = True
					break
				try:
					event = json.loads(payload)
				except ValueError as e:
					raise LLMError(f"Failed to parse llm-server event {payload[:200]!r}: {e}") from e
				if "error" in event:
					raise LLMError(f"Failed to generate: {event['error']}")
				yield event
			if not saw_done:
				raise LLMError("Failed to generate: stream ended without [DONE]")
		finally:
			conn.close()


def _events(resp: http.client.HTTPResponse, host: str, port: int) -> Iterator[str]:
	"""Yields the payload of each SSE data line."""
	try:
		for raw in resp:
			line = raw.decode("utf-8", errors="replace").strip()
			if line.startswith("data:"):
				yield line[5:].strip()
	except (OSError, http.client.HTTPException) as e:
		raise LLMError(f"Failed to read from llm-server at {host}:{port}: {e}") from e


def _error_text(data: bytes) -> str:
	try:
		return json.loads(data).get("error", data.decode(errors="replace"))
	except (ValueError, AttributeError):
		return data.decode(errors="replace")
