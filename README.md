# Artifice

A terminal chat harness for local models served by [llm-server](../llm-server). The model can read any file and run commands, but the operating system stops it from changing anything.

## Run

```bash
bin/artifice                          # new chat in the current directory, default model qwen3.6-35b-a3b
bin/artifice --model qwen3.5-2b       # any model from `llm list`
bin/artifice -c                       # continue the most recent chat
bin/artifice -r 20261107              # resume a chat by ID prefix
bin/artifice -p "question"            # one-shot: answer to stdout, trace to stderr
```

The server address comes from `--host`/`--port`, else `$LLM_HOST`/`$LLM_PORT`, else `localhost:8741`. Type `/help` in a chat for commands.

Needs Python 3.12+, `bwrap` (package `bubblewrap`) and, optionally, `rg` (package `ripgrep`). Artifice offers to install missing ones with `apt` or `dnf`. On systems where AppArmor restricts unprivileged user namespaces, it offers to install a profile that allows them for bwrap.

## Tools

| Tool | Runs |
|---|---|
| `read_file` | numbered lines with offset and limit; refuses binary and special files |
| `list_directory` | entries with sizes |
| `glob` | `rg --files -g` (falls back to `find`) |
| `grep_search` | `rg` (falls back to `grep -rE`) |
| `run_shell_command` | `bash -c` |

Each tool call runs in its own sandbox. Output over 30,000 characters is cut down to its head and tail.

## Sandbox

Every tool call runs as `systemd-run --user --scope` (memory and task limits, when available) → `bwrap` → `confine.py` → the command.

- **bwrap:** read-only bind of `/` at the same paths, a private `/dev` and `/proc`, and new user, PID, network, IPC, UTS and cgroup namespaces. So there is no network and other processes are invisible. `/tmp` is the chat's scratch directory, `~/.local/share/artifice/scratch/<chat>`, which is writable and persists for the chat. Files older than 48 hours are deleted at startup.
- **confine.py** closes what a read-only mount leaves open:
  - Landlock allows writes only under `/tmp` and the private `/dev`, which blocks writing into FIFOs on the host filesystem.
  - seccomp blocks `socket(AF_UNIX)`, which would otherwise reach dbus, ssh-agent and gpg-agent through socket files. It also blocks io_uring, `open_by_handle_at` and the keyring syscalls.
  - `no_new_privs` stops sudo and setuid binaries.
- Limits: 120 s default timeout (600 s max), 1 GiB per written file, 1 MiB of output kept per command.

The escape tests in `tests/test_sandbox.py` try each of these and must all fail to change anything:

```bash
python3 -m unittest discover -s tests -v
```

## Context

The window size comes from the server. Past 75% of it, old tool outputs are replaced with a note (the newest six are kept) until the history is back under 50%. If that is not enough, the model writes notes on everything before the current request, and those notes replace it, labeled as possibly mistaken. `/compact` does the same for the whole chat.

## Data

Chats are saved as JSON in `~/.local/share/artifice/chats/`, mode 0600 in a 0700 directory. They contain everything the model read, so treat them as sensitive.
