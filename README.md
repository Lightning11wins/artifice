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

Needs Python 3.12+, `sudo`, `setpriv` (util-linux), `bwrap` (package `bubblewrap`) and, optionally, `rg` (package `ripgrep`). Artifice offers to install missing ones with `apt` or `dnf`.

Artifice asks for your sudo password once per session (when sudo's ticket expires, it asks again), because the sandbox starts as root to grant the model read access to every file. See below.

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

Every tool call runs as `sudo`, then `systemd-run --scope` (memory and task limits, when systemd runs), then `bwrap`, then `setpriv`, then `confine.py`, then the command.

- **sudo bwrap** starts as root without a user namespace, so the capability it keeps is real on the host. It makes:
  - a read-only bind of `/` at the same paths;
  - a private `/dev`;
  - the host's `/proc`, read-only, so the model sees every process;
  - new PID, network, IPC, UTS and cgroup namespaces. So there is no network, and PIDs passed to `kill` or `ptrace` resolve in the sandbox, not on the host.

  `/tmp` is the chat's scratch directory, `~/.local/share/artifice/scratch/<chat>`, so the host's `/tmp` is hidden. Scratch is writable and persists for the chat. At startup, Artifice deletes files that have not changed in 48 hours from chats that are not open.

  The command to run reaches the sandbox on stdin, so sudo logs only the fixed sandbox command line, not what the model ran.
- **setpriv** drops to your uid, gid and groups and keeps only `CAP_DAC_READ_SEARCH`. That capability lets the model read every file, root-only ones included, and nothing else.
- **confine.py** closes what a read-only mount and that capability leave open:
  - Landlock allows writes only under `/tmp` and the private `/dev`, which blocks writing into FIFOs on the host filesystem. Its ptrace scoping also hides `/proc/PID/environ`, `fd` and `maps` of host processes.
  - seccomp allows only IPv4, IPv6 and netlink sockets, which the network namespace confines. That blocks unix sockets, which would otherwise reach dbus, ssh-agent and gpg-agent through socket files, and families like vsock and Bluetooth that ignore network namespaces. It blocks `open_by_handle_at`, which with the capability opens files by handle, past the sandbox's mounts. It also blocks io_uring and the keyring syscalls.
  - `no_new_privs` stops sudo and setuid binaries.
- Limits: 120 s default timeout (600 s max), 8 GiB of memory without swap and 512 tasks, 1 GiB per written file, 1 MiB of output kept per command.

The escape tests in `tests/test_sandbox.py` try each of these and must all fail to change anything. They need sudo without a password prompt, so run `sudo -v` first, and run them outside any other sandbox:

```bash
python3 -m unittest discover -s tests -v
```

## Context

The window size comes from the server. Past 75% of it, old tool outputs are replaced with a note (the newest six are kept) until the history is back under 50%. If that is not enough, the model writes notes on everything before the current request, and those notes replace it, labeled as possibly mistaken. `/compact` does the same for the whole chat.

## Data

Chats are saved as JSON in `~/.local/share/artifice/chats/`, mode 0600 in a 0700 directory. They contain everything the model read, so treat them as sensitive.

## License

GPLv3. See [LICENSE](LICENSE).
