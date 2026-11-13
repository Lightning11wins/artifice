# Artifice

A terminal chat harness for local models served by [llm-server](https://github.com/Lightning11wins/llm-server). The model can read any file and run commands, but the operating system stops it from writing anything outside its scratch directory.

## Run

```bash
bin/artifice                          # new chat in the current directory, default model qwen3.6-35b-a3b
bin/artifice --model qwen3.5-2b       # any model from `llm list`
bin/artifice -c                       # continue the most recent chat
bin/artifice -r 20261107              # resume a chat by ID prefix
bin/artifice -p "question"            # one-shot: answer to stdout, trace to stderr
```

The server address comes from `--host`/`--port`, else `$LLM_HOST`/`$LLM_PORT`, else `localhost:8741`. Type `/help` in a chat for commands.

Needs Python 3.12+ (and `/usr/bin/python3`), `sudo`, `timeout` (coreutils), `setpriv` and `unshare` (util-linux), `bwrap` 0.10+ or a distribution build with `--bind-fd` (package `bubblewrap`) and `rg` (package `ripgrep`, from EPEL on RHEL and Rocky). Artifice offers to install `bwrap` and `rg` with `apt` or `dnf` when they are missing.

The sandbox starts as root to grant the model read access to every file. Before the first run, and after each update, install its launcher:

```bash
bin/install-sandbox
```

It copies `artifice/launcher.py` and `artifice/confine.py` to `/usr/local/libexec/artifice/`, owned by root, and adds `/etc/sudoers.d/artifice`, which lets you run that launcher with sudo and no password. Artifice refuses to start while the installed copies differ from its own. See below for what this allows.

### In T3 Code and other ACP clients

`bin/artifice --acp` serves the [Agent Client Protocol](https://agentclientprotocol.com) on stdin and stdout. In T3 Code, add a Local ACP command with executable `/path/to/artifice/bin/artifice`, argument `--acp`, and `LLM_HOST` in Variables if the server is not local. Chats are saved as usual, so `-r` can resume them in the terminal. The client's model picker lists the server's models, and the chosen model loads at the first prompt. `/compact` works as in the terminal. Missing `bwrap` or `rg` fail the start, since there is no terminal to ask in.

## Tools

| Tool | Runs |
|---|---|
| `read_file` | numbered lines with offset and limit; refuses binary and special files |
| `list_directory` | entries with sizes |
| `glob` | `rg --files -g` |
| `grep_search` | `rg` |
| `run_shell_command` | `bash -c` |

Each tool call runs in its own sandbox. Output over 30,000 characters is cut down to its head and tail.

## Sandbox

Every tool call runs as `sudo`, then the launcher, which execs `timeout` (a backstop that kills the sandbox 10 s after the command's timeout) and `setpriv --pdeathsig`, which ties the rest to it, then `systemd-run --scope` (memory and task limits, when systemd runs), then `bwrap`, then `unshare`, then `setpriv`, then `confine.py`, then the command.

- **sudo bwrap** starts as root without a user namespace, so the capability it keeps is real on the host. It makes:
  - a read-only bind of `/` at the same paths;
  - a private `/dev`, with a writable `/dev/shm` for shared memory;
  - the host's `/proc`, read-only, so the model sees every process;
  - new PID, network, IPC, UTS and cgroup namespaces. So there is no network, and PIDs passed to `kill` or `ptrace` resolve in the sandbox, not on the host.

  `/tmp` is the chat's scratch directory, `~/.local/share/artifice/scratch/<chat>`, so the host's `/tmp` is hidden. A chat started under `/tmp` or `/dev` starts in your home directory instead. Scratch is writable and persists for the chat. At startup, Artifice deletes files that have not changed in 48 hours, and then empty directories, from chats that are not open.

  The command to run reaches the launcher on stdin, so sudo logs only the launcher, not what the model ran.
- **The launcher** takes a request from stdin and builds the rest of the chain from fixed programs. Anything running as you can run it without a password, so a request can only choose what you could choose anyway: the command runs as you, the scratch directory must be yours and is opened with your permissions, and the command counts against your own limits (`user-UID.slice`), which the per-command limits only tighten. So any program running as you can use it to read every file on the computer, as the model can. It can also write to a world-writable directory inside one you cannot enter, if it first moves that directory into the scratch, which needs write access to the directory holding it.
- **unshare** makes the mounts private, so a mount the host adds later, such as a USB drive, never appears in the sandbox without the read-only flag.
- **setpriv** drops to your uid, gid and groups and keeps only `CAP_DAC_READ_SEARCH`. That capability lets the model read every file, root-only ones included, and nothing else.
- **confine.py** closes what a read-only mount and that capability leave open:
  - Landlock allows writes only under `/tmp` and the private `/dev`, which blocks writing into FIFOs on the host filesystem. Its ptrace scoping also hides `/proc/PID/environ`, `fd` and `maps` of host processes.
  - seccomp allows only IPv4, IPv6 and netlink sockets, which the network namespace confines. That blocks unix sockets, which would otherwise reach dbus, ssh-agent and gpg-agent through socket files, and families like vsock and Bluetooth that ignore network namespaces. It blocks `open_by_handle_at`, which with the capability opens files by handle, past the sandbox's mounts. It also blocks io_uring, the keyring syscalls, inotify and fanotify, whose per-user limits host programs share, file leases and `fcntl` commands newer than Linux 6.12, which would make host programs wait in `open()`, and `fallocate`'s keep-size mode and the ioctls that use it, which skip the file size limit.
  - `no_new_privs` stops sudo and setuid binaries.
- Limits: 120 s default timeout (600 s max; if Artifice itself is killed, the sandbox ends 10 s after the timeout), 8 GiB of memory without swap and 512 tasks, 1 GiB per written file, 1 GiB in `/dev/shm`, 4 GiB and 100,000 entries in scratch, 1 MiB of output kept per command. Pipe buffers, epoll watches and queued signals still count against per-user limits that host programs share.
- Scratch limits are checked every second while a command runs and when it exits, and a directory Artifice cannot read counts as full. Disk use counts deleted files that a command keeps open, through the free space of scratch's filesystem, so other writes to that filesystem during a command count too. A command can overshoot the limits by what it writes in a second; while scratch is over, later commands are killed after a second until the model deletes files. A hard cap would need a disk quota or a dedicated filesystem.
- Reading still has effects that a read-only sandbox cannot stop. Reading a FIFO takes data meant for its reader, reading `/sys/kernel/tracing/trace_pipe` drains the kernel trace buffer, and a lock taken on a read-only file, such as dpkg's or rpm's lock, makes host programs wait for it until the command ends or times out. A filesystem the host unmounts during a command stays mounted in the sandbox until the command ends, so its device stays busy. Reading an automount point mounts it on the host, though that first read fails in the sandbox, whose mounts are private. A command that crashes reaches the host's crash reporter, if one is set up.

The escape tests in `tests/test_sandbox.py` try each of these and must all fail to change anything. `tests/test_app.py` covers the agent loop and CLI with a fake server. The escape tests run the repo's launcher, not the installed one, so they need sudo without a password prompt: run `sudo -v` first, and run them outside any other sandbox:

```bash
python3 -m unittest discover -s tests -v
```

## Context

The window size comes from the server. Past 75% of it, old tool outputs are replaced with a note (the newest six are kept) until the history is back under 50%. If that is not enough, the model writes notes on everything before the current request, and those notes replace it, labeled as possibly mistaken. After that, reasoning from the current request's earlier steps is dropped, and the newest output is replaced only when no reply would fit beside it. If the server still rejects the request, the longest earlier message is replaced with a note until it fits. A request still too long is refused and replaced with a note, so the earlier conversation's notes are replaced only behind a short request. `/compact` does the same for the whole chat. A request made only of short messages and outputs can still be over the limit; start a new chat with `/new`.

## Data

Chats are saved as JSON in `~/.local/share/artifice/chats/`, mode 0600 in a 0700 directory. They contain everything the model read, so treat them as sensitive. A chat can be open in one Artifice session at a time.

## License

GPLv3. See [LICENSE](LICENSE).
