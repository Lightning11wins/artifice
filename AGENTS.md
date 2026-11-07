# Agent notes

- Read `README.md` first; its Sandbox section explains what each layer blocks.
- `bwrap` cannot run inside another sandbox that disables user namespaces, so run `python3 -m unittest discover -s tests -v` and end-to-end checks unsandboxed.
- Any change to `sandbox.py` or `confine.py` needs a test in `tests/test_sandbox.py` that fails without it.
- Test harness changes against `qwen3.5-2b` (fast) before `qwen3.6-35b-a3b`. The server address comes from `LLM_HOST`.
- Indent with tabs. Write docs, comments, and prompts without em dashes.
