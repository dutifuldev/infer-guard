# infer-guard

infer-guard is a command-line guard for local LLM inference processes.
It starts tools such as vLLM, llama.cpp, SGLang, TensorRT-LLM, and TGI with
memory checks, process-group cleanup, and earlyoom-aware launch protection so a
bad model load or benchmark run is less likely to take down an interactive
machine.

## Install

infer-guard is not published yet. Install it from a local checkout:

```bash
cargo install --path .
```

Run a quick machine check:

```bash
infer-guard doctor
```

For high-risk inference profiles, infer-guard requires `earlyoom` by default.
Use `--allow-no-earlyoom` only for tests or controlled machines where you have a
separate safety net.

## Run A Command

Run a server through the guard:

```bash
infer-guard run --profile vllm --min-mem 24G --min-swap 4G -- vllm serve ...
```

`infer-guard run` launches the child in its own process group, watches available
memory and swap, and terminates the group when the configured floor is crossed.
It passes through the child exit code on normal exits and returns `137` when it
kills a workload for memory pressure.

Supported profile names are `auto`, `vllm`, `llama-cpp`, `sglang`, `trtllm`,
`tgi`, and `generic`.

## Install PATH Shims

PATH shims let existing commands run through infer-guard without changing every
script:

```bash
infer-guard install-shims
```

By default this installs guarded shims for common inference tools under
`~/.local/bin`. Make sure `~/.local/bin` appears before the real runtime binary
directory in `PATH`.

To remove the shims:

```bash
infer-guard uninstall-shims
```

## Wrap A Runtime Binary

Use `wrap` when a benchmark or script calls a fixed runtime path directly:

```bash
infer-guard wrap ~/runtimes/vllm/current/.venv/bin/vllm
```

This moves the original executable to `vllm.real` and replaces `vllm` with a
guarded wrapper. Restore the original binary with:

```bash
infer-guard unwrap ~/runtimes/vllm/current/.venv/bin/vllm
```

## Event Logs

Write launch, refusal, exit, and memory-pressure events as JSON lines:

```bash
infer-guard run --event-log ./infer-guard.events.jsonl -- vllm serve ...
```

Event logs are useful when a long benchmark run is killed and you need to know
whether infer-guard refused the launch, saw memory pressure, or observed a
normal child exit.

## Exit Behavior

- Child exits normally: returns the child exit code.
- Missing required earlyoom: returns `3`.
- Preflight memory or swap floor failure: returns `4`.
- Memory-pressure kill after launch: returns `137`.
- Guard configuration or runtime error: returns `2`.

## License

[MIT](LICENSE)
