# infer-guard Implementation Plan

## Goal

Build a small Rust CLI that makes local LLM inference launches hard to OOM.
It should protect humans, agents, benchmark harnesses, and shell scripts that
start vLLM, llama.cpp, SGLang, TensorRT-LLM, TGI, Ollama, LM Studio CLI, or
similar local serving processes.

The default path should be automatic: after shims are installed, existing
commands such as `vllm serve ...` or configured runtime entrypoints run through
the guard without users remembering a special wrapper.

## Non-Goals

- Do not replace earlyoom. Use it as the machine-wide safety net.
- Do not create a persistent service by default.
- Do not silently delete caches, Docker state, model weights, or user files.
- Do not hide remote endpoint auth failures by falling back to local inference.
- Do not tune model quality or inference performance policy.

## Core Design

`infer-guard run -- <command>` is the primitive.

It should:

- verify the intended command is local inference, not a mistaken fallback for a
  remote API target
- require active `earlyoom` by default for high-risk profiles
- preflight `/proc/meminfo`, swap, disk, and existing inference processes
- launch the child in a new process group
- optionally launch inside a transient user cgroup/scope
- monitor `MemAvailable`, `SwapFree`, child liveness, and process tree RSS
- send `SIGTERM`, then `SIGKILL`, to the whole process group on pressure
- write structured JSON event logs explaining launches, kills, and exits
- return `137` when it kills a workload for memory pressure

## CLI Surface

Initial commands:

```bash
infer-guard doctor
infer-guard run [options] -- <command> [args...]
infer-guard install-shims [options]
infer-guard uninstall-shims [options]
infer-guard wrap <path>
infer-guard unwrap <path>
infer-guard inspect
```

Useful run options:

```text
--profile vllm|llama-cpp|sglang|trtllm|tgi|generic
--min-mem 24G
--min-swap 4G
--poll 1s
--term-grace 10s
--require-earlyoom / --allow-no-earlyoom
--event-log PATH
--use-systemd-scope
--memory-high SIZE
--memory-max SIZE
```

Environment overrides for shims:

```text
INFER_GUARD_MIN_MEM
INFER_GUARD_MIN_SWAP
INFER_GUARD_PROFILE
INFER_GUARD_EVENT_LOG
INFER_GUARD_REAL_<TOOL>
```

## Automatic Adoption

There are two adoption layers.

PATH shims:

```text
~/.local/bin/vllm
~/.local/bin/llama-server
~/.local/bin/llama-cli
~/.local/bin/sglang
~/.local/bin/trtllm-serve
~/.local/bin/text-generation-launcher
```

These resolve the real executable later in `PATH` and run it through
`infer-guard run`.

Absolute runtime wrappers:

```bash
infer-guard wrap ~/runtimes/vllm/current/.venv/bin/vllm
```

This moves the target to `vllm.real` and replaces `vllm` with a generated
wrapper. Use this for benchmark scripts that call canonical runtime paths
directly.

`infer-guard unwrap` must restore the original binary exactly.

## Safety Policy

Defaults for interactive GPU workstations:

```text
min_mem = 24G
min_swap = 4G
poll = 1s
term_grace = 10s
require_earlyoom = true
```

The guard should refuse a launch when:

- earlyoom is required but not running
- available memory or swap is already below the configured floor
- another local inference process is already consuming unsafe memory
- the command looks like a local fallback for an intended remote endpoint
- disk is too tight for known compile/model cache growth

The guard may warn, but should not block, for unknown command names when the user
explicitly selects `--profile generic`.

## Cgroup Integration

Phase one should work without systemd or root privileges.

Phase two should support:

```bash
systemd-run --user --scope
```

with optional `MemoryHigh` and `MemoryMax` properties. This is an additional
containment layer, not a replacement for process-group killing or earlyoom.

No persistent system or user service should be installed unless a user
explicitly asks for one.

## Runtime Profiles

Profiles encode command detection and default thresholds.

Initial profiles:

- `vllm`: `vllm`, `api_server`, `gpu_worker`, `python -m vllm`
- `llama-cpp`: `llama-server`, `llama-cli`, `llama-bench`
- `sglang`: `sglang`, `python -m sglang`
- `trtllm`: `trtllm-serve`, TensorRT-LLM launchers
- `tgi`: `text-generation-launcher`
- `generic`: user-selected fallback with explicit thresholds

Profiles should stay conservative. They are safety defaults, not performance
tuning presets.

## Packaging

Primary package:

- Rust binary published through GitHub Releases
- install script that downloads the right release artifact
- `cargo install` support for developers

Later package targets:

- Homebrew tap
- Debian package
- Arch package

The install script should install only the binary by default. Shim installation
must be an explicit `infer-guard install-shims` step.

## Testing

Unit tests:

- memory-size parsing
- process tree discovery
- shim generation
- wrapper unwrap round trip
- event log serialization

Integration tests:

- child exits normally
- child ignores `SIGTERM` and receives `SIGKILL`
- guard kills a process group when a fake memory source crosses threshold
- PATH shim resolves the real binary without recursing
- in-place wrapper restores the exact original executable

Do not require real OOM conditions in CI. Use injectable memory readers and fake
child processes.

## Milestones

1. Rust project skeleton, `doctor`, config parsing, and CI.
2. `run` with process group supervision and `/proc/meminfo` monitoring.
3. JSON event logs and exit-code contract.
4. PATH shim installer and uninstall flow.
5. In-place wrap and unwrap flow for absolute runtime binaries.
6. Profile detection for vLLM and llama.cpp.
7. Optional `systemd-run --user --scope` integration.
8. Release artifacts and installer.
9. Update Codex/Claude skills to prefer the released binary over bundled shell
   scripts.
