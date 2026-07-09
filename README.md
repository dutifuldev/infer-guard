# infer-guard

infer-guard is a local LLM inference launch guard.
It starts tools such as vLLM, llama.cpp, SGLang, and TensorRT-LLM behind memory
and process-group protection so model loads, kernel compiles, and benchmarks do
not take down an interactive machine.

Planned usage:

```bash
infer-guard doctor
infer-guard run -- vllm serve ...
infer-guard install-shims
infer-guard wrap ~/runtimes/vllm/current/.venv/bin/vllm
```

`infer-guard run` starts the child in its own process group, watches memory and
swap, and terminates the process group before the machine runs out of room. For
high-risk local inference profiles, it requires `earlyoom` by default.

## Install

This repository is not published yet. For development:

```bash
cargo install --path .
```

## Commands

Check the local machine:

```bash
infer-guard doctor
```

Run a local inference command through the guard:

```bash
infer-guard run --profile vllm --min-mem 24G --min-swap 4G -- vllm serve ...
```

Install guarded PATH shims:

```bash
infer-guard install-shims
```

Wrap a canonical runtime binary used by existing scripts:

```bash
infer-guard wrap ~/runtimes/vllm/current/.venv/bin/vllm
```

The implementation plan is in [docs/implementation-plan.md](docs/implementation-plan.md).

## License

[MIT](LICENSE)
