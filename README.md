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

The implementation plan is in [docs/implementation-plan.md](docs/implementation-plan.md).

## License

[MIT](LICENSE)
