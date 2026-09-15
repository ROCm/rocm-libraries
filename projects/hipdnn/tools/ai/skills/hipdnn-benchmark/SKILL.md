---
name: hipdnn-benchmark
description: "Run hipDNN graph benchmarks, engine comparisons, and reference validation through dnn-benchmarking."
argument-hint: "[dnn-benchmark arguments]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Benchmark

Use `dnn-benchmarking` as the benchmarking interface for hipDNN.

1. Read the **Install from Released Wheels** section at `https://github.com/ROCm/dnn-benchmarking/tree/users/sareeder/hipdnn-studio-integration` and open its linked releases page. Use the newest release for that branch.
2. Run `rocminfo` and select the requirements file whose `gfx` suffix matches the GPU. Create a fresh Python 3.12+ virtual environment, then install that requirements URL exactly as the README shows.
3. Verify the installation before benchmarking: `python -c "import hipdnn_frontend, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"`. Import `hipdnn_frontend` before `torch` with this preview release to avoid loading an incompatible hipDNN library first.
4. **Bring your own hipDNN:** use a dnn-benchmarking source checkout and its `setup_env.py` whenever you want to test local hipDNN or provider changes, select a different rocm-libraries revision, or target a GPU without release wheels. Do not use the released runtime for these cases because it cannot contain your changes. Read the checkout's `AGENTS.md` or `CLAUDE.md`, then pass the documented workspace, source-revision, and CMake options to `setup_env.py`.
5. Run the requested benchmark. For a smoke check with no graph supplied, download the sample from the selected release tag: `curl -LO https://raw.githubusercontent.com/ROCm/dnn-benchmarking/<tag>/graphs/sample_conv_fwd.json`. Then run `dnn-benchmark --graph sample_conv_fwd.json --warmup 2 --iters 5 -o result.json`.
6. Report the release tag, GPU architecture, command, graph, engines or backend, result artifact, and exit status.

Do not guess a release tag or mix files from different revisions.
