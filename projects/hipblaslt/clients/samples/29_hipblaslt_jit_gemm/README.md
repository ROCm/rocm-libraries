# Run a JIT solution through the public GEMM APIs

This example compiles one supplied TensileLite recipe and executes the result
through both `hipblasLtMatmul` and `hipblaslt_ext::Gemm`. It owns its application
buffers and uses the installed JIT headers. The library owns compilation,
module loading, helper preparation and retained algorithm state.

The application follows five steps:

1. Configure the TensileLite backend with `jit::tensilelite::createBackend`.
2. Capture GEMM descriptors and host scalars with `jit::makeGemmRequest`.
3. Compile an owned solution with `jit::getJitAlgo`.
4. Adapt the solution to a GEMM algorithm with `jit::getGemmAlgo`.
5. Allocate the reported workspace and execute through the C and C++ APIs.

The [JIT API guide](../../../docs/jit.md) explains backend settings, build
requirements, concurrency, lifetime and failure behavior. The
[component roadmap](../../../tensilelite/docs/jit-roadmap.md) places the sample
in the full request, prediction, compilation and execution flow.

## Build and run

Using an existing configured build and Python environment, enable
`HIPBLASLT_ENABLE_JIT` as described in the API guide, then build this sample:

```bash
cmake --build "$project_build" --target _rocisa hipblaslt-jit-gemm --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
fixtures="$project_root/projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_data"
"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$project_python" "$project_root/projects/hipblaslt/tensilelite" \
  "$PYTHONPATH" "$fixtures/single_solution_splitk.yaml" /tmp/jit-gemm-splitk \
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++
```

Use the compiler and architecture appropriate for the local device. The sample
uses M=256, N=128, K=512, column-major NN FP16 input/output, FP32 accumulation,
alpha=1.25 and beta=0.5. The supplied recipe must support that problem.

The output path must not exist, and its parent directory must exist. Generator
diagnostics are retained in `<output>.log`, with process scratch files in
`<output>.cwd`. The sample poisons output storage before each run and compares
every element with an independently computed CPU reference.

The separate `hipblaslt-jit-api-test` executable checks changed inputs, copied
algorithms, workspace failures, invalid tokens, device identity, and retained
algorithms after another bundle is loaded. Those regression cases live under
`clients/tests/jit`; the sample illustrates application use.

## Optional automatic selection

With the provider's `Options::configPath` empty, TensileLite asks Origami for
ranked candidates and compiles the first one its validators accept. This
selection policy stays inside the provider; the generic request and solution
APIs are unchanged. The sample continues to use the supplied recipe. The
[benchmark guide](../../bench/README.jit.md) demonstrates automatic selection
and explains its current model limits and failure diagnostics.
