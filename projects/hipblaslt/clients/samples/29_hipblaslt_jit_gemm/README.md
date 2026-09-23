# Generate an explicit GEMM algorithm

`hipblaslt_ext::experimental::getJitGemmAlgo`, declared in the installed
`hipblaslt/hipblaslt-ext.hpp`, compiles a TensileLite YAML recipe and returns
an algorithm for C `hipblasLtMatmul` or C++ `hipblaslt_ext::Gemm` execution.
The application supplies an explicit `GenerateOptions::configPath`.
Generation and support checks finish before any GPU work is submitted.

## Build

`HIPBLASLT_ENABLE_JIT_GEMM` is disabled by default. The declarations remain
available when disabled, and selection returns `HIPBLAS_STATUS_NOT_SUPPORTED`.
The enabled implementation requires the host library, Linux, ROCm, Boost
headers, TensileLite's Python dependencies, and the local rocisa extension.
Using an existing configured build and Python environment:

```bash
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT_GEMM=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --target _rocisa hipblaslt-jit-gemm --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
```

Use the compiler and target appropriate for the local device. Generated
bundles do not depend on a prebuilt hipBLASLt device library.

## Run the sample

```bash
fixtures="$project_root/projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_data"
"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$project_python" "$project_root/projects/hipblaslt/tensilelite" \
  "$PYTHONPATH" "$fixtures/single_solution_splitk.yaml" /tmp/jit-gemm-splitk \
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++ --k 512
```

The sample exercises both execution APIs. It defaults to M=256, N=128,
K=128, column-major NN FP16 input/output, FP32 accumulation, alpha=1.25,
and beta=0.5. `--m`, `--n`, `--k`, `--trans-b`, and `--amax` select the test
problem. The selected YAML must support that problem. `--normal-api both`
is also accepted.

The output path must not exist, and its parent directory must exist. Generator
diagnostics are retained in `<output>.log`, with process scratch files in
`<output>.cwd`. The sample checks every output against an independently
computed CPU reference, poisons output storage before each run, and exercises
copied algorithms, changed inputs, workspace errors, invalid tokens, device
identity, and retained algorithms after another bundle is loaded.

## Algorithm lifetime and validation

The returned heuristic result contains the required workspace size. Supply
that workspace and follow the ordinary handle and stream requirements of the
execution API. All helper entrypoints are resolved before submission.
Stream-K uses the handle's stream-specific synchronization region;
MultipleBufferSingleKernel and output-amax use its shared synchronization
storage. The ordinary concurrency requirements apply.

Copies of an algorithm remain usable on its generating device within the same
process. Its modules are retained until process exit. Save the recipe and
manifest for reproduction; the opaque algorithm cannot be persisted as a
prebuilt library index, and no persistent bundle reload/cache is provided.

An empty output (M=0 or N=0) needs no algorithm and returns NOT_SUPPORTED without
generation. K=0 still generates the beta*C operation. Physical MX layouts are
checked before selection and again at support/launch: gfx950 requires the
pre-swizzled block32 UE8M0 descriptor and `HostPreSwizzle`, while gfx1250
requires its ordinary block-scale descriptors and `InMemorySwizzle`.

`test_jit_normal_helper_failures.py` removes helper modules or symbols from a
valid split-K bundle, checks that C/extension paths leave D and workspace
untouched, and verifies that failed reinitialization preserves the previous
extension algorithm. Explicit recipes use TensileLite's target and solution
validators. Output-amax currently requires one batch, GlobalSplitU=1 and
StreamK=0. Generation never benchmarks recipes or substitutes another recipe
when the supplied one fails.

The [single-solution documentation](../../../tensilelite/docs/single-solution.md)
describes the recipe, bundle, and Python builder contracts.
