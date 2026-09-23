# Request and execute JIT solutions

The installed `hipblaslt/hipblaslt-jit.hpp` exposes a backend-neutral request API.
`getJitAlgo` accepts an operation request and configured backend and returns an
owned solution bundle. The bundle includes the kernels and helpers needed for
that operation. This sample requests GEMM; no attention provider is implemented.

The sample uses the following public interfaces:

1. `jit::tensilelite::createBackend` configures the TensileLite provider using
   options from `hipblaslt/hipblaslt-jit-tensilelite.hpp`. Python, compiler, recipe
   and output paths belong to this provider.
2. `jit::makeGemmRequest` captures the existing GEMM descriptors and host scalars.
3. `jit::getJitAlgo` compiles a solution on the current device.
4. `jit::getGemmAlgo` adapts that GEMM solution to an ordinary hipBLASLt algorithm.
5. `hipblasLtMatmul` or `hipblaslt_ext::Gemm` checks workspace and executes it.

Compilation and support checks finish before any GPU work is submitted. The
sample owns its application buffers and calls `hipblasLtMatmul` and `Gemm`.

## Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. The declarations remain
available when disabled, and selection returns `HIPBLAS_STATUS_NOT_SUPPORTED`.
The enabled implementation requires the host library and ROCm. The TensileLite
provider also requires its Python dependencies and the local rocisa extension.
Provider processes use POSIX spawning on Linux and `CreateProcessW` on Windows;
additional Python import paths use the platform separator (`:` or `;`).
Using an existing configured build and Python environment:

```bash
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
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
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++
```

The sample exercises both execution APIs. It uses M=256, N=128,
K=512, column-major NN FP16 input/output, FP32 accumulation, alpha=1.25,
and beta=0.5. The selected YAML must support that problem.

The output path must not exist, and its parent directory must exist. Generator
diagnostics are retained in `<output>.log`, with process scratch files in
`<output>.cwd`. The sample checks every output against an independently
computed CPU reference and poisons output storage before each run.
The separate `hipblaslt-jit-api-test` executable checks copied algorithms, changed
inputs, workspace errors, invalid tokens, device identity, and retained algorithms
after another bundle is loaded.

## Algorithm lifetime and validation

The returned heuristic result contains the required workspace size. Supply
that workspace and follow the same handle, stream, and workspace sharing rules
as `hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. All helper entrypoints are resolved before submission.
Stream-K uses the handle's stream-specific synchronization region;
MultipleBufferSingleKernel and output-amax use its shared synchronization
storage. Registry synchronization protects algorithm lookup; it does not protect
application buffers or make simultaneous calls on one `Gemm` object safe.

Copies of an algorithm remain usable on its generating device within the same
process. Its modules are retained until process exit. Reuse within one program invocation needs no recompilation. A different program
invocation compiles its own solution: no persistent cache or bundle reload is
provided. Save recipes and diagnostic manifests for reproduction; never persist
the opaque algorithm bytes or treat them as a prebuilt library index.

An empty GEMM output (M=0 or N=0) returns NOT_SUPPORTED from the request factory
without compilation. K=0 can use a recipe that implements beta*C. TensileLite
owns its datatype, instruction and scale-layout restrictions. The library
propagates provider support failures, including a mismatch between the supplied
physical MX scale layout and the compiled solution.

`clients/tests/jit/test_helper_failures.py` removes helper modules or symbols from a
valid split-K bundle, checks that C/extension paths leave D and workspace
untouched, and verifies that failed reinitialization preserves the previous
extension algorithm. `clients/tests/jit/test_bundle_failures.py` checks malformed
envelopes, missing code, mismatched solution identity and unsupported problems
through the same public API. Both scripts run in the shared JIT workflow.

Explicit recipes use TensileLite's target and solution
validators. Output-amax currently requires one batch, GlobalSplitU=1 and
StreamK=0. Generation never benchmarks recipes or substitutes another recipe
when the supplied one fails.

The [single-solution documentation](../../../tensilelite/docs/single-solution.md)
describes the recipe, bundle, and Python builder contracts.

The private provider loader reads `loader.bin`, a bounded, versioned envelope
published alongside the human-readable `manifest.json`. Corruption tests modify
the consumed envelope or code objects; JSON is a diagnostic record.

A searchable JIT solution library and persistent code cache are future work.
Such a library could index solutions by problem description and look up compatible
code before requesting generation. The current API retains explicitly selected
algorithms in one process; it does not search a JIT collection or reuse code from
an earlier program invocation.
