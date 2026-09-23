# Request and execute JIT solutions

The installed `hipblaslt/hipblaslt-jit.hpp` exposes a backend-neutral request API.
`getJitAlgo` accepts an operation request and configured backend and returns an
owned solution bundle. The bundle includes the kernels and helpers needed for
that operation. GEMM is implemented; an attention request adapter and provider
remain future work.

An application configures the TensileLite provider through
`jit::tensilelite::createBackend`, using options from
`hipblaslt/hipblaslt-jit-tensilelite.hpp`. Python, compiler, recipe and output
paths belong to this provider. It then uses `jit::makeGemmRequest` to capture
existing GEMM descriptors and host scalars, and `jit::getJitAlgo` to compile on
the selected device. Finally, `jit::getGemmAlgo` adapts the solution to the
algorithm accepted by `hipblasLtMatmul` and `hipblaslt_ext::Gemm`.

The application owns its buffers and workspace. The request owns descriptor
values and host scalars; it does not take ownership of device pointers.
Compilation and support checks finish before GPU work is submitted.

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
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --target _rocisa hipblaslt-jit-api-test --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
```

Use the compiler and target appropriate for the local device. Generated
bundles do not depend on a prebuilt hipBLASLt device library.

The `jit` CMake preset enables this feature for a new configuration. The
commands above show how to enable it in an existing build. The shared workflow
runs the public API regression executable and an independent test backend.

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

`../clients/tests/jit/test_helper_failures.py` removes helper modules or symbols from a
valid split-K bundle, checks that C/extension paths leave D and workspace
untouched, and verifies that failed reinitialization preserves the previous
extension algorithm. `../clients/tests/jit/test_bundle_failures.py` checks malformed
envelopes, missing code, mismatched solution identity and unsupported problems
through the same public API. Both scripts run in the shared JIT workflow.

Explicit recipes use TensileLite's target and solution
validators. Output-amax currently requires one batch, GlobalSplitU=1 and
StreamK=0. Generation never benchmarks recipes or substitutes another recipe
when the supplied one fails.

The [single-solution documentation](../tensilelite/docs/single-solution.md)
describes the recipe, bundle, and Python builder contracts.

The private provider loader reads `loader.bin`, a bounded, versioned envelope
published alongside the human-readable `manifest.json`. Corruption tests modify
the consumed envelope or code objects; JSON is a diagnostic record.

A searchable JIT solution library and persistent code cache are future work.
Such a library could index solutions by problem description and look up compatible
code before requesting generation. The current API retains explicitly selected
algorithms in one process; it does not search a JIT collection or reuse code from
an earlier program invocation.


The [component roadmap](../tensilelite/docs/jit-roadmap.md) describes the complete flow and remaining planning, search and cache work.
