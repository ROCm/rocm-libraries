# TensileLite JIT backend

This is a source guide for contributors and integration developers, maintained
under the existing hipBLASLt code/documentation reviewer rules. The
[JIT guide](JIT.md) describes the overall just-in-time (JIT) design, the
components around this backend and the [roadmap](JIT.md#roadmap).

TensileLite is the live JIT backend. This page describes its current direct
entry point, which compiles one explicit YAML (YAML Ain't Markup Language)
recipe, and its planned role behind the Jit backend interface.

**Status:** the direct entry point below is internal.
`hipblaslt-jit-tensilelite.hpp` is in `library/src/amd_detail/`; it is not
installed, and only the JIT tests and `hipblaslt-bench --jit-gemm` use it. After
[roadmap](JIT.md#roadmap) step 5, applications reach TensileLite generation
through `hipblasLtMatmulAlgoGetHeuristic` with `HIPBLASLT_JIT`.

## Current direct entry point

The experimental TensileLite API compiles one supplied YAML recipe, loads its
generated kernel and helpers, and returns a `hipblasLtMatmulHeuristicResult_t`.
Use that result with `hipblasLtMatmul` or the C++ `hipblaslt_ext::Gemm` class.
Compilation happens synchronously before general matrix multiplication (GEMM)
execution.

In-tree code includes `"hipblaslt-jit-tensilelite.hpp"` with
`library/src/amd_detail` on its include path and calls
`hipblaslt_ext::experimental::jit::tensilelite::getGemmAlgo`. The call takes the
ordinary GEMM descriptors, scalars and buffers, generation options, and a
workspace limit. TensileLite must accept the recipe for that problem and device.
The `direct-gemm` case in the [JIT tests](clients/tests/jit/README.md) shows the
whole flow.

### Build requirements

Enable the host library and `HIPBLASLT_ENABLE_JIT` in the existing configured
build. For example, from the repository root with `project_build` set to that
build directory:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_JIT=ON
cmake --build "$project_build" --target _rocisa hipblaslt --parallel
```

Generation needs a Python interpreter with the TensileLite dependencies and
access to the checkout's TensileLite modules and built `rocisa` extension. Supply
that interpreter and the compiler/offload-bundler paths through `Options`.
The generated path loads its own code objects; it does not require a prebuilt
hipBLASLt device library.

A build with JIT disabled does not compile or export the entry point.

### Describe and compile the GEMM

Create the handle, matrix descriptors, input/output buffers and stream using the
ordinary hipBLASLt/HIP APIs. Keep the handle's device current when compiling.
The following fragment assumes those objects exist and that `check` reports a
failed HIP or hipBLASLt status:

```cpp
#include "hipblaslt-jit-tensilelite.hpp"
#include <stdexcept>

namespace tl = hipblaslt_ext::experimental::jit::tensilelite;

tl::Options options;
options.pythonExecutable       = pythonExecutable;
options.tensileSourceDirectory = tensileSourceDirectory;
options.pythonPath             = additionalPythonPaths;
options.configPath             = recipePath;
options.outputPath             = freshOutputDirectory;
options.cxxCompiler            = compilerPath;
options.offloadBundler         = offloadBundlerPath;
// An empty architecture uses the current device's architecture.

hipblasLtMatmulHeuristicResult_t selected{};
tl::Diagnostics diagnostics;
auto status = tl::getGemmAlgo(handle, desc, &alpha,
                              A, layoutA, B, layoutB, &beta,
                              C, layoutC, D, layoutD,
                              options, maxWorkspaceBytes,
                              selected, diagnostics);
if(status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(diagnostics.message);
```

`configPath` is required and names an explicit single-solution YAML recipe.
The output directory must be new and its parent must exist. Additional Python
import paths use the platform separator: `:` on POSIX and `;` on Windows.
An explicit architecture must match the current device, including any required
target features. The recipe, requested GEMM and available workspace must agree;
a valid build alone does not make a kernel usable for every problem.

### Execute and reuse the result

Allocate `selected.workspaceSize` bytes when it is nonzero, then pass the
returned algorithm to the existing execution API:

```cpp
void* workspace = nullptr;
if(selected.workspaceSize)
    check(hipMalloc(&workspace, selected.workspaceSize));

check(hipblasLtMatmul(handle, desc, &alpha,
                      A, layoutA, B, layoutB, &beta,
                      C, layoutC, D, layoutD,
                      &selected.algo, workspace, selected.workspaceSize, stream));
```

For the C++ extension, construct `Gemm` with the same problem and prepare the
algorithm before running it:

```cpp
#include <hipblaslt/hipblaslt-ext.hpp>

hipblaslt_ext::Gemm gemm(handle, desc, &alpha,
                       A, layoutA, B, layoutB, &beta,
                       C, layoutC, D, layoutD);
gemm.setMaxWorkspaceBytes(selected.workspaceSize);
check(gemm.initialize(selected.algo, workspace, false, stream));
check(gemm.run(stream));
```

Repeated executions reuse the compiled algorithm. Keep application buffers,
workspace and ordinary descriptors alive for the operations that use them, and
synchronize before freeing storage. The usual handle, workspace and stream
concurrency rules still apply. Initializing and running on the same stream also
satisfies recipes whose synchronization state is bound to that stream.

The library retains the compiled algorithm's modules until process exit, so
copies of the algorithm remain usable in that process on their original
device. Algorithm bytes and indices are not a persistent library format today.
The API does not expose loading a retained bundle in a later process. Compile
before stream capture.

### Artifacts and failures

The output directory contains `bundle/manifest.json`, the private loader
envelope, the serialized solution library and all generated code objects.
Generator diagnostics remain in the sibling `<output>.log`; the child working
directory is `<output>.cwd`. Use a new output path for another compilation.

On failure, `getGemmAlgo` clears the result, sets `result.state`, and returns a
status; `Diagnostics::message` provides details when available. A missing recipe
is rejected before starting generation. The loader checks artifact identities
and targets, and execution preparation resolves required helper symbols before
submission. Invalid recipes, missing artifacts or unsupported problems produce a
failure rather than an executable result. Empty output needs no GEMM algorithm
and returns `HIPBLAS_STATUS_NOT_SUPPORTED`.

The direct entry point always requires a recipe. Origami prediction is available
through the generic TensileLite provider when `Options::configPath` is empty;
see [Origami modeled inputs](JIT.md#origami-modeled-inputs).

## Planned backend role

The rows below are planned; the code does not implement them yet. Each row names
the roadmap step that changes it.

| Concern | Current | Planned |
| --- | --- | --- |
| Entry point | Internal direct `tensilelite::getGemmAlgo`, and the generic provider created by `tensilelite::createBackend`, both used by tests and the benchmark. | An internal backend behind Jit, reached from the heuristic query. The headers remain for unit tests (step 2). |
| Backend input | An explicit recipe, or problem facts plus Origami-ranked candidates in the generic provider. | Algorithm parameters (M, N, K, datatypes, scale types, layout, activation and the rest of the GEMM description) plus the gfx target, with ranked candidates from the Predictor (step 2). |
| Unmodeled knobs | TensileLite defaults and derivation. | Supplied through TuningKnowledge, which initially returns the same TensileLite defaults (step 2). |
| Backend output | A complete bundle: `Tensile.SingleSolution` assembles, links and compiles code objects with the configured compiler and offload bundler. | Assembly, HIP helper source and metadata only. hipBLASLt builds raw, uncompressed code objects through AMD comgr (step 3). |
| Persistence | Process-local; each program invocation compiles again. | Solutions are published into the per-`ProblemType` JIT solution library and reused across processes (step 4). |

TensileLite remains one of several independent backends. rocRoller and
HipKittens are future backends behind the same interface; they do not route
through TensileLite. `Tensile.SingleSolution` and `Tensile.JitGemm` are the
current Python entry points; the [single-solution guide](tensilelite/SINGLE_SOLUTION.md)
describes them. Step 3 changes what the backend produces.

[Validation](JIT.md#validation) describes the test coverage. Source support and
cross-compilation do not establish numerical results on another GPU or native
Windows execution.
