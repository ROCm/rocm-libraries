# TensileLite JIT backend

This is a source guide for contributors and integration developers, maintained
under the existing hipBLASLt code/documentation reviewer rules. The
[JIT guide](JIT.md) describes the overall just-in-time (JIT) design, the
components around this backend and the [roadmap](JIT.md#roadmap).

TensileLite is the live JIT backend. This page describes its current direct
entry point, which compiles one explicit YAML (YAML Ain't Markup Language)
recipe, its role behind the Jit backend interface, and how heuristic queries
use it.

**Status:** the direct entry point below is internal.
`hipblaslt-jit-tensilelite.hpp` is in `library/src/amd_detail/`; it is not
installed, and only the JIT tests use it. Applications reach TensileLite
generation through `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` with `HIPBLASLT_JIT`; see
[heuristic integration](JIT.md#heuristic-integration).

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
that interpreter and the compiler path through `Options`; TensileLite uses the
compiler only to probe assembler capabilities. hipBLASLt builds the code object
in process with comgr, so the JIT build also needs ROCm's `amd_comgr` package
(see [Build](JIT.md#build)). The generated path loads its own code object; it
does not require a prebuilt hipBLASLt device library.

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
device. These algorithm bytes are not a library index and do not outlive the
process. The generic `jit::getLibraryAlgos` publishes solutions into the
[persistent solution library](JIT.md#persistent-solution-library) instead, and
returns solution indices that later processes run. Compile before stream
capture.

### Artifacts and failures

The output directory contains the source bundle `bundle/`: `library/` holds the
serialized one-solution library, `sources/` holds the main kernel assembly and
the helper HIP source with its headers, and `manifest.json` records provenance.
Generator diagnostics remain in the sibling `<output>.log`; the child working
directory is `<output>.cwd`. When
[`HIPBLASLT_JIT_DEBUG`](JIT.md#diagnostics-with-hipblaslt_jit_debug) is in
effect, the backend passes the generator `--debug` and
`--debug-dir <output>.cwd/jit-debug`, so that directory holds the generator's
`events.jsonl` and `timing.json`. Use a new output path for another
compilation.

On failure, `getGemmAlgo` clears the result, sets `result.state`, and returns a
status; `Diagnostics::message` provides details when available. A missing recipe
is rejected before starting generation. The bundle reader enforces containment
and size limits. The builder checks that the code object targets the device and
defines the entry's kernel; when comgr fails, the message names the `comgr.log`
kept in the Jit scratch directory. Execution preparation resolves required
helper symbols before submission. Invalid recipes, missing artifacts or
unsupported problems produce a failure rather than an executable result. Empty
output needs no GEMM algorithm and returns `HIPBLAS_STATUS_NOT_SUPPORTED`.

The direct entry point always requires a recipe. Origami prediction is available
through `tensilelite::createBackend` when `Options::configPath` is empty; see
[Origami modeled inputs](JIT.md#origami-modeled-inputs).

## Role behind Jit

`TensileLiteBackend` implements the Jit backend interface. Both entry points
create it through `tensilelite::createBackend`, which configures a Jit with the
Origami predictor, the TensileLite defaults as tuning knowledge, the comgr
builder and the Tensile loader. Without a recipe, the backend consumes the
predictor's `origami.gemm.dp.v1` prediction and writes it as the
`Tensile.JitGemm` request; with a recipe, Jit skips prediction and the backend
passes the recipe to `Tensile.SingleSolution`. Knobs the model does not predict
keep TensileLite defaults and derivation. Both run with `--source-only` and the
code-object version of the Jit request. When Jit asks for more than one
solution, the request adds `requested_solutions`, and `Tensile.JitGemm`
publishes one bundle per accepted candidate as `bundle-<rank>`, each with a
different kernel. The kernels that Jit excludes become `exclude_kernel_names`,
which `Tensile.JitGemm` compares with the kernel names of the published
library, so ranked selection moves on to the next kernel the caller lacks. The backend
returns each bundle's one-solution library entry, main kernel assembly and
helper source, and builds and loads nothing; the comgr builder builds one code
object per solution, and the Tensile loader checks support and loads it.

## Heuristic queries

With `HIPBLASLT_JIT` set, the heuristic queries reach this backend without an
application `Options`: hipBLASLt configures it once per process from the
[built-in tool paths](JIT.md#tool-paths-and-scratch-files), with Origami
prediction, and publishes its solutions into the JIT solution library.
`tensilelite::getGemmAlgo` and `tensilelite::createBackend` remain internal
entry points for the tests.

TensileLite remains one of several independent backends. HipKittens and other
future backends sit behind the same interface; they do not route through
TensileLite. `Tensile.SingleSolution` and `Tensile.JitGemm` are the
current Python entry points; the [single-solution guide](tensilelite/SINGLE_SOLUTION.md)
describes them.

[Validation](JIT.md#validation) describes the test coverage. Source support and
cross-compilation do not establish numerical results on another GPU or native
Windows execution.
