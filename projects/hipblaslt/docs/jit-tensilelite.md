# Compile an explicit TensileLite recipe for GEMM

The experimental TensileLite API compiles one supplied YAML recipe, loads its
generated kernel and helpers, and returns a `hipblasLtMatmulHeuristicResult_t`.
Use that result with `hipblasLtMatmul` or the C++ `hipblaslt_ext::Gemm` class.
Compilation happens synchronously before GEMM execution.

Include `<hipblaslt/hipblaslt-jit-tensilelite.hpp>` and call
`hipblaslt_ext::experimental::jit::tensilelite::getGemmAlgo`. The call takes the
ordinary GEMM descriptors, scalars and buffers, generation options, and a
workspace limit. TensileLite must accept the recipe for that problem and device.

## Build requirements

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

The public declaration remains available when JIT is disabled. Calling it then
returns `HIPBLAS_STATUS_NOT_SUPPORTED` and a diagnostic naming the build option.

## Describe and compile the GEMM

Create the handle, matrix descriptors, input/output buffers and stream using the
ordinary hipBLASLt/HIP APIs. Keep the handle's device current when compiling.
The following fragment assumes those objects exist and that `check` reports a
failed HIP or hipBLASLt status:

```cpp
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
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

## Execute and reuse the result

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
device. Algorithm bytes and indices are not a persistent library format. The
API does not expose loading a retained bundle in a later process. Compile before
stream capture.

## Artifacts and failures

The output directory contains `bundle/manifest.json`, the private loader
envelope, the serialized solution library and all generated code objects.
Generator diagnostics remain in the sibling `<output>.log`; the child working
directory is `<output>.cwd`. Use a new output path for another compilation.

On failure, `getGemmAlgo` clears the result, sets `result.state`, and returns a
status; `Diagnostics::message` provides details when available. A missing recipe is rejected before starting
generation. The loader checks artifact identities and targets, and execution
preparation resolves required helper symbols before submission. Invalid recipes,
missing artifacts or unsupported problems produce a failure rather than an
executable result. Empty output needs no GEMM algorithm and returns
`HIPBLAS_STATUS_NOT_SUPPORTED`.

## Current scope and follow-up work

This entry point is the direct TensileLite path for an explicit GEMM recipe.
The separate generic API stack adds a common backend/operation interface above
this implementation. Prediction, a searchable JIT library, persistent caching,
and shared KFA metadata are follow-up work. They are not prerequisites for using
the direct recipe interface.

Consult the PR validation results for the configurations actually tested.
Source support and cross-compilation do not establish numerical results on
another GPU or native Windows execution.
