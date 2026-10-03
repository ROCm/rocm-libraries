# TensileLite JIT backend

This is a source guide for contributors and integration developers, maintained
under the existing hipBLASLt code/documentation reviewer rules. The
[JIT guide](JIT.md) describes the parts of just-in-time (JIT) generation that
do not depend on a backend: the Jit stages, the comgr builder, the JIT solution
library, heuristic integration and `HIPBLASLT_JIT_DEBUG`. The
[target design and roadmap](JIT_ROADMAP.md) place this backend in the overall
design. The [single-solution guide](tensilelite/SINGLE_SOLUTION.md) covers the
Python builder, recipes, bundles and ranked candidate validation, and the
[TensileLite JIT test guide](clients/tests/jit/README.tensilelite.md) covers the
tests that generate with it. The [design notes](jit-design/README.md) hold the
KernelFromAnywhere (KFA) assessment.

TensileLite is the live JIT backend. This page describes how it is built and
how it generates, its direct entry point, which compiles one explicit YAML
(YAML Ain't Markup Language) recipe, its role behind the Jit backend
interface, and how heuristic queries use it.

**Status:** the direct entry point below is internal.
`hipblaslt-jit-tensilelite.hpp` is in `library/src/amd_detail/`; it is not
installed, and only the JIT tests use it. Applications reach TensileLite
generation through `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` with `HIPBLASLT_JIT`; see
[heuristic integration](JIT.md#heuristic-integration).

## Build

`cmake/hipblaslt-jit-tensilelite.cmake` adds this backend to a build with
`HIPBLASLT_ENABLE_JIT=ON`. Its option `HIPBLASLT_JIT_TENSILELITE`, `ON` by
default, makes it the backend of heuristic queries, compiles its
[tool paths](#tool-paths) into the library and adds the tests that generate
with it; `OFF` leaves the build without it. Besides what [Build](JIT.md#build)
lists, the backend requires its Python dependencies and the local rocisa
extension. Additional Python import paths use the platform separator (`:` or
`;`). From the repository root, with a Python environment that has the
TensileLite dependencies:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
project_python=/path/to/venv/bin/python
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --target _rocisa hipblaslt-jit-generic-api-test --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
export LD_LIBRARY_PATH="$project_build/library:$project_build/tensilelite${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use the compiler and target appropriate for the local device. Set both Python
executable keys to the same interpreter, and use the built rocisa extension only
with that interpreter. TensileLite uses the compiler only to probe assembler
capabilities; hipBLASLt builds the code object in process with comgr. The
generated path loads its own code object; it does not require a prebuilt
hipBLASLt device library. A build with JIT disabled does not compile or export
the entry points.

## Generation

The host runs the Python generator as a child process: POSIX spawning on Linux
and `CreateProcessW` on Windows. With `--source-only`, `Tensile.SingleSolution`
writes the main kernel assembly, the helper HIP source and headers, and the
serialized one-solution library, and publishes them as a source bundle with a
provenance manifest. The host reads the bundle, builds one code object with
comgr, checks support and workspace with TensileLite's predicates, and resolves
the helper symbols before the first submission.

| Component | Current input, output and connection |
| --- | --- |
| One-solution builder | One recipe and target produce a source bundle through `Tensile.SingleSolution --source-only` and the existing TensileLite generators and validators: the main kernel assembly, the helper HIP source and its headers, and the one-solution library entry. It builds no code objects in this mode. |
| Ranked recipe selector | Supplied candidates and problem facts produce validated recipes or rejection reasons. `Tensile.JitGemm` calls the builder without running a model, and publishes the first `requested_solutions` accepted candidates (default 1) that `exclude_kernel_names` does not name, each with a different kernel. It compares the kernel names that the published library uses. |
| TensileLite backend | `hipblaslt-jit-tensilelite.cpp`. It writes the `Tensile.JitGemm` request from the prediction, or passes an explicit recipe through, runs the generator with `--source-only`, and returns each published bundle's library entry, main kernel assembly and helper source. It builds and loads nothing. |
| Process backend | `hipblaslt-jit-tensilelite-backend.cpp` defines `makeDefaultProcessBackend` for heuristic queries: the TensileLite backend configured from the [tool paths](#tool-paths), the Origami predictor and the tuning library knowledge. |
| Direct entry point and test | An explicit recipe and GEMM descriptors produce a checked algorithm. `hipblaslt-jit-direct-gemm-test` exercises C/C++ execution independently of the generic entry point, and `hipblaslt-jit-generic-gemm-test` runs the generic flow with this backend. |

With `createBackend`, an empty `Options::configPath` makes Jit run the Origami
predictor before generation; the backend forwards the
[`origami.gemm.dp.v1` and `origami.gemm.persistent.v1` modeled inputs](JIT.md#origami-modeled-inputs)
and the `tensilelite.tuned.v1` seeds to `Tensile.JitGemm`, which applies each
candidate's contract's rejection rules. The selector
validates ranked candidates in order and publishes the first supported recipe,
or the first N when Jit asks for N solutions, skipping a ranked candidate that
would repeat an accepted kernel. It does not benchmark candidates or invent a
recipe when selection fails, and never substitutes another recipe when the
supplied one fails. Explicit recipes continue through `Tensile.SingleSolution`
without the prediction contract. Save recipes and diagnostic manifests for
reproduction.

TensileLite owns its datatype, instruction and scale-layout restrictions;
output-amax currently requires one batch, GlobalSplitU=1 and
TileProcessingStrategy=None.

The backend's version in the
[cache key](JIT.md#persistent-solution-library) is a hash of the hipBLASLt
version and library file, the backend options, the Python interpreter and
compiler files, the recipe, and the TensileLite and rocisa Python sources.

Besides the entry points that [JIT.md](JIT.md#entry-points) lists,
`libhipblaslt.so` exports `jit::tensilelite::createBackend` and
`jit::tensilelite::getGemmAlgo` from `hipblaslt-jit.hpp` and
`hipblaslt-jit-tensilelite.hpp` for the tests. For the internal entry points,
Python, compiler, recipe and output paths belong to `tensilelite::Options`;
heuristic queries use the [tool paths](#tool-paths) instead.

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
The `direct-gemm` case in the
[TensileLite JIT tests](clients/tests/jit/README.tensilelite.md) shows the whole
flow.

Generation needs a Python interpreter with the TensileLite dependencies and
access to the checkout's TensileLite modules and built `rocisa` extension; see
[Build](#build). Supply that interpreter and the compiler path through
`Options`.

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
[`HIPBLASLT_JIT_DEBUG`](JIT.md#diagnostics-with-hipblaslt_jit_debug) has
`timing` or `progress` on, the backend passes the generator `--debug` with those
categories and
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
through `tensilelite::createBackend` when `Options::configPath` is empty.

## Role behind Jit

`TensileLiteBackend` implements the Jit backend interface. Both entry points
create it through `tensilelite::createBackend`, which configures a Jit with the
Origami predictor, the tuning library knowledge, the comgr
builder and the Tensile loader. Without a recipe, the backend consumes the
predictor's prediction and writes it as the `Tensile.JitGemm` request; with a recipe, Jit skips prediction and the backend
passes the recipe to `Tensile.SingleSolution`. Knobs that neither the model nor
a tuned seed supplies keep TensileLite defaults and derivation. Both run with `--source-only` and the
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
[tool paths](#tool-paths), with Origami prediction, and publishes its solutions
into the JIT solution library. `tensilelite::getGemmAlgo` and
`tensilelite::createBackend` remain internal entry points for the tests.

Its failure reports name the variable to set for a configure failure and the
generator log for a generation failure, and its shortfall warning says how many
ranked candidates TensileLite accepted:

```text
hipblaslt error: JIT configure failed for GEMM M=256 N=128 K=512 batch=1 opA=OP_N opB=OP_N A=R_16F B=R_16F C=R_16F D=R_16F compute=COMPUTE_32F epilogue=EPILOGUE_DEFAULT: Python not found at /nonexistent; set HIPBLASLT_JIT_PYTHON
hipblaslt error: JIT generate failed for GEMM M=256 N=128 K=512 ... EPILOGUE_DEFAULT: TensileLite generator: Provider exited with code 1; see /tmp/hipblaslt-jit-hG2kQx/tensilelite.log
hipblaslt warning: JIT returned 1 of 2 requested solutions for GEMM M=256 N=128 K=512 ... EPILOGUE_DEFAULT: Origami ranked 72 parameter candidates; the first 2 candidates accepted by TensileLite were compiled
```

TensileLite remains one of several independent backends. HipKittens and other
future backends sit behind the same interface; they do not route through
TensileLite. `Tensile.SingleSolution` and `Tensile.JitGemm` are the
current Python entry points; the [single-solution guide](tensilelite/SINGLE_SOLUTION.md)
describes them.

### Tool paths

A JIT build compiles the generator's tool paths into the library as defaults:
the configured `Python_EXECUTABLE`, the source tree's `tensilelite` directory,
the directory above the built rocisa extension as the import path, and
`CMAKE_CXX_COMPILER`. `HIPBLASLT_JIT_PYTHON`, `HIPBLASLT_JIT_TENSILE_SOURCE`,
`HIPBLASLT_JIT_PYTHONPATH` and `HIPBLASLT_JIT_CXX` override them. The defaults
point into the build and source trees, so an installed library without those
trees reports a configure failure that names the variable to set. The tool
paths and files are part of the TensileLite
[cache key](JIT.md#persistent-solution-library), so a process with different tools
uses a different key directory. hipBLASLt checks the tools and computes the key
once per process. Generation runs in the
[scratch directory](JIT.md#scratch-files), and with
[`HIPBLASLT_JIT_DEBUG`](JIT.md#diagnostics-with-hipblaslt_jit_debug) set, the
generator's event and timing files are in that directory too.

## Diagnostics

With this backend, the `HIPBLASLT_JIT_DEBUG` lines that
[JIT.md](JIT.md#diagnostics-with-hipblaslt_jit_debug) describes carry the
generator's own work:

- `setup` adds the tool paths as `python`, `tensile_source` and `cxx`, and the
  time to check them (`tool_check`) and to create the backend (`backend`),
  which hashes the generator sources.
- `generation` adds the generator `module` and its `exit`, and the generator's
  own stage times as `child` (`unattributed` is generator time outside its
  stages). Within `backend`, its durations include request writing, the
  generator and bundle reading.
- `progress` adds events relayed from the generator:

| `ev` | When and what |
| --- | --- |
| `child.start`, `child.exit` | The generator process starts, with its module and log, and exits, with `started`, `code`, `signal`, the events relayed and the malformed lines `dropped` |
| `child.request`, `child.stage`, `child.candidate`, `child.done` | Events relayed from the generator: what it was asked for, the start and end of each stage, each candidate it tried and its final status, with `child_pid`, its sequence number `seq` and `child_t_ms` on the clock of `t_ms`. A rejected candidate within 500 ms of the last one relayed is counted in the next one's `rejected_so_far` instead |
| `child.heartbeat` | Every 10 seconds while the generator writes no event: the seconds of silence, the events so far and the open stage |

```text
hipblaslt jit-debug {"v":1,"cat":"progress","ev":"child.candidate","pid":4242,"tid":1,"t_ms":2357.843,"q":"4242.1","gen":"4242.g1","child_pid":4250,"seq":9,"child_t_ms":2306.568,"rejected_so_far":0,"rank":0,"index":0,"of":72,"id":135,"outcome":"selected"}
```

With `timing` or `progress` on, hipBLASLt passes `--debug <categories> --debug-dir <dir>`
to `Tensile.JitGemm` or `Tensile.SingleSolution`, where `<dir>` is `jit-debug`
in the generator's working directory inside the scratch directory. The
generator appends its progress events to `events.jsonl` there as they happen
and writes its stage times to `timing.json` when it finishes. hipBLASLt reads
`events.jsonl` every 100 ms while the generator runs and once after it exits,
relays each complete line, and adds `timing.json` to the `generation` line.
Unset, empty, with `HIPBLASLT_JIT` off, or in a build without JIT, the variable
adds no generator arguments either.

When the generator is killed, the events it wrote before are relayed,
`child.exit` reports the signal, the `generation` line has `child_timing`
`"missing"` in place of `child`, `generation.end` reports `failed`, the
`generate failed` report names the log as for any failure, and the kept scratch
directory holds `jit-debug/events.jsonl`:

```text
hipblaslt jit-debug {"v":1,"cat":"progress","ev":"child.exit","pid":4242,"tid":1,"t_ms":2354.601,"q":"4242.1","gen":"4242.g1","started":true,"code":0,"signal":9,"events":13,"dropped":0,"ns":{"child":2198082498}}
hipblaslt jit-debug {"v":1,"cat":"progress","ev":"generation.end","pid":4242,"tid":1,"t_ms":2355.121,"q":"4242.1","gen":"4242.g1","outcome":"failed","generated":0,"published":0,"loaded":0,"failures":1,"ns":{"total":2199557457}}
```

## Validation

The [TensileLite JIT test guide](clients/tests/jit/README.tensilelite.md)
describes the shared driver, `.github/scripts/test_hipblaslt_jit.py`, which
runs the tests that generate with this backend on a GPU of the requested
architecture. Its heuristic routes cover what the
[JIT guide's validation](JIT.md#validation) lists, through the generator, and a
generator killed during generation. `heuristic-knowledge` checks the tuned
seeds of the device's knowledge file, split-K seeds, and the speed against the
catalog on gfx950. `heuristic-knowledge-install` checks an installed file's
lookup. The `code-object-gfx1250` and `jit-gemm-gfx1250` routes run compile-only
for gfx1250 on any host; the second generates heuristic solutions with
`Tensile.JitGemm` and builds them with comgr. `jit-gemm-knowledge-gfx942` and
`jit-gemm-knowledge-gfx1250` do the same for the tuned seeds of those
architectures' knowledge. The [benchmark guide](clients/bench/README.jit.md) describes the
prediction and benchmark checks.

The shared `hipblaslt-jit-gemm-ci.yml` workflow configures gfx90a, gfx942,
gfx950 and gfx1250 runners. A configured target is a coverage request; the
workflow's run results show which native executions completed. Numerical
results require execution on the target GPU, and cross-compilation establishes
generation and compilation only. Source support and cross-compilation do not
establish numerical results on another GPU or native Windows execution.
