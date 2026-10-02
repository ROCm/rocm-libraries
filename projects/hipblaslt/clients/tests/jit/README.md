# Validate the JIT implementation

The JIT tests cover both internal entry points: the direct TensileLite call
with an explicit YAML recipe and the generic backend/request/solution interface
layered above it. Neither header is installed. The tests include them from
`library/src/amd_detail` and link against `libhipblaslt.so`, which still exports
the entry points. Each path reaches the existing C/C++ GEMM execution APIs. The
`direct-gemm` and `generic-gemm` binaries run each path once from start to
finish; the other cases exercise failures, repeated calls and ownership changes.
The heuristic routes cover `HIPBLASLT_JIT` through the public heuristic queries.
The [roadmap](../../../JIT.md#roadmap) distinguishes the implemented layers from
planned work.

## Build and run from a checkout

Use the project's existing configured build and Python environment. The build
must provide the source hipBLASLt host library, the local rocisa extension, and
the Python dependencies needed by TensileLite. Configure the target for the GPU
on which the tests will run; a compiler target is not a substitute for that GPU.
Set `project_build`, `project_python`, `PYTHONPATH` and `LD_LIBRARY_PATH` as in
the [JIT build instructions](../../../JIT.md#build), then, from the repository
root:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_TESTING=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_CLIENT=ON
cmake --build "$project_build" --parallel 8 --target \
  _rocisa hipblaslt-bench hipblaslt-jit-direct-gemm-test hipblaslt-jit-generic-gemm-test \
  hipblaslt-jit-api-test hipblaslt-jit-generic-api-test hipblaslt-jit-mock-backend-test \
  hipblaslt-jit-component-test hipblaslt-jit-debug-test hipblaslt-jit-process-test \
  hipblaslt-jit-artifacts-test hipblaslt-jit-code-object-test hipblaslt-jit-library-test \
  hipblaslt-jit-heuristic-test
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/jit-validation"
```

`HIPBLASLT_JIT_TESTING=ON` links the mock backend that
`hipblaslt-jit-mock-backend-test` replays bundles through. The tests that need
no generator are also CTest tests:
`ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu` runs the ones
that need no GPU, and `-L jit-gpu` runs the rest.

Choose a fresh output directory. When other work shares the host, set
`HIP_VISIBLE_DEVICES` to keep the tests on one GPU. The driver checks that the shared library and
Python modules come from the checkout/build, points device-library lookup at
an empty directory, and points `HIPBLASLT_JIT_LIBRARY_PATH` into the output
directory so that no case uses the default JIT solution library. It removes
`HIPBLASLT_JIT`, every other `HIPBLASLT_JIT_*` variable and every
`AMD_COMGR_*` variable from the inherited environment, and points
`XDG_CACHE_HOME` into the output directory, so comgr's own cache, which keeps
its default, never writes outside it. It records commands, logs,
generated bundles, and a `summary.json`. A failed case makes the driver return
a failing status.

The default run tests the disabled configuration last. It reconfigures the same
build directory with `HIPBLASLT_ENABLE_JIT=OFF`, rebuilds the disabled consumer,
the heuristic test and the benchmark, and checks that the extension API links
and that `HIPBLASLT_JIT` is ignored with one warning. Restore `ON` and rebuild
before continuing enabled development. To run a focused enabled check without
reconfiguration, select cases explicitly, for example `--case jit-component`
or `--case bench`. `--case` can be repeated; this smoke run covers both entry
points and the shared execution assertions:

```bash
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/jit-smoke" \
  --case direct-gemm --case generic-gemm --case generic-api
```

`--case helper-failures` and `--case bundle-failures` also build a valid split-K
source bundle before damaging its files, and `--case mock-backend`,
`--case mock-backend-library`, `--case jit-library`,
`--case jit-library-concurrency` and `--case code-object` build the same bundle
to replay, publish or rebuild it.

## What each layer checks

| Driver case | Behavior under test |
| --- | --- |
| `process-runner` | Shell-free process arguments, environment and working directory; output capture, failures and descriptor cleanup |
| `artifact-loader` | Source bundles read by directory convention: file ordering and roles, a missing `sources` or `library` directory, duplicate or corrupt library entries, missing main assembly, nested entries, empty or oversized files, the file-count cap, native Unicode paths, compressed library bytes, and path and symbolic-link containment |
| `jit-component` | Jit over fake stages, without a GPU: count limiting, excluded kernels, prediction only for backends that consume it, the stage of each failure, publish and load ordering, scratch lifetime, concurrent generation, and the TensileLite default seeds; with `HIPBLASLT_JIT_DEBUG=all`, the order of the generation events and the outcome and failure stage of each solution |
| `jit-debug` | The `HIPBLASLT_JIT_DEBUG` line writer, without a GPU: value parsing and its warning, JSON escaping and truncation, the line size cap, per-process file names, lines from several threads and processes intact in one file, rate limiting with aggregate lines, and the relay of a child's event file |
| `code-object` | comgr builds, loaded and run on the GPU: assembly and HIP relocatables, multi-source and mixed links, code-object versions, linker flags, target rewriting, a missing ROCm path, concurrent builds and malformed inputs, plus the `splitk-api` bundle's main kernel and 26 helpers assembled, compiled, linked into one code object, loaded and resolved |
| `code-object-gfx1250` | The hardware-free part of `code-object` for gfx1250, on any host |
| `jit-gemm-gfx1250` | Compile-only on any host: `Tensile.JitGemm` generates two ranked gfx1250 solutions from a heuristic request with the arguments hipBLASLt passes, skipping a ranked candidate that repeats an accepted kernel, and comgr assembles, compiles and links each one into a wave32 code object that uses the gfx1250 WMMA instruction |
| `mock-backend` | The in-process mock backend replaying the `splitk-api` source bundle through Jit and the comgr builder: C/C++ numerics, owned scalar values, copied algorithms outliving their owners, name lookups, 65 streams, insufficient workspace, forged tokens and indices, the wrong device, NOT_SUPPORTED for a non-GEMM request or another ProblemType, generation and build faults, and bundle lifetime |
| `mock-backend-library` | `getLibraryAlgos` publishes the mock solution into a fresh JIT solution library and returns a reserved index, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. A second process then runs that index before any lookup, and `getLibraryAlgos` finds it there with a backend that aborts the process if it generates |
| `jit-library` | The JIT solution library without a GPU: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, hash collisions, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a publisher killed after each publication step; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `direct-gemm` | Direct explicit-recipe TensileLite call followed by checked C and C++ GEMM execution |
| `generic-gemm` | Backend/request/solution flow followed by checked C and C++ GEMM execution |
| `generic-api` | Shared execution, ownership and failure assertions from the direct API test, selected through the generic interface |
| `streamk-api`, `amax-api`, `splitk-api` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation |
| `alpha-zero-api` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `helper-failures` | A missing helper source or renamed helper symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `bundle-failures` | Damaged source bundles are rejected through the public API: a foreign target, an escaping symbolic link, missing sources or main assembly, an undefined main kernel, invalid assembly or helper source (the message names the comgr log), corrupt or truncated library entries, missing helper source or symbols, and unsupported problems |
| `heuristic-fallback-c`, `heuristic-fallback-cpp` | `HIPBLASLT_JIT=1` with an empty device library: the C or C++ heuristic query returns only JIT indices for one and three requested solutions, each checked through `hipblasLtMatmul` or `Gemm`, publishes them, and reports any shortfall as a warning |
| `heuristic-forced` | `HIPBLASLT_JIT=2` with an empty and with the build's device library: both queries return only JIT indices with checked numerics |
| `heuristic-cache-hit` | A second process whose generator fails if it runs gets the first process's published index from both queries and from `hipblasLtMatmul` without an algorithm, with no JIT report; a third process with `HIPBLASLT_JIT=0` resolves that index through `getAlgosFromIndex` and runs it with checked numerics |
| `heuristic-distinct` | With one solution already published, a request for three in mode 2 returns that solution first and two new ones, three distinct kernels in all, with no JIT report |
| `heuristic-unsupported` | A problem the backend cannot rank (K=0) in modes 1 and 2: exactly one `hipblaslt error: JIT predict failed` line naming the reason across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and nothing is published |
| `heuristic-concurrent` | In modes 1 and 2, four processes of four threads each start the same query through a file barrier: every query returns the same two distinct solutions with checked numerics and no JIT report, and the library holds exactly those two entries with the allocator just past them |
| `heuristic-null-algo` | `hipblasLtMatmul` without an algorithm runs a JIT solution with checked numerics in modes 1 and 2, and does not use JIT in mode 0 |
| `heuristic-capture` | With one solution published in mode 2, `hipblasLtMatmul` without an algorithm inside a global, thread-local and relaxed capture of its stream, in modes 1 and 2, whose generator fails if it runs: for the published size the capture stays active and the instantiated graph replays twice with checked numerics; for an unpublished size the call returns the status it returns outside a capture when generation fails, the graph is empty, the capture stays active, and exactly one `hipblaslt error: JIT generation skipped during stream capture` line names the problem; nothing is published. With the build's device library, when mode 0 runs the unpublished size, mode 1 captures and replays a pre-tuned solution without a JIT report |
| `heuristic-capture-query` | In modes 1 and 2, with a fresh JIT library, the C and the C++ heuristic query each run inside a global, thread-local and relaxed capture of the GEMM stream, followed in the capture by a launch of the returned algorithm: the query returns one JIT solution and publishes it, the capture stays active, and the instantiated graph replays twice with checked numerics, with no JIT report |
| `heuristic-report` | In modes 1 and 2, a missing Python and a failing generator each print exactly one `hipblaslt error: JIT` line across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and the generator log named in the report is kept |
| `heuristic-partial-fill` | With the build's device library and one solution published in mode 2, a request for one more than the pre-tuned count in mode 1 whose generator fails if it runs returns that JIT solution once and every pre-tuned solution whose kernel differs from it, and reports the shortfall as a warning. It first queries 4096 solutions without JIT, and prints SKIP when that query fails or returns none (the build has no device library for the problem) or when it returns all 4096 (the device library leaves no shortfall) |
| `heuristic-override` | With two solutions published in mode 2, a `HIPBLASLT_TUNING_OVERRIDE_FILE` whose first line names the library's git revision and whose entry names one of them: requests for three in mode 1 return that solution first and the other JIT solution second through both queries, with checked numerics and no kernel repeated, for either solution named |
| `heuristic-provider-order` | With the build's device library in mode 1, for a size with an Equality match and for the default size, which has none: a request the Equality results fill returns the mode 0 result without consulting JIT, and `hipblasLtMatmul` without an algorithm runs the Equality solution; larger requests return the Equality results followed by JIT solutions, or only JIT solutions for the default size; a second process whose generator fails if it runs returns the same results, and its `hipblasLtMatmul` without an algorithm runs the same first solution, a JIT one for the default size; with generation failing, a request for two more returns those results followed by the next pre-tuned results whose kernels JIT does not use. It prints SKIP when no candidate size has an Equality match, the default size has one, or either returns fewer than eight pre-tuned results |
| `heuristic-debug-timing` | `HIPBLASLT_JIT_DEBUG=timing` in mode 1: one `process` and one `setup` line, no progress lines, a `generation` line whose stage times add up and include the generator's own times, one `solution` line per published solution with its HIP compile times, and `query` lines whose `from` counts add up to the returned count; the results equal a run without the variable. A second process gets cache hits and no `generation` line, mode 2 queries take every result from JIT, and of two threads that start together the one that waits names the generation it waited for |
| `heuristic-debug-progress` | `HIPBLASLT_JIT_DEBUG=progress` in mode 1: query, lookup, generation, relayed generator, build and publish events in order, with no timing lines or durations; inside a stream capture in mode 2 an unpublished size gives `capture.skip` and the unchanged report |
| `heuristic-debug-off` | `HIPBLASLT_JIT_DEBUG` unset, empty or `0`: no lines, no `--debug` argument to the generator and the same results; `0` and an unknown name each print one warning and leave the names they accompany in effect; with `HIPBLASLT_JIT=0`, any value leaves the output unchanged |
| `heuristic-debug-file` | `HIPBLASLT_JIT_DEBUG_FILE` with `%i` writes one owner-only file per process and nothing to stderr; two processes sharing one file leave every line intact; a file that cannot be opened prints one warning and the lines go to stderr |
| `heuristic-debug-killed-child` | In modes 2 and 1, a generator killed with SIGKILL once it starts kernel source generation: the relayed events up to that point, `child.exit` with signal 9, a failed `generation.end`, one `generate failed` report, nothing published, the scratch directory kept with the generator's event file, and in mode 1 the pre-tuned results |
| `bench` | `HIPBLASLT_JIT=2` through the benchmark's ordinary heuristic query: genuine Origami ranking and first-valid selection, unchanged numerical checks, compilation outside timing, publication and reuse, and one error report when ranking or validation cannot produce a recipe |
| `disabled-api` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library; the disabled benchmark and heuristic test print one warning that `HIPBLASLT_JIT` is ignored, and the heuristic results match a run without it |

The C/C++ routes use `hipblasLtMatmul` and `hipblaslt_ext::Gemm`. They are distinct
from the benchmark case, which drives the same interfaces through benchmark
problem setup and timing. A standalone generator test establishes compilation;
GPU execution is needed to establish numerical results.

## Direct and generic GEMM tests

`hipblaslt-jit-direct-gemm-test` and `hipblaslt-jit-generic-gemm-test` each
compile one supplied TensileLite recipe for
M=256, N=128, K=512, column-major NN FP16 input/output, FP32 accumulation,
alpha=1.25 and beta=0.5. They then execute the result through `hipblasLtMatmul`
and `hipblaslt_ext::Gemm`, poisoning the output before each run and comparing
every element with a CPU reference. A passing run reports both
`hipblasLtMatmul PASS` and `hipblaslt_ext::Gemm PASS` with zero maximum error.
No prebuilt device library is required.

The direct test makes one `tensilelite::getGemmAlgo` call. The generic test
follows five steps:

1. Configure the TensileLite backend with `jit::tensilelite::createBackend`.
2. Capture GEMM descriptors and host scalars with `jit::makeGemmRequest`.
3. Compile an owned solution with `jit::getJitAlgo`.
4. Adapt the solution to a GEMM algorithm with `jit::getGemmAlgo`.
5. Allocate the reported workspace and execute through the C and C++ APIs.

Both binaries take the same seven arguments. The driver supplies
`single_solution_splitk.yaml` from
`tensilelite/Tensile/Tests/unit/test_data`, or its `_gfx1250` variant on
gfx1250. To run one directly from the repository root:

```bash
tensile_source="$PWD/projects/hipblaslt/tensilelite"
"$project_build/clients/staging/hipblaslt-jit-direct-gemm-test" \
  "$project_python" "$tensile_source" "$PYTHONPATH" \
  "$tensile_source/Tensile/Tests/unit/test_data/single_solution_splitk.yaml" \
  "$(mktemp -d)/jit-direct-gemm" gfx950 "$ROCM_PATH/bin/amdclang++"
```

Use the compiler and architecture for the local device. The output path must
not exist and its parent must exist. Generator diagnostics are retained in
`<output>.log`, with process scratch files in `<output>.cwd` and generated files
under `<output>/bundle`. With `Options::configPath` empty in
`createBackend`, Jit instead asks the Origami predictor for ranked candidates
and TensileLite generates the first one its validators accept; the
[benchmark guide](../../bench/README.jit.md) covers that selection and its
limits.

## Mock backend and Jit component tests

`hipblaslt-jit-mock-backend-test` takes one argument, a source bundle directory
that `Tensile.SingleSolution --source-only` wrote; the driver passes
`<output>/splitk-api/bundle`. It creates the mock backend with
`jit::mock::createBackend` from `hipblaslt-jit-mock.hpp`, so generation runs no
Python and no subprocess, and checks the same FP16 problem as the direct and
generic tests. With `--library` after the bundle it runs the
`mock-backend-library` checks instead, and starts its second process itself.
That mode refuses to run unless `HIPBLASLT_JIT_LIBRARY_PATH` is set, so that it
never publishes into the default library.
`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU. `hipblaslt-jit-debug-test` takes a
fresh output directory too, needs no GPU and starts its child processes itself.
Both are built only with `HIPBLASLT_ENABLE_JIT=ON`.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly and
needs no GPU. It takes the `splitk-api` source bundle, whose library entry it
publishes under several kernel names, and a scratch directory for the libraries
it creates; it ignores `HIPBLASLT_JIT_LIBRARY_PATH`. Adding
`--writers N --per-writer M` runs the multi-process check instead: N writer
processes each publish M entries shared by all writers and M of their own,
while one reader process looks them up.

## Heuristic tests

`hipblaslt-jit-heuristic-test` uses only the public API, so it is built with
and without JIT. It queries `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` for an FP16 GEMM with FP32 accumulation
(M=256, N=128, K=512 by default), prints one JSON line per query with the
status, the returned solution indices, their workspace sizes and kernel names,
and then runs every returned algorithm and compares the output with a CPU
reference. `--api c|cpp|both|none`, `--requested`, `--m`, `--n`, `--k`,
`--handles`, `--queries` and `--workspace` shape the queries, `--null-algo`
also checks `hipblasLtMatmul` without an algorithm,
`--capture global|thread-local|relaxed` runs that call, and each query followed
by a launch of its first result, inside its own capture of the GEMM stream in
that mode, prints the capture status before the capture ends, the captured node
count and, for a query, the launch status, and replays the graph twice with
checked numerics,
`--from-index i,j,...` resolves indices through
`hipblaslt_ext::getAlgosFromIndex` and runs them,
`--tuned` prints whether `hipblaslt_ext::matmulIsTuned` finds an Equality
solution for the problem, `--git-revision` prints `hipblasLtGetGitRevision`,
and `--no-run` skips execution. `--threads N` runs the whole sequence in N threads,
each with its own handles; with `--barrier DIR` each thread claims a
`ready-<n>` file in `DIR` after creating its first handle and waits for
`DIR/go` before its first query, so that threads in several processes start
together.

`test_heuristic.py <binary> <route> <fresh-output>` runs the binary for one
driver route in fresh processes, each with its own JIT solution library,
temporary and cache directories under the output directory, and an empty
`HIPBLASLT_TENSILE_LIBPATH` unless the route uses the build's device library.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names a fresh results directory, and either `--target`
selects a compile-only run for that target ID or `--gpu` also loads and runs
the results on device 0, which must match the target. `--ffm` runs the GPU part
on the simulator that `HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select. Simulator
runs are manual: neither the shared driver nor CI runs `--ffm`.
`--bundle` adds the checks for a TensileLite source bundle, or for a full-build
bundle kept with `--keep-build-tmp`, whose code objects are then compared with
the comgr-built ones. `--only` selects tests by name.

`test_gfx1250_jit_gemm.py <code-object-test> <request> <compiler> <fresh-output>`
runs the `jit-gemm-gfx1250` route. The driver passes
`jit_gemm_request_gfx1250.json` from `tensilelite/Tensile/Tests/unit/test_data`:
the request hipBLASLt writes for its default FP16 problem, with the Origami
ranking of the six best gfx950 candidates retargeted to gfx1250.

## Algorithm error-status tests

`hipblaslt-test` includes the `AlgoErrors.smoke_*` tests in
`clients/tests/src/algo_errors_gtest.cpp`. They check the error statuses for a
solution index that names no solution, which a reserved JIT index that no JIT
library holds also gets, with a 128×128×128 FP16 GEMM and index 2^30 − 1, the
last index below the reserved range:

- `Gemm::initialize` and `GroupedGemm::initialize` with an index that names no
  solution return `HIPBLAS_STATUS_INVALID_VALUE`.
- `hipblasLtMatmul` with that index does not succeed; it returns
  `HIPBLAS_STATUS_INTERNAL_ERROR`.
- `hipblasLtMatmulAlgoGetHeuristic` with `requestedAlgoCount` 0 returns
  `HIPBLAS_STATUS_INVALID_VALUE` and sets `*returnAlgoCount` to 0.

Build the `hipblaslt-test` target and run it with
`--gtest_filter='AlgoErrors.*'` and `HIPBLASLT_JIT` unset; the shared driver
does not run these tests. The index checks are reached only when the build's
device library loads. Without one the calls fail earlier, and the tests still
pass.

## Shared automation and remaining coverage

The shared `hipblaslt-jit-gemm-ci.yml` workflow builds the JIT test binaries and
the benchmark, then runs the driver: direct and generic GEMM tests, the mock
backend and Jit component checks, the JIT solution library checks, the comgr
code-object and cache checks, the heuristic routes, and the prediction/benchmark
layer. The workflow obtains native
runner labels from the shared GPU map for gfx90a, gfx942, gfx950 and gfx1250.
Missing native runners fail setup instead of silently substituting another GPU.
The SDK supplies build dependencies; project libraries are built from the source
under review.

A configured workflow is a coverage request; its run results show which GPU
executions completed. Linux host tests of Windows argument encoding do not
establish Windows process or HIP execution coverage, and the workflow does not
run native Windows jobs.

The direct entry point always requires an explicit recipe. Empty-recipe
prediction is available only through the generic TensileLite backend.

The current predictor has deliberate gaps. On gfx950, the MX cases verify that
ranked recipes lacking required subtile choices fail with candidate-specific
reasons. Complex, zero-K, alpha-zero, and mixed MAC-type cases verify the absence
of a usable ranking and the absence of generation. These are diagnostic checks,
not successful numerical executions. Explicit supported recipes have their own
builder/runtime coverage. None of these tests establishes the activation
specialization or tuning-blueprint behavior described in the roadmap.
