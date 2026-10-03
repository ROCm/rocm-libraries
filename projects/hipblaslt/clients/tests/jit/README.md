# Validate the JIT implementation

The JIT tests check the Jit stages, the `HIPBLASLT_JIT_DEBUG` lines, the comgr
code-object builder, the source-bundle reader, the JIT solution library, the
generic entry point through the existing C/C++ GEMM execution APIs, and
`HIPBLASLT_JIT` through the public heuristic queries. They need no generator:
the mock backend replays the gfx950 source bundles committed in
[`data`](data/README.md). The JIT headers are not installed. The tests include
them from `library/src/amd_detail` and link against `libhipblaslt.so`, which
exports the entry points.

## Build and run from a checkout

Configure the target for the GPU on which the tests will run; a compiler target
is not a substitute for that GPU. The Python tests run with the interpreter
CMake finds, which needs `msgpack`; set `project_python` to such an
interpreter, and `project_build` and `LD_LIBRARY_PATH` as in the
[JIT build instructions](../../../JIT.md#build), then, from the repository root:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_TESTING=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
ctest --test-dir "$project_build/clients/tests/jit" -L jit-gpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU and `-L jit-gpu` the rest. When
other work shares the host, set `HIP_VISIBLE_DEVICES` to keep the tests on one
GPU. Each test writes under `clients/tests/jit/scratch` in the build directory,
which CTest empties before the tests run.

`HIPBLASLT_JIT_TESTING=ON` links the mock backend that
`hipblaslt-jit-mock-backend-test` replays bundles through. In a build without a
generator backend it is also the process's JIT backend: heuristic queries rank
candidates with Origami and replay the bundles `HIPBLASLT_JIT_TEST_REPLAY` lists,
separated by `:` (`;` on Windows). `HIPBLASLT_JIT_TEST_FAULT` set to `generate`,
`build`, `record` or `trap` injects the mock's fault, and `record` appends each
request to the file `HIPBLASLT_JIT_TEST_RECORD` names. In any build with
`HIPBLASLT_JIT_TESTING=ON`, `HIPBLASLT_JIT_TEST_BACKENDS` replaces the build's
backends with mock backends, without a predictor, for heuristic queries. It
lists them separated by `;`, each as `id[+flag...]=bundle[,bundle...]`; the id
also names the backend in `HIPBLASLT_JIT_BACKENDS` and in reports. `optin`
makes it opt-in, `unavailable` fails its configuration, and `unsupported`,
`generate` and `trap` make it reject every problem, fail generation or abort
the process if it generates. The CTest tests are:

- `jit-cpu`: `jit-source-bundle`, `jit-component`, `jit-debug` and
  `jit-code-object`, and with `HIPBLASLT_ENABLE_YAML=OFF` also `jit-knowledge`,
  `jit-library`, `jit-library-concurrency` and `jit-bundle-freshness`. The last fails when
  the committed bundles no longer match the generator;
  [their README](data/README.md) says how to regenerate them. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-code-object-gpu`, and with `HIPBLASLT_JIT_TESTING=ON` in a
  build for gfx950 also `jit-mock-backend`, `jit-mock-backend-library`,
  `jit-bundle-failures`, `jit-helper-failures` and `jit-api-splitk`,
  `jit-api-streamk`, `jit-api-amax` and `jit-api-alpha-zero`, and
  `jit-heuristic-multi-<route>` for each multi-backend route in the table
  below. In a build without a generator backend it also has
  `jit-heuristic-<route>` for each other heuristic route in the table, run
  through the test backend replaying the `rank-1`, `rank-2` and `splitk`
  bundles.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | Source bundles read by directory convention: file ordering and roles, a missing `sources` or `library` directory, duplicate or corrupt library entries, missing main assembly, nested entries, empty or oversized files, the file-count cap, native Unicode paths, compressed library bytes, and path and symbolic-link containment |
| `jit-component` | Jit over fake stages, without a GPU: count limiting, excluded kernels, prediction only for backends that consume it, the stage of each failure, publish and load ordering, scratch lifetime, concurrent generation, and the catalog knowledge seeds; with `HIPBLASLT_JIT_DEBUG=all`, the order of the generation events, the outcome and failure stage of each solution, and the `prediction` line of a ranked and a failed prediction |
| `jit-knowledge` | The tuning knowledge files, without a GPU, on fixture files it writes: nearest-set order, ties, the cap and at most two sets per tile shape; the ProblemType with the fewest extra epilogue features; branch order by PCI chip ID, CU count, fallback chip and generic, falling through a branch without the ProblemType; only the requested group's block inflated, and a corrupt block disabling only its group; the lookup per architecture and in a flat directory, with one `knowledge` debug line for a loaded file, a wrong architecture or schema, a missing file and `HIPBLASLT_JIT_KNOWLEDGE=none`; and a version that changes with the content. On modeled gfx942, gfx950 and gfx1250 devices, Origami ranks the catalog's data-parallel and Hybrid Stream-K candidates together: a Stream-K launch splits the grid over the output tiles, its hybrid mode is dynamic only on gfx950, an equal-latency data-parallel candidate comes first, and without workspace or with an auxiliary output only data-parallel candidates remain. `--decode <file>` reports the first-use time of a file's largest group |
| `jit-debug` | The `HIPBLASLT_JIT_DEBUG` line writer, without a GPU: value parsing and its warning, JSON escaping and truncation, the line size cap, per-process file names, lines from several threads and processes intact in one file, and rate limiting with aggregate lines |
| `jit-code-object`, `jit-code-object-gpu` | comgr builds, compile-only for gfx950 or loaded and run on the GPU: assembly and HIP relocatables, multi-source and mixed links, code-object versions, linker flags, target rewriting, a missing ROCm path, concurrent builds and malformed inputs, plus the `splitk` bundle's main kernel and 26 helpers assembled, compiled, linked into one code object, and on the GPU loaded and resolved |
| `jit-mock-backend` | The in-process mock backend replaying the `splitk` source bundle through Jit and the comgr builder: C/C++ numerics, owned scalar values, copied algorithms outliving their owners, name lookups, 65 streams, insufficient workspace, forged tokens and indices, the wrong device, NOT_SUPPORTED for a non-GEMM request or another ProblemType, generation, build and record faults, a replay after an Origami prediction, rejected mock options, and bundle lifetime |
| `jit-mock-backend-library` | `getLibraryAlgos` publishes the mock solution into a fresh JIT solution library and returns a reserved index, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. A query for two solutions generates only for the shortfall and skips the published kernel. A second process then runs that index before any lookup, and `getLibraryAlgos` finds it there with a backend that aborts the process if it generates |
| `jit-library` | The JIT solution library without a GPU: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, hash collisions, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a publisher killed after each publication step; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `jit-bundle-freshness` | Each committed bundle's manifest records the kernel-argument and persistent-loop layout versions in `GlobalParameters.py` and the builder's code-object version; its library loads in the host library and names its main kernel, and comgr builds it for its target. A copy with either layout version changed is reported stale |
| `jit-api-splitk`, `jit-api-streamk`, `jit-api-amax` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation, on the replayed bundle of that name |
| `jit-api-alpha-zero` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `jit-helper-failures` | A missing helper source or renamed helper symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `jit-bundle-failures` | Damaged source bundles are rejected through the public API: a foreign target, an escaping symbolic link, missing sources or main assembly, an undefined main kernel, invalid assembly or helper source (the message names the comgr log), corrupt or truncated library entries, missing helper source or symbols, and unsupported problems |
| `jit-heuristic-fallback-c`, `jit-heuristic-fallback-cpp` | `HIPBLASLT_JIT=1` with an empty device library: the C or C++ heuristic query returns only JIT indices for one and three requested solutions, each checked through `hipblasLtMatmul` or `Gemm`, publishes them, and reports any shortfall as a warning |
| `jit-heuristic-forced` | `HIPBLASLT_JIT=2` with an empty and with the build's device library: both queries return only JIT indices with checked numerics |
| `jit-heuristic-cache-hit` | A second process whose backend fails if it generates gets the first process's published index from both queries and from `hipblasLtMatmul` without an algorithm, with no JIT report; a third process with `HIPBLASLT_JIT=0` resolves that index through `getAlgosFromIndex` and runs it with checked numerics |
| `jit-heuristic-distinct` | With one solution already published, a request for three in mode 2 returns that solution first and two new ones, three distinct kernels in all, with no JIT report |
| `jit-heuristic-unsupported` | A problem the predictor cannot rank (K=0) in modes 1 and 2: exactly one `hipblaslt error: JIT predict failed` line naming the reason across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and nothing is published |
| `jit-heuristic-concurrent` | In modes 1 and 2, four processes of four threads each start the same query through a file barrier: every query returns the same two distinct solutions with checked numerics and no JIT report, and the library holds exactly those two entries with the allocator just past them |
| `jit-heuristic-null-algo` | `hipblasLtMatmul` without an algorithm runs a JIT solution with checked numerics in modes 1 and 2, and does not use JIT in mode 0 |
| `jit-heuristic-capture` | With one solution published in mode 2, `hipblasLtMatmul` without an algorithm inside a global, thread-local and relaxed capture of its stream, in modes 1 and 2, whose backend fails if it generates: for the published size the capture stays active and the instantiated graph replays twice with checked numerics; for an unpublished size the call returns the status it returns outside a capture when generation fails, the graph is empty, the capture stays active, and exactly one `hipblaslt error: JIT generation skipped during stream capture` line names the problem; nothing is published. With the build's device library, when mode 0 runs the unpublished size, mode 1 captures and replays a pre-tuned solution without a JIT report |
| `jit-heuristic-capture-query` | In modes 1 and 2, with a fresh JIT library, the C and the C++ heuristic query each run inside a global, thread-local and relaxed capture of the GEMM stream, followed in the capture by a launch of the returned algorithm: the query returns one JIT solution and publishes it, the capture stays active, and the instantiated graph replays twice with checked numerics, with no JIT report |
| `jit-heuristic-report` | In modes 1 and 2, a backend that fails to configure and one that fails to generate each print exactly one `hipblaslt error: JIT` line across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and the log named in the report is kept |
| `jit-heuristic-partial-fill` | With the build's device library and one solution published in mode 2, a request for one more than the pre-tuned count in mode 1 whose backend fails if it generates returns that JIT solution once and every pre-tuned solution whose kernel differs from it, and reports the shortfall as a warning. It first queries 4096 solutions without JIT, and prints SKIP when that query fails or returns none (the build has no device library for the problem) or when it returns all 4096 (the device library leaves no shortfall) |
| `jit-heuristic-override` | With two solutions published in mode 2, a `HIPBLASLT_TUNING_OVERRIDE_FILE` whose first line names the library's git revision and whose entry names one of them: requests for three in mode 1 return that solution first and the other JIT solution second through both queries, with checked numerics and no kernel repeated, for either solution named |
| `jit-heuristic-provider-order` | With the build's device library in mode 1, for a size with an Equality match and for the default size, which has none: a request the Equality results fill returns the mode 0 result without consulting JIT, and `hipblasLtMatmul` without an algorithm runs the Equality solution; larger requests return the Equality results followed by JIT solutions, or only JIT solutions for the default size; a second process whose backend fails if it generates returns the same results, and its `hipblasLtMatmul` without an algorithm runs the same first solution, a JIT one for the default size; with generation failing, a request for two more returns those results followed by the next pre-tuned results whose kernels JIT does not use. It prints SKIP when no candidate size has an Equality match, the default size has one, or either returns fewer than eight pre-tuned results |
| `jit-heuristic-debug-timing` | `HIPBLASLT_JIT_DEBUG=timing` in mode 1: one `process` and one `setup` line, no progress lines, a `generation` line whose stage times add up, one `solution` line per published solution with its HIP compile times, and `query` lines whose `from` counts add up to the returned count; the results equal a run without the variable. A second process gets cache hits and no `generation` line, mode 2 queries take every result from JIT, and of two threads that start together the one that waits names the generation it waited for |
| `jit-heuristic-debug-progress` | `HIPBLASLT_JIT_DEBUG=progress` in mode 1: query, lookup, generation, build and publish events in order, with no timing lines or durations; inside a stream capture in mode 2 an unpublished size gives `capture.skip` and the unchanged report |
| `jit-heuristic-debug-off` | `HIPBLASLT_JIT_DEBUG` unset, empty or `0`: no lines and the same results; `0` and an unknown name each print one warning and leave the names they accompany in effect; with `HIPBLASLT_JIT=0`, any value leaves the output unchanged |
| `jit-heuristic-debug-file` | `HIPBLASLT_JIT_DEBUG_FILE` with `%i` writes one owner-only file per process and nothing to stderr; two processes sharing one file leave every line intact; a file that cannot be opened prints one warning and the lines go to stderr |
| `jit-heuristic-multi-both` | Two mock backends, A replaying `rank-1` and `rank-2` and B replaying `splitk`, in mode 2: a request for four returns A's two solutions, then B's, with one shortfall warning that lists both backends' counts; each backend publishes under its own key directory, and a second process whose backends abort if they generate returns the same indices with no report |
| `jit-heuristic-multi-order` | `HIPBLASLT_JIT_BACKENDS` reverses the order, selects one backend, and ignores an identifier the build lacks with exactly one warning |
| `jit-heuristic-multi-optin` | An opt-in backend serves only when `HIPBLASLT_JIT_BACKENDS` names it, and an unnamed one is not configured, so its configuration failure is not reported |
| `jit-heuristic-multi-unavailable` | A backend whose configuration fails prints one configure warning naming it, and the other serves |
| `jit-heuristic-multi-count` | A request for one returns only the first backend's solution, a request for two one of each; when the first fails to generate, the second still serves and the one failure warning names the first |
| `jit-heuristic-multi-domain` | A backend that rejects the problem is skipped without a report; when both reject it, exactly one error says that no enabled backend supports the problem |
| `jit-heuristic-multi-exclude` | Two backends replaying the same bundle return its kernel once |
| `jit-heuristic-multi-fellshort` | Over two queries, a backend that fell short does not generate again and the next backend still serves, with no report |
| `jit-heuristic-multi-capture` | With only the second backend's solution published, `hipblasLtMatmul` without an algorithm inside a global capture, whose backends abort if they generate, runs that solution and the graph replays with checked numerics and no report |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |

The C/C++ routes use `hipblasLtMatmul` and `hipblaslt_ext::Gemm`. Compilation
alone does not establish numerical results; those need GPU execution.

## Mock backend and Jit component tests

`hipblaslt-jit-mock-backend-test` takes one argument, a source bundle
directory; CTest passes `data/gfx950/splitk`. It creates the mock backend with
`jit::mock::createBackend` from `hipblaslt-jit-mock.hpp`, so generation runs no
Python and no subprocess, and checks an FP16 problem with M=256, N=128, K=512,
the record fault, a replay after an Origami prediction and rejected mock
options. With `--library` after the bundle it runs the
`jit-mock-backend-library` checks instead, and starts its second process
itself. That mode refuses to run unless `HIPBLASLT_JIT_LIBRARY_PATH` is set, so
that it never publishes into the default library.
`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU. `hipblaslt-jit-debug-test` takes a
fresh output directory too, needs no GPU and starts its child processes itself.
Both are built only with `HIPBLASLT_ENABLE_JIT=ON`.

`hipblaslt-jit-api-test --replay BUNDLE` runs the public execution checks on a
solution the mock backend replays from `BUNDLE`; `--m`, `--n`, `--k`,
`--amax`, `--alpha-zero` and `--workspace-fallback` shape the problem.
`test_bundle_failures.py` and `test_helper_failures.py` take that binary, a
valid split-K source bundle, a fresh output directory and `--replay`; they
damage copies of the bundle and replay them.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly and
needs no GPU. It takes a split-K source bundle, `data/gfx950/splitk` in CTest,
whose library entry it publishes under several kernel names, and a scratch
directory for the libraries it creates; it ignores
`HIPBLASLT_JIT_LIBRARY_PATH`. Adding `--writers N --per-writer M` runs the
multi-process check instead: N writer processes each publish M entries shared
by all writers and M of their own, while one reader process looks them up.
`hipblaslt-jit-bundle-freshness-test` takes the `data` directory and a scratch
directory.

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

`test_heuristic.py <binary> <route> <fresh-output> --backend test --replay BUNDLE`
runs the binary for one route in fresh processes, each with its own JIT
solution library, temporary and cache directories under the output directory,
and an empty `HIPBLASLT_TENSILE_LIBPATH` unless the route uses the build's
device library. `--replay` is repeated once per bundle. The build's JIT backend
must be the test backend: the routes replay those bundles, and its record fault
stands in for a backend that fails if it generates. In a build without a device
library, `partial-fill` and `provider-order` print SKIP, and `capture` skips its
pre-tuned part. The `multi-*` routes take neither `--backend` nor `--replay`:
they set `HIPBLASLT_JIT_TEST_BACKENDS` to the committed bundles, so they run in
any build with `HIPBLASLT_JIT_TESTING=ON`.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names a fresh results directory, and either `--target`
selects a compile-only run for that target ID or `--gpu` also loads and runs
the results on device 0, which must match the target. `--ffm` runs the GPU part
on the simulator that `HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select. Simulator
runs are manual; CTest does not run `--ffm`. `--bundle` adds the checks for a
source bundle. `--only` selects tests by name.

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
`--gtest_filter='AlgoErrors.*'` and `HIPBLASLT_JIT` unset; CTest does not run
these tests. The index checks are reached only when the build's device library
loads. Without one the calls fail earlier, and the tests still pass.
