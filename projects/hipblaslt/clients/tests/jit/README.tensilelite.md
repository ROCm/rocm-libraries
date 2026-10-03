# Validate the TensileLite JIT backend

These tests generate with the
[TensileLite backend](../../../JIT_TENSILELITE.md). The
[JIT test guide](README.md) covers the tests that need no generator, their
CTest registration, and the test binaries and scripts that these routes share
with them. The shared driver, `.github/scripts/test_hipblaslt_jit.py`, runs
both kinds on a GPU of the requested architecture: the cases below generate
their bundles first, and the replaying cases then use the generated `splitk-api`
bundle instead of the committed one.

## Build and run from a checkout

Use the project's existing configured build and Python environment. The build
must provide the source hipBLASLt host library, the local rocisa extension, and
the Python dependencies needed by TensileLite. Configure the target for the GPU
on which the tests will run; a compiler target is not a substitute for that GPU.
Set `project_build`, `project_python`, `PYTHONPATH` and `LD_LIBRARY_PATH` as in
the [TensileLite build instructions](../../../JIT_TENSILELITE.md#build), then,
from the repository root:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_TESTING=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_CLIENT=ON
cmake --build "$project_build" --parallel 8 --target \
  _rocisa hipblaslt-bench hipblaslt-jit-direct-gemm-test hipblaslt-jit-generic-gemm-test \
  hipblaslt-jit-tensilelite-api-test hipblaslt-jit-generic-api-test hipblaslt-jit-api-test \
  hipblaslt-jit-mock-backend-test \
  hipblaslt-jit-component-test hipblaslt-jit-debug-test hipblaslt-jit-debug-child-test \
  hipblaslt-jit-process-test \
  hipblaslt-jit-source-bundle-test hipblaslt-jit-code-object-test hipblaslt-jit-library-test \
  hipblaslt-jit-bundle-freshness-test hipblaslt-jit-heuristic-test hipblaslt-jit-knowledge-test \
  hipblaslt_jit_knowledge
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/jit-validation"
```

`hipblaslt_jit_knowledge` writes the knowledge files of `GPU_TARGETS`.
`heuristic-knowledge-install` runs only with `--installed <prefix>`, the prefix
of a runtime-component install from a separate JIT-on build tree without a
device library:

```bash
cmake --install "$install_build" --prefix "$install_prefix" --component runtime
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/jit-install" \
  --installed "$install_prefix" --case heuristic-knowledge-install
```

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

## What each driver case checks

`source-bundle`, `jit-component`, `code-object`, `mock-backend`,
`mock-backend-library`, `jit-library` and `jit-library-concurrency` check what
the CTest tests of the same names with a `jit-` prefix check, as the
[JIT test guide](README.md#what-each-test-checks) describes. Each
`heuristic-<route>` case checks what `jit-heuristic-<route>` checks, with the
TensileLite generator: a Python wrapper that fails if it runs stands in for
the backend that fails if it generates, `heuristic-report` causes its configure
failure with a missing Python and its generation failure with `/bin/false` as
the Python, `heuristic-debug-timing` also checks the generator's own stage
times, `heuristic-debug-progress` the relayed generator events, and
`heuristic-debug-off` that the generator gets no `--debug` argument.

| Driver case | Behavior under test |
| --- | --- |
| `process-runner` | Shell-free process arguments, environment and working directory; output capture, failures and descriptor cleanup |
| `jit-debug-child` | The generator child's side of `HIPBLASLT_JIT_DEBUG`, without a GPU: the `--debug` value a child gets, which names only `timing` and `progress`, its `timing.json` read or reported missing or invalid, and its event file relayed while it runs as `child.*` lines, with malformed, out-of-order, oversized and partial lines dropped, rejected candidates coalesced, the per-child cap, a child killed mid-line, and a heartbeat after 10 s of silence |
| `code-object-gfx1250` | The hardware-free part of `code-object` for gfx1250, on any host |
| `jit-gemm-gfx1250` | Compile-only on any host: `Tensile.JitGemm` generates two ranked gfx1250 solutions from a heuristic request with the arguments hipBLASLt passes, skipping a ranked candidate that repeats an accepted kernel, and comgr assembles, compiles and links each one into a wave32 code object that uses the gfx1250 WMMA instruction |
| `jit-gemm-knowledge-gfx942`, `jit-gemm-knowledge-gfx1250` | Compile-only on any host: the C++ matcher's tuned seeds for a device of that architecture's CU count (304 and 192) travel as `tensilelite.tuned.v1` candidates. Every seed is generated, or shares an accepted seed's kernel, and comgr builds each kernel. The gfx942 route uses FP16 NT, whose seeds mostly have `GlobalSplitU=-1` |
| `heuristic-knowledge` | `HIPBLASLT_JIT=2` with the build's knowledge file: near a tuned row, several tuned seeds rank first, and the selected one keeps its parameters through derivation. Fixture files with one `GlobalSplitU=-1` MultipleBufferSingleKernel or `GlobalSplitU=4` MultipleBuffer seed pass both APIs. Without workspace, no split-K seed is ranked. With `HIPBLASLT_JIT` unset or `0`, results match and, under `strace`, the file is never opened. On gfx950, `hipblaslt-bench` is faster with knowledge than with `HIPBLASLT_JIT_KNOWLEDGE=none` at 2048³ and (512, 4096, 16384), by median of five runs with a 2% margin |
| `heuristic-knowledge-install` | Each installed knowledge file is in its architecture's directory under `lib/hipblaslt/library`, and the installed library's default lookup loads it |
| `jit-gemm-persistent-gfx942`, `jit-gemm-persistent-gfx1250` | Compile-only on any host: the catalog ranked by Origami for a device of that architecture's CU count (304 and 192) holds Hybrid Stream-K candidates under `origami.gemm.persistent.v1`. The two best are generated as Stream-K kernels with `WorkGroupMapping=0` and `WorkGroupMappingXCC=-1`, and comgr builds each |
| `heuristic-persistent` | `HIPBLASLT_JIT=2` with the catalog only, at (256, 256, 8192) where a Stream-K candidate ranks first: both APIs run it with checked numerics, its manifest marks the launch, mapping and stagger as chosen by the runtime, and it passes again with `TENSILE_PERSISTENT_HYBRID_FORCE_MODE=1` forcing dynamic assignment. Without workspace, no Stream-K candidate is ranked. gfx90a has none to rank, so the driver skips it there |
| `direct-gemm` | Direct explicit-recipe TensileLite call followed by checked C and C++ GEMM execution |
| `generic-gemm` | Backend/request/solution flow followed by checked C and C++ GEMM execution |
| `generic-api` | Shared execution, ownership and failure assertions from the direct API test, selected through the generic interface |
| `streamk-api`, `amax-api`, `splitk-api` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation |
| `alpha-zero-api` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `helper-failures` | A missing helper source or renamed helper symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `bundle-failures` | Damaged source bundles are rejected through the public API: a foreign target, an escaping symbolic link, missing sources or main assembly, an undefined main kernel, invalid assembly or helper source (the message names the comgr log), corrupt or truncated library entries, missing helper source or symbols, and unsupported problems |
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

`hipblaslt-jit-tensilelite-api-test` and `hipblaslt-jit-generic-api-test` build
`jit_api_test.cpp` generating with TensileLite, through the direct and the
generic entry point; they take the same seven arguments and then the options of
`hipblaslt-jit-api-test`. `hipblaslt-jit-process-test` checks the generator
process launcher without a GPU.

## Generating failure and heuristic routes

Without `--replay`, `test_bundle_failures.py` and `test_helper_failures.py`
take `hipblaslt-jit-tensilelite-api-test`, and a substitute generator writes
the damaged copies of the bundle. `test_bundle_failures.py` also takes
`--architecture`, the target the bundle was generated for.

`test_heuristic.py` without `--backend` runs a route with the TensileLite
generator of the build. `debug-killed-child` runs only that way: its Python
wrapper kills the generator, then itself, once kernel source generation starts.
So do `knowledge`, which takes `--knowledge <dir>` with the device's knowledge
file and optionally `--bench <hipblaslt-bench>`, and `knowledge-install`, which
takes `--installed <prefix>`.

## Compile-only routes

`test_gfx1250_jit_gemm.py <code-object-test> <request> <compiler> <fresh-output>`
runs the `jit-gemm-gfx1250` route. The driver passes
`jit_gemm_request_gfx1250.json` from `tensilelite/Tensile/Tests/unit/test_data`:
the request hipBLASLt writes for its default FP16 problem, with the Origami
ranking of the six best gfx950 candidates retargeted to gfx1250.
`hipblaslt-jit-code-object-test --bundle` also accepts a full-build bundle kept
with `--keep-build-tmp`, whose code objects are then compared with the
comgr-built ones.

`test_knowledge_jit_gemm.py <knowledge-test> <code-object-test> <request>
<compiler> <architecture> <cu-count> <build> <logic> <fresh-output>
[--transpose-b]` runs the `jit-gemm-knowledge-<architecture>` routes on the
same request. It uses `<build>/Tensile/library/<architecture>`'s knowledge
file, or extracts one from the `asm_full` directory `<logic>` when the build
has none; gfx942 takes about 2.5 minutes. `hipblaslt-jit-knowledge-test
--nearest <file> <core-key> <m> <n> <batch> <k> <cu-count>` prints the seeds,
one JSON line each.

`test_persistent_jit_gemm.py <knowledge-test> <code-object-test> <request>
<compiler> <architecture> <cu-count> <fresh-output>` runs the
`jit-gemm-persistent-<architecture>` routes on the same request.
`hipblaslt-jit-knowledge-test --predict <architecture> <cu-count> <m> <n> <k>`
prints the catalog's Origami ranking of an FP16 NN problem with workspace, one
request candidate per JSON line.

## Regenerate the committed bundles

When `jit-bundle-freshness` reports stale bundles, regenerate all of them from
the repository root with the Python interpreter of a hipBLASLt build that has
`_rocisa`:

```bash
"$project_python" projects/hipblaslt/clients/tests/jit/regenerate_bundles.py --build "$project_build"
```

It runs the generator commands that [the bundles' README](data/README.md)
lists, with `PYTHONPATH` set from `--build`, copies each bundle without symbolic
links, removes `provenance.compiler_path` from its manifest and rewrites that
README.

## Shared automation and remaining coverage

The shared `hipblaslt-jit-gemm-ci.yml` workflow builds the JIT test binaries and
the benchmark, then runs the driver: direct and generic GEMM tests, the mock
backend and Jit component checks, the JIT solution library checks, the comgr
code-object and cache checks, the heuristic routes, and the prediction/benchmark
layer. It then builds without the TensileLite backend, once without and once
with the test backend, and runs their CTest tests. The workflow obtains native
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
specialization described in the
[roadmap](../../../JIT_ROADMAP.md#roadmap).
