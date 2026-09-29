# Validate the JIT implementation

The JIT tests cover both internal entry points: the direct TensileLite call
with an explicit YAML recipe and the generic backend/request/solution interface
layered above it. Neither header is installed. The tests include them from
`library/src/amd_detail` and link against `libhipblaslt.so`, which still exports
the entry points. Each path reaches the existing C/C++ GEMM execution APIs. The
`direct-gemm` and `generic-gemm` binaries run each path once from start to
finish; the other cases exercise failures, repeated calls and ownership changes.
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
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_CLIENT=ON
cmake --build "$project_build" --parallel 8 --target \
  _rocisa hipblaslt-bench hipblaslt-jit-direct-gemm-test hipblaslt-jit-generic-gemm-test \
  hipblaslt-jit-api-test hipblaslt-jit-generic-api-test hipblaslt-jit-mock-backend-test \
  hipblaslt-jit-component-test hipblaslt-jit-process-test hipblaslt-jit-artifacts-test \
  hipblaslt-jit-code-object-test
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/jit-validation"
```

Choose a fresh output directory. When other work shares the host, set
`HIP_VISIBLE_DEVICES` to keep the tests on one GPU. The driver checks that the shared library and
Python modules come from the checkout/build, and points device-library lookup
at an empty directory. It records commands, logs, generated bundles, and a
`summary.json`. A failed case makes the driver return a failing status.

The default run tests the disabled configuration last. It reconfigures the same
build directory with `HIPBLASLT_ENABLE_JIT=OFF`, rebuilds the disabled consumer
and benchmark, and checks their rejection diagnostics. Restore `ON` and rebuild
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
source bundle before damaging its files, and `--case mock-backend` and
`--case code-object` build the same bundle to replay or rebuild it.

## What each layer checks

| Driver case | Behavior under test |
| --- | --- |
| `process-runner` | Shell-free process arguments, environment and working directory; output capture, failures and descriptor cleanup |
| `artifact-loader` | Source bundles read by directory convention: file ordering and roles, a missing `sources` or `library` directory, duplicate or corrupt library entries, missing main assembly, nested entries, empty or oversized files, the file-count cap, native Unicode paths, compressed library bytes, and path and symbolic-link containment |
| `jit-component` | Jit over fake stages, without a GPU: count limiting, excluded kernels, prediction only for backends that consume it, the stage of each failure, publish and load ordering, scratch lifetime, concurrent generation, and the TensileLite default seeds |
| `code-object` | comgr builds, loaded and run on the GPU: assembly and HIP relocatables, multi-source and mixed links, code-object versions, linker flags, target rewriting, a missing ROCm path, concurrent builds, malformed inputs and the comgr cache policy, plus the `splitk-api` bundle's main kernel and 26 helpers assembled, compiled, linked into one code object, loaded and resolved |
| `code-object-gfx1250` | The hardware-free part of `code-object` for gfx1250, on any host |
| `comgr-cache` | In fresh processes with private cache directories: `HIPBLASLT_JIT=1` leaves no comgr cache, `AMD_COMGR_CACHE=1` creates one, which shows the check can detect it, and a user-set `AMD_COMGR_CACHE` is kept under `HIPBLASLT_JIT=1` |
| `mock-backend` | The in-process mock backend replaying the `splitk-api` source bundle through Jit and the comgr builder: C/C++ numerics, owned scalar values, copied algorithms outliving their owners, name lookups, 65 streams, insufficient workspace, forged tokens and indices, the wrong device, NOT_SUPPORTED for a non-GEMM request or another ProblemType, generation and build faults, and bundle lifetime |
| `direct-gemm` | Direct explicit-recipe TensileLite call followed by checked C and C++ GEMM execution |
| `generic-gemm` | Backend/request/solution flow followed by checked C and C++ GEMM execution |
| `generic-api` | Shared execution, ownership and failure assertions from the direct API test, selected through the generic interface |
| `streamk-api`, `amax-api`, `splitk-api` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation |
| `alpha-zero-api` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `helper-failures` | A missing helper source or renamed helper symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `bundle-failures` | Damaged source bundles are rejected through the public API: a foreign target, an escaping symbolic link, missing sources or main assembly, an undefined main kernel, invalid assembly or helper source (the message names the comgr log), corrupt or truncated library entries, missing helper source or symbols, and unsupported problems |
| `bench` | Genuine Origami ranking and first-valid selection, unchanged numerical checks, compilation outside timing, and explicit failure when ranking or validation cannot produce a recipe |
| `disabled-api` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library; the disabled benchmark reports that JIT is unavailable |

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
generic tests.
`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names a fresh results directory, and either `--target`
selects a compile-only run for that target ID or `--gpu` also loads and runs
the results on device 0, which must match the target. `--ffm` runs the GPU part
on the simulator that `HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select.
`--bundle` adds the checks for a TensileLite source bundle, or for a full-build
bundle kept with `--keep-build-tmp`, whose code objects are then compared with
the comgr-built ones. `--expect-comgr-cache present|absent` adds the comgr cache
check that the `comgr-cache` route runs; that check expects `XDG_CACHE_HOME`
and `HOME` to name existing scratch directories. `--only` selects tests by name.

## Shared automation and remaining coverage

The shared `hipblaslt-jit-gemm-ci.yml` workflow builds the JIT test binaries and
the benchmark, then runs the driver: direct and generic GEMM tests, the mock
backend and Jit component checks, the comgr code-object and cache checks, and
the prediction/benchmark layer. The workflow obtains native
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
builder/runtime coverage. None of these tests establishes the future searchable
JIT library, persistent cache, exact epilogue specialization, or tuning-blueprint
behavior described in the roadmap.
