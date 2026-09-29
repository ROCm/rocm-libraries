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
  hipblaslt-jit-component-test hipblaslt-jit-process-test hipblaslt-jit-artifacts-test
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
test bundle before damaging its artifacts, and `--case mock-backend` builds the
same bundle to replay it.

## What each layer checks

| Driver case | Behavior under test |
| --- | --- |
| `process-runner` | Shell-free process arguments, environment and working directory; output capture, failures and descriptor cleanup |
| `artifact-loader` | Bounded envelope parsing, truncation and malformed fields, native Unicode paths, compressed library bytes and path containment |
| `jit-component` | Jit over fake stages, without a GPU: count limiting, excluded kernels, prediction only for backends that consume it, the stage of each failure, publish and load ordering, scratch lifetime, concurrent generation, and the TensileLite default seeds |
| `mock-backend` | The in-process mock backend replaying the `splitk-api` bundle through Jit: C/C++ numerics, owned scalar values, copied algorithms outliving their owners, name lookups, 65 streams, insufficient workspace, forged tokens and indices, the wrong device, NOT_SUPPORTED for a non-GEMM request or another ProblemType, generation and build faults, and bundle lifetime |
| `direct-gemm` | Direct explicit-recipe TensileLite call followed by checked C and C++ GEMM execution |
| `generic-gemm` | Backend/request/solution flow followed by checked C and C++ GEMM execution |
| `generic-api` | Shared execution, ownership and failure assertions from the direct API test, selected through the generic interface |
| `streamk-api`, `amax-api`, `splitk-api` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation |
| `alpha-zero-api` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `helper-failures` | Missing helper modules or symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `bundle-failures` | Corrupt envelopes, library identities, code objects and unsupported problems are rejected through the public API |
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
and TensileLite compiles the first one its validators accept; the
[benchmark guide](../../bench/README.jit.md) covers that selection and its
limits.

## Mock backend and Jit component tests

`hipblaslt-jit-mock-backend-test` takes one argument, a bundle directory that
`Tensile.SingleSolution` wrote; the driver passes `<output>/splitk-api/bundle`.
It creates the mock backend with `jit::mock::createBackend` from
`hipblaslt-jit-mock.hpp`, so generation runs no Python and no subprocess, and
checks the same FP16 problem as the direct and generic tests.
`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU.

## Shared automation and remaining coverage

The shared `hipblaslt-jit-gemm-ci.yml` workflow builds the JIT test binaries and
the benchmark, then runs the driver: direct and generic GEMM tests, the mock
backend and Jit component checks, and the prediction/benchmark layer. The workflow obtains native
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
