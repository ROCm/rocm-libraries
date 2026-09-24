# Validate the JIT implementation

The JIT tests cover both application entry points: the direct TensileLite call
with an explicit YAML recipe and the optional generic backend/request/solution
API layered above it. Each path reaches the existing C/C++ GEMM execution APIs.
Samples `29_hipblaslt_jit_gemm` and `30_hipblaslt_generic_jit_gemm` illustrate the
direct and generic paths; the tests here exercise failures, repeated calls and
ownership changes. The [component roadmap](../../../JIT_ROADMAP.md)
distinguishes the implemented layers from planned work.

## Build and run from a checkout

Use the project's existing configured build and Python environment. The build
must provide the source hipBLASLt host library, the local rocisa extension, and
the Python dependencies needed by TensileLite. Configure the target for the GPU
on which the tests will run; a compiler target is not a substitute for that GPU.
From the repository root, with `project_build` set to the existing build:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_CLIENT=ON
cmake --build "$project_build" --parallel 8 --target \
  _rocisa hipblaslt-bench hipblaslt-jit-gemm hipblaslt-generic-jit-gemm \
  hipblaslt-jit-api-test hipblaslt-jit-generic-api-test hipblaslt-jit-backend-test \
  hipblaslt-jit-process-test hipblaslt-jit-artifacts-test
python .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output /tmp/jit-validation
```

Choose a fresh output directory. The driver checks that the shared library and
Python modules come from the checkout/build, and points device-library lookup
at an empty directory. It records commands, logs, generated bundles, and a
`summary.json`. A failed case makes the driver return a failing status.

The default run tests the disabled configuration last. It reconfigures the same
build directory with `HIPBLASLT_ENABLE_JIT=OFF`, rebuilds the disabled consumer
and benchmark, and checks their rejection diagnostics. Restore `ON` and rebuild
before continuing enabled development. To run a focused enabled check without
reconfiguration, select cases explicitly, for example `--case alternate-backend`
or `--case bench`. `--case helper-failures` and `--case bundle-failures` also
build a valid split-K test bundle before damaging its artifacts.

## What each layer checks

| Driver case | Behavior under test |
| --- | --- |
| `process-runner` | Shell-free process arguments, environment and working directory; output capture, failures and descriptor cleanup |
| `artifact-loader` | Bounded envelope parsing, truncation and malformed fields, native Unicode paths, compressed library bytes and path containment |
| `alternate-backend` | An independent HIP provider with no Tensile metadata, owned scalar values, C/C++ execution, helper preparation and bundle lifetime; a non-GEMM request passes through generic compilation |
| `sample` | Direct explicit-recipe TensileLite call followed by checked C and C++ GEMM execution |
| `generic-sample` | Backend/request/solution flow followed by checked C and C++ GEMM execution |
| `generic-api` | Shared execution, ownership and failure assertions from the direct API test, selected through the generic interface |
| `streamk-api`, `amax-api`, `splitk-api` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation |
| `alpha-zero-api` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `helper-failures` | Missing helper modules or symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `bundle-failures` | Corrupt envelopes, library identities, code objects and unsupported problems are rejected through the public API |
| `bench` | Genuine Origami ranking and first-valid selection, unchanged numerical checks, compilation outside timing, and explicit failure when ranking or validation cannot produce a recipe |
| `disabled-api` | Installed API declarations remain usable; the disabled library and benchmark report that JIT is unavailable |

The C/C++ routes use `hipblasLtMatmul` and `hipblaslt_ext::Gemm`. They are distinct
from the benchmark case, which drives the same interfaces through benchmark
problem setup and timing. A standalone generator test establishes compilation;
GPU execution is needed to establish numerical results.

## Shared automation and remaining coverage

The shared `hipblaslt-jit-gemm-ci.yml` workflow starts with the basic direct
sample/CI layer. The generic layer adds generic API and alternate-provider
checks, followed by the separate generic sample and prediction/benchmark layers. The workflow obtains native
runner labels from the shared GPU map for gfx90a, gfx942, gfx950 and gfx1250.
Missing native runners fail setup instead of silently substituting another GPU.
The SDK supplies build dependencies; project libraries are built from the source
under review.

A configured workflow is a coverage request. Consult its actual run results
and the PR's Test Result section to see which GPU executions completed. Linux
host tests of Windows argument encoding do not establish Windows process or
HIP execution coverage. Native Windows validation remains TBD.

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
