# JIT implementation roadmap

This roadmap is source-level guidance for hipBLASLt/TensileLite contributors
and integration developers. It separates current review-stack behavior from
planned work; it is not a released API/support statement or a commitment to
ROCm release-document publication. Markdown changes follow the existing
@ROCm/hipblaslt-reviewers and @ROCm/hipblaslt-docs-reviewers rules in the
monorepo [.github/CODEOWNERS](../../.github/CODEOWNERS). Documentation changes
accompany the code/API changes they describe. The Confluence copy is a
discussion view of the versioned source; release-document integration is TBD.

## Builder and application integration

The standalone Python/CLI builder in [#12459](https://github.com/ROCm/rocm-libraries/pull/12459)
compiles a supplied recipe into a bundle. It does not call hipBLASLt or execute
GPU work. In the later direct integration [#12564](https://github.com/ROCm/rocm-libraries/pull/12564),
`tensilelite::getGemmAlgo` invokes the builder, loads the main kernel and helpers,
checks support, and returns an algorithm. The application then passes that
algorithm to `hipblasLtMatmul` or `Gemm.initialize/run`. Calling matmul alone
does not initiate compilation. The generic API is a separate layer above this
explicit-recipe flow.

Just-in-time (JIT) compilation lets an application generate GPU code for its
operation when it runs. The first reviewable stack accepts an explicit YAML
recipe through the direct TensileLite API, compiles and loads the complete
kernel/helper bundle, and executes it through existing hipBLASLt C/C++ GEMM APIs.
The generic request API is a separate stack above that working basic path.
Prediction and benchmark integration follow in their own later layer.
AIHPBLAS-4801 remains partial: separate planning/cache protocols and broader
modeled-input coverage are deferred.

## Component flow and current status

An application describes an operation and selects a configured backend. The
backend compiles a bundle containing the GPU kernels and helpers needed to run
it. An operation adapter converts that solution to the type accepted by an
execution API. For GEMM, those APIs are `hipblasLtMatmul` and the C++ `Gemm` class.

Within the TensileLite provider, an explicit YAML recipe goes to the one-solution
builder. Automatic selection adds two steps: a prediction model ranks tuning
parameters, then a selector checks candidates in that order and passes the first
valid recipe to the builder. No usable ranking or an exhausted ranking must fail
without inventing a replacement recipe. Compilation does not benchmark recipes.

| Component | Status | Input, output and connection |
| --- | --- | --- |
| One-solution builder | Implemented | One YAML recipe and target produce a complete kernel/helper bundle through `Tensile.SingleSolution` and the existing validators/compiler |
| Ranked recipe selector | Implemented | Supplied candidates and problem facts produce one validated recipe or rejection reasons; `Tensile.JitGemm` calls the builder without running a model |
| Direct TensileLite API and sample | Implemented in the basic stack | Explicit YAML and GEMM descriptors produce a checked algorithm; sample `29_hipblaslt_jit_gemm` exercises C/C++ execution |
| Generic JIT interface and TensileLite provider | Implemented | Opaque `Request` and configured `Backend` produce an owned `Solution`; provider settings stay outside the common types |
| GEMM request and execution adapters | Implemented | `makeGemmRequest` captures existing descriptors; `getGemmAlgo` connects a compiled GEMM solution to C/C++ execution |
| Generic sample | TBD in a later layer | Application buffers and descriptors pass through the generic API using an explicit recipe and checked C/C++ execution |
| Provider prediction and benchmark | TBD in a later layer | Origami ranks matrix instructions, reduction depths and cache hints; a private plan is consumed immediately by the selector/builder before benchmark checks and timing |

## Planned components

The entries below are TBD. Their purpose and connections define the next work;
they are not available behavior in the current API.

| Planned component | Input and output | Intended interaction |
| --- | --- | --- |
| Searchable `JustInTime` solution library and equality-first priority | A problem description selects existing tuned equality results before JIT is considered | The library would prefer a matching tuned result, then search compatible JIT entries before requesting another compilation |
| Separate planning and prediction-input protocol | Operation, target and specialization facts produce a structured compilation plan | The library could inspect the plan and check for existing code before invoking a backend compiler; the initial provider design combines these steps |
| Persistent code cache | A plan identity and compatibility information locate a stored bundle | A cache hit would supply the bundle to the loader; a miss would compile and then store it |
| Exact epilogue specialization | The requested bias, activation and output operations identify a compiled specialization | Compile only that epilogue, removing generic runtime activation dispatch; modeling its cost is separate prediction work |
| Tuning blueprints | Stored knowledge supplies choices that the performance model does not predict | A provider would combine those choices with predicted parameters before validation |
| Additional operation adapters and providers | An operation-specific description becomes a generic request and an executable result | Attention is a possible later operation; no Attention request factory or provider is implemented |

The backend interface is designed as a private interface compiled into the library. A stable plugin
binary interface and dynamic provider discovery are also TBD. Keeping provider
options and tuning schemas out of the public common types leaves room to add
these components without turning every request into a TensileLite recipe.

## Where to start

The [single-solution guide](tensilelite/SINGLE_SOLUTION.md) explains how to compile a supplied
recipe and inspect its complete bundle.
Its ranked-selection section describes candidate validation and rejection diagnostics.
The [direct API guide](JIT_TENSILELITE.md) covers the basic path and its sample.
The [JIT API guide](JIT.md) explains backend configuration, request ownership, GEMM adapters and execution lifetime.

## Review stack and evidence

The basic stack is [process execution (#12552)](https://github.com/ROCm/rocm-libraries/pull/12552),
[artifact loading (#12563)](https://github.com/ROCm/rocm-libraries/pull/12563),
[direct GEMM (#12564)](https://github.com/ROCm/rocm-libraries/pull/12564), and
[sample/CI (#12565)](https://github.com/ROCm/rocm-libraries/pull/12565).
The optional [generic API (#12461)](https://github.com/ROCm/rocm-libraries/pull/12461)
is based on that final basic tip. The direct API and sample remain available above it.

The basic driver passed all ten routes on native gfx950, including numerical
C/C++ execution, expected failures and disabled-JIT behavior. The gfx1250 SIA4
fixture was generated and compiled with a compatible compiler. The shared
workflow configures native gfx90a, gfx942, gfx950 and gfx1250 runners; those
configured targets are distinct from completed local evidence.

KFA standardization, performance timing policy and additional prediction work
are separate follow-ups and do not gate the basic explicit-recipe path.
