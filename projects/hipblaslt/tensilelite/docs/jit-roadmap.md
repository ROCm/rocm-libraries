# JIT implementation roadmap

Just-in-time (JIT) compilation lets an application request GPU code for its
operation when it runs. This roadmap connects the one-solution builder to a
generic request interface, a TensileLite provider, and hipBLASLt execution.
The table distinguishes components available at this layer from later work.
GEMM is the initial operation; the common request and backend types are intended
to support other operations without describing them as matrix multiplication.

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
| Generic JIT interface and TensileLite provider | Implemented | Opaque `Request` and configured `Backend` produce an owned `Solution`; provider settings stay outside the common types |
| GEMM request and execution adapters | Implemented | `makeGemmRequest` captures existing descriptors; `getGemmAlgo` connects a compiled GEMM solution to C/C++ execution |
| Public sample | Implemented | Application buffers and descriptors pass through the generic API using an explicit recipe and checked C/C++ execution |
| Provider prediction and benchmark | TBD in a later layer | Origami ranks matrix instructions, reduction depths and cache hints; a private plan is consumed immediately by the selector/builder before benchmark checks and timing |

## Planned components

The entries below are TBD. Their purpose and connections define the next work;
they are not available behavior in the current API.

| Planned component | Input and output | Intended interaction |
| --- | --- | --- |
| Searchable `JustInTime` solution library and equality-first priority | A problem description selects existing tuned equality results before JIT is considered | The library would prefer a matching tuned result, then search compatible JIT entries before requesting another compilation |
| Separate planning and prediction-input protocol | Operation, target and specialization facts produce a structured compilation plan | The library could inspect the plan and check for existing code before invoking a backend compiler; the initial provider design combines these steps |
| Persistent code cache | A plan identity and compatibility information locate a stored bundle | A cache hit would supply the bundle to the loader; a miss would compile and then store it |
| Exact epilogue specialization | Concrete output operations, such as bias and activation, become specialization inputs | Planning would include these operations when selecting or compiling code; the initial prediction model does not account for their cost |
| Tuning blueprints | Stored knowledge supplies choices that the performance model does not predict | A provider would combine those choices with predicted parameters before validation |
| Additional operation adapters and providers | An operation-specific description becomes a generic request and an executable result | Attention is a possible later operation; no Attention request factory or provider is implemented |

The backend interface is designed as a private interface compiled into the library. A stable plugin
binary interface and dynamic provider discovery are also TBD. Keeping provider
options and tuning schemas out of the public common types leaves room to add
these components without turning every request into a TensileLite recipe.

## Where to start

The [single-solution guide](single-solution.md) explains how to compile a supplied
recipe and inspect its complete bundle.
Its ranked-selection section describes candidate validation and rejection diagnostics.
The [JIT API guide](../../docs/jit.md) explains backend configuration, request ownership, GEMM adapters and execution lifetime.
The [public sample](../../clients/samples/29_hipblaslt_jit_gemm/README.md) shows application code that requests and runs a solution.
