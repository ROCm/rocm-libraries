# JIT implementation roadmap

This roadmap describes just-in-time (JIT) kernel generation for hipBLASLt contributors
and integration developers. It separates the current downstream implementation from
proposed work; it is not a released application programming interface (API) or support
statement. General matrix multiplication (GEMM) is the implemented operation.
Markdown follows the existing @ROCm/hipblaslt-reviewers and
@ROCm/hipblaslt-docs-reviewers rules in [.github/CODEOWNERS](../../.github/CODEOWNERS).
The [discussion document](jit-design/confluence-roadmap-draft.md) develops the design
choices and is copyable into Confluence. Release-document integration is undecided.

## Current behavior

The standalone Python builder compiles one supplied recipe into a complete bundle.
`tensilelite::getGemmAlgo` invokes that builder, loads the main kernel and helpers,
checks support and workspace, and returns an algorithm. The application passes it
to `hipblasLtMatmul` or `Gemm.initialize/run` to execute. An ordinary matmul call
does not initiate this new TensileLite JIT path. The existing rocRoller integration
has its own runtime generation path and can compile during normal library use.

The generic API layers opaque `Request`, `Backend`, and `Solution` handles above
the direct path. Both share the TensileLite runtime and algorithm registry.
The direct entry point requires an explicit recipe. With the generic TensileLite
provider, an empty `Options::configPath` requests provider-private Origami ranking.
The selector validates candidates in that order and compiles the first supported
recipe; it does not benchmark or invent a recipe when selection fails.

| Component | Current input, output, and connection |
| --- | --- |
| One-solution builder | One YAML (YAML Ain't Markup Language) recipe and target produce complete main/helper artifacts through `Tensile.SingleSolution` and existing generators, validators, and compiler tools. |
| Ranked recipe selector | Supplied candidates and problem facts produce one validated recipe or rejection reasons; `Tensile.JitGemm` calls the builder without running a model. |
| Direct API and sample | Explicit recipe and GEMM descriptors produce a checked algorithm. Sample `29_hipblaslt_jit_gemm` exercises C/C++ execution independently of the generic API. |
| Generic API and adapters | `makeGemmRequest` captures existing descriptors and host scalars; `getJitAlgo` returns an owned solution and `getGemmAlgo` adapts it to existing execution. Sample `30_hipblaslt_generic_jit_gemm` covers this flow. |
| Provider prediction and benchmark | Origami ranks matrix instructions, macro tiles/reduction depths and cache hints, and supplies all applicable workgroup-mapping/stagger outputs for the data-parallel domain. Unsupported translations reject the candidate; selection and compilation precede correctness checks and execution timing. See the [modeled-input inventory](JIT.md#origami-modeled-inputs). |

AIHPBLAS-4549 covers the builder/direct integration. AIHPBLAS-4801 is partial:
usable generic handles do not complete the reusable planning/cache protocol.
AIHPBLAS-4551 covers initial prediction and its remaining model work.

## Reuse existing operation and executable types

The generic handle names do not require parallel problem or execution frameworks.
Internal `GemmRequest` already contains `RocblasltContractionProblem`, an operation
tag, and owned alpha/beta values. Device buffers remain application-owned.
Retain this existing GEMM payload rather than introduce another public problem
type solely to accommodate a second compiler. Public `GemmProblemType` is narrower
than the full operation description.

Tensile's `Problem`, `ProblemInputs`, and `Solution` bases are generic but skeletal.
`ContractionProblemGemm` and `ContractionSolution` carry concrete GEMM semantics.
Reuse those semantics and assess extensions only for missing common concerns.
The generic `Solution` / private `CompiledSolution` still need to retain target,
backend, request, workspace, and executable-bundle lifetime; a matmul algorithm
is an adaptation token, not a general owning executable object.

Other operations require distinct payloads, support checks, argument binding,
workspace policy, and helper sequences. Share `KernelArguments`, `KernelInvocation`,
and the Heterogeneous-compute Interface for Portability (HIP) runtime's
`SolutionAdapter` where sufficient. Neither generic handles
nor a new library type alone make GEMM execution operation-independent.

## Common metadata across generators

KernelFromAnywhere (KFA), implemented as Gemm-From-Anywhere (GFA), already ingests
custom kernels as normal `ContractionSolution`s. JIT also loads normal serialized
solutions and uses existing predicates, workspace rules, preparation, and launch.
The convergence target is the same versioned KFA schema and execution semantics
across producers, consumed by the library. Metadata values need not be identical.

| TensileLite | rocRoller | Other generators, proposed |
| --- | --- | --- |
| Current generic provider; generated metadata is partial and generated/custom argument paths differ. | Existing independent runtime generation/cache/launch path; one separate assembly example uses KFA ingestion. It is not a generic JIT provider today. | No other production generic provider is implemented; external KFA examples cover specific artifacts. |
| Complete emitted metadata for supported profiles, then share the KFA consumer. | Export or normalize supported artifacts into the same contract and consumer, preserving predicates and synchronization scratch. | Implement the same producer contract and any missing operation adapter. |

Origami ranks existing solutions or prospective configurations; it is a selector,
not a generator backend. rocRoller's current route ranks `KernelType` configurations,
uses a handle-owned cache, generates/loads on a miss, and launches through
`CommandKernel`. Convergence with KFA is proposed, not already achieved by that cache
or the checked-in rocRoller assembly fixture.

The [KFA assessment](jit-design/kfa-producer-convergence.md) records producer gaps
and source evidence. Complete arguments and their application binary interface (ABI),
predicates, grid/workgroup/cluster units, local data share (LDS), ordered helpers,
workspace initialization, synchronization, and lifetime must agree before dispatch
is shared. Proceed through emission, strict validation, packing/launch/workspace/helper
equivalence, numerical validation, then migration of proven profiles and simplification.

## Alternative under assessment: fallback selection

An optional `JustInTime` library type (AIHPBLAS-4550) could call the shared generation
service after existing lookup returns **zero compatible results**. It is not implemented
or selected as a replacement for the explicit API, which preserves deterministic
backend choice, deliberate generation, and prewarming before stream capture.

Default Tensile construction orders Equality, Range, Prediction, GridBased, FreeSize,
and TruePred under hardware/operation/problem predicates; modes and available
branches affect traversal. Equality uses equality matching. Prediction uses Origami
to rank existing solutions. This ranking differs from provider recipe prediction.
Any applicable existing selector should finish before the proposed coverage fallback.

A final JIT row is insufficient: `findTopSolutions` accumulates results and could
compile merely to fill a requested top-N count. An absent root library or operation
branch, and earlier rocRoller returns, also escape a leaf. Define the complete lookup
boundary, existing reduced-precision-to-32-bit computation retry order, backend/compilation context,
workspace/capability checks, latency, errors, enumeration/index queries, and cache
compatibility before choosing placement. Zero-result fallback covers missing support;
it does not seek faster generated alternatives when a usable kernel already exists.

## Planned components

| Component / tracking | Intended connection and remaining contract |
| --- | --- |
| Planning/input protocol — AIHPBLAS-4801 | Operation, target, and specialization facts produce a reusable plan; the initial provider combines planning and compilation. Preserve modeled inputs while keeping backend tuning schemas private. |
| Persistent cache — AIHPBLAS-4552 | Plan identity and compatibility locate an existing bundle before compilation. Define target/schema/toolchain/specialization identity, grouping, invalidation, concurrent publication, and reclamation. Process-local retention and the rocRoller handle cache are separate mechanisms. |
| Exact epilogue — AIHPBLAS-4553 | Compile the requested bias/activation/output specialization; this is separate from current epilogue correctness and modeling its cost. |
| Tuning blueprints — AIHPBLAS-4554 | Combine stored choices for parameters outside the model with predicted parameters before validation. Existing defaults are not a blueprint database. |
| More operations and providers | Add concrete profiles and adapters after demonstrating their execution contracts; non-GEMM KFA support, a stable external plugin binary interface, and dynamic provider discovery remain undefined. |
| Timing/progress | Independent `HIPBLASLT_JIT_DEBUG` categories `timing`, `progress`, or `timing,progress`. Unset/empty adds no collection, observer, or files. Preserve bundle/error behavior and benchmark timing boundaries. |

AIHPBLAS-4548 is the umbrella. These mappings describe scope, not ticket closure.

## Guides and recorded evidence

The [single-solution guide](tensilelite/SINGLE_SOLUTION.md) covers recipes, bundles,
and ranked candidate validation. The [direct guide](JIT_TENSILELITE.md) and
[generic guide](JIT.md) describe implemented APIs, ownership, and execution.
The [discussion document](jit-design/confluence-roadmap-draft.md) compares designs
and lists the decisions needed before implementation.

The earlier basic and generic review stacks are closed without merging; development
continues on `users/jolabega/downstream-hipblaslt-jit-develop` with no associated pull
request. See [SESSION_HANDOFF.md](../../SESSION_HANDOFF.md) for historical revisions
and evidence. Recorded direct validation passed ten routes on native Linux gfx950;
generic validation passed twelve routes with affected failure checks rerun. The
`ScheduleIterAlg=4` gfx1250 fixture has generation/compilation evidence only. Configured
workflow targets are distinct from completed native runs; Windows execution is unverified.
The new design proposals have not been implemented or runtime-tested. rocRoller analysis
is source-based; the current local build has it disabled.
