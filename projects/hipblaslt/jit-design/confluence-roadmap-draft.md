# hipBLASLt JIT: direct TensileLite workflow and roadmap

> Preserved discussion draft from the upstream review phase. Development now
> continues on `users/jolabega/downstream-hipblaslt-jit-develop`; PR links and
> review-state statements below identify the historical source snapshot. This
> page has not been published to Confluence.

**Review basis:** Published basic tip #12565 (`b2510705`) and dependent generic/prediction/documentation tip #12430 (`d742375d`), verified September 24, 2026. Their executable and test source matches the validated implementation; subsequent changes relocate and clarify contributor documentation. “Implemented” below means present in this review stack, not merged, released or approved as product naming.

This page is for hipBLASLt and TensileLite contributors and integration developers. It is the discussion copy of the versioned contributor roadmap and separates current behavior from planned work. Source Markdown follows the existing `@ROCm/hipblaslt-reviewers` and `@ROCm/hipblaslt-docs-reviewers` CODEOWNERS rules. These contributor files sit outside the ROCm release-documentation source tree; this page makes no release publication or support commitment.

## Purpose and delivery scope

The first delivery is a direct hipBLASLt-to-TensileLite path: an application supplies one YAML recipe, hipBLASLt compiles and loads the resulting solution, and the application executes it through the existing C or C++ GEMM interface. This is the basic AIHPBLAS-4549 integration.

A solution can contain a main kernel and several helpers. Successful compilation alone does not establish support for a requested GEMM: the runtime also checks the device, problem, workspace and required executable symbols.

A separate stack adds generic request/backend/solution interfaces under AIHPBLAS-4801 and initial provider prediction under AIHPBLAS-4551. It extends the basic implementation and preserves the direct API and sample. Persistent caching, KFA metadata convergence, broader prediction coverage and timing/progress diagnostics remain follow-up work; none is required to run the explicit-recipe path.

## Direct workflow available in the basic stack

```text
GEMM descriptors, scalars and buffers + explicit YAML recipe
                              |
              tensilelite::getGemmAlgo
                              |
              Tensile.SingleSolution
                  compile one solution
                              |
       Load serialized solution + main/helper code objects
                              |
              Check device/problem/workspace
                              |
              hipblasLtMatmulHeuristicResult_t
                         /             \
             hipblasLtMatmul       Gemm.initialize / run
```

Include `<hipblaslt/hipblaslt-jit-tensilelite.hpp>` and use `hipblaslt_ext::experimental::jit::tensilelite::getGemmAlgo`. The call takes the usual GEMM descriptors, host scalars and buffers, `tensilelite::Options`, and a workspace limit. It returns the algorithm and required workspace in a `hipblasLtMatmulHeuristicResult_t`.

`Options::configPath` is required and names an explicit recipe. Generation options also identify the Python interpreter/import paths, compiler, fresh output directory and target. An empty architecture means the current device; an explicitly requested target must match that device. The direct entry point does not interpret a missing recipe as a request for prediction.

Compilation is synchronous and invokes `Tensile.SingleSolution` without benchmarking. The loader uses the generated serialized solution and every listed code object. Support checks reuse Tensile's predicates and workspace rules; preparation constructs the ordered launches and resolves required main/helper symbols before submission.

The standalone Python/CLI builder only compiles artifacts; it does not load or execute GPU work. The direct host call performs generation, loading and validation. The application then submits the returned algorithm to the existing execution API. Calling `hipblasLtMatmul` alone does not select a recipe, compile code or request prediction.

`Tensile --build-only` already generates and compiles kernels without benchmarking, including a YAML containing one recipe, and writes code objects plus a serialized solution library. `SingleSolution` reuses those generators and compiler tools while requiring exactly one recipe. It returns named artifacts and solution identity to Python, records all main/helper code objects and target/toolchain provenance in a manifest, and publishes the bundle only after every required artifact succeeds. The direct hipBLASLt integration consumes that bundle through a child-process invocation. Skipping benchmarking and compiling without a GPU are shared capabilities; the distinction is the bounded callable interface and complete bundle contract.

The sample in `clients/samples/29_hipblaslt_jit_gemm` demonstrates the whole flow for FP16 GEMM: configure the direct call, allocate the returned workspace, run both execution APIs, and check against a CPU reference. Applications own their descriptors, buffers, workspace and synchronization. The library owns generation, loading and retained algorithm state. The sample and runtime do not require a prebuilt hipBLASLt device library.

## Lifetime, failures and build behavior

The returned algorithm is bound to its originating process and device. The library retains its loaded modules until process exit so algorithm copies remain usable. Repeated execution reuses the compiled result. Stored algorithm bytes or indices are not portable identities, and retained build files are not a persistent JIT cache.

Compile before stream capture. Keep application buffers and workspace valid until their operations complete. C matmul prepares each execution; C++ callers initialize their `Gemm` with the selected algorithm and initialize again after changing its problem, pointers or captured scalar values. Existing handle, stream, workspace and object concurrency requirements continue to apply. Stream-K preparation can bind synchronization state to its preparation stream.

A failed selection clears the result and returns a status, with diagnostic details when available. A failed build publishes no completed bundle; a subsequent load/support failure may leave completed artifacts for diagnosis. Missing helpers and insufficient workspace are rejected before submitting the operation. Failed C++ reinitialization preserves its prior prepared execution context. Ordinary GPU execution errors keep the execution API's semantics.

Empty M/N output returns `HIPBLAS_STATUS_NOT_SUPPORTED` without compilation. K=0 can use an explicit recipe implementing beta*C. Datatype, scale-layout and instruction support still depend on the chosen solution; output-amax currently requires one batch, GlobalSplitU=1 and StreamK=0.

`HIPBLASLT_ENABLE_JIT` defaults to `OFF`; enabling it requires the host library. The public direct declaration remains available when disabled and returns `HIPBLAS_STATUS_NOT_SUPPORTED`. The output bundle includes the serialized library, manifest, private loader envelope and code objects. Generator logs remain alongside the fresh output directory.

## Generic extension above the basic path

The generic layer is a partial implementation of AIHPBLAS-4801. Its purpose is to describe an operation independently of the backend that produces executable code. The public experimental interface in `hipblaslt/hipblaslt-jit.hpp` uses opaque, copyable `Request`, `Backend` and `Solution` handles:

```cpp
hipblasStatus_t getJitAlgo(int device, const Request& request,
    const Backend& backend, size_t maxWorkspaceBytes,
    Solution& solution, Diagnostics& diagnostics);
```

`makeGemmRequest` captures an existing GEMM description; the generic `getGemmAlgo` adapter turns a compatible solution into the result accepted by `hipblasLtMatmul` and `Gemm`. The caller selects a configured backend. Compiler settings, tuning fields and predictor schemas remain in provider-specific options and implementation rather than the common handles. The provider boundary is for compiled-in implementations and does not establish a stable external plugin ABI.

The request owns descriptor values and copied host scalars while application buffers retain their usual lifetime requirements. `Solution` copies share the compiled bundle; adapting one to a GEMM algorithm retains the executable state in the process-local registry. Sample `clients/samples/30_hipblaslt_generic_jit_gemm` demonstrates this generic flow separately from the direct sample.

| Component | Responsibility |
| --- | --- |
| Application | Describe the operation, choose direct or generic selection, own buffers/workspace, and execute through the operation API. |
| Generic coordinator | Validate request/backend/device, invoke the provider, retain the returned solution and connect the operation adapter. |
| Backend/provider | Produce supported code and executable metadata, including any provider-private prediction and parameter translation. |
| Operation adapter | Connect a compatible solution to the existing operation descriptors and execution interface. |
| Prepared execution | Bind current arguments/workspace and retain the modules and ordered invocations required for execution. |

GEMM is the implemented operation. An independent provider fixture demonstrates the generic seam; it is not a second production compiler. Additional operations require their own factories and execution adapters. Generic attention execution and dynamic provider discovery are future work.

The generic provider currently combines planning and compilation. Its provider-private plan does not complete the future library-controlled planning/cache contract. This distinction is why AIHPBLAS-4801 remains partial even when the common handles and GEMM adapters are usable.

## Prediction as a separate extension

The upper prediction/benchmark layer uses provider-private Origami ranking and the candidate selector. An empty `Options::configPath` requests prediction only through generic `tensilelite::createBackend`; an explicit path requests that recipe. Prediction ranks candidates and compiles the first supported recipe. It does not benchmark candidate kernels or manufacture a default recipe after ranking/validation fails. This provider behavior does not change the direct API's required recipe.

The initial modeled choices are `MatrixInstruction`, `DepthU`, and `NonTemporalA/B`; TensileLite defaults and derivation supply other settings. The policy remains bounded: complete modeled-field coverage, unmodeled tuning choices and epilogue cost estimates need further work. Existing gfx950 MX rankings lack necessary subtile choices, and gfx1250 latency estimates are not calibrated. A supported explicit recipe and a usable prediction are separate capabilities.

`hipblaslt-bench` completes prediction/compilation before correctness checks, warmup and execution timing. Kernel correctness and prediction quality are evaluated separately. This extension belongs to AIHPBLAS-4551 and stays above the basic direct delivery.

## Roadmap and component interactions

AIHPBLAS-4548 is the umbrella. The rows below describe contributions and remaining contracts, not ticket-closure claims.

| Component / tracking | Status | Purpose and connection |
| --- | --- | --- |
| Builder and direct host integration — AIHPBLAS-4549 | Basic implementation available for review | Explicit YAML reaches SingleSolution; complete artifacts load and execute through C/C++ GEMM. Direct sample and focused CI demonstrate application use. |
| Generic interfaces/adapters — AIHPBLAS-4801 | Partial, separate dependent stack | Common operation/backend/solution handles and GEMM adaptation. A reusable planning/cache protocol and complete modeled-input contract remain open. |
| Modeled prediction — AIHPBLAS-4551 | Initial policy in the upper stack; broader work deferred | Provider-private Origami ranking supplies ordered choices to validation and compilation, including choices independent of installed solution indices. |
| `JustInTime` solution library — AIHPBLAS-4550 | Future | Give matching tuned equality results priority, then consult compatible JIT solutions before requesting compilation. Define ordering with other selection paths. Explicit selection calls alone do not provide this library. |
| Planning/input protocol — AIHPBLAS-4801 remainder | Future | Turn operation, target and specialization facts into reusable backend planning/recipe identity before compilation, preserving relevant modeled inputs while keeping predictor schemas private. |
| Persistent cache — AIHPBLAS-4552 | Future | Use plan/compatibility identity for lookup; load on a hit, compile/store on a miss. Define ProblemType grouping, code-object merging, equality-like metadata, invalidation and concurrent publication. |
| Exact epilogue — AIHPBLAS-4553 | Future | Compile exactly the requested bias/activation/output specialization without generic activation dispatch. This is separate from current epilogue correctness and future epilogue cost modeling. |
| Tuning blueprints — AIHPBLAS-4554 | Future | Combine stored choices for unmodeled parameters with predicted choices before validation. Existing defaults are not a blueprint database. |
| KFA metadata convergence | Deferred follow-up | Have TensileLite emit the agreed KFA encoding and use shared selection and execution for generated and existing KFA kernels. Prove argument, helper, workspace, predicate and launch equivalence before consolidating paths. |
| `HIPBLASLT_JIT_DEBUG` diagnostics | Deferred follow-up | Add independent `timing` and `progress` categories for final duration reports and live compilation-stage transitions. Keep default behavior free of the new instrumentation. |
| More operations and production backends | Future | Add concrete request/adapter and producer implementations as their execution contracts are demonstrated. |

For KFA convergence, the producer work must retain the full executable contract: argument order/types/padding, target and predicates, helpers and their order, grid/cluster units, LDS, workspace and hidden synchronization state. Sharing serialized metadata alone does not prove equivalent selection or execution. The current direct implementation has not completed that convergence.

The proposed diagnostic spelling is `HIPBLASLT_JIT_DEBUG=timing`, `progress`, or `timing,progress`. Timing would report compilation stages and the overall call; progress would report stage transitions while work is running. Neither category implies the other, and unset/empty would add no timing collection, observer or debug files. Reports must preserve existing logs, bundle semantics, exit status and benchmark timing boundaries. These names describe planned behavior, not an implemented option.

## Decisions to track

1. **Planning and selection:** define how library selection asks a provider for a reusable plan, keeps equality tuning first, and handles unsupported requests or compilation limits.
2. **Cache identity and ownership:** distinguish operation lookup from artifact identity; include target/features, toolchain, schema, specialization and blueprint compatibility. Agree invalidation, concurrent publication and reclamation.
3. **KFA equivalence:** select feature groups for producer/consumer convergence and compare packed argument bytes and ordered invocations before numerical validation. Preserve helpers, distinct layouts, workspace and copied-algorithm lifetime.
4. **Specialization and model quality:** agree which fields affect compiled identity and evaluate ranking quality separately from numerical correctness.
5. **Observability:** define stage boundaries, parent/child duration accounting and live progress transport without changing public ABI or making report failures change compilation results.

## Validation and references

The basic path has native Linux gfx950 evidence: direct C/C++ correctness, split-K, Stream-K/workspace, amax/alpha-zero, helper/artifact failures, algorithm copies/lifetime and disabled-build behavior. All ten basic driver routes passed. The integrated generic API passed twelve routes, with thirteen affected failure cases rerun after final exception-handling changes. Both direct and generic samples passed C/C++ checks over 32,768 elements with zero maximum error.

The prediction layer passed sixteen targeted benchmark checks: three numerical C/mixed/C++ cases, two expected prediction/recipe failures and eleven negative checks. Six retained API/sample/provider/disabled driver routes also passed, and the enabled build was restored. These are targeted results, not a full benchmark sweep. Documentation-only relocation preserves the validated executable and test source.

The gfx1250 SIA4 fixture has generation/compilation evidence only. The workflow schedules gfx90a, gfx942, gfx950 and gfx1250; this configuration does not establish successful execution on every target. Native Windows execution remains unverified. Earlier upper-stack CI results belong to their recorded historical heads and do not validate the restructured stack automatically.

- Basic foundations: [single-solution builder #12459](https://github.com/ROCm/rocm-libraries/pull/12459), [candidate selector #12460](https://github.com/ROCm/rocm-libraries/pull/12460), [process launcher #12552](https://github.com/ROCm/rocm-libraries/pull/12552), [artifact reader #12563](https://github.com/ROCm/rocm-libraries/pull/12563), [direct runtime #12564](https://github.com/ROCm/rocm-libraries/pull/12564), [direct sample/CI #12565](https://github.com/ROCm/rocm-libraries/pull/12565).
- Contributor guides at the final documentation tip: [versioned roadmap](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT_ROADMAP.md), [standalone builder](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/tensilelite/SINGLE_SOLUTION.md), [direct API](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT_TENSILELITE.md), [generic API](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT.md), and [validation](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/clients/tests/jit/README.md).
- Separate upper work: [generic interface #12461](https://github.com/ROCm/rocm-libraries/pull/12461), [generic usage #12462](https://github.com/ROCm/rocm-libraries/pull/12462), [prediction/benchmark #12463](https://github.com/ROCm/rocm-libraries/pull/12463), [documentation tail #12430](https://github.com/ROCm/rocm-libraries/pull/12430).
- Related design context: [Dynamic Local Kernel Libraries for hipBLASLt](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1915912619/Dynamic+Local+Kernel+Libraries+for+hipBLASLt). This reference requires authentication; its page and subpages were unavailable for this document review.
