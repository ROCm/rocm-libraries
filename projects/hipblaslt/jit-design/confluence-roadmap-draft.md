# hipBLASLt JIT: direct TensileLite workflow and roadmap

**Implementation basis:** Basic tip #12565 (`b2510705`) and dependent generic/prediction/documentation tip #12430 (`d742375d`), validated September 24, 2026 and consolidated on `users/jolabega/downstream-hipblaslt-jit-develop`. Their executable and test source remains the implementation baseline; the earlier pull request (PR) stack is closed without merging. “Implemented” below means present in this downstream checkout, not merged, released or approved as product naming.

**Design update, September 28, 2026:** this revision adds type reuse, backend and selection proposals to the original workflow document. The current interfaces remain unchanged. This document is ready to copy into Confluence; it has not been published there.

This page is for hipBLASLt and TensileLite contributors and integration developers. It is the discussion copy of the versioned contributor roadmap and separates current behavior from planned work. Source Markdown follows the existing `@ROCm/hipblaslt-reviewers` and `@ROCm/hipblaslt-docs-reviewers` CODEOWNERS rules. These contributor files sit outside the ROCm release-documentation source tree; this page makes no release publication or support commitment.

Terminology: just-in-time (JIT) generation; application programming interface (API); general matrix multiplication (GEMM); KernelFromAnywhere (KFA), implemented as Gemm-From-Anywhere (GFA). KFA currently describes GEMM custom kernels; other operation profiles remain proposed.

## Purpose and delivery scope

The first delivery is a direct hipBLASLt-to-TensileLite path: an application supplies one YAML (YAML Ain't Markup Language) recipe, hipBLASLt compiles and loads the resulting solution, and the application executes it through the existing C or C++ GEMM interface. This is the basic AIHPBLAS-4549 integration.

A solution can contain a main kernel and several helpers. Successful compilation alone does not establish support for a requested GEMM: the runtime also checks the device, problem, workspace and required executable symbols.

The implemented generic layer adds request/backend/solution interfaces under AIHPBLAS-4801 and initial provider prediction under AIHPBLAS-4551. It extends the basic implementation and preserves the direct API and sample. Persistent caching, KFA metadata convergence, broader prediction coverage and timing/progress diagnostics remain follow-up work; none is required to run the explicit-recipe path.

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

The standalone Python/command-line interface (CLI) builder only compiles artifacts; it does not load or execute graphics processing unit (GPU) work. The direct host call performs generation, loading and validation. The application then submits the returned algorithm to the existing execution API. Calling `hipblasLtMatmul` alone does not invoke this new TensileLite/generic JIT path. The separate existing rocRoller route can generate code at runtime.

`Tensile --build-only` already generates and compiles kernels without benchmarking, including a YAML containing one recipe, and writes code objects plus a serialized solution library. `SingleSolution` reuses those generators and compiler tools while requiring exactly one recipe. It returns named artifacts and solution identity to Python, records all main/helper code objects and target/toolchain provenance in a manifest, and publishes the bundle only after every required artifact succeeds. The direct hipBLASLt integration consumes that bundle through a child-process invocation. Skipping benchmarking and compiling without a GPU are shared capabilities; the distinction is the bounded callable interface and complete bundle contract.

The sample in `clients/samples/29_hipblaslt_jit_gemm` demonstrates the whole flow for 16-bit floating-point (FP16) GEMM: configure the direct call, allocate the returned workspace, run both execution APIs, and check against a central processing unit (CPU) reference. Applications own their descriptors, buffers and workspace and must follow existing handle/stream synchronization rules; the handle owns internal synchronization state. The library owns generation, loading and retained algorithm state. The sample and runtime do not require a prebuilt hipBLASLt device library.

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

`makeGemmRequest` captures an existing GEMM description; the generic `getGemmAlgo` adapter turns a compatible solution into the result accepted by `hipblasLtMatmul` and `Gemm`. The caller selects a configured backend. Compiler settings, tuning fields and predictor schemas remain in provider-specific options and implementation rather than the common handles. The provider boundary is for compiled-in implementations and does not establish a stable external plugin application binary interface (ABI).

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

### Reuse of existing problem and solution types

The internal `GemmRequest` already embeds `RocblasltContractionProblem`, adds an operation tag, and owns captured alpha/beta values. A different generator does not require another GEMM problem schema. The current `Request` handle supplies ownership and operation identification; public `GemmProblemType` alone is narrower than this complete payload.

| Existing type or mechanism | Current behavior and planned reuse |
| --- | --- |
| Tensile `Problem`, `ProblemInputs`, and `Solution` bases | Generic but skeletal; we will assess extensions for missing semantics rather than assume they already own compilation identity and executable lifetime. |
| `ContractionProblemGemm` and `ContractionSolution` | We will reuse the existing GEMM predicates, size mapping, workspace policy and ordered preparation across compatible producers. These types remain GEMM-specific. |
| Generic `Solution` / private `CompiledSolution` | Current handles retain backend, target, request and bundle state, plus workspace requirements. We will preserve that ownership around existing operation solutions as producer support expands. A matmul algorithm token does not replace general executable ownership. |
| `KernelArguments`, `KernelInvocation`, HIP `SolutionAdapter` | We will reuse existing argument, module and launch mechanisms where sufficient. HIP is the Heterogeneous-compute Interface for Portability runtime. |

A new operation needs its own payload, support rules, argument binding, workspace policy and complete helper sequence. We will generalize shared mechanics only where those requirements justify it. Non-GEMM KFA profiles and adapters are future work; another compiler alone does not require a new executor or another public `Problem` type.

### Alternative generators

| Role | TensileLite | rocRoller | Other generators, proposed |
| --- | --- | --- | --- |
| Current integration | Production generic provider and direct API; explicit recipe or provider-private predicted candidates. | Separate runtime selects a `KernelType`, ranks configurations with Origami, checks a handle-owned cache, and generates/loads on a miss. It is not a generic provider today. | No second production generic provider is implemented; the independent provider is a test fixture. |
| Current execution | Serialized `ContractionSolution`, existing predicates/solve logic, retained bundle and HIP adapter. | `CommandArguments`, predicates and synchronization scratch feed `CommandKernel::launchKernel`; this route precedes Tensile solution lookup. | Checked-in external KFA examples demonstrate particular artifacts and supported profiles. |
| Planned convergence | TensileLite will emit complete KFA metadata and use the shared consumer for proven profiles. | The planned rocRoller adapter will export or normalize supported artifacts into that same contract and consumer. | Additional providers would implement the same producer contract and any required operation adapter. |

On a cache miss, rocRoller's `RocRollerGemmKernel::generate` calls `CommandKernel::generateKernel` and `loadKernel`. A separate checked-in rocRoller assembly fixture uses Tensile KFA ingestion; it does not prove automatic metadata export from runtime generation. The rocRoller dispatch branch also supports precompiled custom code objects from other producers with handwritten argument packing. Any common adapter must preserve its predicates and Stream-K `ZeroedBeforeAndAfter` scratch contract; caller-visible workspace alone is not the complete synchronization requirement.

## Prediction as a separate extension

The prediction/benchmark layer uses provider-private Origami ranking and the candidate selector. An empty `Options::configPath` requests prediction only through generic `tensilelite::createBackend`; an explicit path requests that recipe. Prediction ranks candidates and compiles the first supported recipe. It does not benchmark candidate kernels or manufacture a default recipe after ranking/validation fails. This provider behavior does not change the direct API's required recipe.

The modeled contract covers `MatrixInstruction`, macro tile/`DepthU`, `NonTemporalA/B`, all four workgroup-mapping outputs and all three stagger outputs for the current data-parallel candidate domain. [The capability table](../JIT.md#origami-modeled-inputs) records their translations and mode limits. Unsupported mappings and derived changes reject a candidate; only unpredicted settings use defaults and derivation. Stream-K enablement is a caller input, not an Origami prediction. Broader candidate domains, unmodeled tuning choices and epilogue cost estimates need further work. Existing gfx950 microscaling (MX) rankings lack necessary subtile choices, and gfx1250 latency estimates are not calibrated. A supported explicit recipe and a usable prediction are separate capabilities.

`hipblaslt-bench` completes prediction/compilation before correctness checks, warmup and execution timing. Kernel correctness and prediction quality are evaluated separately. This extension belongs to AIHPBLAS-4551 and stays above the basic direct delivery. Origami ranks configurations or existing solutions; it is not a code-generation backend. The installed library’s Prediction selector ranks existing solutions, while this provider predicts recipes that may not exist yet.

## Roadmap and component interactions

AIHPBLAS-4548 is the umbrella. The rows below describe contributions and remaining contracts, not ticket-closure claims.

| Component / tracking | Status | Purpose and connection |
| --- | --- | --- |
| Builder and direct host integration — AIHPBLAS-4549 | Implemented downstream | Explicit YAML reaches SingleSolution; complete artifacts load and execute through C/C++ GEMM. Direct sample and focused continuous integration (CI) demonstrate application use. |
| Generic interfaces/adapters — AIHPBLAS-4801 | Partial implementation | Common operation/backend/solution handles and GEMM adaptation. The applicable data-parallel modeled-input contract is preserved; a reusable planning/cache protocol remains open. |
| Modeled prediction — AIHPBLAS-4551 | Initial policy implemented; broader work deferred | Provider-private Origami ranking supplies ordered choices to validation and compilation, including choices independent of installed solution indices. |
| `JustInTime` solution library — AIHPBLAS-4550 | Alternative under assessment | If adopted, this library would consult compatible generated entries before compiling, only after complete applicable lookup, including Equality and Origami-based Prediction, yields zero compatible results. Placement and policy remain under assessment below. Explicit selection calls alone do not provide this library. |
| Planning/input protocol — AIHPBLAS-4801 remainder | Future | The planning protocol will turn operation, target and specialization facts into reusable backend planning/recipe identity before compilation, preserving relevant modeled inputs while keeping predictor schemas private. |
| Persistent cache — AIHPBLAS-4552 | Future | The cache will use plan/compatibility identity for lookup, load on a hit, and compile/store on a miss. We will define ProblemType grouping, code-object merging, equality-like metadata, invalidation and concurrent publication. |
| Exact epilogue — AIHPBLAS-4553 | Future | The generator will compile exactly the requested bias/activation/output specialization without generic activation dispatch. This is separate from current epilogue correctness and future epilogue cost modeling. |
| Tuning blueprints — AIHPBLAS-4554 | Future | The provider will combine stored choices for unmodeled parameters with predicted choices before validation. Existing defaults are not a blueprint database. |
| KFA metadata convergence | Deferred follow-up | TensileLite will emit the agreed KFA encoding so generated and existing KFA kernels can share selection and execution. We will prove argument, helper, workspace, predicate and launch equivalence before consolidating paths. |
| `HIPBLASLT_JIT_DEBUG` diagnostics | Deferred follow-up | We will add independent `timing` and `progress` categories for final duration reports and live compilation-stage transitions. Default behavior will remain free of the new instrumentation. |
| More operations and production backends | Future | We will reuse existing operation types and execution machinery and add concrete adapters/profiles and producer implementations as their contracts are demonstrated. rocRoller adaptation is proposed above. |

For KFA convergence, producer metadata will retain the full executable contract: argument order/types/padding, target and predicates, helpers and their order, grid/workgroup/cluster units, local data share (LDS), workspace initialization and internal synchronization state. Sharing serialized metadata alone does not prove equivalent selection or execution. The current direct implementation has not completed that convergence.

KFA custom kernels already become normal `ContractionSolution`s; the current JIT loader also reads a normal singleton `MasterSolutionLibrary` and reuses support, solve and launch machinery. We will standardize on the **same versioned metadata schema and execution semantics** across supported producers; concrete symbols, layouts and tuning values may differ:

```text
PLANNED: supported producers will converge on one library-owned consumer
  TensileLite                rocRoller                Other generator
      |                          |                           |
      +--------------------------+---------------------------+
                                 v
                Code objects + common KFA metadata
                                 v
             Library-owned validation / KFA consumer
                                 v
        Existing operation solution / predicates / preparation
                                 v
              Shared argument, module and launch mechanisms
```

Generated `CustomKernel` metadata is partial: workspace policy and helper ordering still come from the full solution, and generated/custom argument paths differ. Generated assembly's `custom.config` currently contains only `InternalSupportParams.KernArgsVersion`; normal kernel ABI metadata is also present, and fuller solution state lives in the serialized solution. Provenance `Source`, `Version` and descriptive `Features` are not schema negotiation or runtime capability gates.

We will complete producer metadata first, then strictly validate it and prove argument-byte, launch, workspace and helper-sequence equivalence before numerical validation and migration of each supported profile. We will preserve symbols/targets, physical layouts, predicates, dynamic shared memory, synchronization, lifetime and diagnostics. Only then will we remove redundant translation or packing. The [KFA assessment](kfa-producer-convergence.md) records detailed gaps and source evidence.

The proposed diagnostic spelling is `HIPBLASLT_JIT_DEBUG=timing`, `progress`, or `timing,progress`. Timing will report compilation stages and the overall call; progress will report stage transitions while work is running. Neither category will imply the other, and unset/empty will add no timing collection, observer or debug files. Reports will preserve existing logs, bundle semantics, exit status and benchmark timing boundaries. These names describe planned behavior, not an implemented option.

### Optional fallback library and selection policy

The explicit API remains useful for deterministic backend choice, deliberate generation and prewarming before stream capture. If adopted, a `JustInTime` library type would provide an alternative entry point to the proposed shared producer/validation service; it is not a selected replacement. It would address missing coverage; when any usable existing kernel is available, zero-result fallback would not search for a faster generated alternative.

Default Tensile construction orders Equality, Range, Prediction, GridBased, FreeSize and TruePred beneath hardware/operation/problem predicates. Equality is matching with equality distance; Prediction uses Origami. Modes and available branches alter traversal. The proposed trigger is **zero compatible results across the complete applicable lookup**, not an unfilled requested count: `findTopSolutions` accumulates across rows, so a final JIT row could otherwise compile merely to fill top-N.

A leaf cannot handle an absent root library or operation branch, or an earlier return from rocRoller. Before adopting the alternative, we will define that full lookup boundary and the existing retry from reduced-precision to 32-bit computation. Current lookup lacks compiler/backend context; fallback would also need configured backends, target/capability/workspace checks, concurrency and compilation-latency policy, diagnostics, unsupported-versus-failed outcomes, and rules for enumeration/index queries. It would compile outside stream capture. A GEMM-templated library does not gain non-GEMM execution by adding a type.

In this alternative, compatible generated-cache lookup would precede compilation. We will define target, toolchain, schema, recipe and specialization identity, invalidation, concurrent publication and reclamation for the planned cache. Process-local algorithm retention and rocRoller's handle cache are separate from the proposed persistent cache. If the optional library is adopted, it and the existing explicit API would share those proposed services while retaining different selection policies.

## Decisions to track

1. **Planning and selection:** we will assess the optional zero-result fallback after existing selection, including Equality and Origami-based Prediction, and define full-lookup placement, reusable provider plans, configured backends and compilation/query/failure policy while retaining the explicit API.
2. **Cache identity and ownership:** we will distinguish operation lookup from artifact identity, include target/features, toolchain, schema, specialization and blueprint compatibility, and agree invalidation, concurrent publication and reclamation.
3. **KFA equivalence:** we will select feature groups for producer/consumer convergence and compare packed argument bytes and ordered invocations before numerical validation, preserving helpers, distinct layouts, workspace and copied-algorithm lifetime.
4. **Specialization and model quality:** we will agree which fields affect compiled identity and evaluate ranking quality separately from numerical correctness.
5. **Observability:** we will define stage boundaries, parent/child duration accounting and live progress transport without changing public ABI or making report failures change compilation results.

## Validation and references

The basic path has native Linux gfx950 evidence: direct C/C++ correctness, split-K, Stream-K/workspace, amax/alpha-zero, helper/artifact failures, algorithm copies/lifetime and disabled-build behavior. All ten basic driver routes passed. The integrated generic API passed twelve routes, with thirteen affected failure cases rerun after final exception-handling changes. Both direct and generic samples passed C/C++ checks over 32,768 elements with zero maximum error.

The prediction layer passed sixteen targeted benchmark checks: three numerical C/mixed/C++ cases, two expected prediction/recipe failures and eleven negative checks. Six retained API/sample/provider/disabled driver routes also passed, and the enabled build was restored. These are targeted results, not a full benchmark sweep. Documentation-only relocation preserves the validated executable and test source.

The gfx1250 `ScheduleIterAlg=4` fixture has generation/compilation evidence only. The workflow schedules gfx90a, gfx942, gfx950 and gfx1250; this configuration does not establish successful execution on every target. Native Windows execution remains unverified. Earlier upper-stack CI results belong to their recorded historical heads. These results do not validate the proposed convergence or fallback changes. rocRoller research is source-based; it is disabled in the existing local build and was not runtime-tested for this document.

- Historical basic foundations: [single-solution builder #12459](https://github.com/ROCm/rocm-libraries/pull/12459), [candidate selector #12460](https://github.com/ROCm/rocm-libraries/pull/12460), [process launcher #12552](https://github.com/ROCm/rocm-libraries/pull/12552), [artifact reader #12563](https://github.com/ROCm/rocm-libraries/pull/12563), [direct runtime #12564](https://github.com/ROCm/rocm-libraries/pull/12564), [direct sample/CI #12565](https://github.com/ROCm/rocm-libraries/pull/12565).
- Contributor guides at the final documentation tip: [versioned roadmap](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT_ROADMAP.md), [standalone builder](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/tensilelite/SINGLE_SOLUTION.md), [direct API](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT_TENSILELITE.md), [generic API](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT.md), and [validation](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/clients/tests/jit/README.md).
- Historical upper work: [generic interface #12461](https://github.com/ROCm/rocm-libraries/pull/12461), [generic usage #12462](https://github.com/ROCm/rocm-libraries/pull/12462), [prediction/benchmark #12463](https://github.com/ROCm/rocm-libraries/pull/12463), [documentation tail #12430](https://github.com/ROCm/rocm-libraries/pull/12430).
- Related design context: [Dynamic Local Kernel Libraries for hipBLASLt](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1915912619/Dynamic+Local+Kernel+Libraries+for+hipBLASLt). This was supplied as a format/tone reference; its authenticated page and subpages were unavailable for this document review. The deliverable here is the copyable draft.
