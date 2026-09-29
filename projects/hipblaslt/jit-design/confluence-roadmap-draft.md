# hipBLASLt JIT: heuristic-driven generation, design and roadmap

**Implementation basis:** Basic tip #12565 (`b2510705`) and dependent generic/prediction/documentation tip #12430 (`d742375d`), validated September 24, 2026 and consolidated on `users/jolabega/downstream-hipblaslt-jit-develop`, plus the later change that makes the C++ predictor emit the complete data-parallel Origami contract `origami.gemm.dp.v1`. The earlier pull request (PR) stack is closed without merging. “Implemented” below means present in this downstream checkout, not merged, released or approved as product naming.

**Design update, September 29, 2026:** this revision records the approved target design and the decisions settled since the September 28 draft. Just-in-time (JIT) generation moves behind the existing heuristic query, controlled by the environment variable `HIPBLASLT_JIT`, with a persistent JIT solution library and in-process code-object construction. The explicit JIT entry points have left the public application programming interface (API). Roadmap step 1 is Done; the other steps are Planned. This document is ready to copy into Confluence; it has not been published there.

This page is for hipBLASLt and TensileLite contributors and integration developers. It is the discussion copy of the versioned [JIT guide](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/JIT.md), which holds the same design with a rendered Mermaid diagram. Confluence does not render Mermaid Markdown, so this page shows the diagram as text and as an edge table. Source Markdown follows the existing `@ROCm/hipblaslt-reviewers` and `@ROCm/hipblaslt-docs-reviewers` CODEOWNERS rules. These contributor files sit outside the ROCm release-documentation source tree; this page makes no release publication or support commitment.

Terminology: just-in-time (JIT) generation; application programming interface (API); general matrix multiplication (GEMM); KernelFromAnywhere (KFA), implemented as Gemm-From-Anywhere (GFA); AMD code object manager (comgr).

## Summary

Today, JIT generation is an explicit path behind internal entry points. The JIT test binaries and `hipblaslt-bench --jit-gemm` call an entry point with GEMM descriptors, hipBLASLt compiles one solution through TensileLite, and the caller passes the returned algorithm to `hipblasLtMatmul` or `hipblaslt_ext::Gemm`. An ordinary heuristic query or matmul call does not initiate this path. The separate, existing rocRoller integration has its own runtime generation path.

In the target design, `hipblasLtMatmulAlgoGetHeuristic` and `GemmInstance::algoGetHeuristic` consult the pre-tuned libraries first. When `HIPBLASLT_JIT` enables it, they fill a shortfall from a persistent JIT solution library and then from newly generated solutions. Generators emit source and metadata only; hipBLASLt builds code objects through comgr. The explicit JIT entry points are already internal interfaces used by tests.

## Current behavior

| Component | Current behavior |
| --- | --- |
| Direct TensileLite entry point | `hipblaslt_ext::experimental::jit::tensilelite::getGemmAlgo` in the internal `library/src/amd_detail/hipblaslt-jit-tensilelite.hpp` compiles one required, explicit YAML (YAML Ain't Markup Language) recipe and returns a `hipblasLtMatmulHeuristicResult_t`. The `hipblaslt-jit-direct-gemm-test` binary exercises it. |
| Generic entry point | The internal `library/src/amd_detail/hipblaslt-jit.hpp` provides `makeGemmRequest`, `getJitAlgo` and the generic `getGemmAlgo`; `tensilelite::createBackend` configures the provider. The `hipblaslt-jit-generic-gemm-test` binary exercises it. Neither header is installed or included from `hipblaslt-ext.hpp`; the five functions stay exported from `libhipblaslt.so` so the tests and benchmark can link. |
| Builder | `Tensile.SingleSolution` compiles one recipe into a complete bundle, invoking the assembler, linker, HIP compiler driver and offload bundler. The host runs it as a child process, then loads the serialized library and all code objects and checks support and workspace. |
| Provider prediction | With an empty `Options::configPath`, the C++ predictor ranks synthetic candidates with Origami and emits the `origami.gemm.dp.v1` modeled contract (matrix instruction, macro tile, `DepthU`, cache hints, workgroup mapping, stagger and launch). `Tensile.JitGemm` validates candidates in order and builds the first supported one. It never benchmarks candidates or invents a recipe. |
| Benchmark | `hipblaslt-bench --jit-gemm` uses the generic provider with prediction and completes generation before correctness checks and timing. |
| Lifetime | Algorithms and modules are retained until process exit on their generating device. No persistent cache or bundle reload exists. |
| Pre-tuned selection | Default construction orders Equality, Range, Prediction, GridBased and FreeSize selectors, followed by TruePred rows. The Prediction library type (C++ `ProblemPredictionLibrary`) ships 507 pre-tuned solution YAML files in per-architecture `Origami` directories under `Logic/asm_full/` (gfx950 489, gfx1250 7, gfx1250v0 7, navi32 4) and ranks them at runtime with `origami::rank_configs`. Origami does not construct kernels at package time. |
| xf32 retry | When a problem that requests xf32 math finds no solution, `getBestSolutions` repeats the lookup with FP32 math. |
| Shortfall fill | When `getBestSolutions` returns fewer results than `requestedAlgoCount`, the heuristic code calls `getAllSolutions`, excluding GridBased and Prediction, and appends supported, non-duplicate solutions. |
| rocRoller | When it applies (`HIPBLASLT_USE_ROCROLLER=1`, or by default for eligible block-scaled problems), rocRoller returns its own results before Tensile lookup, using a handle-owned cache and generating on a miss. It is not a generic JIT provider today. |

## Target design

### Diagram

```text
HEURISTIC QUERY                                          JIT  BACKENDS AND PREDICTION

EqualityLibrary ---------+                                    TensileLite ----------------+
                         |                                    HipKittens (future) . . . . +
OrigamiLibrary ----------+                                    rocRoller (future) . . . . .+
(LibraryType: Prediction)|                                    OtherBackends (future) . . .+
                         v                                                                |
                  SolutionLibrary <--------------------  Jit <----------------------------+
                         |    if not satisfied by         ^
                         |    pre-tuned libraries         |
                         v                                +------ Predictor <------ Origami
                  AlgoGetHeuristic                                    ^
                                                                      +------------------- TuningKnowledge

Solid lines: connections in the target design.
Dotted lines: future backends. Each backend connects only to Jit; none depends on TensileLite.
```

| From | To | Meaning |
| --- | --- | --- |
| EqualityLibrary | SolutionLibrary | Existing pre-tuned Equality matching. |
| OrigamiLibrary (LibraryType: Prediction) | SolutionLibrary | Existing pre-tuned solutions ranked by Origami at runtime. |
| SolutionLibrary | AlgoGetHeuristic | Results returned by `hipblasLtMatmulAlgoGetHeuristic` and `GemmInstance::algoGetHeuristic`. |
| Jit | SolutionLibrary | Only if the request is not satisfied by the pre-tuned libraries (`HIPBLASLT_JIT=1`), or as the only source (`HIPBLASLT_JIT=2`). |
| TensileLite | Jit | Live backend. |
| HipKittens | Jit | Future backend; explicitly deferred in this pass (dashed). |
| rocRoller | Jit | Future backend (dashed). |
| OtherBackends | Jit | Future backends (dashed). |
| Predictor | Jit | Ranked candidate configurations. |
| Origami | Predictor | Analytical ranking. |
| TuningKnowledge | Predictor | Values for knobs the model does not predict; TensileLite defaults initially. |

### Components

| Component | Role |
| --- | --- |
| AlgoGetHeuristic | The public heuristic entry points, C `hipblasLtMatmulAlgoGetHeuristic` and C++ extension `GemmInstance::algoGetHeuristic`. No JIT-specific public call is required. |
| SolutionLibrary | The existing solution-library lookup, fed by EqualityLibrary, OrigamiLibrary and, when enabled, the JIT solution library. |
| OrigamiLibrary | The existing `LibraryType: Prediction`. The name is a design label, not a new type. |
| Jit | hipBLASLt code that calls a backend-specific JIT interface and builds a library of JIT-generated kernels. The design does not introduce a `JitInterface` type name. |
| JIT interface | Input: algorithm parameters (for GEMM: M, N, K, datatypes, scale types, layout, activation and the rest of the operation description) plus the gfx target. Output: solutions. Implementations are backend specific. |
| Backends | Independent generators: TensileLite (live); rocRoller, HipKittens and others (future extension points; HipKittens deferred in this pass); a new in-process mock backend in tests to prove swappability. |
| Predictor | Ranks candidates for Jit from Origami and TuningKnowledge. The current C++ predictor (`hipblaslt-jit-tensilelite-predictor.cpp`) already ranks synthetic candidates with Origami and emits `origami.gemm.dp.v1`. |
| TuningKnowledge | A new interface. It initially supplies TensileLite defaults for unmodeled knobs; real tuning data is later work under AIHPBLAS-4554. |
| Code-object builder | hipBLASLt C++ that assembles and links through comgr, adapted from rocRoller's `InProcessAssembler`. |
| JIT solution library | The persistent cache, loaded as a second master library. |

### Existing types and KFA metadata

The type-reuse guidance from the September 25 revision still applies. The GEMM payload remains `RocblasltContractionProblem`, and selection and execution reuse `ContractionProblemGemm`, `ContractionSolution`, `KernelArguments`, `KernelInvocation` and the HIP `SolutionAdapter` where sufficient. Another generator does not require a second public GEMM problem model. KFA custom kernels already become normal `ContractionSolution`s; KFA metadata is the candidate common encoding for the metadata that generators emit, but the plan of record does not yet select the format. KFA convergence remains deferred; the [KFA assessment](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/jit-design/kfa-producer-convergence.md) records producer gaps and the proof sequence required before dispatch is shared.

### Code-object construction

Generators emit assembly or HIP source plus metadata only. hipBLASLt C++ assembles and links through comgr, adapted from rocRoller's `InProcessAssembler`: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`, then `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE`. HIP helper source uses comgr HIP compilation. The output is raw, uncompressed executable code objects: comgr cannot bundle or compress, and `hipModuleLoad` accepts raw executable and linkable format (ELF) objects. hipBLASLt disables comgr's own on-disk cache (`~/.cache/comgr`) for these builds, so the JIT solution library is the only persistent cache of generated code.

### JIT solution library

| Property | Design |
| --- | --- |
| Location | The directory named by `HIPBLASLT_JIT_LIBRARY_PATH`. By default hipBLASLt creates a per-user directory with mode 0700: `/tmp/hipblaslt-jit-<uid>/` on Linux, `%TEMP%\hipblaslt-jit-<user>` on Windows. |
| Layout | One library per `ProblemType`, mimicking the TensileLibrary layout of a library file plus code objects. |
| Publication | New solutions merge in under a file lock with atomic rename, so several processes can share a directory. |
| Loading | Loaded at runtime as a second master library. |
| Indices | Cached solutions use real solution indices from a reserved index range. Tests may still use process-local JIT tokens. |
| Cache key | The gfx target and its target features, the backend identifier and version, the comgr version, the compiler environment settings that affect output, and the library schema version. A library whose key does not match is ignored; hipBLASLt never deletes cache entries automatically. |

### Tool paths

The Python interpreter, TensileLite source directory and import paths, and (until comgr replaces them) the compiler and offload bundler become build-time defaults baked into the library. The existing `HIPBLASLT_JIT_PYTHON`, `HIPBLASLT_JIT_TENSILE_SOURCE`, `HIPBLASLT_JIT_PYTHONPATH`, `HIPBLASLT_JIT_CXX` and `HIPBLASLT_JIT_OFFLOAD_BUNDLER` environment variables override them. Today only `hipblaslt-bench` bakes these defaults; a heuristic query has no application options, so the library must own them.

### Heuristic integration

| `HIPBLASLT_JIT` | Behavior |
| --- | --- |
| `0` or unset (default) | Off. Heuristic queries behave as they do today. |
| `1` | Fallback. JIT runs only when the existing lookup leaves the result short of `requestedAlgoCount`. |
| `2` | Forced. JIT is the only source; the query skips Equality, Origami (Prediction), all other libraries and rocRoller's early path. It looks up the JIT solution library first, then generates. |

In fallback mode, the existing lookup runs to completion first: `getBestSolutions`, including rocRoller's early path and the xf32-to-FP32 retry, then the existing `getAllSolutions` shortfall fill. rocRoller results count toward `requestedAlgoCount`. If the result is still empty or has fewer than `requestedAlgoCount` solutions, the query consults the JIT solution library, then generates and publishes. Jit generates as many solutions as are needed to reach `requestedAlgoCount`, as `AlgoGetHeuristic` does for pre-tuned results.

JIT failures are always reported. In fallback mode, failing to reach `requestedAlgoCount` is a hard error unless the existing heuristic contract already allows returning fewer results; step 5 verifies that contract. In forced mode, a failure returns zero results and is still reported. A build with `HIPBLASLT_ENABLE_JIT=OFF` ignores `HIPBLASLT_JIT` with a one-time warning.

### Public API changes

Step 1, which is Done, made these changes. `getJitAlgo`, `makeGemmRequest`, `getGemmAlgo` (generic and TensileLite direct) and `createBackend` left the public and extension API. `hipblaslt-jit.hpp` and `hipblaslt-jit-tensilelite.hpp` are no longer installed or included from `hipblaslt-ext.hpp`; they are internal headers under `library/src/amd_detail/` used by the JIT tests. Samples 29 and 30 became the `hipblaslt-jit-direct-gemm-test` and `hipblaslt-jit-generic-gemm-test` binaries under `clients/tests/jit`, run by `.github/scripts/test_hipblaslt_jit.py`. `hipblaslt-bench --jit-gemm` keeps working through the internal header until step 5 removes it.

### Ticket mapping

AIHPBLAS-4548 is the umbrella epic. These mappings describe scope, not ticket closure.

| Ticket | Scope |
| --- | --- |
| AIHPBLAS-4801, Interface for JIT backends | The backend interface and the predict, cache lookup and active module path. |
| AIHPBLAS-4552 | JIT solution library (cache). |
| AIHPBLAS-4551 | Prediction. |
| AIHPBLAS-4550 | JustInTime library type. |
| AIHPBLAS-4553 | Exact epilogue. |
| AIHPBLAS-4554 | Tuning blueprints. |
| AIHPBLAS-4549 | Builder and direct integration (current behavior). |

## Roadmap

The steps are planned in this order. The related-ticket column associates each step with the ticket scopes above for review; it is not a ticket-closure plan.

| Step | Status | Scope | Related tickets |
| --- | --- | --- | --- |
| 1. Demote the public API | Done | Stop installing the two JIT headers and including them from `hipblaslt-ext.hpp`; move them to internal headers for unit tests. Convert samples 29 and 30 to test binaries run by the shared driver. `--jit-gemm` uses the internal header until step 5. | AIHPBLAS-4801 |
| 2. Jit component and interfaces | Planned | Jit, the backend interface, the mock backend, and the Predictor and TuningKnowledge interfaces. | AIHPBLAS-4801, AIHPBLAS-4551, AIHPBLAS-4554 |
| 3. comgr code-object builder | Planned | In-process assembly/link and helper compilation; TensileLite emits assembly, helper source and metadata only. | AIHPBLAS-4801 |
| 4. JIT solution library | Planned | Per-`ProblemType` cache under `HIPBLASLT_JIT_LIBRARY_PATH` with locked merge, atomic rename and reserved indices. | AIHPBLAS-4552 |
| 5. Heuristic integration | Planned | `HIPBLASLT_JIT` modes and failure rules in `hipblasLtMatmulAlgoGetHeuristic` and `GemmInstance::algoGetHeuristic`; library-owned tool-path defaults; the JIT-off warning; removal of `hipblaslt-bench --jit-gemm`. | AIHPBLAS-4550, AIHPBLAS-4801 |
| 6. Validation sweep | Planned | Shared JIT driver plus heuristic, cache and mode coverage. | AIHPBLAS-4548 |

On the development host, gfx950 hardware (8× MI355X) is available for native numerical validation, and gfx1250 kernels run on the FFM MI450 simulator. Simulator runs are not native gfx1250 hardware evidence.

Outside these steps, exact epilogue specialization (AIHPBLAS-4553), real tuning data (AIHPBLAS-4554), rocRoller and HipKittens backends, KFA metadata convergence, `HIPBLASLT_JIT_DEBUG` timing/progress diagnostics and additional operations remain future work.

## Changes from the September 25 revision

- The September 25 revision assessed an optional `JustInTime` library that would generate only after the complete lookup returned **zero** compatible results, and it warned against compiling to fill a requested top-N count. The approved design also triggers when the result has fewer than `requestedAlgoCount` solutions, and generates enough to fill the count.
- The September 25 revision kept the explicit API as the primary application entry point for deterministic backend choice and prewarming. The approved design removes it from the public API; applications reach JIT through the heuristic query.
- The September 25 revision listed persistent-cache grouping and concurrent publication as undefined. The approved design sets one library per `ProblemType` with locked merge and atomic rename, a cache key under which mismatched libraries are ignored rather than deleted, and 0700 per-user default directories.
- Today the TensileLite generator builds its own code objects with external tools. The approved design has generators emit source and metadata only, and hipBLASLt builds the code objects through comgr.
- The September 25 revision treated rocRoller integration as a proposed KFA adaptation. The approved design adds rocRoller as a future independent backend behind the Jit interface, and `HIPBLASLT_JIT=2` skips rocRoller's existing early path.

## Decisions settled since the September 28 draft

1. **Fallback and rocRoller:** rocRoller results count toward `requestedAlgoCount` in `HIPBLASLT_JIT=1`, and the fallback runs after the xf32-to-FP32 retry and the `getAllSolutions` fill. Forced mode consults the JIT solution library, then generates.
2. **Failure policy:** JIT failures are always reported. In fallback mode, an unfulfilled count is a hard error unless the existing heuristic contract allows fewer results. Forced mode returns zero results on failure.
3. **Tool paths:** build-time defaults baked into the library, overridable by the existing `HIPBLASLT_JIT_*` environment variables.
4. **Cache identity and location:** the cache key listed above; mismatched libraries are ignored and never deleted automatically; 0700 per-user default directories on Linux and Windows. comgr's own cache is disabled for these builds.
5. **Build and bench behavior:** a JIT-off build ignores `HIPBLASLT_JIT` with a one-time warning; `hipblaslt-bench --jit-gemm` is removed in step 5.
6. **Mock backend:** a new in-process test backend.

## Decisions to track

1. **Heuristic-path latency:** how compilation latency, concurrent generation of the same problem and stream capture are handled inside a heuristic query.
2. **Result contract:** whether the existing heuristic contract allows returning fewer than `requestedAlgoCount` results (verified in step 5), and how unsupported and failed outcomes map to statuses.
3. **Reserved indices:** the size and stability of the reserved index range, and enumeration and index-query behavior for it.
4. **Backend choice:** how Jit selects among several backends once more than TensileLite exists.
5. **Metadata format:** whether generator metadata converges on KFA `custom.config` and replaces the current private loader envelope.

## Validation and references

Recorded evidence applies to the current behavior only. The basic path has native Linux gfx950 evidence: all ten direct driver routes passed. The integrated generic API passed twelve routes, with thirteen affected failure cases rerun after final exception-handling changes. Both samples, now the direct and generic GEMM test binaries, passed C/C++ checks over 32,768 elements with zero maximum error. The prediction layer passed sixteen targeted benchmark checks. The gfx1250 `ScheduleIterAlg=4` fixture has generation/compilation evidence only. The workflow schedules gfx90a, gfx942, gfx950 and gfx1250; this configuration does not establish successful execution on every target. Native Windows execution remains unverified. None of these results validates the target design. rocRoller research is source-based.

- Versioned guides on the downstream branch: [JIT guide](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/JIT.md), [TensileLite backend](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/JIT_TENSILELITE.md), [standalone builder](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/tensilelite/SINGLE_SOLUTION.md), [KFA assessment](https://github.com/ROCm/rocm-libraries/blob/users/jolabega/downstream-hipblaslt-jit-develop/projects/hipblaslt/jit-design/kfa-producer-convergence.md).
- Historical basic foundations: [single-solution builder #12459](https://github.com/ROCm/rocm-libraries/pull/12459), [candidate selector #12460](https://github.com/ROCm/rocm-libraries/pull/12460), [process launcher #12552](https://github.com/ROCm/rocm-libraries/pull/12552), [artifact reader #12563](https://github.com/ROCm/rocm-libraries/pull/12563), [direct runtime #12564](https://github.com/ROCm/rocm-libraries/pull/12564), [direct sample/CI #12565](https://github.com/ROCm/rocm-libraries/pull/12565).
- Historical upper work: [generic interface #12461](https://github.com/ROCm/rocm-libraries/pull/12461), [generic usage #12462](https://github.com/ROCm/rocm-libraries/pull/12462), [prediction/benchmark #12463](https://github.com/ROCm/rocm-libraries/pull/12463), [documentation tail #12430](https://github.com/ROCm/rocm-libraries/pull/12430).
- Contributor guides at the final implementation tip: [versioned roadmap](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT_ROADMAP.md) and [generic API](https://github.com/ROCm/rocm-libraries/blob/d742375dbbef5ef44e9890e199741de70d573f14/projects/hipblaslt/JIT.md).
- Related design context: [Dynamic Local Kernel Libraries for hipBLASLt](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1915912619/Dynamic+Local+Kernel+Libraries+for+hipBLASLt). This was supplied as a format/tone reference; its authenticated page and subpages were unavailable for this document review. The deliverable here is the copyable draft.
