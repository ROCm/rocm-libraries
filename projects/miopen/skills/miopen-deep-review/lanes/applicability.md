# Lane: solver applicability (`APPL`)

Find problems a solver accepts but cannot serve: undefined behavior, wrong results, crashes or errors that surface after the
kernel is built. The contract: a solver (or fusion plan, or find/immediate path) accepts a problem only if it serves every
property that can reach it, and declines the rest before a kernel runs. Applicability means `IsApplicable` and its
`IsApplicableXXX` helpers, `IsValidPerformanceConfig` / `GetDefaultPerformanceConfig` / `Search` validity for tunable solvers,
fusion `IsApplicable`, the find-mode and enable-switch filters, and the checks `GetSolution` makes.

## Reachability first

A property counts only if it reaches the changed solver. Trace the layers that exist for it:

1. **Enable switches and find filters.** `conv::IsAlgorithmDisabled` (`projects/miopen/src/conv/solver_finders.cpp`) honours
   `MIOPEN_DEBUG_CONV_{GEMM,DIRECT,FFT,WINOGRAD,IMPLICIT_GEMM}` and `MIOPEN_CONV_DIRECT_MAX_SIZE`; each finder's `IsEnabled` repeats it. Per-solver
   `MIOPEN_DEBUG_*` switches sit inside `IsApplicable` (`conv_direct_naive_conv_fwd.cpp`, `conv_hip_implicit_gemm_fwd_v4r1.cpp`).
   `SearchForAllSolutions` / `SearchForSolutions` (`projects/miopen/src/include/miopen/find_solution.hpp`) skip by
   `GetEnvFindOnlySolver()` (`MIOPEN_DEBUG_FIND_ONLY_SOLVER`) and, when `ExecutionContext::use_dynamic_solutions_only` is set
   (Dynamic Hybrid find, `projects/miopen/src/convolution.cpp`), by `IsDynamic()`. These filters only remove solvers; they never replace a gate.
2. **`IsApplicable`.** Called by those searches after the filters, by `GetSolutions` and `GetSolutionsFallback`
   (`projects/miopen/src/hip/convolutionocl.cpp`) per find-db or ranked candidate, and by `GetSolutionWorkspaceSize`.
3. **Perf config.** For tunable solvers `FindSolutionImpl` accepts a perf-db or user-string config only after
   `IsValidPerformanceConfig`; the LGBM picker walks ranked candidates the same way; the last resort is
   `GetDefaultPerformanceConfig`, which `solver.hpp` requires to be valid for every applicable problem.
4. **Fusion.** `FusionPlanDescriptor` solvers (`projects/miopen/src/fusion.cpp`, `src/include/miopen/fusion/solvers.hpp`,
   `fusion/utils.hpp`) are gated by their own `IsApplicable(FusionContext, FusionDescription)`; `GetSolutions` for fusion
   re-checks find-db ids; the fallback loop and `debug::GetAllApplicableFusionSolutions` iterate `GetAllFusionSolvers()`.
5. **Library or kernel side.** `CkImplLibLoader::IsApplicable` (`src/include/miopen/solver/ck_impl_lib_loader.hpp`),
   `mlir::IsMlirSupportedHardware`, rocBLAS, hipBLASLt: see Method step 2.

A frontend or driver check is no guard: `miopenConvolution*Immediate` and `miopenConvolution*CompileSolution` take a solution id
from the caller. Public-API validation (`Problem::ValidateGroupCount`, tensor validation) does not read solver limits.

## Method

1. List the properties (table below) that reach each changed solver, fusion solver or perf-config rule.
2. For each, find the read in `IsApplicable`, or a proven delegation. An external library (rocBLAS, hipBLASLt, CK static or
   library instances, MLIR) counts only when shown to reject before launch **and** a test pins that. A solver that only
   forwards to `loader.IsApplicable(...)` is delegation: read what the loader function checks.
3. A changed shared predicate or helper (`IsFP8Supported`, `GfxHasMissingFp32Intrinsics`, `static_ck::IsComposableKernelSupportedHardware`,
   an `Is*Supported` or `Is*Applicable` function, a limit constant) is a changed gate. `grep` every caller and read each caller's
   execution path: every value the predicate now admits must be handled there (`accept-then-fail`), not only by the predicate.
4. Compare siblings: fwd / bwd / wrw, 2D / 3D, grouped / ungrouped, asm / HIP / CK / MLIR, plain / fused. Only applicability
   gates here; the rest of the change is `SIBLING`.
5. Check every path that builds or sizes a solution without the gate, and name each as a `bypass-path` or clear it:
   `GetSolution` / `FindSolution` reached with a stored record; `GetWorkspaceSize`; `ConvolutionDescriptor::CompileSolution` and
   the `*Immediate` entry points, which call `LoadOrPrepareInvoker` on the caller's solver id (the workspace-size query is the
   only one that calls `IsApplicable`); `GetSolutionsFallback` (AI ranked list from `ai::immed_mode::PredictSolver` in
   `projects/miopen/src/conv/heuristics/ai_heuristics.cpp`, then WTI scan); find-db and user-db records (`FindDbRecord`,
   workspace values not trusted); perf-db strings and AI/LGBM candidate lists; `MIOPEN_DEBUG_FIND_ONLY_SOLVER` runs.
6. Decline tests: one per property value with boundary rows (below, at, above the limit), each a legal problem, failing when
   the guard is removed. Pattern: `test/gtest/unit_conv_solver_<Solver>.cpp` over `ConvTestCase`
   (`test/gtest/unit_conv_solver.hpp`). Offer the guard removal to `VER` as the mutation.
7. State, for every claimed hazard, the smallest legal problem the solver accepts (shape, strides, dtype, direction, arch).
8. When support depends on a contract the tree cannot show (prebuilt asm `.s` / `.inc`, vendored or library CK, rocBLAS
   internals), the missing positive gate is the hazard. A reachable value or combination the path does not visibly handle is an
   applicability finding, not a `CORR` handoff. Tier it by its worst plausible outcome, mark the kernel side `[INFERENCE]` in
   **Proof**, and never lower the tier because the kernel cannot be read. The verdict stays `UNVERIFIED` until the device check
   under **Approval requests** runs. A mechanism you can describe but not decide from the tree is a finding with `[INFERENCE]` and an
   Approval request, never only a Handoff.
9. No GPU or ROCm on the host: decide from code, keep every device-dependent finding `UNVERIFIED`, and file an Approval request
   (command, arch, smallest hazardous shape, expected duration, where it runs). CPU-only applicability tests on a `MockHandle`
   prove predicates, never kernel results. Never run GPU work from this lane.

## Hazard classes

Closed set; name each finding's class in `Actual`.

| Class | Look for |
|---|---|
| `never-read` | A reachable property `IsApplicable` never reads (dilation, groups, layout, non-packed strides, dtype of one operand, `attribute.deterministic`) |
| `loose-check` | Read, but admits values the kernel cannot serve: 32-bit narrowing of offsets, element counts or byte spans, divisibility, vector-width and tile limits, `int16` shader fields |
| `sibling-drift` | Applicability gate only: one direction, dimension or backend gates it (`AllTensorsDimsFitIntoInt`, `HasNonPackedTensors`, an arch list); a sibling does not |
| `accept-then-fail` | `IsApplicable` passes; `GetSolution`, perf-config selection, compilation, `GetInvoker` or launch throws or returns a failed `ConvSolution` |
| `bypass-path` | Immediate mode, solution-id compile, fallback, stored find-db or user-db records, perf-db strings, or `MIOPEN_DEBUG_FIND_ONLY_SOLVER` reach a solver without its `IsApplicable` or `IsValidPerformanceConfig` |
| `blackbox-delegation` | A library or kernel trusted to reject that does not, or is unproven (CK library loader, MLIR, rocBLAS, prebuilt asm) |
| `over-decline` | Reverse: a servable problem refused, an arch list not widened, or a default config invalid so a solver is dropped to naive |

## Property checklist

"Checked" means the solver reads the property and declines unsupported values. Standard decline test unless stated: from a legal
served problem change only the property, assert `IsApplicable` is false, and assert the unchanged problem is accepted.

| Property | Checked means | Decline test |
|---|---|---|
| Data and compute type | Every tensor dtype (`IsFp32/IsFp16/IsBfp16/IsInt8/IsFp8/IsBfp8`, which test any of in, weights, out), mixed combinations and cast types (`IsTensorsCasted`), `UseTF32()` / `IsTF32Supported`, fp8 fnuz vs OCP (`UsesIeeeExponentBias`) are in the served set per arch | One row per unsupported dtype per role; mixed triples; TF32 on a non-TF32 arch |
| Dims, element counts, byte spans | Rank (`Is3d`), `AllDimsFitIntoInt` / `AllLengthsFitIntoInt`, element and byte spans in `size_t` against the real addressing width (`IsIndexRangeLargeEnough` 2 GiB, asm gtc wrw limit, `conv_wino_fury_RxS.cpp` `IsShaderConstraintsMet`). A per-dimension fit (`AllDimsFitIntoInt`) does not bound flattened element counts or byte spans: when the kernel indexes with 32-bit values (kernel side `[INFERENCE]` if unreadable) check the product. Extents the kernel rounds up (batch padding, gemm-k splits, tile multiples) are bounded after rounding, per perf config, not on the logical tensor | Rows at limit-1, limit, limit+1 on each tensor, from a legal shape |
| Strides, layout, packing | `IsLayoutDefault/NHWC/NCHWc` (plus each solver's own 3D NDHWC test) as the addressing formula needs; `HasNonPackedTensors` declined where the kernel assumes packed; zero or padded strides | Non-packed row stride, permuted order, each layout |
| Groups, depthwise | Group count versus channels, per-group tile divisibility, depthwise-only solvers reject `group_count == 1` and the reverse | Groups 1, C, a non-divisor |
| Stride, dilation, padding | Filter-size, dilation and stride limits; asymmetric pad (`IsAsymmetricPadH/W`); pad values against shader field width | Dilation 2, asymmetric pad, large pad at the field limit |
| Alignment, vector width | Channel and spatial extents divisible by the vector or tile width (K, C, N modulo); pointer alignment not assumed | Extent one below, at and above each multiple |
| Arch and features | Predicate complete for the property: wave size, instruction (MFMA, WMMA, dot, fp8, atomics), XNACK/SRAMECC suffix. Lists and executed runs are `ARCH`'s | Row per arch the predicate matches that the property cannot run on |
| Workspace and memory | `GetWorkspaceSize` in `size_t`, no int narrowing, never below the config that runs (`FindRunSolution` vs maximum), `IsEnoughWorkspace` honoured | Largest accepted shape; workspace buffer one byte short |
| Deterministic and find mode | `ConvolutionDescriptor::attribute.deterministic` read by wrw solvers that use split-k or atomics; `use_dynamic_solutions_only` gets only `IsDynamic()` solvers whose N/H/W are run-time args | Deterministic wrw per solver; dynamic-hybrid with a static solver |
| Fusion ops | Op order, count and activation mode matched by the solver; fused conv dtype, layout and arch gates equal the unfused solver's (`WinoCommonIsApplicable`) | Extra, reordered or unsupported-activation plan; fused vs unfused row |
| Perf-config bounds | `IsValidPerformanceConfig` checks every tile, split and vector value against the problem; `GetDefaultPerformanceConfig` valid for every problem `IsApplicable` accepts; `Search` yields only valid configs | Default config on the extreme accepted shapes; a stored config from another problem |

## Severity

Per [../policy.md](../policy.md) tiers; kind `defect` unless noted.

| Situation | Tier |
|---|---|
| Reachable hazard giving UB, a memory fault, wrong results or a crash | `BLOCKING` |
| Reachable `accept-then-fail` or `over-decline` | `IMPORTANT` (`BLOCKING` when it exposes UB or wrong results) |
| Latent hazard in code the change does not touch | `FUTURE WORK`, kind `source-claim` |

A reachable claim at `IMPORTANT` or above carries its smallest legal problem and the reachability argument over the layers
above in **Proof**.

## Hazard knowledge directory

When the `Hazard knowledge:` field of `dispatch-context.md` (or the dispatch `Overrides:`) names a directory, read its README,
its property checklist and the page for the changed solver. Cite matching history in findings and say whether a finding matches
an open candidate row there by ID, and write `Hazard knowledge: <directory read>` under **Coverage**. Otherwise run from this
brief and write `Hazard knowledge: none supplied` under **Coverage** (a non-file line, free of `|`).

## Handoffs

- Missing decline tests: `TESTB`. Weak decline tests (illegal problem, boundary rows missing): `TESTQ`.
- Mutation proofs: `VER`.
- Arch gate lists and executed arch coverage: `ARCH`.
- Cross-sibling consistency of the change: `SIBLING`.
- Correctness faults that are not applicability: `CORR`; observable semantics: `SEMANTICS`.
