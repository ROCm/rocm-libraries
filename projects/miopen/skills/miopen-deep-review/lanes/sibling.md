# Lane: sibling consistency (`SIBLING`)

MIOpen copies logic across direction, dimension, backend and data type. Check that each logical change in the diff reached every
counterpart that shares its contract, or that the author had a reason to leave one out.

## How it differs from `DUP`

`DUP` proposes a shared form for code that repeats. `SIBLING` accepts the copies as they are and checks that the change was applied
to all of them. A finding here is a behavior difference between two code paths that must agree, never a refactoring proposal.
A copy that should be merged is a `DUP` handoff (`DUP: <locations>`).

## Method

1. Split the diff into logical changes (a guard added, a limit changed, a bug fixed, a perf-config rule, a build option, an
   instantiation). One row per change, not per hunk.
2. Find counterparts of every changed file and function:
   - Direction: `fwd` / `bwd` / `wrw` files and classes (`src/solver/conv/conv_hip_implicit_gemm_grouped_{fwd,bwd,wrw}_xdlops.cpp`,
     `conv_asm_implicit_gemm_gtc_{fwd,bwd,wrw}_nhwc.cpp`, `conv_mlir_igemm_{fwd,bwd,wrw}*.cpp`).
   - Dimension and grouping: 2D vs `3d_` (`conv_hip_implicit_gemm_3d_grouped_*`, `conv_3d_depthwise_fwd.cpp`), grouped vs ungrouped.
   - Backend: asm, HIP, OpenCL, CK (`conv_ck_grouped_conv_fwd.cpp`, `src/ck_impl/`), MLIR, GEMM, Winograd, implicit GEMM; fused
     variants (`src/solver/conv_*_fused.cpp`, `src/include/miopen/fusion/solvers.hpp`) of the unfused solver.
   - Data type: `dm_conv{fp32,fp16,bfp16,int8,fp8,bfp8}.cpp` in `driver/`, `_FP16`/`_BFP16`/`_FP32`/`_I8` gtest instantiations,
     kernel `.s`/`.inc` variants (`Conv_Winograd_v30_3_1_*`, `Conv_Winograd_v40_6_0_*`), `MIOPEN_USE_FP16/BFP16/FP32` branches.
   - Test twins: `test/gtest/group_conv{2d,3d}_{fwd,bwd,wrw}.cpp`, `perf_config_HipImplicitGemm3DGroup{Fwd,Bwd,Wrw}Xdlops.cpp`,
     `unit_conv_solver_<Solver>.cpp`; the same case in `test/gtest` and legacy `test/` driver tests.
   - Backends the build supports: HIP vs HIPNOGPU, CK on/off, MLIR on/off, Windows; a change under one `#if` needs a look at the
     other side.
3. Discover by name first (`git grep -l <stem>` with the direction or dimension token swapped), then by shared helper: grep a distinctive
   token of the changed code (a helper, constant, build-option string, env var, error text) across `src/`, `test/`, `driver/`, `fin/`
   and add every hit not in the diff to the candidate list. Solver ids and registrations (`src/solver.cpp`) are not counterparts; they
   are `CORR`'s.
4. For each (change, sibling) pair decide: applies, does not apply (name the reason in the sibling's contract), or unknown. Open the
   sibling and quote the exact line that lacks or has the change.

## Report

Put a table in **Checked, no finding** or in the finding's **Proof**: `change | sibling path:line | applies? | state at $HEAD`.
A sibling the change applies to and left untouched is the finding.

| Tier | When |
|---|---|
| `IMPORTANT` | Same contract, different behavior: a guard, bound, default, error status, workspace size, perf-config rule or dtype handling fixed in one sibling and still wrong or absent in another. Kind `defect`; name the input that now behaves differently |
| `IMPORTANT` | The change adds a test or instantiation for one sibling and not its twin, and the twin's behavior is not otherwise covered. Kind `test-gap` (hand to `TESTB`) |
| `SUGGESTION` | Cosmetic divergence: naming, comment, ordering, formatting of the same logic |
| `FUTURE WORK` | The sibling already diverged at `$BASE` and the diff does not touch it; cite both `$BASE` lines |

A deliberate asymmetry (wrw has no such constraint, 3D is unsupported, the arch does not have the instruction) is not a finding when the
PR text or a nearby comment says so or the sibling's own `IsApplicable` already rejects the case; record it under **Checked, no
finding**. If a sibling reaches the same behavior through a shared helper the diff already changed, it is covered.

## Handoffs

`DUP:` copies that should be one. `CORR:` a sibling whose applicability or perf-config contract differs. `TESTB:` twin tests or
dtype instantiations missing. `ARCH:` a gate changed in one sibling only. A host with no build still completes this lane by reading;
a divergence whose effect needs a run stays `UNVERIFIED`, with an Approval request for the GPU run.
