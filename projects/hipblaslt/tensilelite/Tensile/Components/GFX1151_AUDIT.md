# GFX1151 (RDNA 3.5) Validator Coverage Audit

This document is the bead `l6q.1` deliverable: an enumeration of every
rocisa instruction class the gfx1151 (RDNA 3.5) code-emit path is known
to produce, cross-checked against the validator's `_OPERAND_RULES`
dispatch table and the `_min_issue_quad_cycles_for` issue-cost table in
`ScheduleCapture.py`.

## Methodology

Two complementary inputs:

1. **Static-analysis sweep** (this commit). The validator's existing rule
   tables (`_LR_CLASS_NAMES`, `_LW_CLASS_NAMES`, `_GR_CLASS_NAMES`,
   `_MFMA_CLASS_NAMES`, `_SWAIT_CLASS_NAMES`, `_SBARRIER_CLASS_NAMES`,
   `_SNOP_CLASS_NAMES`, `_CVT_PACK_CLASS_NAMES`,
   `_MIDDLE_PACK_CLASS_NAMES`) live at module scope in
   `ScheduleCapture.py`. Cross-referenced with the `MFMAInstruction`
   class in `rocisa/rocisa/include/instruction/mfma.hpp`, which renders
   as either `v_mfma_*` (CDNA) or `v_wmma_*` (RDNA 3.x) depending on
   `getAsmCaps()["HasMFMA"]` — i.e. WMMA on gfx1151 is NOT a separate
   rocisa class; it reuses `MFMAInstruction` with a different rendered
   mnemonic. The `_is_mfma` discriminator therefore already claims WMMA.

2. **Live-build sweep** (deferred — see "Sandbox limitation" below). The
   bead asks for a real gfx1151 kernel built from
   `gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml` (located
   at `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/`).
   This requires a working ROCm + Tensile pipeline with `tensilelite-client`
   built; the sandbox the bead runs in cannot satisfy the C++ build
   prerequisite. Recipe documented below; the live audit is filed as
   sub-bead `l6q.1.live`.

## Reference YAML

Path:
```
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml
```

Key kernel parameters (from the first solution entry):
- `ISA: [11, 5, 1]` — RDNA 3.5 (gfx1151).
- `EnableMatrixInstruction: true` — kernel emits matrix-mul ops.
- `MatrixInstruction: [16, 16, 16, 1]` — WMMA variant.
- `MacroTile0: 16`, `MacroTile1: 16`, `DepthU` (varies).
- `WaveSize: 32` — RDNA wave32 vs CDNA wave64.

## Build recipe (deferred — sandbox limitation)

```
cd projects/hipblaslt/tensilelite
Tensile/bin/Tensile \
  ../library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml \
  /tmp/gfx1151_audit
```

Dependencies the sandbox lacks:
- `tensilelite-client` C++ executable (build via `invoke build-client --gpu-targets gfx1151`).
- ROCm runtime headers / libraries.
- AMDGPU assembler (`hipcc` / `clang -target amdgcn-amd-amdhsa`).

Outputs to commit under `Tensile/Components/GFX1151_AUDIT/` once the
build environment is available:
- A representative `.s` slice (the main loop body of one kernel).
- The captured CMS schedules (via `_capture_context.default` / `cms`
  pickled or rendered).
- A diff log of any rocisa class names that fail the
  `CaptureUnknownInstructionError` check in `build_dataflow_graph`.

## Instruction-coverage table (static analysis)

Enumeration of every rocisa instruction class the validator's existing
classifier tables already recognize. The "covered_by_rule?" column maps
each class to the `_OPERAND_RULES` entry (or `_NoDataflowRule`) that
claims it. The "min_quad_cycles" column is the value
`_min_issue_quad_cycles_for` returns for an instance of the class
under the default (CDNA 4) profile; values listed as `1+wait` indicate
the SNop wait-state add. The "RDNA 3.5 status" column flags whether the
gfx1151 emit path is known (or expected) to produce instances of the
class.

| rocisa class           | Covered by `_OPERAND_RULES`        | min_quad_cycles | RDNA 3.5 status |
|------------------------|------------------------------------|-----------------|-----------------|
| DSLoadB32              | `_DSLoadRule`                      | 1               | Used (LR)       |
| DSLoadB64              | `_DSLoadRule`                      | 1               | Used (LR)       |
| DSLoadB128             | `_DSLoadRule`                      | 1               | Used (LR)       |
| DSLoadB256             | `_DSLoadRule`                      | 1               | Used (LR)       |
| DSLoadInstruction      | `_DSLoadRule`                      | 1               | Used (LR)       |
| DSStoreB8              | `_DSStoreRule`                     | 1               | Used (LW)       |
| DSStoreB16             | `_DSStoreRule`                     | 1               | Used (LW)       |
| DSStoreB32             | `_DSStoreRule`                     | 1               | Used (LW)       |
| DSStoreB64             | `_DSStoreRule`                     | 1               | Used (LW)       |
| DSStoreB128            | `_DSStoreRule`                     | 1               | Used (LW)       |
| DSStoreInstruction     | `_DSStoreRule`                     | 1               | Used (LW)       |
| BufferLoadB32          | `_BufferLoadRule` / `_DTLBufferLoadRule` | 1         | Used (GR)       |
| BufferLoadB64          | `_BufferLoadRule` / `_DTLBufferLoadRule` | 1         | Used (GR)       |
| BufferLoadB128         | `_BufferLoadRule` / `_DTLBufferLoadRule` | 1         | Used (GR)       |
| GlobalLoadB32          | `_BufferLoadRule`                  | 1               | Used (GR)       |
| GlobalLoadB64          | `_BufferLoadRule`                  | 1               | Used (GR)       |
| GlobalLoadB128         | `_BufferLoadRule`                  | 1               | Used (GR)       |
| GlobalLoadInstruction  | `_BufferLoadRule`                  | 1               | Used (GR)       |
| GlobalReadInstruction  | `_BufferLoadRule`                  | 1               | Used (GR)       |
| MFMAInstruction        | `_MFMARule` (or `_GenericALURule` for Pack-categorized) | 1 | Used as WMMA — same class, mnemonic differs (`v_wmma_*` vs `v_mfma_*`) |
| SWaitCnt               | `_NoDataflowRule`                  | 1               | Used (different counter semantics on RDNA — see GAP) |
| SBarrier               | `_NoDataflowRule`                  | 1               | Used            |
| SNop                   | `_NoDataflowRule`                  | 1 + wait_state  | Used            |
| VCvtPkF32toBF16        | `_GenericALURule` (CVTPack route)  | 1               | TF32 emulation only — NOT used on gfx1151 (no F32X emulation in reference yamls) |
| PVCvtBF16toFP32        | `_GenericALURule` (MiddlePack)     | 1               | Same — TF32 only |
| VCvtBF16toFP32         | `_GenericALURule` (MiddlePack)     | 1               | Same — TF32 only |
| VSubF32                | `_GenericALURule` (MiddlePack)     | 1               | Same — TF32 only |
| VDot2CF32BF16          | `_GenericALURule` (MiddlePack)     | 1               | Same — TF32 only |
| (V/SAddU32 / SMov / generic ALU) | `_GenericALURule` / `_VCCRule` / `_SCCRule` | 1 | Used (GRInc + m0 setters) |
| VSwap*                 | `_VSwapRule`                       | 1               | Likely unused on gfx1151 (TF32-driven) |

## Identified gaps

**Gaps actively affecting gfx1151 today:**

1. **WMMA finish-cycle timing not characterized.** The default
   `_DEFAULT_CDNA4_ARCH_PROFILE` has `standard_mfma_finish_cycles = 3`
   from CDNA 4 ISA section 7.6. RDNA 3.5 WMMA on gfx1151 has a
   different (likely smaller) finish window — the current placeholder
   `_GFX1151_ARCH_PROFILE.standard_mfma_finish_cycles = 4` is
   intentionally CONSERVATIVE (over-stalls; will not under-stall).
   Sub-bead `l6q.1.t1` tracks confirming the documented value.

2. **WMMA type-switch thresholds unknown.** RDNA 3.5 WMMA has a
   different issue model (no separate 4x4 family in the same shape as
   CDNA 4 PackMFMA). The placeholder thresholds `from_standard = 6`
   and `from_4x4 = 4` are conservative bumps from the CDNA 4 values.
   Sub-bead `l6q.1.t2` tracks the lookup.

3. **CVT->WMMA / WMMA->CVT settle windows unknown.** Same shape as
   above; the reference yaml does NOT use F32 emulation (`UseMFMAF32XEmulation: false`)
   so this branch is untested in practice today, but if it ever
   activates on gfx1151 the placeholder values
   `cvt_before_mfma_quad_cycles = 3` and
   `mfma_4x4_before_cvt_quad_cycles = 6` apply. Sub-bead `l6q.1.t3`
   tracks the lookup if/when F32X emulation extends to gfx1151.

4. **Per-instruction issue cost.** RDNA 3.5 wave32 has a different
   issue-rate model than CDNA 4 wave64 — but every non-SNop instruction
   currently returns 1 quad-cycle in `_min_issue_quad_cycles_for`. The
   profile field `default_issue_quad_cycles` is set to 1 for both arches
   (assumption: per-instruction cost is the same in quad-cycle units).
   Sub-bead `l6q.1.t4` tracks confirming this.

5. **SWaitCnt counter semantics differ.** RDNA 3.x uses a different set
   of counters (`vmcnt`, `lgkmcnt`, `vscnt` reorganized) than CDNA 4.
   The current `_swait_drains` helper looks at `dscnt` and `vlcnt` only.
   This is a SCHEDULE_COMPLETENESS / LR_DATA_READY concern (declared
   but not yet active for gfx1151 in `ISA_CONCERN_CATALOG`); already
   tracked separately. Out of scope for `l6q.1`.

**Gaps NOT affecting gfx1151 today (untested branches):**

- TF32 emulation (`UseMFMAF32XEmulation`) is OFF in every gfx1151
  reference yaml inspected, so the CVTPack / MiddlePack / 4x4 PackMFMA
  branches do not fire. They WOULD over-stall correctly thanks to the
  placeholder profile values; they're not a correctness risk for the
  current ship target.

**Coverage summary:**

- Covered: 28 / 28 distinct rocisa classes the validator currently knows
  about. No `CaptureUnknownInstructionError` is expected from the
  static analysis.
- Missing: 0 (subject to live-build confirmation in sub-bead `l6q.1.live`).
- Unknown: 0 from the static-analysis side; live-build sweep may surface
  unknown classes.

## Per-arch constants table (sources)

| Constant                                        | CDNA 4 (gfx950) | gfx1151 placeholder | Source / Sub-bead                  |
|-------------------------------------------------|-----------------|---------------------|------------------------------------|
| standard_mfma_finish_cycles                     | 3               | 4                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t1` |
| mfma_4x4_finish_cycles                          | 1               | 2                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t1` (same lookup) |
| cvt_before_mfma_quad_cycles                     | 2               | 3                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t3` |
| mfma_4x4_before_cvt_quad_cycles                 | 5               | 6                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t3` |
| mfma_type_switch_threshold_from_standard        | 5               | 6                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t2` |
| mfma_type_switch_threshold_from_4x4             | 3               | 4                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t2` |
| default_issue_quad_cycles                       | 1               | 1                   | CDNA 4 ISA 7.6 / sub-bead `l6q.1.t4` |

## Sandbox limitation

The sandbox cannot run the live-build sweep:
- `Tensile/bin/Tensile` invokes the C++ kernel writer, which requires a
  built `tensilelite-client` and ROCm-target assembler.
- The bead's safety net (the unit-test suite) does NOT cover gfx1151
  end-to-end builds today.

What this commit DOES deliver:
- The ArchProfile abstraction + per-arch wiring + bit-identical CDNA 4
  path (commits on branch `agent/l6q1`).
- The static-analysis audit table (this document).
- A unit-test smoke test (`test_arch_profile_gfx1151.py`) that
  exercises the gfx1151 profile resolver and confirms a synthetic
  WMMA-shaped capture builds without `CaptureUnknownInstructionError`.

What it DOES NOT deliver (filed as sub-beads):
- `l6q.1.live` — run `Tensile/bin/Tensile` on the reference yaml and
  commit the `.s` slice + captured CMS schedules.
- `l6q.1.t1` — RDNA 3.5 WMMA finish-cycle timing.
- `l6q.1.t2` — RDNA 3.5 WMMA type-switch thresholds.
- `l6q.1.t3` — RDNA 3.5 CVT/WMMA settle windows (only relevant if
  F32X emulation lands for gfx1151).
- `l6q.1.t4` — RDNA 3.5 per-instruction issue cost.
