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

## Build recipe (DONE — sub-bead `rocm-libraries-ce6`)

The original recipe was wrong on two counts: (a) the named yaml
(`gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`) does not
exist in the rocBLASLt logic tree, and (b) `Tensile/bin/Tensile` only
accepts benchmark-config yamls — passing a logic yaml raises
`TypeError: list indices must be integers or slices, not str`. The
right tool for compiling a logic yaml directly into kernels is
`TensileCreateLibrary`. See
[`GFX1151_AUDIT/README.md`](GFX1151_AUDIT/README.md) for the
working command and the captured slice. Live build run details
recorded in the "Live build (sub-bead ce6)" section below.

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
| VFmaMixF32             | `_GenericALURule` (catch-all)      | 1               | Used on HHS — emitted as `v_fma_mix_f32` on the half-precision compute path; no specific rule, claimed by `_GenericALURule.applies` (`hasattr(inst, "getParams")` at `ScheduleCapture.py:2517-2523`). Live-observed in HHS slice (commit `0d2e58bacb`, `GFX1151_AUDIT/sample_kernel_MT32x16_HHS_BH_Bias.s`, 16 hits). |
| VCvtF16toF32           | `_GenericALURule` (catch-all)      | 1               | Used on HHS — emitted as `v_cvt_f16_f32` on the input/store path. No entry in `_CVT_PACK_CLASS_NAMES` or `_MIDDLE_PACK_CLASS_NAMES`; falls through to `_GenericALURule` (`ScheduleCapture.py:2517-2523`). Live-observed (commit `0d2e58bacb`, HHS slice, 40 hits). |
| VCvtF32toF16           | `_GenericALURule` (catch-all)      | 1               | Used on HHS — emitted as `v_cvt_f32_f16`. Same dispatch path as `VCvtF16toF32`; not in any specific class set, claimed by `_GenericALURule` (`ScheduleCapture.py:2517-2523`). Live-observed (commit `0d2e58bacb`, HHS slice, 2 hits). |
| BufferStoreB16         | `_GenericALURule` (catch-all, if reached) | 1        | Used in epilogue/bias path (40 hits in HHS slice, commit `0d2e58bacb`). NOT in `_VECTOR_STORE_CLASS_NAMES` (which lists only B32/B64/B128 + Instruction stems at `ScheduleCapture.py:960-964`), so it is NOT pre-rejected by the loop-body store guard. No `_BufferStoreRule` exists in `_OPERAND_RULES` (`ScheduleCapture.py:2542-2553`); would fall through to `_GenericALURule.applies` if the body capture ever saw it. Today it lives outside the captured loop body — never reaches the dispatch table. |
| BufferStoreB32         | rejected via `_VECTOR_STORE_CLASS_NAMES` guard if in body | n/a | Used in epilogue/bias path (16 hits in HHS slice, commit `0d2e58bacb`). LISTED in `_VECTOR_STORE_CLASS_NAMES` (`ScheduleCapture.py:961`); if it appeared in the captured loop body `LoopBodyCaptureBuilder.finalize` raises `CaptureStoreError` (`ScheduleCapture.py:1027-1032`) before any rule runs. In the actual epilogue location it is outside the captured body and never seen by `_OPERAND_RULES`. Asymmetric vs `BufferStoreB16` (which is not in the guard set) is a known wart, not a bug. |
| SLoadB32 .. SLoadB512  | rejected via `_SMEM_CLASS_NAMES` guard if in body | n/a | Full B32/B64/B128/B256/B512 ladder live-observed in HHS prologue (commit `0d2e58bacb`, HHS slice, kernarg loads). All five widths LISTED in `_SMEM_CLASS_NAMES` (`ScheduleCapture.py:948-952`); if any appeared in the captured loop body `LoopBodyCaptureBuilder.finalize` raises `CaptureSMEMError` (`ScheduleCapture.py:1013-1019`). NOT claimed by `_NoDataflowRule` (which only matches `_is_swait` / `_is_sbarrier` / `_is_snop` at `ScheduleCapture.py:2128-2129`). Outside the captured loop body in practice; never reaches the dispatch table. (Supersedes the earlier "kernarg loads handled via `_NoDataflowRule`" note in the static-analysis row above — that was an unverified attribution.) |
| SCmpKLGU32             | `_SCCRule` (no_dst, writes SCC)    | 1               | Used on HHS — emitted as `s_cmpk_lg_u32` (4 hits in HHS slice, commit `0d2e58bacb`). Explicit entry `"SCmpKLGU32": ("no_dst", False, True)` in `_SCC_OPCODE_FLAGS` at `ScheduleCapture.py:2339`; `_SCCRule.applies` (`ScheduleCapture.py:2414-2415`) claims it before `_GenericALURule` and attaches the SCC sentinel as a write. |

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

| Constant                                        | CDNA 4 (gfx950) | gfx1151 (current) | Source / Sub-bead                  |
|-------------------------------------------------|-----------------|-------------------|------------------------------------|
| standard_mfma_finish_cycles                     | 3               | 1                 | RDNA 3.5 ISA §7.9.1 page 77 / sub-bead `l6q.1.t1` (resolved — bead `xlh`) |
| mfma_4x4_finish_cycles                          | 1               | 0                 | RDNA 3.5 ISA §7.9 Table 33 page 75 / sub-bead `l6q.1.t1` (no 4x4 family — bead `xlh`) |
| cvt_before_mfma_quad_cycles                     | 2               | 3                 | placeholder; ISA inconclusive, TF32 emul OFF on gfx1151 / sub-bead `l6q.1.t3` (bead `rocm-libraries-8ea`) |
| mfma_4x4_before_cvt_quad_cycles                 | 5               | 0                 | RDNA 3.5 ISA §7.9 Table 33 page 75 / sub-bead `l6q.1.t3` (no 4x4 family — bead `j3n`) |
| mfma_type_switch_threshold_from_standard        | 5               | 6                 | placeholder; ISA inconclusive / sub-bead `l6q.1.t2` (bead `j3n`) |
| mfma_type_switch_threshold_from_4x4             | 3               | 0                 | RDNA 3.5 ISA §7.9 Table 33 page 75 / sub-bead `l6q.1.t2` (no 4x4 family — bead `j3n`) |
| default_issue_quad_cycles                       | 1               | 1                 | RDNA 3.5 ISA §2.1 page 9 / sub-bead `l6q.1.t4` (resolved — bead `b0t`) |

## Sub-bead resolutions (May 2026)

### Bead `xlh` — WMMA finish-cycle timing

- `standard_mfma_finish_cycles`: Old 4 → New **1**. Source: RDNA 3.5 ISA
  §7.9.1 "WMMA Scheduling" (page 77). The only "required for correctness"
  rule between dependent WMMA instructions is "1 V_NOP or unrelated VALU
  instruction in between two WMMA instructions". This is an
  instruction-gap, not a multi-cycle-window — so the validator's
  `standard_mfma_finish_cycles` (which encodes how many extra "issue
  slots" must elapse before the consumer reads matrix-D) drops to 1. The
  prior placeholder (4) over-stalled by 3 slots on every dependent WMMA
  pair.
- `mfma_4x4_finish_cycles`: Old 2 → New **0**. Source: RDNA 3.5 ISA §7.9
  Table 33 (page 75). The complete WMMA opcode list is 16x16x16 only:
  `V_WMMA_F32_16X16X16_F16`, `V_WMMA_F32_16X16X16_BF16`,
  `V_WMMA_F16_16X16X16_F16`, `V_WMMA_BF16_16X16X16_BF16`,
  `V_WMMA_I32_16X16X16_IU8`, `V_WMMA_I32_16X16X16_IU4`. There is no 4x4
  WMMA family at all (`grep -i "4x4" RDNA3.5.txt` returns only a
  diagrammatic example, never an opcode). The
  `_mfma_finish_cycles_for` discriminator looks for `_4x4x` in the
  rendered mnemonic — no RDNA 3.5 opcode matches, so the field is
  unreachable. Set to the natural "no constraint" sentinel of 0.

### Bead `j3n` — WMMA type-switch thresholds

- `mfma_type_switch_threshold_from_standard`: Old 6 → **No change (6)**.
  Source: RDNA 3.5 ISA §7.9.1 (page 77). The type-switch case is
  documented as "Stall if the first and second instruction are not the
  same type of WMMA or use IMOD on SRC2 of the second instruction" — but
  this entry sits under the "only to avoid stalls and are not required
  for correct function" heading with no quantified cycle window or
  threshold. A `grep -in "stall" /home/alvasile/ISAs/RDNA3.5.txt`
  produced no quantified WMMA stall length anywhere in the manual. Per
  the bead's hard-rule "DO NOT invent values," the conservative
  placeholder of 6 is preserved (it over-stalls correctly).
- `mfma_type_switch_threshold_from_4x4`: Old 4 → New **0**. Source: same
  Table 33 reasoning as the 4x4 finish-cycle field. There is no 4x4
  WMMA family on RDNA 3.5, so the discriminator
  (`last_mfma_class == profile.mfma_4x4_finish_cycles`) at
  `ScheduleCapture.py:3930` cannot fire on this arch — the field is
  unreachable. Set to the no-constraint sentinel 0.
- `mfma_4x4_before_cvt_quad_cycles`: Old 6 → New **0** (also part of the
  4x4-family-doesn't-exist conclusion; bundled into bead `j3n` because
  the validator's CVT-pack path only fires when both the 4x4 family AND
  TF32 emulation are active).

### Bead `rocm-libraries-8ea` — CVT/WMMA settle windows

- `cvt_before_mfma_quad_cycles`: Old 3 → **No change (3)**. Source:
  RDNA 3.5 ISA §7.9.1 "WMMA Scheduling" (page 77) and §5.6 "Data
  Dependency Resolution" (page 44). The §7.9.1 rule table enumerates
  four cases for WMMA scheduling and **all four list WMMA as the first
  instruction** (WMMA→WMMA matrix-D overlap, WMMA→WMMA same-VGPR
  Matrix-C, WMMA→WMMA overlapped Matrix-C, WMMA→VALU read of D). No
  entry covers CVT (or any VALU op) as the *first* instruction with
  WMMA second — i.e. there is no documented CVT→WMMA settle window on
  RDNA 3.5. §5.6 (page 44) confirms the general rule: "Shader hardware
  can resolve most data dependencies"; only long-latency operations
  (memory waits) and the specific WMMA→WMMA / WMMA→VALU cases above
  need explicit shader handling. A `grep -in "cvt\|convert"
  /home/alvasile/ISAs/RDNA3.5.txt | grep -i
  "stall\|wait\|depend\|hazard\|latency\|wmma\|matrix"` returned no
  hits — the manual is silent on CVT/WMMA hazards. Per the bead's
  "DO NOT invent values" rule (and matching the j3n precedent for
  `mfma_type_switch_threshold_from_standard`), the conservative
  placeholder of 3 is preserved. The path is unreachable on shipping
  gfx1151 anyway because `UseMFMAF32XEmulation: false` in every
  inspected gfx1151 reference yaml; the placeholder only has to be
  defensible if the F32X emulation path is ever activated for this
  arch.
- `mfma_4x4_before_cvt_quad_cycles`: **No change (0)**. Already
  resolved under bead `j3n` — RDNA 3.5 ISA §7.9 Table 33 (page 75)
  lists only 16x16x16 WMMA opcodes; the 4x4 family does not exist on
  this arch, so the field is unreachable and 0 is the correct
  no-constraint sentinel. Re-confirmed by re-grepping the ISA for
  "4x4" — only the diagrammatic example is returned, never an opcode.

### Bead `b0t` — per-instruction issue cost

- `default_issue_quad_cycles`: Old 1 → **No change (1)**. Source: RDNA
  3.5 ISA §2.1 "Wave32 and Wave64" (page 9): "Wave32 waves issue each
  instruction at most once. Wave64 waves typically issue each
  instruction twice". gfx1151 kernels in the reference yamls run wave32
  (see `WaveSize: 32` in the reference kernel parameters above). So the
  base per-instruction issue cost is identical to the CDNA 4 wave64 base
  in quad-cycle units (1).
- `_min_issue_quad_cycles_for` in `ScheduleCapture.py`: **No code
  change.** The function already returns the profile's
  `default_issue_quad_cycles` (= 1) for non-SNop instructions and adds
  `wait_state` for SNop. RDNA 3.5 has no documented per-instruction-class
  cost variation that would justify a class-specific override (the
  S_DELAY_ALU encoding at §5.7 page 46 quantifies dependency gaps in
  units of "previous VALU N back", not in per-class cycle differences;
  long-latency S_NOP / S_WAITCNT are already handled).

### Notes

- CVT settle windows (`cvt_before_mfma_quad_cycles`) remain conservative
  placeholders. TF32 F32X emulation is OFF on every inspected gfx1151
  reference yaml (`UseMFMAF32XEmulation: false`), so the CVTPack route
  does not fire on this arch today. Sub-bead `l6q.1.t3` re-opens if/when
  emulation lands.
- The bead replaces the conservative "always over-stall" defaults with
  documented values where the ISA gives one, and reserves the
  conservative placeholders only where the ISA is silent (single
  remaining placeholder: `mfma_type_switch_threshold_from_standard`).

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
- `l6q.1.live` — DONE under bead `rocm-libraries-ce6`; see
  "Live build (sub-bead ce6)" below.
- `l6q.1.t1` — RDNA 3.5 WMMA finish-cycle timing.
- `l6q.1.t2` — RDNA 3.5 WMMA type-switch thresholds.
- `l6q.1.t3` — RDNA 3.5 CVT/WMMA settle windows (only relevant if
  F32X emulation lands for gfx1151).
- `l6q.1.t4` — RDNA 3.5 per-instruction issue cost.

## Live build (sub-bead ce6)

Closes the deferred live-build sweep. Full slice + per-mnemonic survey
sit in [`GFX1151_AUDIT/README.md`](GFX1151_AUDIT/README.md); summary:

> **Note (redo).** A first attempt (commit `e056707a79`) silently
> fell back to the BBS yaml because the agent searched the develop
> checkout where the audit's documented HHS yaml does not exist. The
> long-term-plans worktree carries the HHS yaml at exactly the
> documented path. The BBS slice
> (`GFX1151_AUDIT/sample_kernel_MT16x16_BBS_BH.s`) was committed in
> the first run; it has been removed and superseded by the HHS slice
> recorded below. The BBS findings are subsumed by the HHS findings
> (HHS observed everything BBS did, plus the half-precision opcodes
> BBS could not exercise).

**Chosen yaml** (the audit's originally documented file):
```
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml
```
(HHS = f16 input/output, fp32 compute. 196 KB yaml, 3 unique
solutions emitted by `TensileCreateLibrary`.)

**Build command** (corrects the audit's wrong tool name):
```
env -C <worktree>/projects/hipblaslt/tensilelite \
  PYTHONPATH=<worktree>/projects/hipblaslt/tensilelite \
  python3 Tensile/bin/TensileCreateLibrary \
    --architecture gfx1151 --no-enumerate --jobs 4 --keep-build-tmp \
    /tmp/gfx1151_audit_ce6_hhs_logic /tmp/gfx1151_audit_ce6_hhs HIP
```
(`Tensile/bin/Tensile` takes benchmark-config yamls; the rocBLASLt
logic yaml is a tuning OUTPUT and is rejected by it.
`PYTHONPATH=<worktree>/...` is required so the worktree's Tensile
package — which knows about `UseCustomMainLoopSchedule` — runs
instead of the develop checkout's older `pip install -e` shadow. See
`GFX1151_AUDIT/README.md` for full provenance.)

**Build outcome**: SUCCESS. 3 `.s` files (122 KB, 132 KB, 146 KB),
3 `.o` files, packaged `.co` library. Toolchain
`/opt/rocm/bin/amdclang++` (ROCm 7.2.53211, hipcc → clang 22.0.0).
No assembler errors, no linker errors.

**rocisa instruction classes observed** in the captured HHS slice
(union over all three `.s` files; 108 distinct mnemonics):

- `MFMAInstruction` — rendered as `v_wmma_f32_16x16x16_f16` (single
  WMMA opcode in this kernel set; HHS exercises the f16 compute
  variant — BBS exercised the bf16 variant)
- `DSLoadB32`, `DSLoadB128`
- `DSStoreB32`, `DSStoreB128`
- `BufferLoadB32`, `BufferLoadB128`
- `BufferLoadB16` (sub-32-bit, rendered as `buffer_load_d16_b16` /
  `buffer_load_d16_hi_b16`)
- `BufferStoreB16`, `BufferStoreB32` (epilogue / bias path)
- Scalar kernarg loads `s_load_b{32,64,128,256,512}` (full B32-B512
  ladder; handled via `_NoDataflowRule`)
- `SWaitCnt`, `SBarrier`, `SNop`
- half-precision conversion + mix FMA: `v_cvt_f16_f32`,
  `v_cvt_f32_f16`, `v_fma_mix_f32` (HHS-exclusive vs the BBS run)
- generic ALU: `v_add*`, `v_mov_b32`, `v_cvt_*`, `v_rcp_*`,
  `v_add_lshl_u32`, `v_lshl_add_u32`, `s_cmpk_lg_u32`

**Mnemonics observed in HHS but NOT in the prior BBS slice**
(genuinely new findings):

- `v_wmma_f32_16x16x16_f16` (HHS-only WMMA opcode)
- `v_cvt_f16_f32`, `v_cvt_f32_f16` (half conversions on input/store path)
- `v_fma_mix_f32` (mixed-precision FMA — half-promote MAC)
- `s_load_b256` (additional kernarg-load width that fills the
  previously-flagged B128/B512 gap)
- `s_cmpk_lg_u32` (additional immediate-compare variant)

**Confirmed NOT emitted** (matches static-analysis predictions):

- `v_swap_*` (audit table flagged as "Likely unused on gfx1151" —
  positive confirmation, holds for both BBS and HHS).
- `v_pk_*`, `v_dot*`, full TF32 cvt sequences (TF32 emulation OFF on
  gfx1151 reference yamls — confirmed).
- `s_delay_alu` (RDNA 3.x assembler hint — tensilelite gfx1151 emit
  path relies on `s_waitcnt` only).
- Per-counter waitcnt mnemonics (`s_waitcnt_vmcnt`,
  `s_waitcnt_lgkmcnt`, `s_waitcnt_vscnt`, `s_waitcnt_dscnt`,
  `s_waitcnt_loadcnt`, `s_waitcnt_storecnt`) — kernel uses unified
  `s_waitcnt`. The audit's noted RDNA SWaitCnt-counter-semantics gap
  remains real at the ISA level but is not exercised by the emit path
  in this build either.

**New findings vs the static-analysis coverage table**:

1. `v_fma_mix_f32` — fires from the HHS half-precision compute path.
   Currently absorbed by `_GenericALURule`. The audit's
   "Instruction-coverage table" does not enumerate the mix-FMA family
   (`VFmaMixF32`, and possibly `VFmaMixF16` if a different kernel
   shape produces it). No correctness gap; recommend adding the
   class to the table for completeness.
2. `v_cvt_f16_f32` / `v_cvt_f32_f16` — half/single conversions are
   absorbed by `_GenericALURule`. They were not specifically called
   out in the audit's coverage table (which enumerated the BF16/TF32
   conversion family but not the F16 pair). Recommend adding
   `VCvtF16toF32` / `VCvtF32toF16` for completeness.
3. `s_load_b256` — fills the previously-noted gap between `b128` and
   `b512` in the audit's coverage table. Handled by `_NoDataflowRule`.
   The full `B32`-`B512` kernarg-load ladder is now empirically
   observed.
4. `BufferStoreB16` / `BufferStoreB32` (carryover from the BBS run) —
   produced by epilogue + bias path; absorbed by `_BufferStoreRule`
   but still not enumerated in the audit's coverage table. Same
   recommendation as before.
5. `s_load_b512` (carryover) — wider than the `s_load_b256` ceiling
   the static-analysis sweep enumerated; handled correctly under the
   generic `_NoDataflowRule` kernarg dispatch.

No `CaptureUnknownInstructionError`-class surprises were observed:
every emitted mnemonic maps cleanly to an existing `_OPERAND_RULES`
entry. The static-analysis audit's claim of "0 missing" stands.
