# GFX1151 audit slice (sub-bead `rocm-libraries-ce6`, redo)

Captured assembly slice from the live-build sweep that closes the
"Live build" sub-bead deferred by the static-analysis audit in
`../GFX1151_AUDIT.md`.

## Source yaml (corrected)

```
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml
```

This is the yaml the audit doc originally named. The earlier run
(commit `e056707a79`) silently fell back to the BBS variant
(`gfx1151_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml`) because
the agent searched the develop checkout where the HHS yaml does not
exist. The long-term-plans worktree contains it at exactly the
documented path. The BBS slice
(`sample_kernel_MT16x16_BBS_BH.s`) has been removed and is superseded
by the HHS slice committed alongside this README.

HHS = `f16` input/output, `fp32` compute. 196 KB yaml, 3 unique
solutions emitted. The BBS variant exercised the `_bf16` WMMA opcode;
HHS exercises the `_f16` opcode and additionally hits the FMA-mix
half-to-float promotion path.

## Build command

```
mkdir -p /tmp/gfx1151_audit_ce6_hhs_logic
cp .../gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml \
   /tmp/gfx1151_audit_ce6_hhs_logic/

env -C <worktree>/projects/hipblaslt/tensilelite \
  PYTHONPATH=<worktree>/projects/hipblaslt/tensilelite \
  python3 Tensile/bin/TensileCreateLibrary \
    --architecture gfx1151 \
    --no-enumerate \
    --jobs 4 \
    --keep-build-tmp \
    /tmp/gfx1151_audit_ce6_hhs_logic \
    /tmp/gfx1151_audit_ce6_hhs \
    HIP
```

`PYTHONPATH=<worktree>/projects/hipblaslt/tensilelite` is required
because the system-installed Tensile (`pip install -e` against the
develop checkout) shadows the worktree's package when `Tensile/bin/...`
is invoked directly. Without the override, the develop checkout's
older Tensile runs and cannot parse the worktree's logic yaml
(`UseCustomMainLoopSchedule=1 but CMS is not supported`).

Toolchain: `/opt/rocm/bin/amdclang++` (ROCm 7.2.53211, hipcc resolving
to clang 22.0.0). Build completed in seconds, producing 3 `.s` files
(122 KB, 132 KB, 146 KB), 3 `.o` files, and a packaged `.co` /
`.hsaco` library under `/tmp/gfx1151_audit_ce6_hhs/library/gfx1151/`.

## Slice contents

`sample_kernel_MT32x16_HHS_BH_Bias.s` — the smallest of the three
emitted kernels (122 KB), built from the HHS Cijk_Alik_Bljk variant
with `MacroTile 32x16x64`, `MatrixInstruction 16x16x1`, wave32, MIWT
`1_1`. Header confirms `.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"`,
`.amdhsa_wavefront_size32 1`, 128 VGPRs, 74 SGPRs, 14.75 KB LDS. The
other two kernels are not committed to keep the slice under 500 KB;
they share the same instruction-class profile.

## Instruction-class survey (HHS)

108 distinct mnemonics across the three emitted `.s` files (vs 107 in
the BBS slice; same shape, different WMMA + half-conversion opcodes).
All mnemonics map to existing `_OPERAND_RULES` entries.

| Mnemonic                         | rocisa class (validator)         | Notes                       |
|----------------------------------|----------------------------------|-----------------------------|
| `v_wmma_f32_16x16x16_f16`        | `MFMAInstruction` (WMMA render)  | HHS-only opcode             |
| `v_fma_mix_f32`                  | `_GenericALURule`                | half-promote MAC, HHS-only  |
| `v_cvt_f16_f32`, `v_cvt_f32_f16` | `_GenericALURule`                | half/single conv, HHS-only  |
| `s_load_b256`                    | `_NoDataflowRule`                | scalar kernarg fetch, HHS-only in this kernel set |
| `s_cmpk_lg_u32`                  | `_GenericALURule`                | HHS-only sk-cmp variant     |
| `ds_load_b32`, `ds_load_b128`    | `DSLoadB32`, `DSLoadB128`        |                             |
| `ds_store_b32`, `ds_store_b128`  | `DSStoreB32`, `DSStoreB128`      |                             |
| `buffer_load_b32`, `buffer_load_b128` | `BufferLoadB32`, `BufferLoadB128` |                        |
| `buffer_load_d16_b16`, `buffer_load_d16_hi_b16` | sub-32-bit `BufferLoad` (b16 form) |              |
| `buffer_store_b16`, `buffer_store_b32` | `_BufferStoreRule`          | epilogue / bias path        |
| `s_load_b{32,64,128,256,512}`    | scalar kernarg loads (`_NoDataflowRule`) | full B32-B512 ladder |
| `s_waitcnt`                      | `SWaitCnt`                       | unified waitcnt only        |
| `s_barrier`                      | `SBarrier`                       |                             |
| `s_nop`                          | `SNop`                           |                             |
| `v_cvt_{f32,f64}_u32`, `v_cvt_u32_{f32,f64}` | `_GenericALURule`    | index/strides               |
| `v_add*`, `v_mov_b32`, `v_rcp*`, `v_add_lshl_u32`, `v_lshl_add_u32` | `_GenericALURule` | |

### Mnemonics observed in HHS but NOT in BBS slice

- `v_wmma_f32_16x16x16_f16` (HHS uses f16 compute input vs BBS bf16)
- `v_cvt_f16_f32`, `v_cvt_f32_f16` (half conversions on input/store path)
- `v_fma_mix_f32` (mixed-precision FMA)
- `s_load_b256` (additional kernarg-load width)
- `s_cmpk_lg_u32` (additional immediate-compare variant)

### Mnemonics observed in BBS but NOT in HHS slice (for completeness)

- `v_wmma_f32_16x16x16_bf16`
- `v_add3_u32`, `v_bfe_u32`, `v_cmp_u_f32`, `v_fmac_f32`

## Confirmed NOT emitted (matches static-analysis predictions)

- `v_swap_*` (audit table flagged "Likely unused on gfx1151" — confirmed)
- `v_pk_*`, `v_dot*`, full TF32 cvt sequences (TF32 emulation OFF)
- `s_delay_alu` (gfx1151 emit path relies on `s_waitcnt` only)
- Per-counter waitcnt mnemonics (`s_waitcnt_vmcnt`, `s_waitcnt_lgkmcnt`,
  `s_waitcnt_vscnt`, `s_waitcnt_dscnt`, `s_waitcnt_loadcnt`,
  `s_waitcnt_storecnt`) — kernel uses unified `s_waitcnt`. The audit's
  noted RDNA SWaitCnt-counter-semantics gap remains real at the ISA
  level but is not exercised by the emit path.

## Audit-coverage gaps surfaced by the HHS run

1. `v_fma_mix_f32` — present in HHS but not BBS. Currently absorbed by
   `_GenericALURule`. The static-analysis coverage table does not
   enumerate the mix-FMA family; recommend adding `VFmaMixF32` (and
   `VFmaMixF16`, if it can fire from a different kernel) to the table
   for completeness. No correctness gap.
2. `s_load_b256` — fills the previously-noted gap between `b128` and
   `b512` in the audit table. Handled by `_NoDataflowRule`. Worth
   noting in the table that the full `B32`-`B512` kernarg-load ladder
   is now empirically observed.
3. `BufferStoreB16` / `BufferStoreB32` (carryover from BBS run) —
   produced by epilogue + bias path; absorbed by `_BufferStoreRule`
   but still not enumerated in the audit's coverage table. Same
   recommendation as the BBS slice.

No `CaptureUnknownInstructionError`-class surprises: every HHS
mnemonic maps cleanly to an existing `_OPERAND_RULES` entry, matching
the static-analysis audit's "0 missing" claim.
