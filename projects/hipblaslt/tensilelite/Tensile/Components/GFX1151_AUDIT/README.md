# GFX1151 audit slice (sub-bead `rocm-libraries-ce6`)

Captured assembly slice from the live-build sweep that closes the
"Live build" sub-bead deferred by the static-analysis audit in
`../GFX1151_AUDIT.md`.

## Source yaml

```
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml
```

Chosen over the audit's originally-named yaml
(`gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`) because
the latter does not exist in the rocBLASLt logic tree. Of the two
nearest equivalents the audit doc suggested, this one is the smaller
(1639 lines vs 10560 for the GridBased AuxH alternative; six unique
solutions vs ~80) so it gives full instruction coverage with the
fastest build cycle.

## Build command

The audit doc's recipe specified `Tensile/bin/Tensile --build-only`,
but `Tensile/bin/Tensile` only accepts benchmark-config yamls (with
top-level `GlobalParameters` / `BenchmarkProblems` keys). The
rocBLASLt logic yaml is the OUTPUT of tuning, not a benchmark config,
so passing it directly raises
`TypeError: list indices must be integers or slices, not str`. The
correct tool for compiling a logic yaml directly into kernels is
`TensileCreateLibrary`, which is what produced this slice:

```
mkdir -p /tmp/gfx1151_audit_ce6_logic
cp .../gfx1151_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml \
   /tmp/gfx1151_audit_ce6_logic/

Tensile/bin/TensileCreateLibrary \
  --architecture gfx1151 \
  --no-enumerate \
  --jobs 4 \
  --keep-build-tmp \
  /tmp/gfx1151_audit_ce6_logic \
  /tmp/gfx1151_audit_ce6 \
  HIP
```

Toolchain: `/opt/rocm/bin/amdclang++` (ROCm 7.2.53211, hipcc resolving to
clang 22.0.0). Build completed in ~7 seconds, producing 6 `.s` files,
6 `.o` files, and a packaged `.co` / `.hsaco` library under
`/tmp/gfx1151_audit_ce6/library/gfx1151/`.

## Slice contents

`sample_kernel_MT16x16_BBS_BH.s` — the smallest of the six emitted
kernels (132 KB), built from the BBS (bf16 input, bf16 output, fp32
compute) Cijk_Alik_Bljk variant with MacroTile 16x16, MatrixInstruction
16x16x1, wavefront size 32. Header confirms
`.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"`,
`.amdhsa_wavefront_size32 1`, 96 VGPRs, 76 SGPRs, 12.5 KB LDS.

The other five kernels (MT16x16 alt, MT32x16, MT64x16, MT64x64,
MT80x128) are not committed to keep the slice under 500 KB; they share
the same instruction-class profile.

## Instruction-class survey

Mnemonics observed across all six emitted `.s` files (extracted via
regex sweep over the assembly source):

| Mnemonic                       | rocisa class (validator)         | Already in audit table? |
|--------------------------------|----------------------------------|-------------------------|
| `v_wmma_f32_16x16x16_bf16`     | `MFMAInstruction` (WMMA render)  | yes                     |
| `ds_load_b32`, `ds_load_b128`  | `DSLoadB32`, `DSLoadB128`        | yes                     |
| `ds_store_b32`, `ds_store_b128`| `DSStoreB32`, `DSStoreB128`      | yes                     |
| `buffer_load_b32`, `buffer_load_b128` | `BufferLoadB32`, `BufferLoadB128` | yes              |
| `buffer_load_d16_b16`, `buffer_load_d16_hi_b16` | sub-32-bit `BufferLoad` (rendered via b16 form) | partial (validator treats all `BufferLoad*` via `_BufferLoadRule`; specific d16-hi mnemonic not enumerated in audit table) |
| `buffer_store_b16`, `buffer_store_b32` | not currently in `_OPERAND_RULES` LR/LW/GR sets — emitted by epilogue / bias path | NOT enumerated |
| `s_load_b32` ... `s_load_b512` | scalar kernarg loads (handled by `_NoDataflowRule` via generic dispatch) | partially — `s_load_b512` is the widest variant |
| `s_waitcnt`                    | `SWaitCnt`                       | yes (semantics gap noted) |
| `s_barrier`                    | `SBarrier`                       | yes                     |
| `s_nop`                        | `SNop`                           | yes                     |
| `v_cvt_{f32,f64}_u32`, `v_cvt_u32_{f32,f64}` | generic ALU (index/strides) | covered by `_GenericALURule` |
| `v_add*`, `v_mov_b32`, `v_rcp*`, `v_add3_u32`, `v_add_lshl_u32` | generic ALU | covered by `_GenericALURule` |

What is NOT observed (and was not expected to fire on gfx1151):

- `s_delay_alu` — RDNA 3.x lets the assembler emit this for VALU
  dependency hints. Tensilelite's gfx1151 path apparently does not
  emit it (relies on `s_waitcnt` only). Not a validator concern today.
- `v_swap_*` — confirmed unused (audit table entry "Likely unused on
  gfx1151" is now positive).
- `v_pk_*`, `v_dot*`, TF32 cvt sequences — confirmed unused (TF32
  emulation OFF, as the audit static analysis predicted).
- Per-counter waitcnt variants (`s_waitcnt_vmcnt`, `s_waitcnt_lgkmcnt`,
  `s_waitcnt_vscnt`, `s_waitcnt_dscnt`, `s_waitcnt_loadcnt`,
  `s_waitcnt_storecnt`) — not emitted; the unified `s_waitcnt`
  covers all counters. The audit's noted SWaitCnt-counter-semantics
  gap is real (RDNA encoding differs) but the kernel emit path does
  not exercise the per-counter mnemonics in this build.

## New findings vs static-analysis table

1. `buffer_store_b16` / `buffer_store_b32` — store-side BufferStore
   instructions appear in the epilogue. The audit table only
   enumerates BufferLoad classes. The validator's `_OPERAND_RULES`
   table does have a separate buffer-store dispatch (see
   `_BufferStoreRule` in `ScheduleCapture.py`) but the audit doc's
   coverage table did not list `BufferStoreB16`/`BufferStoreB32`
   explicitly — they are produced by the bias / epilogue path. No
   correctness gap, but the audit table should mention them.

2. `s_load_b512` — wider scalar load than the `s_load_b256` ceiling
   the static-analysis sweep enumerated. Currently classified as
   `_NoDataflowRule` (no register-dependency tracking on kernarg
   loads) which is correct; flagged for record only.

No `CaptureUnknownInstructionError`-class surprises: every emitted
mnemonic maps cleanly to an existing `_OPERAND_RULES` entry under one
of the rocisa class names already covered by the static-analysis
audit.
