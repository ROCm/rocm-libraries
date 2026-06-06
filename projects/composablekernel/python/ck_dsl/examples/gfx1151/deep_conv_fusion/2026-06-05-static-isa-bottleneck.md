# Static ISA bottleneck attribution — gfx1151 deep-fusion best config (2026-06-05)

## Question

Is the kernel **memory/latency-bound** (prior characterization) or
**control/compute-bound** (current hypothesis)?

**Verdict: compute-bound — specifically integer-VALU-bound in the
quantize / int4-pack / maxpool epilogue. NOT global-memory-bandwidth-bound.**
The earlier "memory/latency-bound" label is overturned by the static ISA.

## Method (constraint: no on-device counters)

rocprofv3/ATT/PMC are unavailable on the Windows gfx1151 board, and the gfx950
dev host can't run gfx11. Substitute = static disassembly of the exact
`gfx11-generic` hsaco (disassembly is arch-independent).

- hsaco: best config
  `t512x32_pt2x64_w4x2_wmma16x16x16_directa_schcompv4_nativeiu8_i8i4_realquant`
- `llvm-objdump -d --mcpu=gfx11-generic best.hsaco` → 3116 instructions
- Resources: **LDS 42080 B/WG** (one WG/CU, LDS-limited occupancy),
  **VGPR 110** (no spills), **SGPR 26**, wave32, no scratch.

## Instruction mix (static)

| Category            | Count | %     |
|---------------------|-------|-------|
| **VALU** (v_*)      | 2577  | 82.7% |
| SALU / control (s_*)| 275   | 8.8%  |
| **LDS** (ds_*)      | 200   | 6.4%  | (65 load / 135 store)
| WMMA (v_wmma)       | 56    | 1.8%  | (40 iu8 conv0 + 16 iu4 conv1)
| **global_load**     | **7** | 0.2%  |
| global_store        | 1     | <0.1% |

s_waitcnt: 68 total — **lgkmcnt 58 vs vmcnt 13**.

## Interpretation

1. **Not memory-bandwidth-bound.** Only **7 global_loads** and 13 vmcnt waits in
   the entire kernel. Footprint + W0 + W1 loads are coalesced into a handful of
   wide transfers (`global_load_b64` ×6, `b128`), exactly what `batch_loads`
   intends. There is no global-memory traffic structure that could dominate
   wall-clock. HBM bandwidth is not the limiter.

2. **VALU-dominated (82.7%).** The hot VALU is integer bit-manipulation, not
   WMMA math (WMMA is only 1.8%). The dominant opcodes are the
   quantize / int4-pack / maxpool epilogue:
   - **int4 pack / nibble assembly:** `v_lshrrev_b32` 257, `v_bfe_u32` 192,
     `v_and_b32` 163, `v_lshlrev_b32` 145, `v_add3_u32` 128, `v_or3_b32` 102,
     `v_and_b16` 126, `v_bfe_i32` 96, `v_or_b16` 50 — bitfield extract/shift/mask/merge.
   - **maxpool (2×2):** `v_max_i32` 128, `v_max3_i16` 48, `v_cmp_lt_i32` 117,
     `v_cndmask_*` 67+63+20.
   - **quant/clamp:** `v_min_u32` 192, `v_sub_nc_u32` 128.
   The frag loads themselves are free bitcasts (confirmed earlier:
   `_load_frag_iu8_from_lds`, `_load_frag_iu4_packed_from_lds`), so this VALU is
   genuinely the arithmetic of ReLU→requantize→nibble-pack→pool, not unpack overhead.

3. **Secondary: LDS latency, not global latency.** lgkmcnt waits (58) far exceed
   vmcnt (13). The 135 `ds_store` + 65 `ds_load` are the conv0→conv1 scatter and
   int4 output staging. With one WG/CU resident (LDS-limited), these LDS
   round-trips are the wait points feeding the VALU pipe — consistent with a
   compute/control-bound kernel whose stalls are on-chip (LDS), not off-chip (HBM).

4. **Control flow is light.** Only 7 `s_cbranch_execz` + a few `execnz` (boundary
   predication), 3 `s_barrier`. So "control-bound" in the branch sense is minor;
   the "compute-bound" reading is the accurate half of the hypothesis.

## Recommended next lever (data-supported)

Attack the **integer-VALU epilogue**, since it is 82.7% of static instructions
and the WMMA/global paths are already cheap:

1. **int4 pack via byte-permute.** The `lshr/lshl/and/or3/bfe` chain assembling
   nibbles is the single largest opcode cluster (~1000 ops). Try `v_perm_b32` /
   packed `v_pk_*` to fold two int4 results per op instead of shift-mask-or.
2. **maxpool via packed-int max.** Replace the `v_max_i32` + `v_cmp_lt` +
   `v_cndmask` pattern (≈300 ops) with `v_max3_i16` / `v_pk_max_i16` over packed
   lanes where the 2×2 reduction allows.
3. **fold quantize clamp.** `v_min_u32` (192) + `v_sub_nc_u32` (128) suggest a
   clamp expressed as min+sub; a single `v_med3` or `v_pk` clamp may halve it.

Each must be A/B'd interleaved on the board (`compare_configs.py`) and re-verified
bit-exact (`--verify` → `max_abs_diff=0`) at toy / multi-CTA / full, with an ISA
diff confirming the VALU count actually dropped.

## Caveat

Static counts ≠ dynamic cycles. But the mix is so lopsided (2577 VALU vs 7
global_load) that the qualitative verdict is robust: this is a compute-bound,
integer-VALU-limited kernel, with LDS (not HBM) latency as the secondary stall.

## Lever 2 result: pk_maxpool (2026-06-06)

Implemented `--pk-maxpool` (spec flag `pk_maxpool`, default off): widen conv1 i8
codes to i16 and reduce the 2x2 maxpool with `vector.smax`, now lowered via the
`llvm.smax.v<N>i16` intrinsic (lower_llvm.py `_op_vector_smax`) instead of
`icmp sgt`+`select`, plus a new `vector.sext` op (ir.py / lower_llvm / lower_hip).

- **Static (gfx11-generic):** VALU 2633->2540 (-93, -3.5%), TOTAL 3116->3043
  (-73), SALU 275->295 (+20). The maxpool reduction now emits 24x `v_max_i16`
  (hardware int max) instead of scalar cmp/select.
- **NOT packed:** `v_pk_max_i16` count is still 0. The backend keeps the i16
  lanes scalar because each came from a per-lane `sext` of i8 LDS data (one i16
  per 32-bit VGPR); `v_pk_max_i16` needs two i16 co-located in one VGPR. Reaching
  the true packed form requires storing conv1 codes as packed 2xi16 end-to-end
  ("option b"), or explicit inline-asm packing (overhead-gated).
- **Board A/B (interleaved, full 2160x3840, 3 rounds):** base median 14.261 ms,
  pk median 14.073 ms -> **pk ~1.3% faster**; every pk run beat every adjacent
  base run, spread <0.4% within pk. Bit-exact at toy (0/3072), multi-CTA
  (0/24576), full (0/49766400).
- **Takeaway:** -3.5% static VALU -> +1.3% wall-clock (sub-linear), confirming
  maxpool VALU is partly on the critical path but the kernel is partly
  latency/overlap-bound. A perfect packed maxpool would likely add <=~1% more,
  bounded by the same conversion ratio.
