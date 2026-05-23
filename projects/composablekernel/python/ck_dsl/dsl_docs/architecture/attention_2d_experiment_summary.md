# CK DSL Unified Attention 2D Experiment Summary

Date: 2026-05-21  
Scope: bf16 prefill-2D unified attention, primarily `d64_b32_h64kv8` with sinks on gfx950.

## Baseline

The production tiled-2D path already has an R4 variant in
`instances/attention_tiled_2d.py`:

```text
use_mfma_32x32=True
use_transposed_qk_32x32=True
block_m_per_warp=32
```

R4 is the baseline for performance comparisons.

Initial 142-shape cohort:

| Variant | Geomean latency vs Triton |
|---|---:|
| stock | 1.415x |
| R4 | 1.366x |

## Experiment Results

### R1+R4 Register-PV

Goal: remove the P→LDS round trip and keep P in registers.

Result: not a production candidate.

Findings:

- Naive R1+R4 regressed vs R4.
- The old 16x16 register-PV path removed LDS stores but introduced heavy cross-lane reshaping.
- The standalone `attention_tiled_2d_r1r4.py` fork became stale once useful ideas were moved back into `attention_tiled_2d.py`.

Action:

- Do not keep or select the R1+R4 fork.
- Keep optimizations only as opt-in flags in `attention_tiled_2d.py`.

### Transposed Scalar State and Mask Hoist

Flags:

```text
use_transposed_scalar_state
use_transposed_mask_once
use_transposed_invariant_hoist
```

Goal: reduce redundant transposed softmax state and repeated row/mask work.

Findings:

- Improves R4+sinks parity:
  ```text
  max_abs 0.0625 -> 0.015625
  ```
- Reduces generated LLVM IR and scalar operations.
- Does not improve latency alone.

No-SW 71-shape result:

| Variant | Geomean latency vs Triton |
|---|---:|
| R4 | 1.439x |
| R4_s1mask | 1.674x |

Action:

- Do not use `s1mask` alone as a performance optimization.
- Use it when paired with half-local PV, where it improves correctness and helps the combined variant.

### Half-Local PV With Tuned V Layout

Flag:

```text
use_transposed_half_local_pv
```

Goal: reduce P-side cross-lane exchange by making each 32-lane half consume P rows it already owns.

Implementation:

- Half 0 consumes K rows `{0..3, 8..11}`.
- Half 1 consumes K rows `{4..7, 12..15}`.
- V uses matching half-local `ds_read_tr16_b64` transpose reads.
- P and V use the same permuted K order, preserving dot-product correctness.

Target result:

```text
R4:                 0.5816 ms
R4_hlpv:            0.4757 ms
R4_s1mask_hlpv:     0.4848 ms
```

No-SW 71-shape result:

| Variant | Geomean latency vs Triton |
|---|---:|
| R4 | 1.439x |
| R4_hlpv | 1.412x |
| R4_s1mask_hlpv | 1.387x |

SW 71-shape result:

| Variant | Geomean latency vs Triton |
|---|---:|
| R4 | 1.298x |
| R4_hlpv | 1.274x |
| R4_s1mask_hlpv | 1.265x |

Action:

- Keep `R4_s1mask_hlpv` as the best current kernel variant.
- Use selector fallback for high-`num_seqs` SW tail shapes.

### Fast Paged-KV Descriptor

Flag:

```text
use_fast_paged_kv_desc
```

Goal: specialize paged KV address generation for the dominant shape:

```text
bf16, d64, b32, h64kv8, T=64, num_warps=4
```

Findings:

- Parity-clean in targeted validation.
- Helps larger rows slightly, but regresses smaller rows.
- Reduces code/resource size:
  ```text
  VGPRs: 140 -> 115
  SGPRs: 106 -> 44
  SGPR spills: 32 -> 0
  ```

Action:

- Keep as opt-in / selector-controlled.
- Do not enable blindly across all shapes.

### Fast Paged-KV + R4 Register-P Residency Probe

Experimental files:

```text
instances/attention_tiled_2d_fastkv_regp.py
examples/attention/benchmark_prefill2d_fastkv_regp.py
```

Goal: test whether R4 benefits from the fast paged-KV descriptor plus
register-resident P, while leaving the production tiled kernel and existing
benchmark scripts untouched.

Implementation note:

- The R4 transposed path already consumes P from `PT32_n` registers, but the
  generic builder still allocates the unused `P_lds` slab unless
  `use_register_pv` is set.
- The experimental builder wraps the current R4 spec with a proxy that skips the
  `P_lds` allocation and enables `use_fast_paged_kv_desc`.
- The benchmark appends a short shape hash to the emitted kernel symbol before
  compilation. The normal display name omits shape-specialization constants
  such as `num_seqs` / binary-search trip count, and ROCm HSACO/module caching
  can otherwise alias different specialized code objects that share a kernel
  symbol.

Full 142-shape result:

| Variant | Geomean latency vs Triton | Geomean latency vs R4 | Wins vs R4 | Wins vs Triton |
|---|---:|---:|---:|---:|
| R4 | 1.418x | 1.000x | - | 2/142 |
| R4_fastkv_regp | 1.738x | 1.226x | 1/142 | 2/142 |

Only one shape improved over R4:

```text
d64_b32_h64kv8_q1000_k1041_ns332_tq8192_sw0_sc0_sinks1_bfloat16
R4:             0.831933 ms
R4_fastkv_regp: 0.792376 ms
speedup:        1.050x
```

Action:

- Do not select `R4_fastkv_regp` broadly.
- If revisited, constrain it to explicit per-shape selection and rerun with
  shape-unique kernel symbols to avoid HSACO cache aliasing.

Follow-up combo hypothesis:

The most plausible composition was not plain R4, but the current combo stack:

```text
R4_s1mask_hlpv
+ fast_paged_kv_desc
+ skip_legacy_qreg
+ mask_limit on no-SW
+ register-P proxy that removes the unused P_lds allocation
```

Reasoning:

- In transposed R4, P is already materialized as `PT32_n` registers and the PV
  path consumes those registers directly.
- HLPV also consumes P from those registers; it changes the V/P K ordering, not
  the source residency.
- Therefore the register-P proxy composes mechanically with `s1mask`, HLPV,
  `skipqreg`, and fast paged-KV, but its remaining effect is only to remove the
  dead `P_lds` allocation.

Candidate-shape validation:

| Probe | Shapes | Geomean `combo_regp / combo` | Notes |
|---|---:|---:|---|
| 8-shape representative set | 8 | 1.022x | 3 tiny wins, one large no-SW regression |
| 4-shape longer recheck | 4 | 0.875x | order-sensitive outlier on `ns1/sw0` |
| 4-shape reversed order | 4 | 1.017x | wins mostly disappeared |

Conclusion:

- No robust evidence that removing `P_lds` helps once HLPV/combo is active.
- Apparent wins are sub-1% on most shapes and sensitive to run/order noise.
- Keep the current combo policy as-is; do not add a `combo_regp` selector path
  without a stronger occupancy/resource-driven reason and a full sweep.

### Early-V Schedule

Flag:

```text
use_early_v_schedule
```

Goal: issue the current V async copy immediately after the iter-start K
drain/barrier, before QK, so V can overlap with QK + softmax instead of only
softmax. The next-K prefetch remains after QK, preserving the existing partial
wait before PV that leaves next K pending.

Implementation:

```text
old:
  wait current K
  QK
  issue current V
  issue next K
  softmax
  partial wait current V
  PV

early_v:
  wait current K
  issue current V
  QK
  issue next K
  softmax
  partial wait current V
  PV
```

Full 142-shape result against the current combo stack:

| Variant | Geomean latency vs Triton | Geomean latency vs combo | Wins vs combo |
|---|---:|---:|---:|
| combo | 1.294x | 1.000x | - |
| combo_early_v | 1.255x | 0.970x | 79/142 |

Shape split:

| Subset | Shapes | Wins | Geomean `early_v / combo` |
|---|---:|---:|---:|
| no-SW | 71 | 70 | 0.933x |
| SW=128 | 71 | 9 | 1.009x |
| no-SW, `num_seqs >= 256` | 37 | 37 | 0.930x |
| SW=128, `num_seqs >= 256` | 37 | 5 | 1.008x |

ISA/resource check on
`d64_b32_h64kv8_q1000_k1041_ns332_tq8192_sw0_sc0_sinks1_bfloat16`:

```text
combo:       VGPR 142, SGPR 105, LDS 24576, K-loop wait/barrier 24
combo_early: VGPR 165, SGPR 105, LDS 24576, K-loop wait/barrier 20
```

Action:

- Promising for no-SW prefill shapes, especially larger `num_seqs`.
- Do not use for sliding-window shapes; the earlier V issue tends to regress SW
  by about 1% geomean.
- Candidate selector policy: use `combo_early_v` for no-SW eligible combo
  shapes; keep the existing combo/R4 fallback for SW.

### Sliding-Window T=32 Schedule

Variant:

```text
combo_t32
```

Goal: reduce wasted work on sliding-window shapes by using `T=32` instead of
`T=64`. For `SW=128`, this aligns the useful window with more granular KV
tiles and cuts masked-out KV columns/tail work. The fast paged-KV descriptor is
disabled on this variant because it is intentionally specialized to `T=64`.

Full SW 71-shape result:

| Variant | Geomean latency vs Triton | Geomean latency vs R4 | Wins vs R4 |
|---|---:|---:|---:|
| R4 | 1.447x | 1.000x | - |
| combo | 1.459x | 1.008x | - |
| R4_t32 | 1.367x | 0.945x | - |
| combo_t32 | 1.168x | 0.807x | 71/71 |

Shape split:

| Subset | Shapes | Geomean `combo_t32 / R4` | Geomean `combo_t32 / Triton` |
|---|---:|---:|---:|
| SW, `num_seqs < 64` | 10 | 0.799x | 1.111x |
| SW, `64 <= num_seqs < 256` | 24 | 0.824x | 1.238x |
| SW, `num_seqs >= 256` | 37 | 0.798x | 1.139x |

Combined best measured policy:

```python
if sliding_window == 0:
    use combo_early_v
else:
    use combo_t32
```

Full 142-shape result:

| Policy | Geomean latency vs Triton | Wins vs Triton |
|---|---:|---:|
| no-SW combo_early_v + SW combo_t32 | 1.114x | 20/142 |

Action:

- `combo_t32` is the best measured SW path so far and should replace the prior
  SW R4/combo fallback in experimental selectors.
- Remaining gap to Triton is still mostly SW mid-batch shapes, but the gap is
  much smaller than with T=64.

### Skip Legacy Q Register Gather

Flag:

```text
use_mfma32_skip_legacy_qreg
```

Goal: skip unused legacy 16x16 Q register gather in the 32x32 path.

Findings:

- Parity-clean.
- Removes one prologue wait/barrier pair.
- Runtime win is sub-1%.

Action:

- Safe to include in the measured combined policy.
- Not meaningful alone.

### AGPR Residency

Flag:

```text
use_agpr_alloc_zero
```

Goal: force VGPR-form MFMA / avoid AGPR moves.

Findings:

- Backend support works in micro-probes.
- It removes AGPR moves in some legacy 16x16 shapes.
- Current R4/R4_s1mask_hlpv target already has zero AGPR moves, so it does not improve the current best path.

Action:

- Keep backend/probe infrastructure.
- Do not enable by default for current R4_s1mask_hlpv.

### Grouped KV2 Online Softmax

Flag:

```text
use_grouped_kv2_softmax
```

Goal: process two KV tiles before updating the running output accumulator.

Findings:

- Compiles and small smoke tests looked promising.
- Full 142-shape sweep regressed badly.

142-shape result:

| Variant | Geomean latency vs Triton |
|---|---:|
| R4 | 1.366x |
| grouped-KV2 | 1.647x |

Action:

- Do not use grouped-KV2 in selectors, harnesses, or future benchmark comparisons unless explicitly re-investigating.

### SW-Prefill Specialized Wrapper

Goal: make a dedicated sliding-window prefill kernel path.

Findings:

- Correct, but not better than R4/stock overall.
- Loop unroll and skip-final-K scheduling experiments were parity-clean but slower.

SW 71-shape result:

| Variant | Geomean latency vs Triton |
|---|---:|
| R4 | 1.298x |
| SW-prefill wrapper | 1.372x |

Action:

- Do not select the SW wrapper.
- Use R4_s1mask_hlpv with high-`num_seqs` SW fallback instead.

## Current Best Policy

Best measured practical policy:

```python
if sliding_window > 0 and num_seqs >= 450:
    use R4
else:
    use R4_s1mask_hlpv
```

With additional opt-in stack in the parity harness:

```text
R4_s1mask_hlpv
+ fast_paged_kv_desc
+ skip_legacy_qreg
+ mask_limit for no-SW only
```

Full 142-shape composite:

| Policy | Geomean latency vs Triton |
|---|---:|
| stock | 1.415x |
| R4 | 1.366x |
| previous HLPV policy | 1.295x |
| combo no-SW + combo SW policy | 1.179x |

Equivalent CK speedup vs Triton:

```text
0.848x
```

## Current Gaps To Triton

After AGPR moves are no longer the primary issue for the current best path, remaining gaps are:

- VALU/mask/softmax work.
- Wait/barrier density.
- LDS/transposition overhead on SW tail cases.
- Descriptor/addressing overhead on some shapes.
- Selector policy quality.

## Recommendations

1. Keep `R4_s1mask_hlpv` and the measured combo policy as the current best experimental path.
2. Do not keep the standalone `attention_tiled_2d_r1r4.py` fork.
3. Do not use grouped-KV2.
4. Keep AGPR residency support as backend infrastructure, not a default attention option.
5. Move proven selector policy into `attention_unified.py` only after the benchmark harness policy is stable.
6. Continue using Triton-vs-current-best HSACO/ISA diffs to rank future work.
