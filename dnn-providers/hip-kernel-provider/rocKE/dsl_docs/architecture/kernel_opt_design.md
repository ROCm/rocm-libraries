# Design: R1+R4 combined kernel for rocke unified_attention tiled-2D

**Status**: design proposal — implementation lives in rocke repo, not here.
**Target repo**: `rocm-libraries/dnn-providers/hip-kernel-provider/rocKE/Python/rocke/`
**Target shape (benchmark)**: `d64_b32_h64kv8_q1000_k1035_ns284_tq8192_sw0_sc0_sinks1_bfloat16`
**Target hardware**: gfx950 (MI350X), gfx942 (MI300X) as secondary.

## 1. Goal

Build the rocke tiled-2D `unified_attention` kernel variant whose K-loop matches
Triton's structural pattern:

```
1. load K tile          (ds_read_b64_tr_b16)
2. QK MFMA              (v_mfma_f32_32x32x16_bf16)   ← 32x32 atom
3. softmax              (v_permlane32_swap_b32 × 2)  ← in native 32x32 reg layout
4. load V tile          (ds_read_b128)
5. PV MFMA              (v_mfma_f32_32x32x16_bf16)   ← consumes P from registers
```

This is the union of two existing levers:
- **R4** (`use_mfma_32x32=True` + `use_transposed_qk_32x32=True`) — switches
  both QK and PV to the 32×32×16 BF16 atom.
- **R1** (`use_register_pv=True`) — eliminates the P → LDS round-trip by
  keeping P in VGPRs across the QK→PV chain.

Today they are mutually exclusive (validator
`attention_tiled_2d.py:274-279`).

## 2. Why each lever in isolation underperforms

From `analysis/2026-05-21-experiment-hsaco-structural-diff.md` and the cohort
re-bench (`results/rocke_ua_prefill2d_bf16_r4.csv`):

| variant                | target shape | cohort geomean ratio vs stock |
|------------------------|------------:|------------------------------:|
| Triton                 | 0.405 ms    | 0.71 (1.41× faster)           |
| rocke stock (16×16)   | 0.924 ms    | 1.00                           |
| rocke R1 (BMW=32)     | 0.946 ms    | 0.67× (49% slower)             |
| rocke R1 (BMW=16)     | 0.866 ms    | 0.76× (24% slower)             |
| rocke R4 (no sinks)   | 0.566 ms    | 0.97× geomean (3.5% faster, 39% target) |
| rocke R1+R4 (target)  | TBD         | TBD                             |

R1 alone with the 16×16 atom is structurally a net loss: it removes 32
`ds_write_b16`/iter but replaces them with **288** register-class permutes
(`192 ds_swizzle_b32 + 96 ds_bpermute_b32`), because the 16×16 atom's C-out
lane layout doesn't match its A-in lane layout, so P must be re-shaped before
each PV MFMA.

R4 alone shifts to the 32×32×16 atom (whose C-out lane layout is *natively*
the A-in lane layout for the next chained MFMA), but in the current rocke
code path it still feeds PV with a P that was staged for the 16×16 emit
contract. Net K-loop instruction count must be measured (see §6) but R4 alone
gives only +3.5% geomean on the cohort, suggesting structural overhead
remains.

Combining R1+R4 should give the Triton-style kernel: 32×32 atom + P kept in
VGPRs + no swizzle bus traffic.

## 3. Current code map (rocke)

### 3.1 Selector

`attention_unified.py`:
- `_enable_mfma_32x32(problem)` (line ~515)
- `_enable_transposed_qk_32x32(problem)` (line ~544): gates the 32×32 path on
  `dtype=bf16`, `head_size in {64, 128}`, `max_seqlen_q > 256 and num_seqs >= 2`,
  `not (softcap > 0 or use_sinks)`, `not (sliding_window > 0 and not use_fp8)`,
  `not (use_alibi or use_qq_bias)`, `not use_fp8`.
- `_enable_register_pv(problem)` (find via grep)
- `_select_2d_block_m_per_warp(problem)` (line ~617): empirically picks BMW=32
  for long-prefill no-SW bf16, BMW=16 elsewhere.

### 3.2 Spec validator

`attention_tiled_2d.py:274-279` (paraphrasing):
```python
if use_register_pv and use_mfma_32x32:
    raise ValueError(
        "use_register_pv with use_mfma_32x32 not supported: "
        "the 32x32 path has a separate register-P migration"
    )
```

### 3.3 Kernel emit (the relevant per-iter steps)

- **REGS_PER_LANE**: `attention_tiled_2d.py:319-320`
  - 16×16 atom: `REGS_PER_LANE = 4`
  - 32×32 atom: `REGS_PER_LANE = 16`
- **Sinks m_init**: `attention_tiled_2d.py:1080-1093` — iterates
  `REGS_PER_LANE`. Wiring is structurally in place for 32×32 (16-slot iter)
  but a parity bug exists with `use_sinks=True` (see §5).
- **P repack / PV input feed**: two code branches in the K-loop:
  - default (16×16): pack acc → AGPR readback → `ds_swizzle` → `ds_write_b16`
    → `ds_read_b16` (P-via-LDS)
  - `use_register_pv=True` (16×16): pack acc → AGPR readback →
    `ds_swizzle_b32` × 192 + `ds_bpermute_b32` × 96 (register-side reshape)
  - `use_mfma_32x32=True`: acc lives in VGPRs (no AGPR), but the PV feed path
    currently uses a structurally similar repack — needs HSACO confirmation (§6).
- **PV MFMA**: 16 `v_mfma_f32_32x32x16_bf16` per outer K-iter in the 32×32
  path (vs 32 `v_mfma_f32_16x16x32_bf16` in the 16×16 path).

## 4. Proposed implementation

### 4.1 Header diagnosis

The R4 path's 32×32 PV MFMA accepts its A-operand in the same lane layout
that the QK MFMA's C-out produces. This means a *direct register alias* is
sufficient — no `ds_swizzle`, no `ds_bpermute`, no LDS staging. This is the
specific property the 16×16 atom lacks.

Therefore the R1-for-32×32 emit is **simpler** than the existing R1-for-16×16
emit, not more complex:

| step                         | R1 / 16×16            | R1 / 32×32 (new)    |
|------------------------------|----------------------|--------------------|
| acc readback (AGPR → VGPR)   | 64 `v_accvgpr_read`  | 0 (acc already VGPR) |
| P scale (post-softmax)       | same                 | same                |
| P → PV-A lane reshape        | 192 swizzle + 96 bperm | **0**             |
| LDS staging                  | optional (R1 off)    | 0                   |
| PV MFMA feed                 | from reshaped reg or LDS | direct register alias |

### 4.2 Source change sketch

Three things change in rocke:

**(a) Spec validator** (`attention_tiled_2d.py:274-279`):
- Replace the unconditional error with a branch: allow
  `use_register_pv=True` together with `use_mfma_32x32=True` (the new path),
  keep the error only when 32×32 register-PV emit is requested under
  conditions it doesn't support yet (e.g. `softcap > 0`, sinks unfixed —
  see §5).

**(b) Kernel emit** (`attention_tiled_2d.py`, the body of the K-loop):
- Add a third branch in the "feed PV from P" block, parallel to the existing
  default and `use_register_pv && !use_mfma_32x32` branches:
  - **Precondition**: `use_register_pv && use_mfma_32x32 && use_transposed_qk_32x32`.
  - **Body**: omit the P-repack entirely. The P-tile registers from the QK
    accumulator (after softmax scale) are aliased directly as the A-operand
    for the PV `v_mfma_f32_32x32x16_bf16`.
  - **Register pressure**: with 32×32 acc + P held in VGPRs, the per-warp
    register footprint is dominated by:
    - Q_reg (already there: `Q_BYTES`)
    - 32×32 acc (~16 reg-slots/lane × 4 atoms = 64 VGPRs/warp)
    - P scratch (same 64 VGPRs, can alias acc post-softmax)
    - K, V staging
  - Expected ceiling: ~180-200 VGPRs/warp (vs 168 stock 16×16, vs 236 R1
    16×16, vs 150 Triton). Must verify against HSACO.

**(c) Selector** (`attention_unified.py`, `_enable_register_pv`):
- Currently presumed to gate on `not use_mfma_32x32` (mirroring validator).
- New behavior: when `_enable_transposed_qk_32x32` returns True, also enable
  `use_register_pv` for the same shape class (this is the new fast path).

### 4.3 What does **not** change

- Softmax row reduction: stays as the `v_permlane32_swap_b32` pattern used by
  the R4 path. (Triton uses 2/iter; rocke R4 should match.)
- Q-in-registers (R2): already implemented; no change.
- K, V LDS layout: no change.
- Accumulator initialization, sinks contribution: see §5.
- Validator gates for other features (alibi, qq_bias, fp8, softcap): unchanged.

## 5. Pre-requisite: fix R4 sinks-wiring bug

Currently in `tests/test_rocke_ua_mfma_32x32_sinks.py`, the R4-with-sinks run
on the target shape gives `max_abs_diff = 0.0625` with `mean = 5.3e-5` against
the 16×16 reference. This is too localized to be a generic atom bug — it's a
sink-row offset.

**Hypothesis**: the sinks m_init code at `attention_tiled_2d.py:1080-1093`
iterates `REGS_PER_LANE` slots and applies the sink contribution to slot 0
(or some fixed slot index that was correct for `REGS_PER_LANE=4` but is
misplaced for `REGS_PER_LANE=16`). The 32×32 atom has 4× more reg-slots/lane,
so a slot-0 hard-coded sink write only initializes 1/16 of the
softmax-denominator rows correctly.

**Confirmation experiment** (analysis-only, no source change):
- Dump R4-with-sinks HSACO on the target shape.
- Locate the m_init prologue (look for the sink VGPR write before the
  K-loop entry).
- Verify whether the sink value is written to all 16 reg-slots of the
  per-lane m_init tensor or only to a subset.

**Fix**: the sink contribution should be broadcast to all REGS_PER_LANE
slots of the m_init tensor regardless of atom variant. This is a one-line
fix in the rocke emit (a loop bound or a `tl.broadcast_to`-equivalent
shape, depending on how the m_init tensor is constructed).

This fix is a prerequisite to landing R1+R4 because:
1. The benchmark cohort is bf16 prefill-2D, ~50% of which have `has_sinks=True`.
2. R1+R4 without correct sinks wiring is a net loss across the cohort
   (the sinks shapes would have to fall back to 16×16+stock).

## 6. Pre-requisite analysis: R4 HSACO baseline

Before landing the source change, capture an R4 HSACO on the target shape and
run `src/stage3_extract_isa/compare_ua_hsacos.py` with the existing
`triton + rocke_stock` baseline. This is required to:

1. Confirm R4 already has 0 `ds_write_b16`/iter (expected) and 0
   `v_accvgpr_*`/iter (expected — 32×32 acc lives in VGPRs).
2. Measure the **current** R4 swizzle/bpermute count per iter. If it's
   already 0, then "R1+R4 combined" reduces to "fix R4 sinks" and the
   source change in §4.2 is unnecessary. If it's > 0, the source change is
   justified and the measured count gives the upper bound on what R1+R4
   removes.
3. Measure R4 VGPR/AGPR count to predict R1+R4 register pressure (target:
   stay below the 256-VGPR boundary that drops MI350X occupancy).

**Capture command** (uses existing infrastructure, no new code):
```bash
source setup_env.sh && source .venv/bin/activate
python tests/dump_rocke_ua_hsaco.py \
    --shape-signature d64_b32_h64kv8_q1000_k1035_ns284_tq8192_sw0_sc0_sinks1_bfloat16 \
    --enable-mfma-32x32 \
    --out tests/logs/r4_hsaco_dump
python src/stage3_extract_isa/compare_ua_hsacos.py \
    --label triton       tests/logs/triton_hsaco_dump/kernel_unified_attention_2d.hsaco \
    --label rocke_stock  tests/logs/regpv_pair/stock__...hsaco \
    --label rocke_r4     tests/logs/r4_hsaco_dump/...hsaco \
    --dump-loops
```

(`dump_rocke_ua_hsaco.py` may need a `--enable-mfma-32x32` flag added, or
the R4 monkey-patch from `experiments/bench_rocke_ua_r4_cohort.py` applied
before invocation. The latter is preferred — it doesn't augment kernel code.)

## 7. Validation plan

### 7.1 Parity (gates the kernel emit)

For each `(shape, BMW)` in the target shape × `{16, 32}`:
1. Reference: stock 16×16 path, `use_sinks` per shape.
2. Candidate: R1+R4 path, same `use_sinks`.
3. Pass if `max_abs_diff ≤ 2e-2` (matches existing R4 test tolerance).

Test harness: `tests/test_rocke_ua_register_pv_sinks.py` and
`tests/test_rocke_ua_mfma_32x32_sinks.py` already have the patch/revert
plumbing. A new test
`tests/test_rocke_ua_register_pv_mfma_32x32.py` should:
- Apply both R1 and R4 patches (lift sinks gate AND lift validator).
- Compare against stock 16×16 baseline.
- Run with `--bench` to get latency.

### 7.2 Latency (gates the rollout)

Cohort re-bench using existing harness:
```bash
python experiments/bench_rocke_ua_r1r4_cohort.py \
    --output results/rocke_ua_prefill2d_bf16_r1r4.csv \
    --shape-list /tmp/r4_shape_sigs.txt  # same 142 shapes as R4 cohort
```
(driver mirrors `bench_rocke_ua_r4_cohort.py`, with both patches applied.)

Pass criteria:
- Geomean ratio R1+R4 / stock ≥ 1.20 (cohort-wide ≥ 20% speedup).
- Target shape ratio R1+R4 / stock ≥ 1.50 (target ≥ 50% speedup, closing
  most of the gap to Triton).
- No shape regresses by more than 5%.

### 7.3 Microarchitectural (gates the design)

HSACO three-way compare R1+R4 vs Triton vs stock should show:
- `ds_write_b16` per iter: 0 (matches Triton)
- `ds_swizzle_b32` per iter: 0 (matches Triton)
- `ds_bpermute_b32` per iter: 0 (matches Triton)
- `v_accvgpr_*` per iter: 0 (matches Triton)
- `v_mfma_f32_32x32x16_bf16` per iter: 16 (matches Triton)
- VGPR count: ≤ 200 (vs Triton 150; rocke stock 168; R1 16×16 236)

## 8. Risk and mitigations

| risk                                                  | mitigation                                                                                              |
|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Register pressure pushes past 256-VGPR ceiling        | Test on target shape first; if it overflows, gate R1+R4 on BMW=16 (smaller tile, fewer per-lane regs). |
| 32×32 lane layout assumption wrong (P-out ≠ PV-A)     | HSACO compare in §6 settles this *before* writing kernel code. If false, abort design.                  |
| Sinks bug deeper than slot-broadcast                  | §5 confirmation experiment tells us before we touch kernel code.                                        |
| Sliding-window shape regression (R4 alone gave flat)   | R1+R4 may not help SW shapes; keep SW shapes on R4-only or stock. Validator can gate per-shape.         |
| Combined path interacts with softcap or alibi         | Existing validator gates remain — R1+R4 only enabled where R4 is currently enabled (no softcap, no alibi). |

## 9. Out of scope

- Other dtypes (fp16, fp8): R4 selector already restricts to bf16.
- `head_size ∉ {64, 128}`: existing selector gate.
- 1D unified attention (`backend="default"`): different kernel.
- AITER / vLLM integration: this is purely a rocke kernel-emit change.
- MI300X (gfx942): primary target is gfx950; gfx942 should work
  structurally but needs separate perf validation.

## 10. Suggested sequencing

| step | action                                              | gates             | who           |
|------|-----------------------------------------------------|-------------------|---------------|
| 1    | Run §6 R4 HSACO baseline + 3-way diff               | none              | this repo     |
| 2    | Diagnose §5 R4 sinks bug from HSACO                 | step 1            | this repo     |
| 3    | Fix R4 sinks (rocke one-line / one-loop)           | step 2            | rocke repo   |
| 4    | Re-run R4 cohort with sinks fix; verify parity      | step 3            | this repo     |
| 5    | Decide on R1+R4 based on R4-fixed cohort numbers    | step 4            | review        |
| 6    | If go: implement §4.2 (a)(b)(c) in rocke           | step 5            | rocke repo   |
| 7    | Parity test on target shape (§7.1)                  | step 6            | this repo     |
| 8    | Cohort re-bench (§7.2)                              | step 7            | this repo     |
| 9    | 3-way HSACO confirm (§7.3)                          | step 7            | this repo     |
| 10   | Promote validator gate (default-on for bf16 long-prefill) | step 8 + 9   | rocke repo   |

Step 5 is a real decision point — if R4-with-fixed-sinks already gives
+20% cohort geomean, R1+R4 may be unnecessary (or a tail optimization).
The current R4 number (3.5% cohort geomean) is depressed by the sinks
shapes falling back; the sinks-fixed number could be significantly higher
without R1.

## 11. References

- `analysis/2026-05-21-experiment-hsaco-structural-diff.md` — root cause and
  per-iter histograms
- `analysis/2026-05-21-experiment-triton-vs-rocke-design-gap.md` — design gap analysis
- `analysis/2026-05-20-experiment-rocke-ua-optimization-plan.md` — R1..R4 ladder
- `results/triton_ua_prefill2d_bf16.csv` — Triton baseline (142 shapes)
- `results/rocke_ua_prefill2d_bf16.csv` — rocke stock baseline (142 shapes)
- `results/rocke_ua_prefill2d_bf16_r4.csv` — rocke R4 cohort (142 shapes)
- `tests/test_rocke_ua_mfma_32x32_sinks.py` — R4 validator
- `tests/test_rocke_ua_register_pv_sinks.py` — R1 validator
- `experiments/bench_rocke_ua_r4_cohort.py` — R4 cohort driver
- rocke: `rocm-libraries/dnn-providers/hip-kernel-provider/rocKE/Python/rocke/instances/attention_unified.py`
- rocke: `rocm-libraries/dnn-providers/hip-kernel-provider/rocKE/Python/rocke/instances/attention_tiled_2d.py`
