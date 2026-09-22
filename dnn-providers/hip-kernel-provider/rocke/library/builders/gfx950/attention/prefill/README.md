# Dense flash-attention prefill (gfx950 / MI355X)

Productized dense causal flash-attention prefill kernel for gfx950, authored in the
rocke IR DSL. Forward-only, bf16/fp16, head_dim 64/128, MHA or GQA.

- Kernel: [`kernels/gfx950/attention_dense.py`](../../../../kernels/gfx950/attention_dense.py)
  (`Gfx950AttentionDenseSpec`, `build_attention_dense`, `supports_attention_dense`)
- Host builder / launcher (this dir): `attention_dense_prefill.py`

## Baked-in levers (always-on, no env gates)

- **CK-1 transposed PV** — P feeds the PV MFMA in its native QK-output layout via a
  half-local V load (`pv32_v_load_paired`); the cross-half P-relayout shuffle is gone
  (~96 `ds_bpermute` removed). +35% over the pre-CK-1 winner.
- **LDS bank-conflict padding on K** (`[NBUF, BN, D+8]`) — kills the 8-way conflict on
  the QK K-reads. The dominant base win (+80% over the naive baseline).
- **native `exp2_fast`** (`v_exp_f32`, no overflow guard; softmax argument is always
  `<= 0`).
- **full-population `sched_group_barrier` template** (DS_READ/MFMA/VALU/TRANS per PV
  step).
- **diagonal-only causal masking** — mask-free body over below-diagonal KV tiles
  (~94% at Sq=8192) plus a masked diagonal tail.
- **depth-1 cluster split** fusing exp2 into the PV MFMA loop for MFMA/VALU co-exec.
- **vectorized O store**.
- **qualified wide LDS DMA** — the aligned persistent D128/BN64 path can use two
  `buffer_load_dwordx4 ... lds` operations per operand/wave with 520/544-half
  slab-padded K/V layouts. IGLP-1 owns this loop schedule and K-major PV traversal
  keeps the 256-VGPR kernel spill-free.

Shape (batch / seqlen / heads / head_dim / causal / dtype) is baked at build time
(dense, statically-sized ABI). Tile/resource knobs are `block_n`, `waves_per_eu`,
and `lds_k_group_pad`; persistent scheduling knobs are `num_persistent`,
`persist_decode`, `interleave`, and `wide_lds_dma`.

## Persistent (grid-stride) mode

`Gfx950AttentionDenseSpec(persistent=True, num_persistent=256)` emits a persistent variant:
a 1-D grid of `num_persistent` long-lived CTAs grid-strides over the
`W = (seqlen_q // 256) * Hq * B` work items, so the per-CTA launch/dispatch + scalar
setup + K/V-prime cold-start is amortized once per CU instead of once per query-block.
This closes the causal fixed-cost amortization gap. `num_persistent=256` = one 8-wave
block per CU on MI355X (256 CUs) at 2 waves/SIMD; larger oversubscribes the CUs (tail
loss). The work-item decode is `persist_decode="auto"` by default. Auto selects:

1. **gqa-pair** when its one-phase balance equation
   (`NP == NQB*Hkv*B`, with even `NQB` and GQA ratio) holds;
2. **gqa-pair-2phase** when its two-phase balance equation
   (`NP == NQB*Hkv*B*gqa/2`, with even `NQB`) holds;
3. **hkv-major** when its broader GQA balance condition
   (`gqa*NQB*B >= 2*NP`) holds; or
4. **qb-major** otherwise.

### Balanced GQA-pair decode

`persist_decode="gqa_pair"` maps two neighboring CTAs to one
`(low_qb, high_qb, hkv, batch)` group. Each CTA handles half of the local query
heads at both complementary query blocks. The complementary blocks have constant
combined causal cost, while consecutive local query heads reuse the same K/V head
through L2.

The explicit mode requires persistent causal attention, aligned sequence lengths,
even `NQB` and GQA ratio, and `NP == NQB*Hkv*B`. It composes numerically with
`interleave`, attention sinks, and aligned sliding-window attention. Auto uses it
for aligned non-windowed causal shapes whenever the balance equation holds.
Sliding-window requests fall back to qb-major because complementary query blocks
do not have constant combined work under a finite window. MHA (`Hq == Hkv`) also
falls back because it has no grouped query heads to reuse.

`persist_decode="gqa_pair_2phase"` assigns one query head to each CTA and processes
complementary query blocks in two grid-stride phases. Auto selects it when
`NP == NQB*Hkv*B*gqa/2`; this covers, for example, the D128 Hq32/Hkv8 S4096 and
Hq64/Hkv8 S2048 dashboard shapes at NP=256.

### Wide LDS DMA

`wide_lds_dma=True` is the gfx950 D128/BN64 persistent fast path. It changes the
per-row 32-bit DMA layout into FlyDSL-compatible slabs:

- K: `8 × 2 × 520` half elements per tile (`+8` padding per 512-half slab);
- V: `8 × 2 × 544` half elements per tile (`+32` padding);
- two 128-bit-per-lane DMA instructions per operand/wave;
- 68,096 B total double-buffered LDS, down from 75,776 B;
- IGLP-1 instead of the narrow path's manual scheduling directives.

The dispatcher enables it for aligned persistent causal D128/BN64 fp16 and bf16
shapes without sliding windows, sinks, or ragged lengths. GQA-pair selection is
independent: MHA and GQA shapes whose pair equations do not hold still use wide
DMA with qb-major or hkv-major work ordering.

## Measured (MI355X, bf16, D=128, Hq=128, Hkv=8, causal, Sq=8192)

Absolute MI355X TFLOPS swing **±25–30% with auto-clock**, so only **same-session
ratios are load-bearing**; the table below is one representative session, with each
number pinned to its exact config (grid / decode / V-pad / lazy):

| config | grid | decode | V-pad | lazy | TFLOPS |
|---|---|---|---|---:|---:|
| default grid | one-CTA/q-block | — | 32 | on | ~543 |
| persistent baseline | persistent NP=256 | qb-major | 0 | off | ~877 |
| persistent + V-pad | persistent NP=256 | qb-major | 32 | off | ~912 |
| **persistent (shipped default)** | persistent NP=256 | **hkv-major** | 32 | on | **~948** |

The load-bearing, clock-invariant deltas: hkv-major vs qb-major ≈ **1.04×** (L2 hit
57%→~93%), V-pad 0→32 ≈ **+5%** (clears the transposed-PV bank conflicts), lazy ≈
**+2%**. The shipped default (`persistent=True`, `persist_decode="auto"`,
`lazy_rescale=True`, `lds_v_row_pad=32`) is the last row. All configs are 0 VGPR
spill and parity-identical vs `torch.nn.functional.scaled_dot_product_attention`
(max abs err ~1.46e-3 at Sq=8192).

## Measured Llama-3-8B target (MI355X, fp16)

For `B=1, Sq=Skv=8192, Hq=32, Hkv=8, D=128, causal`, five-round same-GPU
ABBA measurements repeated twice for the final wide-DMA+IGLP kernel:

- unchanged qb-major baseline: 750.2–752.6 TFLOPS median;
- gqa-pair + wide DMA + IGLP-1: 903.2–903.7 TFLOPS median;
- paired ratio: 1.20028–1.20110x;
- max absolute error: `1.92e-4`;
- resources: 256 VGPR, 68,096 B LDS, zero scratch/spills, 256 CTAs;
- static load-to-LDS instructions: 64→16; total ISA instructions: 2,239→2,041.

The gqa-pair step raised estimated L2 hit rate 32.0%→65.1%, halved fetch volume
934→463 MB, and raised MFMA utilization 32.8%→40.5%. The final ATT trace shows
wide DMA + IGLP reducing LDS-wait share 30.3%→7.6% and the mapped PV MFMA
callsite 19.4%→5.3%.

## Ungated dashboard sweep (MI355X)

The capability-gated policy was measured against the previous exact-shape policy
on the 90 unique dense shapes from the 2026-09-02 Solera export. Both runs used
the same n07 GPU and robust per-shape timing (five samples, each targeting at
least 100 ms, with the median reported):

- 90/90 shapes passed numerical parity; 53 used wide DMA and 17 used a balanced
  GQA-pair mapping;
- geometric-mean throughput improved **5.20%** over all 90 shapes and **8.88%**
  over the 55 shapes whose emitted kernel changed;
- the 16 shapes whose pair mapping changed improved **16.3%** geometric mean
  (the 11 D128/wide shapes moving from qb-major to a pair mapping improved
  **19.9%**); wide DMA with unchanged work ordering improved its 37-shape cohort
  by **6.08%**;
- wins against the dashboard FlyDSL values increased from 45/82 to 54/82;
- the largest gains were Hq32/Hkv8 S4096: fp16 **669→845 TFLOPS (+26.2%)** and
  bf16 **689→885 TFLOPS (+28.5%)**;
- no changed shape regressed by 1%; the only negative changed-shape delta was
  fp16 Hq128/Hkv8 S8192 at **-0.4%**, within run noise.

## Usage

```bash
# parity + benchmark (default shapes 256/512/2048/8192)
python attention_dense_prefill.py                 # default (one CTA per query-block/head)
python attention_dense_prefill.py --block-m 128   # explicit query-tile geometry
python attention_dense_prefill.py --persistent    # persistent grid-stride (NP=256)
python attention_dense_prefill.py --bn 128        # sweep block_n
python attention_dense_prefill.py --vpad 16       # explicit V-row LDS pad
python attention_dense_prefill.py --persistent --np 256 --interleave
python attention_dense_prefill.py --persistent --persist-decode gqa_pair
python attention_dense_prefill.py --persistent --persist-decode gqa_pair_2phase
python attention_dense_prefill.py --persistent --persist-decode gqa_pair --wide-lds-dma
```

Programmatic:

```python
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
)

spec = Gfx950AttentionDenseSpec(
    batch=1, seqlen_q=8192, seqlen_kv=8192,
    num_query_heads=128, num_kv_heads=8, head_size=128,
    causal=True, dtype="bf16",
    block_m=256, block_n=64, lds_v_row_pad=32,
    persistent=True, num_persistent=256,   # grid-stride persistent variant
)
kernel = build_attention_dense(spec)       # -> KernelDef; compile with backend="python"
```

Through the dispatcher (opt-in; picks the persistent best-config for large Sq):

```python
from dispatch.attention import AttentionRequest, dispatch_attention, dense_spec_for_request
from kernels.gfx950.attention_dense import run_attention_dense_torch

req = AttentionRequest(
    batch=1, nhead_q=128, nhead_k=8, seqlen_q=8192, seqlen_k=8192,
    hdim_q=128, hdim_v=128, arch="gfx950", dtype="bf16", mask_type=1,
    algorithm="attention_dense",   # opt-in; "auto" keeps the unified 2D/3D path
    # dense_persistent="auto"      # "auto"|"on"|"off"; auto => persistent for large Sq
    # dense_persist_decode="auto"  # also accepts gqa_pair / gqa_pair_2phase
)
res  = dispatch_attention(req)                 # res.spec.kernel_name() -> ...persist256_hkvmaj
spec = dense_spec_for_request(req)             # launch-ready best-config AttentionDenseSpec
run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=1/128**0.5)
```

`dense_persistent="auto"` turns on the persistent grid-stride variant once there is
enough work to fill the grid (`⌈Sq/256⌉·Hq·B >= num_persistent`) — i.e. the large-Sq
prefill regime — so the dispatcher reaches the persistent path, not the default
grid. Aligned causal D128/BN64 shapes enable wide DMA/IGLP; auto then chooses a
balanced pair mapping when its CTA-count equation holds. The kernel name exposes
the decisions through `wdma`, `gqapair`, or `gqapair2` tokens. Callers may also
request either pair mapping explicitly.

## Tuning — lds_k_group_pad

`Gfx950AttentionDenseSpec.lds_k_group_pad` pads each K row-group in LDS to break
bank conflicts on the `do_qk` K reads. The pad must be a multiple of 8:
`smem_load_vN` stamps align 16 unconditionally, so an 8-byte-aligned pitch
preserves `ds_read_b128` alignment. `__post_init__` enforces the invariant.

**Sweep axes.** `pad ∈ {0, 8, 16, 24, 32}` × `block_n ∈ {64, 128}` × three
GQA ratios (Hq=128/Hkv=8, Hq=64/Hkv=8, Hq=32/Hkv=32) × four modes (causal,
swa W=512/1024/2048/4096, varlen, persistent) × three sequence lengths
(S=2048/4096/8192) — 840 total configs, all correctness-passed.

**Bank-conflict model.** A whole-wave view of the K reads treats all 64 lanes
simultaneously. At D=64 with a 2-row-group boundary, the 32-way conflict at
pad=0 drops to a 4-lane-per-bank floor at pad=8. Pad values 8, 16, and 24 all
reach this floor: the 16-lane phase that could distinguish them (pad=8 yields 16
distinct start banks, pad=16 repeats each twice) is too weak to produce a
reliable separation in a single forward pass. This model predicts that 8, 16, and
24 are statistically indistinguishable on most shapes, which the sweep confirms.

**Decision.** Default stays at 8. No shape-conditioned override is added.

pad=24 is the plurality winner (77 shapes vs 65 for pad=8) but gain magnitudes
are 0.001–3.22%, with the majority under 1%. Without an ABBA repeat sweep,
differences at this scale are within run-to-run noise; the original pad=8
selection used ABBA specifically because sub-1% differences are unreliable in a
single forward pass. The 77-shape plurality is not a strong enough signal to
change the default.

pad=16 has the largest single-shape win in the sweep but is directly
contradicted by a large regression on a superficially similar shape at
bn=128 sliding-window. That cliff makes pad=16 unsafe as either a global
default or a shape-conditioned override until the regression is understood.
**The pad=16 regression at S=4096 W=4096 bn=128 Hq=32/Hkv=32 is recorded as a
constraint for future work on this knob.**

pad=0 and pad=32 are consistently worst across all shapes, confirming the
original sweep result.

## TODO / follow-ups

1. **Two-tile pipeline remains register-blocked.** A true Q-reload reduced the
   target kernel from 248 to 231 VGPR, but repeated global-load/address work
   dropped performance to ~398 TFLOPS. `NBUF=3` reached 256 VGPR, 85 spills,
   344 B scratch, and ~288 TFLOPS. Neither experiment ships.
2. **Improve varlen performance** — the packed `varlen` path (default builder,
   `cu_seqlens_q/kv`) currently trails the dense/persistent path and flyDSL on
   ragged batches. Follow-ups: extend the hkv-major L2-locality decode and the
   persistent grid-stride mode to varlen (today persistent is uniform-batch only),
   and improve per-sequence load balance across CTAs so short and long sequences
   in a batch don't serialize.

## Notes

- gfx950-only (uses `ds_read_b64_tr_b16` and `v_exp_f32`).
- Compiles through the rocke LLVM-direct (`backend="python"`) path. The `exp2_fast`
  op is mirrored in the **C++ engine** and covered by a `backend="both"`
  byte-identity gate (`tests/test_attention_ir_cpp_parity.py::
  test_attention_ir_cpp_python_byte_identity`) — the Python and C++ lowerings
  are byte-for-byte identical across every dense variant.
