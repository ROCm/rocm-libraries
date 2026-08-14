# Dense flash-attention prefill (gfx950 / MI355X)

Productized dense causal flash-attention prefill kernel for gfx950, authored in the
rocke IR DSL. Forward-only, bf16/fp16, head_dim 64/128, MHA or GQA.

- Kernel: [`kernels/gfx950/attention_dense.py`](../../../../kernels/gfx950/attention_dense.py)
  (`AttentionDenseSpec`, `build_attention_dense`, `supports_attention_dense`)
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

Shape (batch / seqlen / heads / head_dim / causal / dtype) is baked at build time
(dense, statically-sized ABI). `block_n` (KV tile) and `waves_per_eu` are the only
performance knobs.

## Persistent (grid-stride) mode

`AttentionDenseSpec(persistent=True, num_persistent=256)` emits a persistent variant:
a 1-D grid of `num_persistent` long-lived CTAs grid-strides over the
`W = (seqlen_q // 256) * Hq * B` work items, so the per-CTA launch/dispatch + scalar
setup + K/V-prime cold-start is amortized once per CU instead of once per query-block.
This closes the causal fixed-cost amortization gap. `num_persistent=256` = one 8-wave
block per CU on MI355X (256 CUs) at 2 waves/SIMD; larger oversubscribes the CUs (tail
loss). The work-item decode is `persist_decode="auto"` by default: it resolves to the
**hkv-major** L2-locality decode when balance-safe (GQA and `gqa*NQB*B >= 2*NP`, e.g.
128/8 at Sq=8192) and falls back to **qb-major** otherwise.

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
`lazy_rescale=True`, `ROCKE_DENSE_VPAD=32`) is the last row. All configs are 0 VGPR
spill and parity-identical vs `torch.nn.functional.scaled_dot_product_attention`
(max abs err ~1.46e-3 at Sq=8192).

## Usage

```bash
# parity + benchmark (default shapes 256/512/2048/8192)
python attention_dense_prefill.py                 # default (one CTA per query-block/head)
python attention_dense_prefill.py --persistent    # persistent grid-stride (NP=256)
python attention_dense_prefill.py --bn 128        # sweep block_n
python attention_dense_prefill.py --persistent --np 256 --interleave
```

Programmatic:

```python
from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

spec = AttentionDenseSpec(
    batch=1, seqlen_q=8192, seqlen_kv=8192,
    num_query_heads=128, num_kv_heads=8, head_size=128,
    causal=True, dtype="bf16",
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
    # dense_persist_decode="auto"  # "auto"|"qb_major"|"hkv_major"
)
res  = dispatch_attention(req)                 # res.spec.kernel_name() -> ...persist256_hkvmaj
spec = dense_spec_for_request(req)             # launch-ready best-config AttentionDenseSpec
run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=1/128**0.5)
```

`dense_persistent="auto"` turns on the persistent grid-stride variant once there is
enough work to fill the grid (`⌈Sq/256⌉·Hq·B >= num_persistent`) — i.e. the large-Sq
prefill regime — so the dispatcher reaches the ~948-TFLOPS path, not the default grid.

## TODO / follow-ups

1. **Q-reload (enables the inner-loop unroll)** — the QK B-operand (`q_packs`,
   `K_STEPS × vec8` ≈ 32 VGPR) is held resident across the whole KV loop. ISA
   probing shows the kernel is at **243/256 VGPR, 0 spill** (flyDSL fits a *2-tile*
   pipeline in ~252 VGPR), so a depth-2 software-pipeline unroll — the lever that
   would lift MFMA utilization ~45% → ~70% and close the flyDSL gap — is
   register-blocked. Re-reading Q per use (L2-resident) instead of holding all
   packs frees ~32 VGPR toward that unroll. (Pairs with a P-LDS publish for the
   lagged relayout, another ~16 VGPR.) Standalone it is perf-neutral; it is a
   prerequisite for the 2-tile unroll.
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
