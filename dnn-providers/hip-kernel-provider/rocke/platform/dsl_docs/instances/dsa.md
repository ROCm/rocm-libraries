# DSA lightning indexer

The lightning-indexer forward (gfx942 and gfx950), the scoring stage of DeepSeek
Sparse Attention (DSA), is provided by the arch-neutral
[`lightning_indexer.py`](../../../library/kernels/common/lightning_indexer.py).
The host drivers are in
[`library/builders/common/dsa`](../../../library/builders/common/dsa/).

## Role in the DSA pipeline

DSA replaces one dense attention pass with three stages: the lightning indexer
scores every allowed key against the query, top-k selection keeps the highest k
(2048) keys, and MLA attention runs over only that subset. This instance is the
first stage. It is score-only: it computes a relevance score per key and emits a
score matrix, with no softmax and no value output. The scores feed top-k, so what
matters downstream is the ranking they induce, not their exact magnitudes.

The projection, RoPE, and index-key cache write are caller-side (the reference's
fused qk-norm-rope-quant op); this kernel consumes an already projected and
RoPE'd index-query and an already populated index-key cache.

## Score

For a query position t and key position s:

```
I(t, s) = sum over heads h of  w[t, h] * ReLU( index_q[t, h] . index_k[s] )
```

The index-query is per head (`n_index_heads` heads of width `index_head_dim`);
the index-key is shared across heads, one `index_head_dim` vector per token
(MQA-style), so it carries no head axis. The ReLU clamps each head's contribution
to be non-negative and `w[t, h]` is a learned per-query-token, per-head weight
(signed, shape `[seqlen_q, H_I]`). Scoring is causal: a
query only scores keys at or before its position; future keys are driven to a
sentinel so top-k never selects them.

## Specs and compositions

`IndexerSpec` owns the geometry (`n_index_heads`, `index_head_dim`, `seqlen_q`,
`seqlen_k`, dtype) plus the `body` selector (`"scalar"` | `"mfma"`);
`IndexerTileSpec` owns `block_size` (256 for the scalar body, one wave64 = 64 for
the MFMA body). There is one build entry point, `build_lightning_indexer`, plus
`lightning_indexer_grid` and `lightning_indexer_signature`. Bq / Bk and
head-streaming would appear only with an LDS-staged MFMA variant; the current
register-only MFMA body needs none.

## Numerical contract

The reference is a dense numpy oracle in
[`hostpack.py`](../../../library/builders/common/dsa/hostpack.py)
(`ref_indexer_scores`), computed on bf16-rounded operands so it grades the kernel
on the same precision the hardware uses, and on signed per-query weights. The
on-GPU gate (the manifest `check`) requires every future (masked) key to be the
exact sentinel, every valid score to be finite, and the valid scores within a
tight relative tolerance -- so a dropped head or a broken causal mask fails. Once
a top-k stage lands it will add a selection-set (recall@k) check on top of this
exact-value gate, since the scores ultimately feed a ranking.

## Validation

Run the CPU admission/build checks from `rocke/library`:

```bash
python -m unittest tests.test_dsa_indexer_gfx942_spec
```

Run the on-GPU numeric check on a matching device:

```bash
python -m pytest tests/test_dsa_indexer_gfx942_numeric.py
```

The emitted IR is pinned by `tests/test_dsa_indexer_gfx942_golden.py` and by the
platform representative golden (`lightning_indexer/gfx942/*`).

## Implementation

Two bodies behind the `IndexerSpec.body` field, selected by two dispatch
candidates per arch:

- Scalar (`body="scalar"`): one workgroup per query row; threads stride the key
  range and each key's score is a straight FMA reduction over `index_head_dim`
  followed by ReLU, the per-head weight, the head sum, and the causal bound. No
  LDS. Grid `(seqlen_q, 1, 1)`. This is the correctness oracle and the fallback
  for non-16-aligned shapes and single-query (`seqlen_q=1`) decode. It is
  correctness-first, not a tuned decode path: one workgroup strides the whole key
  range and `seqlen_k` is compile-time (a fresh compile each step as the cache
  grows), so dispatch consumers should not treat it as a production decode kernel.
- MFMA (`body="mfma"`): per head, `scores_h = Q_h @ K^T` over the 16x16x16 bf16
  contract atom, register-to-register (score-only means no value matmul, so no
  P->LDS round-trip and no LDS at all). Each workgroup owns one (16-query,
  16-key) output block, so the launch parallelizes over both dimensions: grid
  `(seqlen_q/16, seqlen_k/16, 1)`, one wave64 per block. Requires 16-aligned
  `seqlen_q` / `seqlen_k` / `index_head_dim`; unaligned requests fall to the
  scalar candidate. This is the fast prefill path.

Both run on gfx942 (CDNA3) and gfx950 (CDNA4): the 16x16x16 bf16 atom and the
register-only body exist on both, so the kernel is arch-neutral and the two arches
differ only in the lowering target (and their own goldens).

Two seams are kept so the fused indexer+top-k kernel is a swap, not a rewrite:
`_emit_score` is the only place a score reaches memory, and the input loads are
the only place inputs are read.

## ABI

```
(index_q: ptr<bf16, global>,   # [seqlen_q, H_I, D_I], projected + RoPE'd
 index_k: ptr<bf16, global>,   # [seqlen_k, D_I], shared across heads
 w:       ptr<f32,  global>,   # [seqlen_q, H_I] per-query per-head weights (signed)
 scores:  ptr<f32,  global>,   # [seqlen_q, seqlen_k] output score matrix
 q_pos_base: i32)              # causal base: pos(q) = q_pos_base + q
```

Block `(block_size, 1, 1)` (256 scalar, 64 mfma). Grid is body-dependent (above).

## Coverage

gfx942 and gfx950, bf16. fp8 (gfx950-first, OCP-native) is a later phase.
Sparsity is a long-context feature, so benchmark shapes sweep `seqlen_k` well
above the selection size; below it the whole context is selected and DSA reduces
to dense MLA.

## Dispatch

`library/dispatch/dsa/` (`dispatch_lightning_indexer`) in its own registry,
separate from unified attention: the DSA op string is admitted only by the DSA
family's request-errors, so a DSA request and a standard-attention request can
never select each other's kernels. The Sk <= k dense-fallback routing is an outer
decision above this registry, not a gate here.

## Host drivers

Torch-free numpy: `hostpack.py` (inputs, bf16 packing, the score oracle) and
`manifest.py` (the `run_manifest` adapter that compiles, launches, and grades on a
gfx942 device). The benchmark
[`benchmark_indexer.py`](../../../library/benchmarks/common/dsa/benchmark_indexer.py)
sweeps `seqlen_k` through the same path; it prints timings and never writes
measured performance numbers into the repo.
