# DSA lightning indexer

The gfx942 lightning-indexer forward, the scoring stage of DeepSeek Sparse
Attention (DSA), is provided by
[`lightning_indexer.py`](../../../library/kernels/gfx942/lightning_indexer.py).
The host drivers are in
[`library/builders/gfx942/dsa`](../../../library/builders/gfx942/dsa/).

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
I(t, s) = sum over heads h of  w_h * ReLU( index_q[t, h] . index_k[s] )
```

The index-query is per head (`n_index_heads` heads of width `index_head_dim`);
the index-key is shared across heads, one `index_head_dim` vector per token
(MQA-style), so it carries no head axis. The ReLU clamps each head's contribution
to be non-negative and `w_h` is a learned per-head weight. Scoring is causal: a
query only scores keys at or before its position; future keys are driven to a
sentinel so top-k never selects them.

## Specs and compositions

`IndexerTileSpec` owns the tiling knob (the scalar v1 has one: `block_size`, the
threads per query row). `IndexerSpec` owns the geometry: `n_index_heads`,
`index_head_dim`, `seqlen_q`, `seqlen_k`, and dtype. There is one build entry
point, `build_lightning_indexer`, plus `lightning_indexer_grid` and
`lightning_indexer_signature`. `Bq` / `Bk` and head-streaming appear with the
later MFMA score body, where the per-head query tile is the binding LDS resource.

## Numerical contract

The reference is a sparse oracle in
[`hostpack.py`](../../../library/builders/gfx942/dsa/hostpack.py)
(`ref_indexer_scores`), computed in numpy on bf16-rounded operands so it grades
the kernel on the same precision the hardware uses. Because the scores feed a
ranking, the meaningful check is that the induced order is preserved well enough
for selection, so the indexer test is read together with the top-k test rather
than on exact score values.

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

## gfx942 implementation

v1 is a scalar-reduction body: one workgroup per query row, threads stride the key
range, and each key's score is a straight FMA reduction over `index_head_dim`
followed by the ReLU, the per-head weight, the head sum, and the causal bound. It
allocates no LDS, so it fits the 64 KiB budget trivially; occupancy is register-
rather than LDS-bound. This is the correctness-first baseline; the MFMA score body
(reusing the 16x16x16 bf16 contract atom) is a later perf hoist under the
optimization runbook.

Two seams are kept so the fused indexer+top-k kernel is a swap, not a rewrite:
`_emit_score` is the only place a score reaches memory, and
`_load_index_q_elem` / `_load_index_k_elem` are the only places inputs are read.

## ABI

```
(index_q: ptr<bf16, global>,   # [seqlen_q, H_I, D_I], projected + RoPE'd
 index_k: ptr<bf16, global>,   # [seqlen_k, D_I], shared across heads
 w:       ptr<f32,  global>,   # [H_I] per-head indexer weights
 scores:  ptr<f32,  global>,   # [seqlen_q, seqlen_k] output score matrix
 q_pos_base: i32)              # causal base: pos(q) = q_pos_base + q
```

Grid `(seqlen_q, 1, 1)`; block `(block_size, 1, 1)`.

## Coverage

gfx942, bf16 only. fp8 (gfx950-first, OCP-native) and gfx950 are later phases.
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
[`benchmark_indexer.py`](../../../library/benchmarks/gfx942/dsa/benchmark_indexer.py)
sweeps `seqlen_k` through the same path; it prints timings and never writes
measured performance numbers into the repo.
