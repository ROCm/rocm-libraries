# Lightning indexer causal-skip: measured A/B

## Scope

This document records a single optimization A/B for the DSA lightning indexer: a
causal-skip that runs zero head iterations for output work that is entirely
causally masked (a future key in the scalar body, a fully-future 16x16 output
block in the MFMA body). Its purpose is to record the measurement method and the
relative result so the keep/revert decision is reproducible. All numbers here are
relative ratios; absolute measurements live only in the protected internal
record.

## What changed

Both bodies set the per-head loop's trip count to zero when the work is fully
masked, so the per-head dot (scalar) or the H_I MMAs and per-head Q loads (MFMA)
are skipped and the sentinel is written by the existing causal select. The output
is bit-for-bit identical to the no-skip baseline; the skip only avoids computing
values that were already the masked sentinel.

## Method

- Baseline: the parent commit (runtime-seqlen indexer, no causal-skip).
- Arm: the same tree with the causal-skip.
- Harness: `library/benchmarks/common/dsa/benchmark_indexer.py`, median of 5 timed
  invocations with min-max spread, verification on.
- Correctness gate (both arms, every shape): the benchmark grades scores against
  the numpy oracle in `builders/common/dsa/hostpack.py` at bf16 input with a 1e-3
  relative tolerance; every run reported `ok`.
- Hardware target: gfx942. dtype bf16. n_index_heads 64, index_head_dim 128.
- Provenance: run both arms with the command block in the PR description; the
  arm is the parent commit with the kernel change stashed.

## Result (relative to the no-skip baseline)

| Shape | q_pos_base | Body | Causal-skip vs baseline |
|---|---|---|---|
| seqlen_q 2048, seqlen_k 2048 | 0 (first-chunk / short-context prefill) | mfma | faster at this configuration |
| seqlen_q 512, seqlen_k 32768 | 0 (DIAGNOSTIC, not production) | mfma | much faster at this configuration |
| seqlen_q 1, seqlen_k 8192 | tail (decode) | scalar | within run-to-run spread (neutral) |

The `512 x 32768, q_pos_base=0` row is a diagnostic configuration, not a
production shape: it scores early queries against a long cache so almost every key
is future, which exaggerates the skip. Real chunked prefill at that cache length
runs with `q_pos_base` near the tail, where the future region is only the small
within-chunk triangle and the skip is near-neutral. It is listed to show the skip
fires and stays correct when the masked region is large.

## Decision: keep

The skip is a real speedup for first-chunk / short-context prefill
(`seqlen_q ~ seqlen_k`) and is within the measured spread (neutral) for decode and
for realistic long-context tail chunks. The added work is a few control-flow ops
and never computes a different result, so there is no measured regression. Kept on
that basis, with the caveat that the dominant DSA long-context-tail case sees
little benefit.
