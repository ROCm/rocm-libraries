# gfx942 GQA head-fold — case study

**Shape.** bf16, `head_size=128`, GQA 32 query heads / 8 KV heads (`num_queries_per_kv = 4`),
sliding window, paged KV, `block_size <= 32`. Production builder
`build_gfx942_4warp_gqa` (`kernels/gfx942/attention_tiled_2d.py`), non-lean D128 branch.

**Result.** KV is read once per KV head instead of once per query head.
**No regression across 24 measured points on two gfx942 parts**; ~4-5% at the
long sequence lengths that dominate prefill, more at short ones (full table in
§4). The kernel is numerically **exact** against the unfolded one.

Reproduce: `python prefill/gqa_head_fold_bench.py`
Algorithm write-up: `ALGORITHM.md` §6.4.

---

## 1. The problem

Without the fold, one workgroup owns **one query head**:

```
grid = (num_query_heads, total_q // 128 + num_seqs, 1)      # block_m = 128
```

The 4 query heads sharing a KV head are 4 *separate* workgroups. Each streams the
same paged K and V from HBM (High Bandwidth Memory — the off-chip DRAM). The same
bytes cross the memory interface up to 4 times, with reuse only by luck of L2
residency.

This was not the first hypothesis. An earlier counter sweep on this same kernel
found every compute and occupancy counter unremarkable, with a single outlier:
L2 misses. That is what redirected the work from scheduling to traffic. (Prior campaign, recorded in the team vault under
`SDPA/gfx942-gqa-fold-hbm-win`; the numbers below are re-measured here for the
kernel this document describes.)

## 2. The change

Pack the 4 heads into one workgroup's 128-row M-tile — 32 tokens × 4 heads instead
of 128 tokens × 1 head:

```
row m  ->  token = qbase + m // FOLD_HEADS ,  head = kv_head * GQAG + m % FOLD_HEADS
grid   =  (num_kv_heads, total_q // 32 + num_seqs, 1)
```

with `FOLD_HEADS = 4`, `TOKBLK = 128 // FOLD_HEADS = 32`. The MFMA atom, the
softmax recurrence and the fp32 accumulator are untouched; only row ownership and
the addresses derived from it change. An inverse map recovers `m` at the O-store,
so the output layout is unchanged.

Parallelism survives the smaller token tile. At `total_q = 8192`, one sequence:

| | grid.x | grid.y | workgroups |
|---|---:|---:|---:|
| unfolded | 32 | 65 | 2080 |
| folded | 8 | 257 | 2056 |

The 4× cut in `grid.x` is repaid by the 4× rise in `grid.y`.

## 3. The mechanism, measured

gfx942 part A, ROCm 7.13, sq=8192, bs=16, 7 launches per arm, `rocprofv3 --pmc
TCC_MISS_sum TCC_HIT_sum`. Both arms are the same builder; the fold-off arm forces
`gfx942_gqa_fold_eligible` false.

| arm | TCC_HIT_sum | TCC_MISS_sum | L2 hit rate |
|---|---:|---:|---:|
| unfolded | 4.270e7 | 1.883e8 | 0.185 |
| **folded** | 1.008e8 | **1.182e8** | **0.460** |

L2 misses — the traffic that reaches HBM — fall **37%**, and the L2 hit rate more
than doubles. This is the intended mechanism, confirmed directly rather than
inferred from the wall clock.

Note it is a 37% cut, **not** the naive 4×. Two reasons: L2 already captured some
of the redundant re-reads in the unfolded kernel (hence its non-zero hit rate), and
Q/O traffic is unchanged by the fold.

## 4. Wall clock

`prefill/gqa_head_fold_bench.py`, bf16 D128 GQA 32/8, `sliding_window=4096`,
warmup 5 / 20 attempts, gfx942 part A, ROCm 7.13:

| sq | bs | nofold ms | fold ms | speedup |
|---:|---:|---:|---:|---:|
| 512 | 16 | 0.1165 | 0.0967 | 1.204× |
| 1024 | 16 | 0.2904 | 0.2497 | 1.163× |
| 2048 | 16 | 0.8466 | 0.7698 | 1.100× |
| 4096 | 16 | 2.8451 | 2.7025 | 1.053× |
| 8192 | 16 | 7.9091 | 7.5086 | 1.053× |
| 16384 | 16 | 18.0150 | 17.0694 | 1.055× |
| 512 | 32 | 0.1137 | 0.0937 | 1.213× |
| 1024 | 32 | 0.2891 | 0.2431 | 1.189× |
| 2048 | 32 | 0.8315 | 0.7523 | 1.105× |
| 4096 | 32 | 2.7812 | 2.6328 | 1.056× |
| 8192 | 32 | 7.7268 | 7.3399 | 1.053× |
| 16384 | 32 | 17.6341 | 16.7663 | 1.052× |

**No regression at any point.** The gain is largest at short sequences and settles
at ~5% from 4096 upward. Both block sizes benefit; `bs=16` is the busier gather
(`BPT = BN // BS = 2` block-table entries per 32-key tile vs 1 at `bs=32`).

**Part sensitivity is real and worth recording.** The same sweep on a second
gfx942 part, with a different memory topology, gives **+0.6% to +13.1%**, settling
at ~4.0-4.2% for long sequences instead of ~5.3%. A traffic optimisation depends
on the memory system it is relieving, so quote the part with the number, and lead
with the sustained long-sequence figure rather than the short-sequence peak. Two
independent runs on that second part agreed to within 0.1 points, so the
difference is the part, not noise.

## 5. Correctness

The fold repacks rows; it must not change any output value.

- **Exact against the unfolded kernel.** At sq=16384 (where the fp32 oracle would
  need a 16 GiB score tensor) fold and no-fold outputs agree with `max_abs = 0`.
- **Against the fp32 windowed paged-attention oracle:** `max_abs = 0.01562` at every
  measured point. That is exactly `2^-6`, one bf16 ulp for outputs in `[2, 4)` —
  i.e. the residual is the rounding of the final bf16 store, not accumulation drift.
  The gate is the repo-wide bf16 attention bound of `4e-2` (`ALGORITHM.md` §10),
  which this clears by 2.6×.
- Guards: `tests/test_gfx942_gqa_head_fold_numeric.py` (on-GPU oracle),
  `tests/dispatch/attention/test_gfx942_gqa_head_fold.py` (CPU emit/grid/predicate),
  and the golden-IR case `attention_d128_swa/gfx942/4warp_gqa_fold`, which is the
  only one of the three that runs on a host with no gfx942 GPU.

## 6. What lost

Recorded so the next person does not re-run them.

- **Five scheduling levers before the diagnosis.** Transposed-V, K-pairing, a
  within-tile waitcnt ladder, PV hoist and grid-reorder all moved instruction counts
  and **none moved wall clock** — because the kernel was traffic-bound, not
  schedule-bound. (Prior dense-kernel campaign; vault
  `SDPA/gfx942-gqa-fold-hbm-win`.)
- **A 2-head fold** netted approximately nothing in that campaign. The win needs the
  full 4-head fold plus the small token tile, not a partial fold.
- **`FOLD_HEADS` cannot simply be widened.** A 4:1 fold needs 4 × 32 = 128 rows,
  exactly the tile; 8:1 would need 8 × 32 = 256 rows, which the tile cannot hold.
  Widening the cohort means re-deriving the row split, so the builder **raises** on
  `GQAG != FOLD_HEADS` rather than silently corrupting addresses.
- **fp16 is untested, not disproven.** The fp16 MFMA atom has the same `32x32x8`
  geometry, so the fold would very likely work; it is excluded only because no fp16
  A/B exists. Widening needs a measurement plus an oracle case, not a predicate edit.

## 7. Caveats

- **C++ engine not covered by these runs.** In the kreb ROCm 7.13 container the C++
  lowerer rejects the `llvm23` flavor and falls back to the Python lowerer, so every
  number here validates the Python-lowered kernel. The C++ path needs a separate run
  on an `llvm20`/`llvm22` image. (The golden-IR case pins Python-lowered IR at all
  three flavors, so it does not close this either — C++/Python byte-identity is a
  separate gate.)
- **One head configuration.** Everything above is 32/8. The predicate admits only
  `num_queries_per_kv == 4`, so that is the whole shipped cohort, but it does mean
  the fold has never run at another ratio.

## 8. Reproduce

```bash
# wall clock A/B (both arms, correctness checked per point)
python rocke/library/builders/gfx942/attention/prefill/gqa_head_fold_bench.py

# numeric guardrail (needs a gfx942 GPU)
pytest rocke/library/tests/test_gfx942_gqa_head_fold_numeric.py -v

# structure + golden IR (no GPU required)
pytest rocke/library/tests/dispatch/attention/test_gfx942_gqa_head_fold.py -q
pytest rocke/platform/tests/test_rocke_ci_static.py -q
```
