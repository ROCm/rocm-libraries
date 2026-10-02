# rocKE attention on gfx1151 (Strix Halo): prefill and decode

Reducing Qwen3-8B inference latency on the Strix Halo iGPU by serving attention with two
rocKE kernels in place of the incumbents — `wmma_fmha_swapqk` for prefill instead of
vLLM's Triton Flash-Attention, and `paged_decode_splitk` for decode instead of AMD's HIP
`paged_attention_rocm` — with every improvement validated end-to-end in a production
vLLM v1 serving path rather than in isolated microbenchmarks alone.

---

## Headline

| | prefill — `wmma_fmha_swapqk` | decode — `paged_decode_splitk` |
|---|---|---|
| Replaces | vLLM Triton Flash-Attention (dense) | AMD HIP `paged_attention_rocm` |
| Shipped config | `qk_douter=False`, `gqa_fuse=4`, `k_lds`, `v_paged`, `block_n=64` | `d_lanes=16`, split-K to `_TARGET_CTAS=256` |
| Regime | compute-bound, ~5500 FLOP/byte | bandwidth-bound, ~4 FLOP/byte |
| Reported as | TFLOP/s | **GB/s** — see [§5.1](#51-why-decode-must-be-reported-in-gbs) |
| Isolated best | **1.57×** at S=1024, **1.14×** at S=8192 | **1.09×** at Sk=32768 |
| End-to-end | **1.073× TTFT** at a 30720-token prompt | **1.024× ITL** at ctx32k |
| Default | on (`ROCKE_MIN_SEQLEN=512`) | off (`ROCKE_PAGED_DECODE=1` to enable) |

**Together, on one Qwen3-8B request at batch 1 —** `generate(30720 prompt tokens, 304
output tokens)` takes **62.09 s** with both kernels gated off and **59.12 s** with both
on: a **1.0503×** speedup on the number a user actually waits for, and **10.83 → 11.06
generated tokens/s**. Across the context ladder the geomean is **1.0201×**. The two wins
are **additive to within 0.04 points** of the product of the two single-kernel speedups
through ctx16k; at ctx32k the product over-predicts by ~0.4 points
([§7](#7-both-kernels-together)).

> All e2e figures on this page are **batch 1**. This part serves one user at a time, so
> batching is not the operating point; where a batched number exists it is labelled.

> The prefill kernel is **~7.8× faster** than the config that shipped when this work
> started (163.2 → 20.9 ms at S=8192). That 7.8× is a *composed* figure spanning two
> measurement sessions. Each individual step below is measured within one interleaved
> session; the sessions overlap on a common config (`qk_douter=False, gqa_fuse=4, bn64`:
> 41.0 ms vs 37.7 ms) and agree to within the ~9–13% inter-session clock drift this box
> exhibits. Never quote a number stitched across sessions as if it were a single
> measurement.

---

## 1. The setup

**Hardware.** AMD Strix Halo, gfx1151, RDNA3.5, 40 CUs, wave32, 103 GiB unified LPDDR5X,
**32 MiB MALL**, 2 MB L2, 32 KB L0 per CU, 64 KB LDS per CU. ROCm 7.2.1,
torch 2.11.0+rocm7.2. Measured ceilings differ by access pattern: ~102 GB/s on the
prefill gather patterns, up to ~220 GB/s on the decode streaming reads.

**Model.** Qwen3-8B fp16, 36 decoder layers, GQA with 32 query heads / 8 KV heads,
head dim 128, causal, native context 32768, ~16.4 GB of weights.

**Baselines.** For prefill, vLLM's `_fwd_kernel` from `ops/triton_prefill_attention.py`
(dense Triton FA). For decode, AMD's HIP `paged_attention_rocm`, which is what
`RocmAttentionImpl` dispatches. For e2e, the whole vLLM v1 path with attention routed to
each.

**The prefill kernel.** `wmma_fmha_swapqk` computes `Sᵀ = K × Qᵀ` instead of `S = Q × Kᵀ`.
That puts the query index on `lane % 16`, so the C→A transpose of `P` before the `P×V`
GEMM becomes **register-local** (`permlanex16` + 2× `v_perm_b32`) — no LDS, no barrier.
On `v_wmma_f32_16x16x16_f16` that is a real structural advantage: 1019 inner-loop
instructions at 72% VALU density and zero barriers, against Triton's 3218 at 34% with
4 barriers and 1168 LDS ops per iteration.

**The decode kernel.** `paged_decode_splitk` is a **two-dispatch** split-K attention over
the paged KV cache: pass 1 has each CTA reduce a slice of the KV range into a partial
`(m, l, acc)` triple; pass 2 reduces the partials with a numerically-stable rescale. It
uses **no WMMA at all** — see [§5.2](#52-gqa-fusion-is-total-and-there-is-no-wmma).

---

## 2. How rocKE was integrated with vLLM

vLLM's backend auto-selection on this box resolves to `ROCM_ATTN`
(`RocmAttentionBackend` / `RocmAttentionImpl`). The rocKE backend is a **subclass of
that exact impl**, dropped in as
`vllm/v1/attention/backends/rocm_rocke_attn.py` and selected with
`VLLM_ATTENTION_BACKEND=ROCKE_GFX1151`. A vendored copy lives at
[`rocke/integrations/vllm/rocm_rocke_attn.py`](../../../../integrations/vllm/rocm_rocke_attn.py).

Because it subclasses the impl vLLM would have chosen anyway, the metadata builder, the
KV-cache shape and the KV-cache write path (`forward_includes_kv_cache_update = False`;
`do_kv_cache_update` is called by `attention.py`, not by `forward`) are all inherited
unchanged. **The only thing that differs is which attention kernel runs.** That is what
makes the A/B honest.

### 2.1 Prefill dispatch

`forward` checks eligibility, then per request:

1. `_full_prefill_layout` — every request must be a full prefill with zero cached
   context (`query_start_loc` diffs equal `seq_lens`).
2. `kv_cache_dtype in ("auto", "float16")`, `alibi_slopes is None`,
   `sliding_window == (-1, -1)`.
3. `seqlen >= ROCKE_MIN_SEQLEN` (512) and `seqlen % block_n == 0`.
4. A kernel compiles for this `(Hq, Hk, D, causal, block_n, k_lds, gqa_fuse, v_paged)`.

Anything that fails routes to Triton. Kernels compile **lazily** on first use (~90 ms)
and are cached on the full config tuple, so an env flip can never serve a stale binary.

### 2.2 Decode dispatch

A decode step is eligible when the batch is **pure decode** (one query token per
request), `kv_cache_dtype` is `auto`/`float16`, no ALiBi, no sliding window, and the KV
bytes touched exceed `ROCKE_PAGED_DECODE_MIN_KV_MIB` (default **32**, i.e. the MALL
size — see [§9.11](#911-decode-below-the-mall-boundary)). Otherwise the step falls
through to AMD's HIP kernel.

The kernel is **off by default** (`ROCKE_PAGED_DECODE=0`) because its win is confined to
long context; the gate above is what keeps it from being a regression everywhere else.

### 2.3 Every gate fails silently

That is the single most dangerous property of this integration, and it burned us
repeatedly (see [§10](#10-measurement-discipline)). The backend therefore exports
`ROCKE_STATS` counters — `swapqk`, `triton_slice`, `fallback`, `paged_v`,
`paged_v_declined`, `paged_decode`, `paged_decode_declined` — plus a
`ROCKE_DECODE_DECLINE` reason map, and **every benchmark asserts on them.** A run that
got faster while dispatching zero rocKE kernels is not a result; it is the incumbent
measured twice.

### 2.4 Knobs

All read once at import and folded into the kernel cache key.

| knob | default | effect |
|---|---|---|
| `ROCKE_MIN_SEQLEN` | `512` | prefill dispatch floor; raise past every prompt to get a matched Triton control |
| `ROCKE_K_LDS` | `auto` | stage K in LDS ([§8.3](#83-staging-k-in-lds-k_lds)) |
| `ROCKE_V_PAGED` | `1` | gather V from the paged cache ([§8.4](#84-paged-v--reading-v-straight-out-of-the-kv-cache)) |
| `ROCKE_V_TRANSPOSED` | `1` | legacy permute path; only reachable with `ROCKE_V_PAGED=0` |
| `ROCKE_BLOCK_N` | `auto` | KV tile width; 128 is retired ([§9.2](#92-block_n128--shipped-then-deleted)) |
| `ROCKE_GQA_FUSE` | `auto` | prefill head fusion |
| `ROCKE_GQA_FUSE_CAP` | `4` | upper bound on the above |
| `ROCKE_QK_DOUTER` | `0` | d-outer QK loop ([§8.1](#81-qk_douter--false--flip-one-default)) |
| `ROCKE_TRITON_MODE` | `slice` | which Triton path declined requests fall back to |
| `ROCKE_PAGED_DECODE` | `0` | enable the decode kernel |
| `ROCKE_PAGED_DECODE_MIN_KV_MIB` | `32` | decode KV-bytes floor; `0` routes unconditionally |

`d_lanes` is **not** an environment variable — it is a `PagedDecodeCfg` field defaulting
to 16 and folded into the kernel cache key. Changing it means editing the config, which
is deliberate: [§8.5](#85-d_lanes--16--a-decode-knob-with-an-interior-optimum) shows its
optimum is interior and shape-dependent, not something to sweep at runtime.

### 2.5 A correctness constraint that shaped the prefill design

The KV loop bound is `loop_stop = seqlen_k // block_n` — plain integer division, **the
tail is truncated, not masked**. So `seqlen_k % block_n == 0` is a *correctness*
requirement, not a perf preference. An early gate tested `seqlen % 32 != 0` while running
a 64-wide tile, so a 2080-token request silently attended to only 2048 keys. Both the
vLLM gate and the in-repo torch op now share the same picker and reject anything
non-divisible.

---

## 3. Prefill: isolated kernel results

Hq=32, Hk=8, D=128, causal, batch 1, fp16. `rocprofv3` dispatch duration, arms
interleaved A B A B, first rep discarded, **minimum across reps**.

### 3.1 The optimization chain

Session A — one interleaved session, all five arms, all five lengths. Times in ms.

| S | ① shipped default<br>`F=1, bn64, qk_douter=True` | ② `qk_douter=False` | ③ + `gqa_fuse=4` | Triton |
|---:|---:|---:|---:|---:|
| 1024 | 0.449 | 0.335 | **0.280** | 0.427 |
| 2048 | 3.924 | 2.320 | **1.199** | 1.853 |
| 3072 | 14.464 | 7.473 | **3.347** | 4.163 |
| 4096 | 30.088 | 15.671 | 7.983 | 7.108 |
| 8192 | 163.219 | 92.590 | 41.044 | 26.737 |

Session B — the `k_lds` head-to-head, one interleaved session. Times in µs.

| S | bn64 (= ③ above) | bn64 + `k_lds` | bn128 | Triton |
|---:|---:|---:|---:|---:|
| 1024 | 279.7 | **269.4** | 396.0 | 423.3 |
| 2048 | 1231.2 | **1112.8** | 1652.4 | 1795.4 |
| 4096 | 7484.4 | **4730.7** | 6579.7 | 6341.1 |
| 8192 | 37708.0 | **20936.7** | 28320.5 | 23945.4 |

### 3.2 Throughput, and why scaling was the real problem

| | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| swapqk (post-fusion, bn64) | 30.7 TF | 27.9 TF | 18.4 TF | **14.6 TF** |
| swapqk + `k_lds` | 31.9 TF | 30.9 TF | 29.1 TF | **26.3 TF** |
| Triton | 20.3 TF | 19.1 TF | 21.7 TF | 23.0 TF |

Read the *shape*, not the ratios. Going S=1024 → 8192 the work grows 64×, and:
bn64 grows **134.8×** (losing efficiency faster than it gains work), bn64+`k_lds` grows
**77.7×** (64× is the floor), Triton grows **56.6×** (it gets *more* efficient, because
its per-iteration cost is flat in S). `k_lds` does not make swapqk's scaling as good as
Triton's — it makes it good enough that swapqk's much lower constant factor is never
given back.

### 3.3 Final speedup vs Triton, shipped config

| S | swapqk + `k_lds` | Triton | speedup |
|---:|---:|---:|---:|
| 1024 | 269.4 µs | 423.3 µs | **1.57×** |
| 2048 | 1112.8 µs | 1795.4 µs | **1.61×** |
| 4096 | 4730.7 µs | 6341.1 µs | **1.34×** |
| 8192 | 20936.7 µs | 23945.4 µs | **1.14×** |

The `v_paged` change ([§8.4](#84-paged-v--reading-v-straight-out-of-the-kv-cache)) is
**kernel-neutral by design** — it costs +0.54%/layer in isolation (22703 → 22825 µs at
S=8192) and buys its entire win by deleting host-side work. It does not appear in this
table because it is not a kernel-throughput lever. Its in-kernel cost is also not
uniform: the paged gather costs ~10% at S ≤ 4096 but **wins ~16% at S=30720**, crossing
over near 8192.

**Correctness** is gated on every timed run: `max_abs_diff = 4.883e-04`, `bad = 0` at
tolerance 2e-2, bit-identical across every config, `block_n`, and `k_lds`/`v_paged` arm.

---

## 4. Prefill: end-to-end results

Qwen3-8B fp16, 36 layers, prefill TTFT. All arms **interleaved in one process**,
minimum of 3 reps, dispatch counters asserted, **all arms token-identical**.

### 4.1 The TTFT ladder, shipped config vs matched Triton control

From the combined sweep ([§7](#7-both-kernels-together), reproduced by
[`e2e_combined_bench.py`](e2e_combined_bench.py)), `prefill_only` arm, B=1:

| context | prompt tokens | Triton TTFT (ms) | swapqk TTFT (ms) | speedup |
|---|---:|---:|---:|---:|
| ctx2k | 2048 | 1596.3 | 1566.4 | 1.0191× |
| ctx4k | 4096 | 2988.4 | 2923.8 | 1.0221× |
| ctx8k | 8192 | 6266.1 | 6052.6 | 1.0353× |
| ctx16k | 16384 | 14485.5 | 13586.7 | 1.0662× |
| ctx32k | 30720 | 34107.2 | 31784.4 | **1.0731×** |

The win **grows with prompt length** because attention is O(S²) while the rest of the
layer is O(S) — its share of prefill rises with S, and so does the share of the win that
survives to TTFT.

### 4.2 The earlier three-arm sweep

An independent session isolating the `v_paged` step. `k1v1` = `k_lds` on with V handed
over as a per-layer `permute(1,2,0).contiguous()`; `k1vp` = `k_lds` on with V gathered
from the paged KV cache (the shipped default); `triton` = swapqk gate raised, same
backend, same plumbing.

**Graph mode** (ms, lower is better):

| S | k1v1 | k1vp | Triton | k1vp vs k1v1 | k1vp vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1024 | 862.3 | 863.6 | 872.3 | +0.15% | **−1.00%** |
| 2048 | 1664.4 | 1655.5 | 1684.2 | −0.53% | **−1.70%** |
| 4096 | 3302.3 | 3228.9 | 3281.2 | −2.22% | **−1.59%** |
| 8192 | 7049.2 | **6775.0** | 7022.0 | −3.89% | **−3.52%** |

**Eager mode** (ms):

| S | k1v1 | k1vp | Triton | k1vp vs k1v1 | k1vp vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1024 | 850.0 | 845.8 | 857.1 | −0.49% | **−1.32%** |
| 2048 | 1687.4 | 1679.0 | 1707.9 | −0.50% | **−1.69%** |
| 4096 | 3356.5 | 3270.5 | 3342.6 | −2.56% | **−2.16%** |
| 8192 | 7070.4 | **6817.1** | 7065.9 | −3.58% | **−3.52%** |

### 4.3 Why 1.14× in the kernel becomes 3.5% end-to-end

Attention is a *minority* of prefill on this part. At S=2112 it is ~5% of the round
(4 req × 36 layers × 2.35 ms ≈ 360 ms of 6500 ms), so even a 1.75× attention win predicts
only +2.4% e2e — which is exactly what the GQA-fusion e2e measurement showed (a null at
S=2112, +7.8% median at S=3968, matching a +7.4% arithmetic prediction).

**Anyone optimizing e2e prefill at short prompts on this part should be looking at the
GEMMs, not at attention.**

Earlier e2e checkpoints, for the arc: `k_lds` alone was worth **~815–823 ms of a ~7.1 s
TTFT at S=8192 (11.5%)** — larger than the isolated benchmark predicted. Sequence-
adaptive `block_n` was worth **6.0%** at 8192-token prefill before `k_lds` retired it.

---

## 5. Decode: design and isolated results

### 5.1 Why decode must be reported in GB/s

Prefill attention runs at roughly 5500 FLOP/byte — far above this part's ridge point of
~270 — so it is compute-bound and TFLOP/s is the meaningful rate. Decode attention runs
at roughly **4 FLOP/byte**: one query token against the whole KV range. It is
bandwidth-bound by two orders of magnitude.

The same measurement therefore reads two completely different ways:

| framing | value at Sk=32768 | reads as |
|---|---|---|
| TFLOP/s | 0.87 TF | **1.5% of peak** — a catastrophe |
| GB/s | 217 GB/s | **~99% of achievable** — essentially done |

The second is the truth. **Never publish a decode attention number as TFLOP/s**, and
never compare it against the prefill kernel's TF figures; they are not the same units of
merit.

### 5.2 GQA fusion is total, and there is no WMMA

Decode has a single query token, so `S = Q·Kᵀ` degenerates from a matmul to an **outer
product**. A `16×16×16` WMMA would have 15 of 16 rows idle. The kernel therefore uses
plain VALU FMAs and fuses **all four** query heads sharing a KV head into one CTA — not
`gqa_fuse=4` as a tunable, but total fusion as the only sensible shape, since the KV tile
is the entire cost and must be read once.

Each CTA owns one `(request, kv_head)` pair and a slice of the KV range. `d_lanes` lanes
cooperate on the head dimension; the partial dot products are combined with
`ds_swizzle_b32` butterflies. That single parameter sets both the swizzle count and the
register pressure — see [§8.5](#85-d_lanes--16--a-decode-knob-with-an-interior-optimum).

### 5.3 Numerical stability: a finite sentinel

The running max is initialised to `_NEG_BIG = -1e30`, not `-inf`. A split whose KV slice
is entirely masked would otherwise produce `exp(-inf − −inf) = exp(nan)`, and the pass-2
rescale would propagate the NaN into a request that was perfectly well-defined. With a
finite sentinel the empty split contributes `l = 0` and drops out of the reduction
arithmetically.

Measured accuracy against a float64 CPU reference: **7.6e-06** max abs error, against
AMD's HIP kernel at **1.04e-05** on the same inputs. The split-K path is *more* accurate
than the incumbent, not merely as accurate.

### 5.4 Isolated decode results

B=1, Hq=32, Hk=8, D=128, fp16, paged cache with `block_size=16`. Wall clock under HIP
graph replay amortised over `--iters` — **not** `rocprofv3` per-dispatch time, because
the rocKE arm is two dispatches to the incumbent's one and per-dispatch timing would hide
launch overhead the shipping path actually pays. Scripts `07` and `08` measure the same
pair independently and cross-check each other.

| Sk | KV bytes | HIP `paged_attention_rocm` | rocKE split-K | 07 says | 08 says | rocKE GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 16 MiB | 29.2 µs | 31.6 µs | 0.925× | 1.000× | 287–302 |
| 4096 | 32 MiB | 43.4 µs | 40.6 µs | 1.069× | 1.040× | 387–402 |
| 8192 | 64 MiB | 84.2 µs | 80.7 µs | 1.043× | 0.978× | 398–417 |
| 16384 | 128 MiB | 351.2 µs | 323.7 µs | 1.085× | 1.085× | 191–207 |
| 32768 | 256 MiB | 679.8 µs | 618.9 µs | 1.098× | **1.094×** | 197–217 |

Two things to read off this table:

- **The win starts at the MALL boundary.** Below ~32 MiB of KV the whole working set
  fits in the 32 MiB MALL, both kernels are cache-resident, and the result is a wash or
  a small loss. Above it both stream from LPDDR5X and the split-K structure pays.
- **The GB/s column is not monotonic** — it peaks in the 4k–8k band where the MALL still
  helps and falls back to ~200 GB/s once the working set is fully DRAM-resident. That
  ~200 GB/s is the real streaming ceiling, and the kernel is at it.

The two scripts disagree by up to 6.5 points at Sk ≤ 8192 (where the effect is ~0) and
agree to 0.4 points at 16k/32k (where it is real). That disagreement pattern is itself
the evidence: the small-Sk numbers are noise around a null.

---

## 6. Decode: end-to-end results

Qwen3-8B, B=1, `decode_only` arm against a matched control (same backend, decode gate
off so AMD's HIP kernel runs). **ITL measured as a slope**, `(t_hi − t_lo)/256` from two
measured generation lengths in the same session — see
[§10](#10-measurement-discipline) rule 14. Same sweep as §7, reproduced by
[`e2e_combined_bench.py --ctx-sweep`](e2e_combined_bench.py).

| context | ITL, HIP (ms) | ITL, rocKE (ms) | speedup |
|---|---:|---:|---:|
| ctx2k | 71.534 | 71.417 | 1.0016× |
| ctx4k | 72.987 | 72.798 | 1.0026× |
| ctx8k | 75.760 | 75.505 | 1.0034× |
| ctx16k | 81.853 | 81.119 | 1.0090× |
| ctx32k | 92.309 | 90.106 | **1.0244×** |

**The isolated 1.094× becomes 2.4% of ITL, and that is arithmetic, not disappointment.**
Fit the control column: `ITL ≈ 69.96 ms + 7.26e-4 ms/token`. The intercept is everything
that is not attention — ~16.4 GB of weights streamed every single step. KV at 30720
tokens is ~4.5 GB against that, so attention is **2.1% of ITL at ctx2k and 24.2% at
ctx32k**. It closes: 36 layers × (679 − 619 µs) ≈ 2.16 ms off a 92.3 ms step; measured
92.309 → 90.106, a 2.20 ms reduction.

The 7.26e-4 ms/token slope implies **203 GB/s** of KV streaming inside the real model,
which agrees with the isolated rig's ~200 GB/s — the kernel does in vLLM what it does on
the bench.

**Batching is where this kernel pays**, because it amortises the weight streaming that
dominates the intercept: at B=32 the same kernel is worth 1.015× ITL. That is not this
part's operating point, and is recorded only so the B=1 numbers above are not mistaken
for the kernel's ceiling.

Counters asserted on every run: `swapqk=0`, `triton_slice=36`, `paged_decode=10908`
(= 36 layers × 303 decode steps), `paged_decode_declined=0`, `ROCKE_DECODE_DECLINE`
empty.

---

## 7. Both kernels together

This is the only measurement that answers "how much faster is my model?". One `LLM`
instance, four arms patched as module attributes between `generate()` calls, all
interleaved in a single session, **token-identical across arms** under greedy decode.
Reproduced by [`e2e_combined_bench.py --ctx-sweep`](e2e_combined_bench.py).

| arm | `_SWAPQK_MIN_SEQLEN` | `_PAGED_DECODE` |
|---|---|---|
| `stock` | `10**9` (→ Triton) | `0` (→ AMD HIP) |
| `prefill_only` | `512` | `0` |
| `decode_only` | `10**9` | `1` |
| `both` | `512` | `1` |

`stock` is a **matched control** — same rocKE backend class, same plumbing, both dispatch
floors raised past every request. It isolates the kernels, which is the point; it is not
byte-identical to stock vLLM with a different backend selected.

**Total request latency, `generate(P, 304)` at B=1**, milliseconds:

| context | P | `stock` | `both` | **speedup** | `prefill_only` | `decode_only` | product |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctx2k | 2048 | 23261.4 | 23212.7 | **1.0021×** | 1.0012× | 1.0012× | 1.0023 |
| ctx4k | 4096 | 25125.4 | 24977.2 | **1.0059×** | 1.0025× | 1.0035× | 1.0061 |
| ctx8k | 8192 | 29231.9 | 28897.7 | **1.0116×** | 1.0077× | 1.0034× | 1.0112 |
| ctx16k | 16384 | 39340.5 | 38142.4 | **1.0314×** | 1.0247× | 1.0069× | 1.0318 |
| ctx32k | 30720 | 62094.5 | 59118.5 | **1.0503×** | 1.0433× | 1.0104× | 1.0542 |

Geomean across the ladder: **1.0201×**.

**The two kernels are additive through ctx16k.** The `both` column and the product of the
two single-kernel columns agree to within **0.04 points** at ctx2k–ctx16k. There is no
interaction term to model: the prefill kernel does not change what decode reads, and
decode does not change what prefill wrote.

At **ctx32k the product over-predicts by ~0.4 points** (1.0542 against a measured
1.0503) — the two wins sub-add slightly at the longest context. That is ~8% of a
5-point win, and it is the one row on this page where the composition is not clean. It
is reported rather than explained; the mechanism is not established.

**Generation rate**, the number a user perceives during streaming:

| context | `stock` | `both` |
|---|---:|---:|
| ctx2k | 13.98 tok/s | 14.00 tok/s |
| ctx8k | 13.20 tok/s | 13.26 tok/s |
| ctx32k | 10.83 tok/s | **11.06 tok/s** |

**ctx2k is a negative control and it behaves like one.** Both kernels are predicted to do
nothing at 2048 tokens — the prompt is short enough that prefill attention is a rounding
error, and the KV working set is inside the MALL. It reports 1.0021× and 13.98 → 14.00
tok/s. An artifact that reports ≈0 where ≈0 is expected is what makes the ctx32k row
credible.

**There is deliberately no single combined scalar.** Total latency is
`≈ TTFT(P) + (G−1)·ITL`, and that reweights the two kernels by an order of magnitude
across the ladder — at ctx32k the win is 1.0433 prefill × 1.0104 decode, so it is
overwhelmingly a prefill win at B=1. Quote a context length, never an average.

### 7.1 Why the sweep runs under PIECEWISE cudagraphs

Switching arms in-process requires the kernel choice not to be frozen in a replayed
graph, and the two phases are **not symmetric**:

- **Prefill** arm-switching works under any mode. FULL capture only applies to uniform
  pure-decode batches; ragged prefill always takes the piecewise path and runs eagerly.
- **Decode** arm-switching requires `PIECEWISE`. Under a FULL mode the decode subgraph is
  captured once and the backend's Python never runs again — the arm freezes **and the
  dispatch counters freeze at zero**, so a counter assertion fires falsely in exactly the
  configuration you most want to test.

Prior FULL-vs-PIECEWISE runs agreed to within 0.4 points, which is the basis for trusting
the PIECEWISE ratio as the headline.

---

## 8. Optimizations that worked

### 8.1 `qk_douter = False` — flip one default

The shipped `SwapQKCfg` default was `qk_douter=True`, tuned at L=16384 / H=24 / D=128
MHA where it was documented as +3.3%. On the Qwen3-8B GQA shape it is a **loss
everywhere**, and the documented +3.3% does not reproduce even at its own shape:

| shape | `True` (shipped) | `False` | cost of the default |
|---|---:|---:|---:|
| Hq24 MHA S2048 dense | 19.68 TF | **20.66 TF** | −4.7% |
| Hq32/Hk8 GQA S2048 causal | 8.78 TF | **13.66 TF** | **−35.7%** |

Confirmed in `SQ_BUSY_CYCLES` with `SQ_WAVES` matched (identical launch geometry):
215.6M → 127.5M on the GQA shape, a **1.71×** penalty. In the S-sweep it is worth
1.34–1.94× depending on length.

The planned fix was "make it shape-aware." The measurement said the *sign* is not
shape-dependent — it loses at every shape tried. The fix was one line: flip the default.

### 8.2 GQA head fusion (`gqa_fuse=4`)

Process all F=4 query heads that share a KV head **in one CTA**, so each K/V tile is
fetched once and consumed four times. `block_size = 32 * n_waves * gqa_fuse` widens the
CTA to 256 threads while `q_rows_per_cta` stays fixed — the point is sharing, not more
work per CTA.

| S=2048 | F=1 | F=4 | Triton |
|---|---:|---:|---:|
| duration | 2354 µs | **1343 µs** (1.75×) | 1853 µs |
| `SQ_BUSY` | 126.4M | 65.4M (1.93×) | — |
| `TA_TA_BUSY` | 25.3M | 12.7M | 11.0M |
| VGPR / scratch | 216 / 0 | 216 / 0 | 256 / 168 B |

`SQ_WAVES` is identical across all arms — same launch geometry, so this is the kernel.
The gap to Triton on the texture-address path (`TA_TA_BUSY`, the discriminator between
the two implementations) closes from **2.30× to 1.16×**. It was thought this would need
a d-blocked epilogue to fit under the 256-VGPR wave32 ceiling; it did not — VGPR stayed
at 216 with zero scratch at F=1, 2 and 4.

**Fusion is a constant factor, not a scaling fix.** It is a 1.46×/1.40× win at
S=1024/2048 but a 0.82×/0.59× *loss* versus Triton at S=4096/8192 (crossover ≈ 3.3K),
because the super-linear growth survives it. That is what motivated §8.3.

Two things fell out of this work: a latent **causal-trim bug** (the kv-tile stop omitted
the `q_block` factor, silently dropping a kv block at `q_block>1` — a wrong answer, not
a slow one), and a **scheduler sign flip** — `sched_mode=pingpong` helps 7% when fused
and hurts 7% when unfused. A scheduling knob's sign is not invariant across a structural
change; re-sweep it, do not inherit it.

### 8.3 Staging K in LDS (`k_lds`)

The diagnosis first. At long S, swapqk was not limited by occupancy (24 waves/CU vs
Triton's 16), not by spills (swapqk spills nothing; **Triton** spills 168 B and wins
anyway), not by serialization (12.8% vs Triton's 26.4%, zero barriers), not by bank
conflicts (zero for both), and **not by DRAM bandwidth** — Triton sits at the ~102 GB/s
roofline while swapqk stalls 40% below it.

It was **L0 vector-cache thrash**. swapqk used no LDS by design, so every WMMA operand
came straight from the memory hierarchy and reuse depended entirely on the 32 KB L0.
The cleanest evidence: three configs at S=8192 executing an **identical** number of
vmem load instructions (33.8M / 33.8M / 34.1M) but differing **5.2× in L2 requests and
3.1× in wall time**.

The fix stages **only K** in LDS; V stays on `v_transposed` + `buffer_gather`. Per tile:

```
s_barrier                 # previous tile's readers are done
coop load K -> LDS        # whole CTA, 4 chunks/thread, global -> VGPR -> ds_write
s_barrier                 # tile visible to all waves
QK:  ds_read K, WMMA      # 2x ds_read_b128 + vec_concat per fragment
PV:  buffer_gather V      # UNCHANGED
```

Result at S=8192: total L2 requests fall **59.0M → 17.5M (3.37×)** for identical work,
and wall time follows at **2.07×**. `GL2C_MISS` falls 3.2× alongside `GL2C_HIT` — the CU
stops *asking* L2 for data it already has. A pre-registered kill criterion (≥2.4× fewer
L2 requests **and** ≥1.15× wall time, else the thesis is falsified) cleared with margin.

Cost: **+1 VGPR, 0 scratch, 17 KB LDS.** Instruction count goes *up* 7.4% — this is a
composition win, not an instruction-count win, and must not be presented as one. vmem
instructions drop 2.7× and the per-CTA K request count drops 512 → 32 for the same
16 KB of unique bytes, because half of the old K requests were pure address-pipe waste
(the A-operand row is `lane % 16`, so lanes *l* and *l+16* issue identical addresses).

The bank pad `_KVPAD = 8` is **derived, not tuned**: row stride `(128+8)` f16 = 68
dwords, `68 mod 32 = 4`, so the 8 lanes serviced per `ds_read_b128` pass tile all 32
banks exactly once. Measured `SQC_LDS_BANK_CONFLICT = 0` against 287M `LDS_IDX_ACTIVE`.

Two consequences beyond the speedup: swapqk now beats Triton at **every** measured
length — the 3.3K crossover is gone — and `block_n=128` is **retired**
([§9.2](#92-block_n128--shipped-then-deleted)).

### 8.4 Paged V — reading V straight out of the KV cache

After `k_lds`, swapqk won 1.14× in isolation but only **tied** Triton end-to-end. The
whole gap was one line in the backend:

```python
v_t = v.permute(1, 2, 0).contiguous() if cfg.v_transposed else v
```

The kernel's `P×V` A-fragment wants V as `[Hk, D, Sk]` — token fastest-varying — so the
backend materialised a transposed copy **every layer, every forward**. Measured cost:

| S | ms/layer | GB/s | MB of V | ms/forward (36 layers) |
|---:|---:|---:|---:|---:|
| 1024 | 0.088 | 47.4 | 2.10 | 3.2 |
| 2048 | 0.413 | 20.3 | 4.19 | 14.9 |
| 4096 | 1.950 | 8.6 | 8.39 | 70.2 |
| 8192 | **5.798** | **5.8** | 16.78 | **208.7** |

Note the bandwidth *degrades* 47.4 → 5.8 GB/s as S grows — ~18× below what the fabric
can do — because the gather stride is `Sk` elements. That is ~209 ms of a ~7.1 s TTFT.

**The insight: the transpose is already done.** vLLM's paged V cache is *already stored
transposed*. `PagedAttention.split_kv_cache` (from `vllm.v1.attention.ops.paged_attn`)
yields `[num_blocks, num_kv_heads, head_size, block_size]` — token fastest-varying,
exactly the order the A-fragment wants. `reshape_and_cache` performs the transpose
during `unified_kv_cache_update`, which runs *before* attention with an explicit data
dependency. **We were paying 5.8 ms/layer to redo work the cache write already did.**

New kernel knobs `v_paged` + `kv_block_size` gather V through the block table:

```
element offset = blk*(Hk*D*bs) + kv_head*(D*bs) + d_col*bs + tok
  voffset = d_col * bs * 2                                  (per-lane, VGPR)
  soffset = (blk*(Hk*D*bs) + kv_head*(D*bs) + tok) * 2      (uniform, SGPR)
```

Two facts made this cheap rather than invasive. `block_size` is always a multiple of 16,
so a 16-key WMMA A-fragment **never straddles a physical block** — the gather keeps its
exact 2× `dwordx4` shape. And paging is *better behaved* than the old addressing: the
per-lane term shrinks from `d_col*seqlen_k` (up to 2.08 MB, a runtime multiply) to
`d_col*bs*2` (≤ 4064 B, a compile-time-constant multiply).

**Result:** the gather costs **+0.54%/layer** at S=8192 (and *wins* ~16% at S=30720)
against ~209 ms of permute deleted. End-to-end that is the TTFT ladder in
[§4.1](#41-the-ttft-ladder-shipped-config-vs-matched-triton-control), with **no
crossover** — neutral at short prompts, and the win grows with S because the permute is
O(S). Shipped **on by default** (`ROCKE_V_PAGED=1`) after the sweep. GPU numerics: 12/12
PASS over B ∈ {1,2} × S × `block_size` ∈ {16,32,64} with shuffled, poison-filled pages.

Two traps this exposed, both worth remembering:

- **The paged gates fail silently into the *permute path*, not into Triton**, so the
  `swapqk` counter cannot see them. `ROCKE_STATS` gained `paged_v` /
  `paged_v_declined` and the harness asserts on them — the very first smoke run was a
  silent fallback that only the assertion caught.
- **`to_sgpr_u32` (readfirstlane) on the raw block id is mandatory, not an
  optimization.** AMDGPU treats every `addrspace(1)` load as divergent, so without it
  the backend wraps every V load in a **32-iteration waterfall loop**. It is invisible
  in the load counts — only the `readfirstlane` count guards it, which is why there is a
  CPU-only regression test on exactly that count.

### 8.5 `d_lanes = 16` — a decode knob with an interior optimum

`d_lanes` sets how many lanes cooperate on the 128-element head dimension. It trades two
costs in opposite directions:

```
ds_swizzle per key = d_lanes * log2(d_lanes) / (kv_block_size / 2)
VGPRs              = 2 * gqa_fuse * head_size / d_lanes
```

More lanes means fewer registers per lane (better occupancy) but more butterfly
reduction steps. The optimum is **interior**, which is the interesting part — it is not
"as many as possible" or "as few as possible":

| `d_lanes` | swizzles/key | VGPR | waves/SIMD | verdict |
|---:|---:|---:|---:|---|
| 32 | 20 | 115 | 4 | occupancy is fine; the swizzles dominate |
| **16** | **8** | **152** | **3** | **won at all 11 measured points** |
| 8 | 3 | 172 | 2 | cheapest reduction, but occupancy collapses |

16 won at **every one of 11 configurations**, not on average — there was no point at
which either neighbour was preferable. Moving the default from 32 to 16 **doubled the
batched e2e win** (to 1.027× at B=32).

**VGPR allocation is invisible in LLVM IR** — it is decided by the backend register
allocator after IR is emitted. The table above was obtained by probing the compiled
`.hsaco` metadata. Reasoning about occupancy from IR-level register counts would have
picked the wrong value.

### 8.6 `num_splits` — a measured target, not a derived one

Split-K needs a split count. The natural derivation is "one CTA per CU", i.e. 40. The
measured optimum is **`_TARGET_CTAS = 256`** — about **6.4 CTAs per CU** — and it held
across a full B ∈ {1..32} × splits ∈ {1..64} sweep: every optimum in the grid landed at
whatever split count produced 256 total CTAs, regardless of batch.

The kernel therefore solves for splits rather than fixing them:
`splits = ceil(_TARGET_CTAS / (B * Hk))`, clamped to the KV range. At B=1 that is
256/8 = 32 splits.

Oversubscribing by 6.4× is what keeps the memory system busy through the tail of each
CTA's KV slice; one-CTA-per-CU leaves the fabric idle whenever a CTA reaches its
epilogue. This is a case where the hardware-derived number was simply wrong and only the
sweep found the right one.

---

## 9. Optimizations that were expected to help and did not

### 9.1 `v_transposed = False` — remove the permute the obvious way

The natural way to kill the §8.4 permute: tell the kernel to consume row-major V. It is
a **wash**, and it was measured *before* the paged idea existed.

The kernel falls back to the row-major V gather — **16× `buffer_load_f16_d16` per
fragment instead of 2× `dwordx4`** — and the two costs nearly cancel, with a crossover
around S=4096:

| S | e2e delta vs `v_transposed=True` (graph / eager) |
|---:|---|
| 1024 | ~0 / ~0 |
| 2048 | −4 / −7 ms |
| 4096 | −46 / −49 ms (row-major wins) |
| 8192 | **+67 / +50 ms** (row-major loses) |

Kernel-only A/B confirms the mechanism: at S=8192 row-major costs **+8.34 ms/layer**
(21921 → 30259 µs, 25.08 → 18.17 TF) ≈ +300 ms over 36 layers, against the ~209 ms of
permute it saves.

**The lesson:** the problem was never the transpose's *existence*, it was its
*implementation*. `v_paged` gets the permute deleted **and** keeps the fast gather,
which is why it has no crossover and this does.

### 9.2 `block_n=128` — shipped, then deleted

Widening the KV tile buys L0 hits (L2 requests per texture load 1.55 → 0.95), so it is
a 13–29% win above S=4096 and a 18–32% loss below it. It shipped behind a measured
sequence-adaptive lookup table and was worth **6.0% e2e** at 8192-token prefill.

**`k_lds` retired it the same day.** LDS buys strictly more L0 relief without bn128's
112 B of scratch: bn64+`k_lds` beats bn128 by 1.39× at S=4096 and 1.35× at S=8192, and
bn128 already lost below that. The adaptive table collapsed to **one row** — which also
*widens* eligibility, since bn64 divides every length bn128 did and more, so fewer
requests fall through to Triton.

`k_lds` and `block_n=128` **compete, they do not stack**: a padded K tile is 17 KB at
bn64 (3 workgroups still fit the 64 KB per-CU LDS, occupancy stays at 24 waves/CU) but
35 KB at bn128, which collapses occupancy to **1 WG/CU**.

`block_n=256` spills 632 B and loses everywhere. Not a candidate.

### 9.3 `kv_lds` — staging K **and** V in LDS

The first attempt at the LDS idea measured **0.38× / 0.35× / 0.29×** at
L=512/2048/4096 — losing worse as L grew, the exact opposite of the hypothesis. It was
recorded as dead. Three root causes, and why the K-only retry was different:

| recorded cause | why K-only at `gqa_fuse=4` differed |
|---|---|
| VGPR 197→256 plus 16 B spill | The coop loader's live payload. `kv_lds` staged K+V across **64** threads = 128 VGPR; K-only across a **256**-thread CTA = 16 VGPR, and the chunk row/col div+mod hoists out of the loop. Measured: **+1 VGPR, 0 scratch.** |
| `ds_load` 0 → 320 | Entirely a **V** problem — the flat LDS V read is 16 uncoalesced scalar `ds_load`s per fragment. Staging K only never touches it. |
| 2 barriers/tile kill pingpong | Still real in principle, and now a harder 8-wave rendezvous. Measured: it did not materialise — pingpong stayed the better arm for both configs at both lengths. |

**The lesson is not "LDS works after all."** The earlier verdict was *correct for its
configuration* (H24 dense, `gqa_fuse=1`, 2 waves sharing the tile) and was invalidated
by an unrelated change — head fusion made the amortisation 4× larger — that nobody
re-checked it against. Negative results have a configuration attached.

### 9.4 `head_adjacent` dispatch swizzle — a prediction that failed, twice over

A falsification probe before building fusion: reorder the dispatch so the F CTAs sharing
a KV head are adjacent in flight. The plan predicted a **null**, with arithmetic (at
216 VGPR the concurrent window already spans ~2 KV heads ≈ 2 MB ≈ L2, so they are
already co-temporal).

**The prediction was wrong** — the swizzle measured **1.27×** (`SQ_BUSY` 118.8M →
93.7M). The K/V refetch cost is *partly* temporal and reordering recovers that part; the
larger part is that the sharing CTAs do not share a **CU**, and only fusion recovers
that. It was **not shipped** anyway: fusion strictly dominates and subsumes it, so a
second dispatch-order knob would be cost without benefit.

### 9.5 `q_block=2` — a "verified 2.05× win" that was a confound

The kernel hard-rejects `qk_douter and q_block > 1`, so **every** `q_block=2` build
silently disabled `qk_douter` — and that was the entire effect. Held constant,
`q_block=2` is *worse* at S=2048 (163.5M vs 127.2M `SQ_BUSY`, and it spills 624 B).

### 9.6 The knob sweep that found nothing

At S=4096 on top of `block_n=32, qk_douter=False` (900.6M `SQ_BUSY` baseline), none of
these beat −7%: `o_nt` 894.1 · `prefetch_v` 909.8 · `v_prefetch=2` 912.6 ·
`bcast_group=4` 984.9 · `o_f16` 1002.9 · `qk_ilp=4` 1426.5 · `qk_ilp=1` 1783.7.

Knob tuning was exhausted here. Everything that worked afterwards (§8.1–§8.4) was
structural.

### 9.7 The "~8% rocKE plumbing win" — real, but not ours

An early round reported the rocKE backend ~8% faster than vLLM's default **while
dispatching zero swapqk kernels**, attributed to rocKE's plumbing.

The effect is real (**7.10%** mean, 6/6 interleaved pairs, thermals flat, output
bit-identical through 64 decode tokens). The attribution was wrong. vLLM contains **two
different functions named `context_attention_fwd`**:

| | `ops/prefix_prefill.py` | `ops/triton_prefill_attention.py` |
|---|---|---|
| kind | **paged** — block-table indirection, cached-context merge, fp8 dequant | **dense** — contiguous `key`/`value` |
| reached by | `RocmAttentionImpl.forward` (vLLM's default) | `RockeAttentionImpl._forward_rocke` |

The paged kernel performs its indirection and cached-context merge **even when there is
zero cached context**, which is exactly a cold full prefill. That is the entire gap.
Measured on its own, the rocKE backend class contributes **nothing** — it is marginally
*slower*. The win belongs upstream in `RocmAttentionImpl`, not in a rocKE file, and its
addressable market shrinks once prefix caching or chunked prefill is on.

### 9.8 Two other prefill retractions worth carrying

- **"swapqk is a +4.8% e2e win"** — measured against the wrong control (vLLM's default
  backend, which changes plumbing, KV handling and kernel at once). Against a matched
  same-backend control it was a **3.4% regression** at the time.
- **"graph mode is ~1.05 s faster than eager"** — an artifact of comparing two separate
  process launches. In matched within-process runs they are within noise (7062 vs 7085
  at S=8192). Consistent with `splitting_ops = attention_ops`: attention is split *out*
  of piecewise cudagraphs, so at prefill sizes launch overhead is negligible.

### 9.9 `d_lanes = 8` — the predicted decode optimum that lost on both sides

Fewer lanes means fewer swizzles — 3 per key instead of 16's 8 — and the reduction cost
is the term the design was written to minimise. It was the predicted winner.

It lost, because VGPR pressure rises as `1/d_lanes`: 172 registers collapses occupancy
to **2 waves/SIMD**, and this is a bandwidth-bound kernel whose entire job is keeping
enough loads in flight to saturate the fabric. The swizzles it saves are VALU work that
was already hidden behind memory latency; the occupancy it costs is not recoverable.

The mirror-image failure at `d_lanes=32` is cleaner still: occupancy is fine at 4
waves/SIMD, but 20 swizzles per key is enough VALU work that it stops being free. **Both
neighbours of 16 fail for opposite reasons**, which is what an interior optimum looks
like when you find one.

### 9.10 Expecting the decode win to track the bandwidth share

The isolated decode kernel wins 1.094× at Sk=32768. Attention is 24.2% of ITL at that
context. The tempting arithmetic — 24.2% of a 9.4% improvement ≈ 2.3% — is close enough
to the measured 2.4% to feel like it works, and it does *not* generalise: at ctx8k the
same arithmetic predicts 0.7% (7.9% share) and measures 0.34%, over-predicting by 2×.
At ctx2k it predicts 0.2% and the measurement is barely above zero.

The rule this produced is [§10](#10-measurement-discipline) rule 15: an isolated kernel
speedup is an *upper bound* on the e2e effect, never an estimate of it. Compute the
share, use it to decide whether the measurement is worth running, and then run it.

### 9.11 Decode below the MALL boundary

The original decode measurement reported the split-K kernel at **0.66–0.80×** — a large
loss — below 32 MiB of KV. That figure was taken at `d_lanes=32`; at the shipped
`d_lanes=16` the same region is a **wash** (0.93–1.00×), and the loss is corrected.

What survives is the boundary itself: below ~32 MiB of KV the working set is
MALL-resident, both kernels are fast, and there is nothing for split-K to recover. That
is why `ROCKE_PAGED_DECODE_MIN_KV_MIB` defaults to 32 rather than 0 — the kernel is
gated to the regime where it was shown to win, not enabled unconditionally and hoped for.

---

## 10. Measurement discipline

Roughly half the elapsed effort on this project went into **measurement defects that
produced confident, plausible, wrong numbers** — several of which had already been
written up as conclusions. The rules that came out of it:

1. **Confirm the kernel actually dispatched.** vLLM's EngineCore is *forked* and
   inherits the parent's mutated `sys.path`; one harness line stripping `rocke-repo`
   from `sys.path` made **every** early e2e number Triton-vs-Triton. Assert on
   `ROCKE_STATS` every time.
2. **Confirm the work was actually performed.** vLLM v1 enables **prefix caching by
   default**, so `[PROMPT] * 4` computes the prefill once and replays it. The harness
   reported 18,339 tok/s; the real figure was 1,189.
3. **Interleave A B A B — a blocked A/B cannot separate the arm from its slot**, and
   running the order forwards then backwards does *not* fix it if an arm keeps landing
   in an extreme slot.
4. **Take the minimum across reps**, never the median, for anything over ~5 ms. The box
   is shared; contention can only inflate. It also up-clocks 609 → 938 MHz as it warms,
   so discard the first rep.
5. **Never stitch a table together from probes run hours apart** — inter-session clock
   drift is ~13%.
6. **Never compare arms across separate process launches** — run-to-run drift is ~1 s at
   S=8192 (~11.9% on e2e), which swamps the effects being measured.
7. **`max_tokens=1` cannot validate a prefill change.** It compares the prefill output
   tensor and never reads the KV cache back. A path that computes the right output but
   corrupts the cache passes it. Use a 64-token decode hash.
8. **GRBM counters are broken on this part** — `GRBM_COUNT` reads a fixed ~1.0M
   regardless of duration, i.e. *below* `GRBM_GUI_ACTIVE`. Use `SQ_BUSY_CYCLES` /
   `SQ_WAVE_CYCLES`. `MemUnitBusy` is also non-functional (returns 319, 656 for a
   documented 0–100 percentage), and gfx1151 exposes **no stall counters at all**.
9. **`rocprofv3` multiplexes counters across dispatches.** One counter per pass, and
   report coverage.
10. **Do not profile the correctness oracle.** A torch/rocBLAS reference emits
    dispatches *larger* than the kernel under test, so a "biggest kernel" heuristic
    silently selects `Cijk_...` — every early Triton number was measuring our own
    reference GEMM.
11. **Verify two arms calling "the same function" resolve to the same module**
    (see §9.7).
12. **Log thermals rather than assuming them.** "It's probably thermal" is easy and
    unfalsifiable; sampling `rocm-smi --showtemp --showpower --showclocks` between arms
    costs nothing and turns it into data. (It was 32–38 °C and the effect was real.)

Three more that decode added:

13. **FULL cudagraphs freeze the arm *and* the counters.** Under a FULL mode the decode
    subgraph is captured once and the backend's Python never runs again — so the
    dispatch counters read zero and a counter assertion fires falsely in exactly the
    configuration you most want to test. Use `PIECEWISE` for any in-process decode A/B
    ([§7.1](#71-why-the-sweep-runs-under-piecewise-cudagraphs)).
14. **Measure ITL as a slope, not an average.** `ITL = (t_hi − t_lo)/(G_hi − G_lo)` from
    two measured generation lengths in one session; the slope cancels prefill
    algebraically. And **start past the admission ramp** — at `gen_lo = 8` the request
    admission ramp leaked into the slope and manufactured a spurious 4.5% regression.
    `gen_lo = 48` is clean.
15. **An isolated kernel speedup is an upper bound on the e2e effect, not an estimate
    of it.** Always compute what fraction of the step the kernel occupies before
    quoting anything ([§9.10](#910-expecting-the-decode-win-to-track-the-bandwidth-share)).

And one reporting rule specific to this pair of kernels:

> **Never confuse the three token rates.** An attention-only rate derived from an
> isolated decode benchmark runs ~4× high (44.8 tok/s at ctx32k against a real 11.06).
> A whole-request `tok_per_s` is dragged down by prefill. Only `decode_tok_per_s`
> (`B × 1000 / ITL`) is the generation rate a user perceives. The three differ by
> multiples, not percentages.

Operational traps: a killed benchmark orphans a `VLLM::EngineCore` holding ~88 GB of the
unified pool — it has PPID 1 and comm `VLLM::EngineCor`, so it does *not* match
`pgrep -f <script>`; find it with `rocm-smi --showpids` and kill by PID. And
**`pkill -f <pattern>` over SSH matches your own remote shell** and kills the session
(exit 255) before it kills the target.

---

## 11. Document index

| Document | What it covers |
|---|---|
| [`case_study_singlewave_fmha.md`](case_study_singlewave_fmha.md) | The earlier single-wave WMMA campaign and the ~11 TF plateau it reached — historical record, superseded by the swapqk rewrite |

---

## 12. Reproducing

Everything runs on the gfx1151 box. vLLM is a **source checkout on `PYTHONPATH`**, not
pip-installed.

```bash
source ~/.pyenv/versions/3.11.9/envs/rocm-env/bin/activate
R=<repo>/dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH=~/vllm:$R/library:$R/platform/python
export ROCKE_CPP_QUIET_FALLBACK=1
cd $R/library
```

**CPU-only tests** (no GPU — these run anywhere):

```bash
python -m pytest tests/test_gfx1151_wmma_fmha_swapqk.py \
                 tests/test_gfx1151_paged_decode_splitk.py
```

The prefill suite covers the `is_valid_spec` rejection matrix, kernel-name/artifact-cache
collisions, the ABI param list (paged adds exactly `VBlockTable` + `bt_stride`, and only
when `v_paged`), the LDS allocation size, the `readfirstlane` waterfall guard, and the
2× `dwordx4` V load shape. The decode suite covers the split-count solver, the
`_NEG_BIG` sentinel, the two-dispatch ABI, and the `d_lanes` swizzle count.

**Prefill correctness + throughput gate:**

```bash
python -m builders.gfx1151.attention.wmma_fmha_swapqk_verify \
    --batch 1 --seqlen-q 8192 --seqlen-k 8192 --causal \
    --heads 32 --kv-heads 8 --head-size 128 \
    --gqa-fuse 4 --block-n 64 --k-lds \
    --v-paged --kv-block-size 16 --shuffle-blocks 1
```

The paged path **poison-fills** the fake cache with `7777.0` before scattering and
shuffles the block table by default, so an addressing bug screams instead of returning
plausibly-clamped zeros. The CPU reference keeps consuming the original row-major V —
the test is that the paged kernel reproduces it.

**Decode correctness + bandwidth gate:**

```bash
python -m builders.gfx1151.attention.paged_decode_splitk_verify \
    --batch 1 --seqlen-k 32768 \
    --heads 32 --kv-heads 8 --head-size 128 \
    --kv-block-size 16 --shuffle-blocks 1
```

Reports max abs error against a float64 CPU reference and the achieved GB/s — **not**
TFLOP/s, for the reason in [§5.1](#51-why-decode-must-be-reported-in-gbs).

**End-to-end.** One sweep produces all three e2e ladders — [§4.1](#41-the-ttft-ladder-shipped-config-vs-matched-triton-control)
from the `prefill_only` arm, [§6](#6-decode-end-to-end-results) from `decode_only`, and
[§7](#7-both-kernels-together) from all four:

```bash
python -m builders.gfx1151.attention.e2e_combined_bench --ctx-sweep   # ~95 min
```

It needs a vLLM source checkout on `PYTHONPATH` exporting the `ROCKE_GFX1151` backend —
that is the `~/vllm` entry above, and the backend it selects is the file committed here as
[`rocke/integrations/vllm/rocm_rocke_attn.py`](../../../../integrations/vllm/rocm_rocke_attn.py).
The sweep builds one `LLM`, patches the arms as module attributes between `generate()`
calls, interleaves them in a single session, and **asserts the `ROCKE_STATS` dispatch
counters on every measurement** — so a silently-declined gate fails the run instead of
being reported as a result. Defaults are the published ones: `PIECEWISE` cudagraphs
([§7.1](#71-why-the-sweep-runs-under-piecewise-cudagraphs)), minimum of 3 reps after a
discarded warmup, ITL as a slope between two measured generation lengths
([§10](#10-measurement-discipline)). Drop `--ctx-sweep` for the four named workloads, and
run it detached — it is long enough that a dropped SSH session would kill it.

Before any GPU run: `uptime` and `rocm-smi --showpids`. If the latter is not *"No KFD PIDs
currently running"*, an orphan is holding ~88 GB and the run will die at startup.

---

## 13. Known gaps

**Prefill**

- **`k_lds` is a bn64-only lever**, by LDS budget. Only D=128 has been timed, only
  causal, batch 1, Hq32/Hk8.
- **Single-buffered by choice.** Double-buffering K would hide the staging latency
  behind the previous tile's WMMA at the cost of 35 KB and the occupancy cliff. Not
  attempted, not measured.
- **The `_SWAPQK_MIN_SEQLEN = 512` floor is plausibly too conservative** — swapqk now
  beats Triton by 1.57× at S=1024, but nothing below S=1024 has been measured.
- **The truncating tail bound is unchanged.** `seqlen_k % block_n != 0` still routes to
  Triton. Paging does not fix it and the two must not be conflated.
- **2 GiB SRD cap.** The paged `soffset` is a 32-bit byte offset; per-layer V cache is
  ~1 GiB at the current 70 GiB total, so it fits, but not with margin. Guarded
  host-side rather than by building an i64 path.
- **The in-repo torch custom op never sets `gqa_fuse`**, so it compiles at F=1 while the
  vLLM backend picks F=4. It also does a device→host→device round trip for the V
  transpose. Both pre-existing, both worth fixing separately.
- **An unexplained rig-vs-model gap at S=30720.** The isolated rig reports 0.976× where
  the model reports +9.8% TTFT — roughly 121 ms/layer unaccounted for. Chunking and a
  pessimal V layout are both ruled out; memory pressure is the leading untested
  hypothesis. **The isolated prefill tables are trustworthy through S=16384 only**; the
  e2e numbers in §4.1 and §7 are unaffected, since they are measured directly.

**Decode**

- **Off by default.** `ROCKE_PAGED_DECODE=1` is required, and the 32 MiB KV floor gates
  it further. Nothing below the MALL boundary has been shown to benefit.
- **Only fp16, only `block_size=16`, only D=128, only non-ALiBi non-sliding-window.**
  Every other combination declines into AMD's HIP kernel.
- **B=1 is the only operating point measured end-to-end here.** The 1.015× at B=32 comes
  from a separate batched run and is not part of the ladder in §7.
- **`_TARGET_CTAS = 256` was swept on one shape.** It held across B ∈ {1..32} for
  Qwen3-8B's Hk=8, but a model with a different KV-head count would change the CTA count
  per split and has not been checked.

**Both**

- **The backend exists in three places** — the vLLM checkout, the vendored copy under
  `integrations/vllm/`, and the benchmark harness — with nothing enforcing they stay
  identical. Divergence would be silent.
