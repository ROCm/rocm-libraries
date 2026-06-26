# Unified paged attention on gfx950, from the math up

This file explains *what* the CK DSL `unified_attention` kernels compute and
*why* they are shaped the way they are on CDNA (gfx950, MI355X, wave64, MFMA).
The [`README.md`](README.md) is the parity + optimization history; this file is
the specification, the data layout, and the two kernel strategies (2D and 3D
split-KV) the production dispatcher chooses between.

> The harness in this folder benchmarks against AITER's Triton
> `unified_attention`; the CK DSL kernels mirror that Triton reference 1:1 in
> semantics (`kernel_unified_attention_2d` / `_3d`), which is why every output is
> bit-identical to Triton once cast back to the working dtype.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $Q$ | $S_q \times H_q \times d$ | query, one batch slice; $H_q$ query heads |
| $K, V$ | paged blocks of $[\,b,\, H_k,\, d\,]$ | key / value cache, $H_k$ KV heads |
| $O$ | $S_q \times H_q \times d$ | output (same shape as $Q$) |
| $S_q, S_k$ | scalar | query / key sequence length (per sequence) |
| $d$ | scalar | head dimension (`head_size`, 64/128/256) |
| $\tau$ | scalar | softmax scale, $\tau = 1/\sqrt{d}$ |
| $b$ | scalar | paged-cache `block_size` (16, 32, or 64) |
| $g$ | scalar | GQA group size, $g = H_q / H_k$ (queries per KV head) |

Attention is **paged and varlen**: the KV cache is stored as fixed-size blocks
scattered in HBM, a per-sequence **block table** maps logical KV positions to
physical blocks, and `cu_seqlens_q` packs many sequences of different lengths
into one ragged batch. This is the vLLM / serving layout, not a dense
`[B, H, S, d]` tensor.

---

## 1. What unified attention computes (the specification)

For one query row $i$ of one $(\text{sequence}, \text{head})$, with score vector
$s = \tau\, Q_i K^{\top} \in \mathbb{R}^{S_k}$:

$$
O_i \;=\; \sum_{j=1}^{S_k} \frac{e^{\,s_j}}{\sum_{j'} e^{\,s_{j'}}}\, V_j .
$$

"Unified" means one kernel family serves the whole serving workload — decode
($S_q = 1$), prefill ($S_q = S_k$), and chunked prefill ($1 < S_q < S_k$) — plus
every bias/mask variant a real deployment needs. The reference
(`ref_paged_attn` in `parity_unified_attention.py`) and the kernels both apply,
in this exact order, before the softmax row-reduction:

1. **scale** $s \leftarrow \tau\, Q_i K^\top$;
2. **softcap** (optional): $s \leftarrow c\,\tanh(s/c)$ — bounds logits to $\pm c$;
3. **ALiBi** (optional): $s_j \mathrel{+}= \text{slope}_h \cdot (k_j - \text{ctx})$ — a linear positional bias;
4. **QQ-bias** (optional): $s_j \mathrel{+}= \text{bias}[q_{\text{local}}, k_j - \text{ctx}]$ inside the query section;
5. **causal mask** $s_j \leftarrow -\infty$ for $k_j > q_i + \text{ctx}$;
6. **sliding window** (optional): also mask $k_j < q_i + \text{ctx} - W$;
7. **attention sinks** (optional): append per-head learned logits as extra
   softmax columns (they soak up probability mass but contribute no value rows).

Each of these is a flag on `UnifiedAttentionProblem` / `UnifiedAttention2DTiledSpec`,
so the dispatcher builds exactly the kernel the shape needs.

---

## 2. The streaming-softmax core (shared by both paths)

Both kernels are flash-attention: they tile the $S_k$ keys, never materialize the
$S_q \times S_k$ score matrix, and maintain three running quantities per query row
updated tile by tile — the running max $m$, the running denominator $\ell$, and
the un-normalized output accumulator $\mathbf{o} \in \mathbb{R}^{d}$:

$$
\begin{aligned}
m' &= \max\!\big(m, \operatorname{rowmax} s^{(t)}\big), &\quad \alpha &= e^{\,m - m'} \\
p^{(t)}_j &= e^{\,s^{(t)}_j - m'}, &\quad \ell' &= \alpha\,\ell + \textstyle\sum_j p^{(t)}_j \\
\mathbf{o}' &= \alpha\,\mathbf{o} + P^{(t)} V^{(t)}, &\quad O_i &= \mathbf{o}^{(N)} / \ell^{(N)}.
\end{aligned}
$$

The rescale $\alpha$ re-bases everything accumulated under the old max to the new
max; the result is **exact**, not approximate. As in all of CK DSL, exponentials
are computed in base 2 — the GPU has a hardware $2^x$ but no $e^x$. The tiled
2D/3D kernels take the raw softmax scale $\tau$ as a runtime param and fold
$\log_2 e$ into it once per kernel on the device
(`qk_scale = scale * 1.4426950408889634`, i.e. $\tau \log_2 e$), so every
`exp2(s - m)` is one instruction. Softcap, ALiBi, and QQ-bias are likewise
applied in the $\log_2$ domain (e.g. `apply_softcap_log2`).

Where the two paths differ is **who owns which keys**, which is dictated entirely
by occupancy — how many CTAs the shape can keep the device busy with.

---

## 3. CDNA mapping: wave64 + MFMA

The matrix unit on gfx950 is `v_mfma_*` over wave64 (64 lanes), with native
$A \cdot B$ accumulation (no transpose, unlike RDNA's WMMA $A B^\top$). The
default (non-transposed) path runs QK/PV on the wide-K `16x16x32` atom
(`mfma_f32_16x16x32_{f16,bf16}`), falling back to `16x16x16` only for a K=16
head-dim tail step in PV. The transposed-32x32 prefill "combo" path runs
QK/PV on gfx950's wide-K `32x32x16` atom (`mfma_f32_32x32x16_{f16,bf16}`, the
gfx950-only 32x32 family). The flash-attention building
blocks ported from CK Tile's `BlockFmhaPipelineQRKSVSAsync` are:

- **Q staged in LDS once** per CTA, reused across the whole K-loop. (The
  transposed combo additionally hoists Q into VGPRs — `Q32_reg` — to drop
  Q's permanent LDS allocation.)
- **K and V streamed cache → LDS each tile**, with the global load issued early
  so the QK MFMA starts the moment the LDS write retires (async DMA, see §7.1).
- **`m`, `l` in registers**; the per-row max/sum reductions are XOR
  butterflies (CK Tile's `block_tile_reduce_xor_sync` pattern) — no LDS
  round-trip. The intra-16-lane / intra-32-lane stages lower to
  `ds_swizzle_b32` SWAP mode (not `ds_bpermute`); only a cross-half (mask 32)
  stage — e.g. the split-KV reduce's wave64 fold, or the transposed scalar
  state's alpha broadcast — uses `ds_bpermute`.
- **`o_acc` in MFMA-accumulator distribution** (per-lane `<4×f32>` per N-tile of
  the head dim), truncated to the output dtype through an LDS-staged shuffle
  epilogue with 16-byte stores.

---

## 4. The 2D path — one CTA owns a query block

The 2D kernel (`attention_tiled_2d.py`) assigns **one CTA per
$(\text{q-block}, \text{kv-head})$** and walks the *entire* KV sequence for that
q-block in the K-loop. `BLOCK_M = 16` packs the GQA group: with $g$ queries per
KV head, the 16 MFMA rows hold $(\text{query-position}, \text{query-head})$ pairs
so one QK MFMA serves the whole group sharing a KV head.

- **Grid:** `(num_kv_heads, total_num_q_blocks, 1)`, block = `64 · num_warps`.
- **Best for:** chunked prefill and short / sliding-window contexts, where the
  q-block × kv-head grid is already large enough to fill the device. It is the
  path the production dispatcher selects for the d64/b32 GQA-8 serving traces.

The boundary masks (causal, sliding-window) are applied per element via
`v_cndmask`. A causal early-exit clamps the K-loop so fully-future tiles are
skipped; the boundary tile straddling the diagonal is still masked elementwise.

### The transposed "combo" 2D variant (the prefill win)

The plain 2D kernel is single-warp-ish per CTA and leaves the device
under-occupied on long prefill. The **combo** rewrite — selected automatically by
`_enable_combo_2d` for the validated bf16 / fp8 d64/b32 GQA-8 family — restructures
the softmax onto a **transposed 32×32 MFMA** layout (`block_m_per_warp=32`,
`use_mfma_32x32`, `use_transposed_qk_32x32`) and layers several co-operating
optimizations, each a spec flag:

| flag | what it does |
|---|---|
| `use_transposed_scalar_state` | keep `m`/`l` in the transposed lane layout (no re-shuffle) |
| `use_transposed_mask_once` | apply the causal mask once per tile, not per MFMA (full-attn only) |
| `use_transposed_half_local_pv` | halve the PV operand staging via the transposed layout |
| `use_mfma32_skip_legacy_qreg` | drop the legacy per-MFMA Q-register reload |
| `use_transposed_mask_limit` | fold the causal row limit into the threshold (full-attn only) |
| `use_fast_paged_kv_desc` | the 64/8-head fast paged-KV descriptor (one i64 block base) |

These flags are **mutually load-bearing** — most are perf-neutral alone and only
pay off together — which is why the harness sweeps them as a bundle (`combo`) and
the dispatcher ships them as one validated policy. `mask_once` / `mask_limit` are
valid only for full attention; under a sliding window the spec validator rejects
them, so the SW combo runs with them off.

### Occupancy is the 2D lever, not instruction count

Static ISA inspection shows the combo kernel is **VALU/SALU-bound** (~800 VALU +
~650 SALU vs only ~16 MFMA per kernel), dominated by the per-element causal-mask
`v_cndmask`. But the binding constraint on this kernel is **how many workgroups
fit per CU** — it is occupancy/latency-bound, and the wins all come from raising
WG/CU, never from cutting instructions. Three levers shaped the tuning (all
detailed in the README):

1. **Raise `waves_per_eu` (the d64/b32 trace family).** That combo is VGPR-limited
   (~137 VGPR → 3 WG/CU at the default `waves_per_eu=2`); `waves_per_eu=3` reaches
   4 WG/CU (+15%), and `waves_per_eu=4` adds ~5% more on full-attention shapes.
2. **K single-buffer for single-batch d128 prefill (the LDS lever).** The
   single-batch d128 combo at `num_warps=2` is not VGPR-limited but **LDS-limited**:
   at `tile_size T = 2·block_size = 64` the K double-buffer + V single-buffer LDS
   (`K_lds[2,T,HD] + V_lds[1,T,HD] = 48 KB`) admits only **1 WG/CU**, while the
   register file already admits two. Keeping the larger `T = 64` tile (good
   long-context per-iter amortization) but **halving K_lds via K single-buffer**
   (`use_k_single_buffer`: `K_lds[1] 16 KB + V_lds[1] 16 KB = 32 KB → 2 WG/CU`,
   VGPR=215 AGPR=0) doubles occupancy and hides the per-iter latency. **V
   double-buffer is OFF on d128** — a V[i+1] prefetch is a net drag there and
   would re-inflate LDS back to 1 WG/CU. The single K slot re-issues its next-K
   prefetch *after* the PV-wait barrier (all QK reads drained) so it cannot
   WAR-race. This took the single-batch d128 prefill cohort from below flash to
   **≈1.10x over torch SDPA flash** (geomean; 1.36x at S1024, 1.02x at S2048,
   but **0.95x at the long S4096 holdout — a small honest loss to flash there**,
   re-measured on verified llvm22). The same cohort still **trails Triton's
   multi-warp 2D kernel (~0.55-0.60x)**; see README. d64 single-batch prefill
   instead keeps `num_warps=4` with a wider `tile_size = 128` to feed its KV
   loop.
3. **Lighten the prelude for sliding window.** SW prunes the K-loop to a handful
   of tiles, so the per-CTA prelude (Q→LDS, binary search, sink init) dominates.
   For **bf16** SW the combo drops to `num_warps=2` (BLOCK_M=64, half the
   prelude, 2× the CTAs) but keeps `tile_size = 2·block_size`; fp8 SW is
   dequant-bound rather than prelude-bound, so it stays at `num_warps=4` and is
   the *only* SW case that also shrinks `tile_size = block_size` (one paged-KV
   block per iter). The compute-bound no-SW combo keeps `num_warps=4` /
   `tile_size = 2·block_size` for both dtypes.

Reducing instruction count by splitting the loop into a no-mask phase + a masked
boundary phase was byte-identical but **~7% slower** (I-cache / code-size cost on
a latency-bound kernel), and was reverted — a representative "fewer instructions
≠ faster" result on this kernel.

---

## 5. The 3D split-KV path — many CTAs share a query block

When a single $(\text{q-block}, \text{kv-head})$ has so much KV that one CTA
walking it serially leaves the device idle (long-context decode), the 3D kernel
(`attention_tiled_3d.py`) **splits the KV sequence into segments** and gives each
segment its own CTA, then reduces:

- **Grid:** `(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)`. Each segment CTA
  runs the §2 recurrence over only `tile_start..tile_end` of the KV sequence and
  writes a *partial* $(m, \ell, \mathbf{o})$ to a workspace
  `segm_output[total_q, num_qh, num_segments, head_size]` (plus `segm_max` and
  `segm_expsum`, all fp32).
- **`reduce_segments`** then combines the per-segment partials into the final
  output. Because flash-attention's $(m, \ell, \mathbf{o})$ merge is associative
  (re-base each segment's accumulator to the global max, sum the denominators),
  the split is **exact** — the final `acc /= L` and the output-dtype cast happen
  in the reduce kernel.

The 3D path turns a serial KV walk into parallel segments + a cheap reduction,
which is the only way to saturate the device on long-context single-query decode.

---

## 6. The dispatcher — which path runs (`select_path`)

`UnifiedAttentionProblem.select_path()` mirrors AITER's own `use_2d_kernel`
selector so CK DSL and Triton make the same algorithmic choice. The rule, in
spirit:

- **2D** when the q-block × kv-head grid already saturates the device
  (`target = num_sms · 4`), or for short context ($S_k \le 512$), or under a
  sliding window — the split-KV segments would only add launch overhead.
- **3D split-KV** otherwise — long, full-context sequences where the 2D grid is
  too small to fill the device.

Once a shape lands on the 2D path, a second tier of routing picks the *kernel
geometry* for it, driven entirely by the occupancy (WG/CU) the shape can sustain:

- **Single-batch (`num_seqs == 1`) d128 long prefill** (bf16/fp16, GQA, no
  bias/SW, $S_q > 256$): the full transposed-32×32 combo with **`num_warps=2`,
  `tile_size T = 2·block_size = 64`, K single-buffer on, V double-buffer off**.
  The K single-buffer halves K_lds so the larger `T=64` tile still fits in the
  **32 KB → 2 WG/CU** budget (see §4 lever 2). This is the LDS-bound occupancy
  story that reversed the old d128-prefill loss vs flash; it now **wins ≈1.10x
  over torch SDPA flash** (1.36x S1024 → 0.95x at the S4096 holdout) but
  **still trails Triton's 2D kernel (~0.55-0.60x)** — see the README cohort
  section. Only same-session ratios are load-bearing (±25-30% auto-clock).
- **Single-batch d64 long prefill**: same combo but **`num_warps=4`,
  `tile_size = 128`** (8·block_size) — d64 is wide-KV-loop-bound, so it wants the
  wider tile and more warps rather than the d128 LDS lever.
- **The d64/b32 GQA-8 serving traces** (multi-seq chunked prefill, sinks): the
  combo with the `waves_per_eu` lever (§4 lever 1) and the prelude-light
  `num_warps=2` SW geometry (§4 lever 3).
- **Decode / long full-context** shapes route to **3D split-KV** before the 2D
  geometry tier even runs — the segment grid is the only way to fill the device
  on single-query decode, and that lane is a clean ≈1.20x win over Triton (3D
  vs 3D, verified llvm22; see README).

This is also why the parity harness reports **three tables** (`auto`/`2d`/`3d`):
`auto` is what production launches, while forcing both backends to the *same*
path (`2d`-vs-`2d`, `3d`-vs-`3d`) is the algorithmically-fair comparison.

---

## 7. Implementation details worth the math

### 7.1 Async DMA K/V (current-V-first, next-K-second)

K and V are streamed to LDS with async DMA. The issue order is deliberately
**current-V then next-K**, so the PV matmul of the current tile only has to wait
on the V stream while next-K loads in the shadow — the QK of the next tile then
finds K already in flight. This overlap is what keeps the kernel HBM-bandwidth-
fed on the long-context, large-cache regime where it wins (§ README).

### 7.2 64-bit paged-KV addressing

A paged-KV byte offset is `physical_block · (block_size · num_kv_heads ·
head_size · dtype_bytes)`. That product overflows a 32-bit hardware buffer
voffset once the cache exceeds **2 GiB** (~65 K bf16 / ~131 K fp8 blocks) — and
production caches are far larger (captured traces ~350 K blocks ≈ 11 GiB).
`_enable_i64_kv_addr` switches the load paths to fold `block · stride` into a
**64-bit base** (only a small within-block offset stays in the 32-bit field)
when `num_kv_blocks · block_stride > 2³¹`, so small caches keep the exact fast
i32 path and only large caches pay the tiny per-block-base cost. Without it,
large caches silently read garbage (verified `max_abs ≈ 1.4`).

### 7.3 fp8 KV cache (bf16-Q + fp8-KV)

The fp8 path stores K/V as fp8 e4m3 (half the HBM bytes of bf16). The
**sync-dequant** loader writes bf16 into the K/V LDS (folding `k_scale` in)
*before* the MFMA, so the transposed bf16 combo runs unchanged on fp8 inputs.
A native fp8×fp8 QK MFMA (no dequant) was implemented and measured but is
lose-lose on the compute-bound full-attention shape — slower *and* less accurate
(quantizing Q to fp8 costs ~1e-2) — because the dequant is already hidden behind
the K/V load latency. The accurate sync-dequant path is the production choice.

### 7.4 Magic division

Per-tile index arithmetic (`pos // block_size`, grid-axis decode) uses CK Tile's
mul-hi **magic division** so a compile-time constant divisor folds to a
`v_mul_hi_u32` + add + shift instead of the ~20-cycle hardware integer divider.

---

## 8. Where the algorithm ends and tuning begins

The math above is fixed and exact: online softmax, the bias/mask order of §1, the
associative split-KV merge. Everything the README tunes — the combo flag bundle,
`waves_per_eu`, `num_warps`, `tile_size`, async-DMA issue order, i64 addressing,
fp8 dequant — changes only **how these steps are scheduled onto gfx950**, never
what is computed. Correctness is pinned by bit-exact parity against Triton and
ULP-level agreement with `ref_paged_attn`; performance is the per-CTA schedule
and the 2D-vs-3D occupancy choice.

---

## 9. Where to go next

- [`README.md`](README.md) — the parity harness, the three-table methodology, the
  prefill-2D optimization history (combo, `waves_per_eu`, i64 addressing, fp8),
  and the substantiated speedups.
- `parity_unified_attention.py` — the canonical parity + benchmark harness
  (default / creative / fmha scenario sets).
- The kernels themselves live in `rocke.instances` (not in this folder):
  `gfx950/attention_tiled_2d.py`, `gfx950/attention_tiled_3d.py`, and
  `common/attention_unified.py` (the dispatcher and shared spec).
