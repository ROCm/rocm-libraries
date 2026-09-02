# Unified paged attention on gfx950, from the math up

This file explains *what* the CK DSL `unified_attention` kernels compute and
*why* they are shaped the way they are on CDNA (gfx950, MI355X, wave64, MFMA).
The [`README.md`](README.md) is the parity + optimization history; this file is
the specification, the data layout, and the two kernel strategies (2D and 3D
split-KV) the production dispatcher chooses between. §8 additionally documents the
**dense prefill kernel** — a non-paged, compile-time-sized peak-TFLOPS sibling
(persistent grid-stride + on-chip ragged support) for the contiguous
`[B, S, H, d]` prefill regime.

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
  (`target = num_cus · 4`), or for short context ($S_k \le 512$), or under a
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

## 8. The dense prefill kernel — peak-TFLOPS sibling (`attention_dense.py`)

The 2D/3D kernels above serve the *paged, varlen* deployment layout. Dense
prefill (a contiguous `[B, S, H, d]` batch, no page table, self-attention
$S_q = S_k$) is a different regime: the shape is known up front and there is no
paging indirection to hide, so a dedicated kernel
(`kernels/gfx950/attention_dense.py`) trades generality for **peak MFMA
throughput**. It computes the exact same streaming-softmax core (§2) but bakes
the shape into the kernel and reshapes the schedule around a 256-row query tile.

- **Compile-time-sized ABI.** `batch / seqlen / heads / head_size / causal /
  dtype` are constants baked at build time (dense, statically-sized). The only
  runtime args are the `q/k/v/o` pointers and the `f32` softmax scale. `block_n`
  (KV tile), `waves_per_eu`, and the persistent knobs are the tuning parameters;
  every algorithmic lever below is always-on.
- **256-row query tile, `32×32×16` MFMA.** `_BLOCK_M = 256` query rows per CTA
  (8 wave64s = 512 threads) run QK/PV on gfx950's wide-K `mfma_f32_32x32x16`
  atom. QK produces $S^\top = K Q^\top$ so the key lands on the per-lane
  accumulator regs — the layout that keeps the softmax a cheap in-lane reduce +
  one `lane^32` exchange and lets PV consume $P$ with no relayout shuffle.
- **head_size 64 and 128** (bf16 / fp16), MHA and GQA including non-power-of-2
  group sizes (e.g. 40/8, 28/4).

### 8.1 The winning levers (all always-on)

| lever | what it does | measured |
|---|---|---|
| **CK-1 transposed PV** | $P$ feeds the PV MFMA in its native QK-output layout via a half-local V load (`pv32_v_load_paired`); the cross-half P-relayout shuffle is gone (~96 `ds_bpermute` removed). | +35% |
| **LDS bank-conflict pad on K** (`[NBUF, BN, D+8]`) | kills the 8-way conflict on the QK K-reads. | +80% (base win) |
| **LDS V pad** (`+32`) | the transposed PV read (`ds_read_b64_tr_b16`) has a stricter bank pattern than K; a `+32` V-row pad fully clears its conflicts. | +~5% |
| **native `exp2_fast`** (`v_exp_f32`, no overflow guard — softmax arg ≤ 0) | one instruction per exp. | +11.5% |
| **depth-1 cluster** | fuses `exp2(s − m)` into the PV-MFMA loop so the softmax VALU/TRANS co-executes in the MFMA shadow (`sched_group_barrier` names the full DS_READ/MFMA/VALU/TRANS population per step). | — |
| **partial-vmcnt software prefetch** | per-tile K/V DMA drains to a *partial* `vmcnt` (keeps the freshest V prefetch in flight across the barrier) instead of a full `vmcnt(0)` serialize. | raises MfmaUtil |
| **PV-only `s_setprio`** | the PV MFMA cluster is bracketed at raised priority so it wins issue slots; paired with the prefetch. | +3.5% |
| **lazy online rescale** | keep the running max as a *lazy* max that only re-anchors when a tile exceeds it by >8 (log2); when every lane is within 8 (a `wave_all` vote) skip the O/ℓ rescale entirely (a 0/1-trip `scf.for` → a wave-uniform scalar branch), cutting the VALU between the QK and PV clusters. Numerically approximate ($P$ bounded by $2^8$) but parity-identical at bf16/fp16 tolerance. | +~2% |

**Diagonal-only causal masking.** For causal, below-diagonal KV tiles need no
mask (~94% of tiles at $S = 8192$); the KV loop is split into a mask-free body
`[1, diag)` and a masked diagonal tail `[diag, n_upper)`, and the causal row
limit clamps `n_upper` so fully-future tiles are never visited.

**head_size and the DMA loader.** One `async_buffer_load_lds` instruction moves
64 lanes × 2 bf16 = 128 elements. At **D=128** that is exactly one padded K/V row
per instruction (the fast path). At **D=64** the loader packs `128//D = 2` rows
per instruction into an *unpadded* contiguous LDS tile (lane $l$ → row
$l/(D/2)$, col $2\,(l \bmod D/2)$); the padded fast path is emitted
byte-identically for D=128, so D=64 support costs the D=128 schedule nothing.

### 8.2 Two grids: default and persistent

Same inner pipeline, different outer work assignment:

- **Default** — one CTA per $(\text{q-block}, \text{query-head}, \text{batch})$.
  Grid `(⌈S_q/256⌉, H_q, B)`. Simple; the per-CTA launch/dispatch + scalar setup
  + K/V-prime cold-start (~4.5 tile-equivalents) is paid once **per query block**.
- **Persistent (grid-stride)** — a 1-D grid of `num_persistent` long-lived CTAs
  (256 = exactly one 8-wave block per CU on MI355X's 256 CUs at 2 waves/SIMD).
  Each CTA grid-strides over the flattened work-item space
  $W = ⌈S_q/256⌉ \cdot H_q \cdot B$, so the fixed per-CTA cold-start is amortized
  once **per CU** instead of once per query block. The inner compute is
  byte-identical to the default path; the difference is the outer work loop, the
  work-item decode, and per-item state reset.

**Persistent work-item decode (`persist_decode`).** How $W$ is unflattened to
$(\text{qb}, h_q, \text{bt})$ decides load balance *and* L2 locality:

- **`qb_major`** — `wi = qb·(H_q·B) + h_q·B + bt`. Putting the triangular causal
  cost index `qb` in the MSB spreads cheap + expensive query blocks across each
  CTA under grid-stride. But every 256-CTA grid-stride phase spans *all* KV
  heads at once → large L2 footprint (57% L2 hit at GQA-8).
- **`hkv_major`** — `wi = hkv·(NQB·g·B) + blk·(g·B) + h_ql·B + bt`, with `blk`
  folded to a **low/high-paired** query-block index (`blk < half → qb = blk`;
  else `qb = NQB−1−(blk−half)`). Putting `hkv` in the MSB keeps each grid-stride
  phase within ~1 KV head so the shared GQA K/V stays L2-resident across its $g$
  query heads (**L2 hit 57% → ~93%, HBM misses 5.9× lower**); the low/high qb
  pairing preserves `qb_major`'s causal-triangle balance. Valid only when the
  CTA grid-strides across both halves of a KV head ($g\cdot NQB\cdot B \ge 2\,NP$).
- **`auto`** (default) — `hkv_major` when it is balance-safe **and** GQA
  ($g>1$), else `qb_major`. Strictly ≥ `qb_major`.

**Measured (MI355X, bf16, D=128, causal, $S = 8192$, 128/8 GQA, 0 spill, err
≈1.46e-3).** Absolute MI355X TFLOPS swing **±25–30% with auto-clock**, so only
**same-session ratios are load-bearing**; the numbers below are one representative
session, each pinned to its config (grid / decode / V-pad / lazy):

| config | grid | decode | V-pad | lazy | TFLOPS |
|---|---|---|---|---:|---:|
| default grid | one-CTA/q-block | — | 32 | on | ≈543 |
| persistent baseline | persistent NP=256 | qb-major | 0 | off | ≈877 |
| persistent + V-pad | persistent NP=256 | qb-major | 32 | off | ≈912 |
| **persistent (shipped default)** | persistent NP=256 | **hkv-major** | 32 | on | **≈948** |

The clock-invariant deltas are the load-bearing part: **hkv/qb ≈ 1.04×** (L2 hit
57%→~93%), **V-pad 0→32 ≈ +5%**, **lazy ≈ +2%** — and the shipped default
(`persistent=True`, `persist_decode="auto"`, `lazy_rescale=True`,
`ROCKE_DENSE_VPAD=32`, the last row) is byte-identical IR to a re-measure, so the
ratio reproduces regardless of the absolute clock. The persistent variant is the
production choice for dense prefill.

### 8.3 Ragged sequence lengths — on-chip boundary padding

The 256-row / `block_n`-key tile geometry assumes aligned lengths. Real prompts
are arbitrary, and this layout **cannot pad on the host** (no room to grow the
dense buffers). Instead a **separate kernel path** (`ragged=True`, its own
`kernel_name`) pads the boundary tiles *on-chip* — everything else is the §8.1
pipeline unchanged. Because seqlen is compile-time, the raggedness is fully known
at build time:

- **Grid ceil'd** — `⌈S_q/256⌉` query blocks (persistent: the work-item count is
  ceil'd), so the partial last query block is covered.
- **OOB query rows → register-zero pad.** Q is loaded through a bounds-checked
  `buffer_load_vN` (buffer-resource extent = the real Q buffer); rows past
  $S_q$ return 0 instead of faulting.
- **OOB keys → LDS-zero pad.** The K/V async DMA already goes through buffer
  resources, so keys past $S_k$ load as 0 into LDS.
- **Masking is nearly free for causal.** A padded key has token index
  $\ge S_k > $ every real query index, so the causal mask ($k_j \le q_i$)
  *already* drops it — causal ragged needs **no extra key mask**. Non-causal
  adds a compile-time `k_j < S_k` key mask on the partial tile.
- **Partial output rows dropped.** The epilogue store is wrapped in a per-lane
  `q_i < S_q` guard, so padded rows never write (and never clobber a neighbouring
  batch's real rows — correct for any $B$).

Self-attention only ($S_q = S_k$); not combined with varlen or the sliding
window (the validator rejects those). The aligned path is emitted
**byte-identically** when `ragged=False`, so ragged support costs the aligned
schedule nothing; the ragged kernel itself runs at **~890 TFLOPS** (≈94% of the
948 aligned headline), the delta being the bounds-checked load + guarded store.

---

## 9. Where the algorithm ends and tuning begins

The math above is fixed and exact: online softmax, the bias/mask order of §1, the
associative split-KV merge. Everything the README tunes — the combo flag bundle,
`waves_per_eu`, `num_warps`, `tile_size`, async-DMA issue order, i64 addressing,
fp8 dequant — changes only **how these steps are scheduled onto gfx950**, never
what is computed. Correctness is pinned by bit-exact parity against Triton and
ULP-level agreement with `ref_paged_attn` (unified paths) / torch SDPA (dense);
for the dense kernel a golden LLVM-IR SHA gate additionally pins that every
tuning change to one variant leaves the others' machine code byte-identical.
Performance is the per-CTA schedule, the 2D-vs-3D occupancy choice, and the
dense default-vs-persistent grid.

---

## 10. Where to go next

- [`README.md`](README.md) — the parity harness, the three-table methodology, the
  prefill-2D optimization history (combo, `waves_per_eu`, i64 addressing, fp8),
  and the substantiated speedups.
- `parity_unified_attention.py` — the canonical parity + benchmark harness
  (default / creative / fmha scenario sets).
- The kernels themselves live in `rocke` (not in this folder): the unified paged
  kernels are `kernels/gfx950/attention_tiled_2d.py`,
  `kernels/gfx950/attention_tiled_3d.py`, and `kernels/common/attention_unified.py`
  (the dispatcher and shared spec); the dense prefill kernel (§8) is
  `kernels/gfx950/attention_dense.py`, gated by the `attention_dense` family of
  the platform parity harness (IR-SHA, in CI via `rocke_golden_static`) plus
  `tests/test_attention_ir_cpp_parity.py` (C++/Python byte-identity).
