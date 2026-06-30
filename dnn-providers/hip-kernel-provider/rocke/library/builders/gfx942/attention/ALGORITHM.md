# gfx942 unified attention, from the math up

A derivation of what the **gfx942 (CDNA3 / MI300X) tiled SDPA-forward** kernels
compute, and *why they are shaped the way they are* — the online-softmax
recurrence, the paged-KV layout the example feeds them, and the CDNA3-specific
matmul choice (the transposed-x8 flash regime) that the harness exercises.

This is written for a reader with a math / CS background who has **not** seen
flash attention before. We start from the definition of attention, derive the
streaming-softmax recurrence that makes it tile-able, and then map that
recurrence onto what the two shipped paths — the **narrow** `16x16x16` default
and the **transposed-x8** flash regime — actually do.

> If you just want to *run* the example, see [`README.md`](README.md). The two
> kernels themselves live in `rocke.instances.gfx942.attention_tiled_2d`
> (prefill) and `..._tiled_3d` (the split-KV decode path); this folder is a
> torch-reference parity + latency harness over them.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $Q$ | $S_q \times d$ | query matrix, one $(\text{batch}, \text{head})$ slice |
| $K$ | $S_k \times d$ | key matrix |
| $V$ | $S_k \times d$ | value matrix |
| $O$ | $S_q \times d$ | output (same shape as $Q$) |
| $S_q, S_k$ | scalar | sequence lengths (query, key) |
| $d$ | scalar | head dimension (`head_size`, here 64 or 128) |
| $\tau$ | scalar | softmax scale, $\tau = 1/\sqrt{d}$ |
| $T$ | scalar | the KV tile width (`block_size`, 64 in this example) |

A subscript like $Q_i$ denotes row $i$; $S_{i,:}$ denotes row $i$ of a score
matrix. $\exp$, $\max$ act rowwise.

---

## 1. What attention computes (the specification)

Scaled dot-product attention, for one head of one batch element, is

$$
O = \operatorname{softmax}\!\left(\tau\, Q K^{\top}\right) V,
\qquad \tau = \frac{1}{\sqrt{d}} .
$$

Written one query row $i$ at a time, with the **score vector**
$s = \tau\, Q_i K^{\top} \in \mathbb{R}^{S_k}$ (one score per key):

$$
O_i \;=\; \sum_{j=1}^{S_k} \underbrace{\frac{e^{\,s_j}}{\sum_{j'} e^{\,s_{j'}}}}_{\text{attention weight } a_j}\, V_j .
$$

So $O_i$ is a **convex combination of the value rows**, weighted by how much
query $i$ attends to key $j$. That is the entire mathematical object. The rest of
this document is about computing it **without ever materializing the
$S_q \times S_k$ score matrix** — which for long sequences (the `long_prefill`
group reaches $S=8192$) does not fit in registers or LDS.

---

## 2. The softmax stability trick

Softmax is invariant to subtracting any constant $c$ from every score:

$$
\frac{e^{\,s_j}}{\sum_{j'} e^{\,s_{j'}}}
= \frac{e^{\,s_j - c}}{\sum_{j'} e^{\,s_{j'} - c}}
\qquad\text{for any } c .
$$

In finite precision $e^{s_j}$ overflows for large $s_j$; choosing
$c = \max_j s_j$ keeps the largest exponent at $e^0 = 1$. **The whole difficulty
of flash attention is that we want this max, but we process keys in tiles and do
not know the global max until we have seen every tile.** Section 3 resolves that.

---

## 3. Online (streaming) softmax — the heart of the algorithm

Process the $S_k$ keys in **tiles** of width $T$. For a fixed query row, write
the score sub-vector of tile $t$ as $s^{(t)} \in \mathbb{R}^{T}$. Maintain three
running quantities, updated tile by tile:

- $m^{(t)}$ — the running **max** of all scores seen through tile $t$,
- $\ell^{(t)}$ — the running **denominator** $\sum e^{s - m^{(t)}}$,
- $\mathbf{o}^{(t)} \in \mathbb{R}^{d}$ — the running **un-normalized output**.

**Initial state**: $m^{(0)} = -\infty$, $\ell^{(0)} = 0$,
$\mathbf{o}^{(0)} = \mathbf{0}$.

**Update for tile $t$.** Extend the running max, then form the rescale factor

$$
m^{(t)} = \max\!\big(m^{(t-1)},\; \max_j s^{(t)}_j\big), \qquad
\boxed{\;\alpha^{(t)} = e^{\,m^{(t-1)} - m^{(t)}}\;}\quad (0 < \alpha^{(t)} \le 1),
$$

and apply it before adding this tile's contribution:

$$
\begin{aligned}
p^{(t)}_j &= e^{\,s^{(t)}_j - m^{(t)}} & &\text{(tile weights, current max)}\\[2pt]
\ell^{(t)} &= \alpha^{(t)}\,\ell^{(t-1)} + \textstyle\sum_j p^{(t)}_j & &\text{(rescale old sum, add new)}\\[2pt]
\mathbf{o}^{(t)} &= \alpha^{(t)}\,\mathbf{o}^{(t-1)} + \textstyle\sum_j p^{(t)}_j\, V^{(t)}_j & &\text{(rescale old output, add new)}
\end{aligned}
$$

**Final normalization** after the last tile: $O_i = \mathbf{o}^{(N)} / \ell^{(N)}$.

By induction over $t$, $\alpha^{(t)}$ re-bases every previously-stored term from
"$-m^{(t-1)}$" to "$-m^{(t)}$", so at $t = N$ the max is global and
$\mathbf{o}^{(N)}/\ell^{(N)}$ equals the true softmax output of Section 1. The
streaming computation is **exact**, not an approximation — it never holds more
than one $T$-wide tile of scores at a time. This is what makes attention
$O(S_k)$ memory instead of $O(S_k^2)$, and it is the recurrence both gfx942
kernels implement.

---

## 4. Causal masking and the early-exit

This example is **causal-only for prefill** (the kernel always applies a causal
mask and has no non-causal prefill mode — see `final_shapes_check.py`). Causal
attention forbids a query at position $q$ from attending to keys $k > q$: in the
tile update, $s_j \leftarrow -\infty$ whenever key $j$ is in the future of the
query row, which makes its weight $e^{-\infty} = 0$.

Naively you still compute every score and mask the upper triangle — ~2× the
matmul the math needs. The standard win is to **stop the K-loop early**: a query
tile only needs KV tiles whose lowest key index is $\le$ its largest query
position, so fully-future tiles are skipped entirely. The `final_shapes_check.py`
reference mirrors the kernel's masking with a **bottom-right-aligned** causal
mask (`torch.triu(..., diagonal = kv_len - query_len + 1)`), which is the correct
alignment when $S_q \ne S_k$ — and a known point of divergence from Torch's
`is_causal=True` (top-left aligned), called out in that script's docstring.

**Decode** (`seqlen_q == 1`) is run non-causal: a single query attends to all
keys, which is identical to causal at $q = 1$.

---

## 5. The data layout the example feeds: paged KV

The harness does **not** hand the kernel a dense $[B, H_q, S_q, d]$ tensor; it
maps each dense SDPA problem onto the **paged-KV** layout the production
dispatcher uses (`parity_unified_attention.py::make_inputs`):

- Each batch element becomes one sequence with
  $(\text{query\_len}, \text{kv\_len}) = (\text{seqlen\_q}, \text{seqlen\_k})$.
- The KV cache is `[num_blocks, block_size, kv_heads, head_size]` with
  `block_size = 64`; a per-sequence **block table** is a contiguous,
  non-overlapping run of `block_size`-token blocks, so each sequence's KV working
  set is genuinely distinct.
- Queries are packed `[total_q, heads, head_size]` with a `cu_seqlens_q` prefix
  sum; `seqused_k` carries each sequence's KV length.

`block_size = 64` is chosen to match the gfx942 flash-regime tile ($T = 64$), so
the wide / narrow geometries below apply cleanly. **Grouped-query attention
(GQA)** is expressed by `heads > kv_heads`: a group of `heads / kv_heads` query
heads shares one KV head (the reference `repeat_interleave`s K/V to match).

The launch grid is recomputed exactly as the production dispatcher does —
`grid = (kv_heads, total_num_q_blocks, 1)` (or transposed under the
`use_q_major_grid` flag), `block = (64 * num_warps, 1, 1)` — so the example
exercises the same build + launch plumbing as the provider.

---

## 6. Two matmul orientations: narrow vs transposed-x8

The recurrence of Section 3 is fixed; the two shipped gfx942 paths differ only in
**how the two matmuls ($QK^\top$ and $PV$) are mapped onto the CDNA3 matrix
unit**. The harness selects between them in `_build_spec`.

### 6.1 The narrow default (`16x16x16`)

In **this harness's** `_build_spec`, every shape that is *not* D128 fp16 uses the
default narrow path: the `mfma_16x16x16` atom, with `num_warps` keyed on
`block_size` (D64 ships `num_warps=4` + `block_m_per_warp=32`; D128 bf16 stays
`num_warps=2`). Scores $S = \tau Q K^\top$ are produced as the MFMA accumulator,
the softmax recurrence runs over them, the probabilities $P$ are staged to LDS
(`P_lds`), and the PV matmul reads them back as the A-operand. This is the
general-purpose path and the one whose LDS footprint the `supports_tiled_2d` gate
models.

> **Note — harness vs production dispatcher.** This narrow/flash split is the
> *harness's* explicit choice (`parity_unified_attention.py::_build_spec` routes
> only D128 fp16 to the flash regime). The production dispatcher's gfx942 fp16
> flash family is broader: `_enable_gfx942_fp16_flash` now admits **D64 fp16 too**
> (`head_size in (64, 128)`), running the same gfx942-legal `32x32x8` atom plus a
> sliced-K / cfvst stack. So when `final_shapes_check.py` runs through the
> production dispatcher (Sections 8–9), a D64 fp16 prefill shape is *not* on the
> narrow path described here — it takes the dispatcher's D64 flash path. The
> bf16 fallback to narrow (Section 6.2) holds in both the harness and the
> dispatcher.

### 6.2 The transposed-x8 flash regime (D128 fp16)

For **D128 fp16**, the harness routes to the flash regime (`use_mfma_32x32x8 = True`
+ `use_transposed_qk_32x32 = True`), which is the central CDNA3 trick. Rather than
computing $S = Q K^\top$, the kernel computes the **transpose**

$$
S^{\top} = K\,Q^{\top}
$$

via the gfx942-legal `mfma_f32_32x32x8_f16` atom (a wider-M MFMA: M=32, N=32,
K-step 8, `<16 x f32>` accumulator). The payoff: the probabilities come out already in the
orientation the PV matmul needs as its B-operand, so $P^{\top}$ stays
**register-resident** and feeds the PV MFMA directly — **no `P_lds` round-trip**.
The output accumulator's LDS region (`Acc_lds`) is epilogue-only and is
backend-aliased into the loop-dead K/V LDS region. Consequences the harness and
provider account for:

- **`use_k_single_buffer`** is set to `num_warps * 32 <= block_size`: when the
  M-tile ($\text{num\_warps} \times 32$) fits the tile width $T$, K is single-buffered;
  wider tiles double-buffer K.
- **`supports_tiled_2d`'s LDS gate over-counts this path** (it assumes `P_lds`
  present + K double-buffered + a full `Acc_lds`), so the harness *skips* the
  generic gate for the flash configs and lets the spec's own `__post_init__` +
  comgr be the arbiters — mirroring the provider's accurate
  transposed-x8 LDS model. The generic gate is applied only to the narrow path.

`use_mfma_32x32x8` is **fp16-only** on this kernel path — the gfx942 spec's
`__post_init__` rejects it for bf16 ("use_mfma_32x32x8 is fp16-only"). Note this
is a *kernel-path* restriction, not a hardware one: gfx942 does have the bf16
`32x32x8` atom (`mfma_f32_32x32x8_bf16` is in the gfx942 MMA catalog and lowers
through the `..._1k` builtin), but the transposed-x8 attention path is only wired
and validated for fp16. That is why D128 **bf16** falls back to the narrow
`16x16x16` path (Section 6.1) rather than the flash regime.

---

## 7. The wide4 default vs the L4 contrast

Within the D128-fp16 flash regime, the harness reproduces the two geometries the
provider ships and demonstrates the difference explicitly (this is the point of
`parity_unified_attention.py`, per its docstring):

| config | `num_warps` | workgroup | K buffering | when |
|---|---:|---:|---|---|
| **wide4** | 4 | 256 threads | double-buffered (`32·4 > 64`) | the provider's analytic default for qualifying D128 fp16 |
| **L4**    | 1 | 64 threads  | single-buffered (`32·1 ≤ 64`) | the `HIPDNN_GFX942_FLASH_WIDE=0` contrast |

The important subtlety the docstring records: **wide4 is the *provider's*
analytic default, not the DSL spec's default.** A spec built with no flash knobs
lands on L4 (WG=64). So `parity_unified_attention.py` sets `num_warps=4` (+
`use_mfma_32x32x8` / `use_transposed_qk_32x32` / `use_k_single_buffer=False`)
*explicitly* to reproduce the shipped peak; `HIPDNN_GFX942_FLASH_WIDE=0` selects
the L4 geometry for an A/B. `HIPDNN_GFX942_FLASH_WIDE=2` selects an intermediate
wide2 (`num_warps=2`, WG=128, `BLOCK_M = 32·2 = 64 ≤ T`, so K single-buffers).

---

## 8. The decode path (3D split-KV)

Decode (`seqlen_q == 1`) is structurally different: there is one query row but a
long KV sequence, so parallelism comes from **splitting the KV dimension** across
threadgroups (the `attention_tiled_3d` path) and combining the partial
softmax-weighted outputs. The harness routes decode through the production
dispatcher `run_unified_attention_torch(..., backend="auto")`
(`final_shapes_check.py`), which selects the 3D path and — for the
overhead-dominated decode regime — can engage an internal CUDA-graph replay
(Section 9).

---

## 9. Host overhead and the graph heuristic

For tiny kernels (decode, short prefill) the per-launch **host** cost (Python
dispatch + kernarg pack + `hipModuleLaunchKernel`) is a large fraction of total
time, so CUDA-graph replay — which removes exactly that host cost — pays off. The
dispatcher encodes this in `_recommend_graph_replay(problem)`
(`attention_unified.py`), verified to return:

| regime | rule | graph? |
|---|---|:---:|
| decode | `max_seqlen_q == 1` | yes (3D) |
| short prefill | `max_seqlen_q <= 768` | yes (2D, or 3D if routed there) |
| long prefill | `max_seqlen_q > 768` | no (kernel-bound; overhead is noise) |

Feature-flagged problems (sinks / alibi / qq-bias / softcap / sliding-window /
fp8) are excluded. The internal graph is engaged only when the caller is **not**
already capturing (a framework that graphs the whole forward takes precedence),
and is toggled by `HIPDNN_GFX942_2D_GRAPH` / `HIPDNN_GFX942_3D_GRAPH` (both
default-on). `final_shapes_check.py` measures each shape in **two** modes — eager
(host + kernel) and graph (kernel-only, host overhead removed symmetrically) — so
a CK-vs-Torch gap can be attributed to kernel time vs launch overhead.

---

## 10. Numerical correctness gate

Both scripts gate the kernel output against an **fp32 paged reference**
(`ref_paged_attn`): gather the per-sequence K/V from the block table,
`repeat_interleave` for GQA, score in fp32, apply the bottom-right causal mask,
softmax, and weight $V$. The tolerance is `max_abs <= 2e-2` (fp16) / `4e-2`
(bf16) — wide enough to absorb the fp16/bf16 MFMA accumulation order, tight
enough to catch a real layout or masking bug. No latency number is reported for a
kernel that fails the gate.

---

## 11. Where the algorithm ends and tuning begins

The recurrence (Sections 3–4) and the paged layout (Section 5) are fixed; the
choices in Sections 6–9 — the matmul orientation, the wide4/L4 geometry, K
buffering, the grid transpose, and graphing — change only *how* the fixed
computation is mapped onto CDNA3, never *what* is computed. Correctness is pinned
by the fp32 gate (Section 10); the rest is the per-shape schedule the harness lets
you A/B through the documented `HIPDNN_GFX942_*` environment variables.
