# The skinny-M decode GEMM, from the math up

A from-the-ground-up explanation of the matmul this example optimizes — what it
computes, *why* it is bandwidth-bound rather than compute-bound, what "skinny M"
does to the tile geometry, and how each lever in the [`README.md`](README.md)
maps onto a concrete part of the kernel. The README is the optimization
*history*; this file is the *specification and the reasoning* behind the kernel's
shape.

> If you just want to *run* the sweep, see [`README.md`](README.md). This file
> explains *why the winning configuration looks the way it does*.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $A$ | $M \times K$ | activations (the decode tokens) |
| $B$ | $N \times K$ | weight matrix; the `C` of `RCR` is B's layout (column-major operand, stored $[N,K]$) |
| $C$ | $M \times N$ | output |
| $M$ | scalar | tokens in the batch — **2** for this decode shape |
| $N$ | scalar | output features — **4096** |
| $K$ | scalar | contraction (input features) — **4096** |
| `tile_m`, `tile_n`, `tile_k` | scalar | the macro-tile a workgroup (CTA) owns |
| `warp_m × warp_n × warp_k` | scalar | the wave grid inside a CTA |
| atom | — | the MFMA instruction, here `mfma_f32_16x16x32_bf16` |

The concrete problem is the **Qwen3-8B `o_proj` decode matmul**: `bf16`,
$M=2$, $N=4096$, $K=4096$, `RCR` layout (row-major $A$, column-major $B$, row-major
$C$ — i.e. $C = A B^{\top}$ with $B$ stored as $[N, K]$).

---

## 1. What the kernel computes (the specification)

It is an ordinary matrix multiply with an f32 accumulator and a bf16 result:

$$
C_{ij} \;=\; \sum_{k=0}^{K-1} A_{ik}\, B_{jk}, \qquad
C \in \mathbb{R}^{M\times N},\; \text{accumulated in f32, stored bf16}.
$$

There is no fusion, no softmax, no quantization — the entire mathematical object
is this sum. **The interesting part is not the math; it is that for $M=2$ this
matmul is bound by how fast you can read $B$ from HBM, not by the multiply-add
throughput.** Everything below follows from that one fact.

---

## 2. Why this shape is bandwidth-bound (the arithmetic intensity lemma)

The total floating-point work is $2MNK$ multiply-adds. The unavoidable memory
traffic (each operand read once, the result written once) is

$$
\text{bytes} \;=\; \underbrace{2 M K}_{A} \;+\; \underbrace{2 N K}_{B} \;+\;
\underbrace{2 M N}_{C} \quad(\text{2 bytes per bf16 element}).
$$

Plug in $M=2,\,N=K=4096$:

$$
\begin{aligned}
A &= 2\cdot 2\cdot 4096 = 16\ \text{KiB} \\
B &= 2\cdot 4096\cdot 4096 = 32\ \text{MiB} \quad(\textbf{99.9\% of the traffic}) \\
C &= 2\cdot 2\cdot 4096 = 16\ \text{KiB}
\end{aligned}
$$

The **arithmetic intensity** — FLOPs per byte — is

$$
\frac{2MNK}{2(MK + NK + MN)} \;=\; \frac{MNK}{MK+NK+MN}
\;\approx\; \frac{M\,NK}{NK} \;=\; M \;=\; 2 \quad(\text{since } NK \gg MK, MN).
$$

**Intensity $\approx M$.** A tall, dense matmul (large $M$) reuses each weight
across many rows and is compute-bound; a decode matmul ($M=2$) reuses each weight
across only 2 rows, so it is **pinned at intensity $\approx 2$ FLOP/byte** — far
below the part's compute:bandwidth ratio. On MI355X (≈8 TB/s HBM, ≈2.5 PFLOP/s
bf16) the crossover intensity is ~300 FLOP/byte; at 2 we are ~150× below it. The
matmul *cannot* be compute-bound. **The whole campaign is therefore about
streaming $B$'s 32 MiB as fast as the part allows, and nothing else moves the
needle.**

The hard floor: $33.5\ \text{MB} / 8\ \text{TB/s} \approx 4.2\ \mu s$. The
sustained-bf16 streaming ceiling the tuned kernel reaches on this part is
~3.3 TB/s ≈ 40 % of peak (the winning config: 33.6 MB / 10.30 µs ≈ 3.26 TB/s,
matching the README's measured ~40 % HBM), which is where it and rocBLAS both
land (DSL 10.30 µs vs rocBLAS 10.18 µs best — within noise; `data/22_confirm_winner.json`).

---

## 3. From spec to kernel: who computes what

The matmul is tiled the standard way, but the tile *shape* is dictated by §2.

- **Grid (independent work).** One CTA per output macro-tile
  $(\texttt{m\_tile}, \texttt{n\_tile})$. With `tile_m=16, tile_n=16` and
  $M=2,\,N=4096$ that is $\lceil M/16\rceil \times \lceil N/16\rceil = 1 \times 256
  = 256$ CTAs — exactly enough to give each of the ~256 CUs one block. Choosing a
  *wider* `tile_n` shrinks this grid (fewer CTAs) and idles CUs; choosing the
  *smallest* `tile_n` maximizes the CTA count that streams $B$ in parallel. This is
  why §3.3 "memory-bound ⇒ small-N tile wins" holds and why the sweep's first
  winner is the smallest-N candidate.
- **The K-loop is the stream.** Each CTA walks $K$ in `tile_k`-wide steps, loading
  one `tile_k` slab of $A$ and $B$ per trip and feeding them to the MFMA. The trip
  count is $K/\texttt{tile\_k}$, so a deeper `tile_k` means fewer loop trips and
  less per-trip overhead (waitcnts, address math) — the "K-pack" lever (§5).
- **The matrix unit.** The atom is `mfma_f32_16x16x32_bf16`: it multiplies a
  $16\times32$ bf16 $A$-fragment by a $32\times16$ bf16 $B$-fragment and
  accumulates a $16\times16$ f32 tile, contracting **32** elements of $K$ per
  instruction. Because intensity is $\approx 2$, the MFMA is *never* the
  bottleneck — it sits idle waiting on the $B$ stream — which is exactly why
  scheduling/pipeline knobs that "hide compute" are neutral here (§6).

---

## 4. The "skinny M" tax (padding, and why it is unavoidable)

The MFMA's $A$-fragment is $16$ rows tall, so `tile_m` must be a multiple of 16.
But $M=2$. The kernel therefore pads $M$ up to 16 and **wastes 14 of every 16
rows** of $A$-side compute and LDS. This is the "skinny-M tax," and it has two
consequences the README's levers chase:

1. **Don't make it worse.** A larger `tile_m` (e.g. 32) multiplies the padding
   while adding no real work — so `tile_m=16` is the floor, and bumping it loses
   (the `t32x128x32` sweep row).
2. **The down/output atomics scale with padded M, not real M.** When a kernel
   variant (e.g. the fused-MoE-style or split-K paths) tiles M, the padded rows
   still issue work; minimizing `tile_m` minimizes that waste.

The tax is intrinsic to the MFMA shape — there is no $1$- or $2$-row matrix atom
— so the kernel pays it and instead spends its optimization budget on the $B$
stream, where the bytes actually are.

---

## 5. The K-loop body, step by step

Fix a CTA owning output tile $(\texttt{m\_tile}, \texttt{n\_tile})$. One K-loop
trip processes a `tile_k`-wide slab:

### 5.1 — Load the $A$ and $B$ slabs

Each trip loads $A[\,:,\,k{:}k{+}\texttt{tile\_k}\,]$ and the CTA's
$B[\texttt{n\_tile},\,k{:}k{+}\texttt{tile\_k}\,]$ slab. Two routes exist:

- **Round-trip (default):** `global_load_dwordx4` → VGPR → `ds_write_b128` → LDS,
  then `ds_read_b128` into the MFMA-input layout. Three hops, extra VGPRs.
- **Direct-to-LDS (DTLA, the `direct_to_lds` lever):**
  `buffer_load_dwordx4 … lds` writes the dword payload **straight into LDS**,
  skipping the VGPR stage. This is the hardware path Tensile's `DTLA1/DTLB1` token
  selects. It frees 32 VGPRs/trip and is what makes a deep `tile_k` (1024) viable
  — the round-trip path's VGPR pressure collapses occupancy at `tile_k ≥ 1024`.

### 5.2 — The MFMA accumulation

For each $16\times32$ sub-slab of the loaded tile:

$$
\texttt{acc} \mathrel{+}= \texttt{MFMA}_{16\times16\times32}\big(A_{\text{frag}},\, B_{\text{frag}}\big),
\qquad \texttt{acc}\in\mathbb{R}^{16\times16}\ (\text{f32}).
$$

The accumulator rides the `scf.for` K-loop as a loop-carried value; it is small
(one $16\times16$ f32 tile per warp), so it lives in registers, not memory.

### 5.3 — The epilogue (`cshuffle`)

After the last K-trip, the f32 accumulator is shuffled through LDS into a
coalesced store layout and written to $C$ as bf16 (`cshuffle` epilogue). For
$M=2$ this is 16 KiB — negligible against $B$'s 32 MiB.

**`tile_k` is the K-pack lever (§5).** Deeper `tile_k` ⇒ fewer K-loop trips ⇒ less
per-trip waitcnt/control overhead, *until* the LDS slab ($\approx 4\cdot\texttt{tile\_k}+512$
bytes on the `t16×16` `mem` pipeline) crowds occupancy below the round-trip-hiding
threshold. The measured sweet spot is `tile_k=512` for the round-trip path and
`tile_k=1024` once DTLA frees the VGPRs — exactly the `MT 16×16×512`/`×1024` tile
hipBLASLt itself picks for this shape.

---

## 6. Why the scheduling knobs are (mostly) neutral

The `mem` / `compv3` / `compv4` pipelines and the `interwave` / `intrawave`
schedulers trade LDS and register budget for **compute–memory overlap** — they
hide MFMA latency behind in-flight loads, or vice versa. But §2 says the MFMA is
already idle waiting on $B$; there is *no compute to hide*. So on the matching
tile these knobs cluster within noise (the three `t16×128×32` rows are identical),
and DTLA — which removes 32 instructions — is even a small *regression* on its
own, because the freed issue slots have no useful overlapping work to fill them.

The corollary, and the campaign's recurring lesson: **a scheduling knob that
crosses no real constraint is at best neutral on a bandwidth-bound kernel.** The
levers that *do* move it all change the byte stream or the device fill, not the
instruction schedule.

---

## 7. The three levers that actually move it

Each attacks the $B$ stream or the device occupancy directly:

### 7.1 — Direct-to-LDS (DTLA), §5.1
Removes the global→VGPR→`ds_write` round-trip for both operands. *Alone* it is
neutral-to-negative, but it frees the VGPRs that let `tile_k` reach 1024 — and
deep-K is what cuts the K-loop trip count. DTLA is enabling, not directly winning.

### 7.2 — Chiplet swizzle, §7
MI355X has **8 XCDs**, each with its own L2. The default WG dispatch scatters
consecutive CTAs across XCDs, so each XCD's L2 sees uncorrelated traffic. At
$M=2$ the **same 16 KiB $A$ tile is reused by every CTA in the M-row** — exactly
the cross-CTA reuse an L2 can hold. The swizzle (`chiplet_aware_super_tile_dynamic`,
the runtime-WGID variant the emitter calls in `gemm_universal.py`) remaps WGIDs so
consecutive WGs land on the *same* XCD, keeping $A$ resident in
one L2. The win is small (~2.1 % in the step-21 sweep, ~2.4 % in the cleaner
step-22 serial re-measure; `data/21_chiplet.json` / `data/22_confirm_winner.json`)
because $A$ is only 16 KiB, but it is real and
above the noise floor — and it only appears at a shape with many CTAs per M-row
(at the small-tile-count shapes of the earlier `extra_levers` step, the swizzle
remapped a single chunk and was a no-op).

### 7.3 — Split-K (the structural follow-up), §§ Steps 23–24
The first two levers make a *saturated* config stream bytes efficiently. But at
decode the grid is **not** saturated: $\lceil M/\texttt{tile\_m}\rceil \approx 1$
m-tile, so the base grid is just $n_\text{tiles}$ CTAs against ~256 CUs — most of
the device is idle. Split-K slices $K$ into `split_k` chunks, launches `split_k×`
more CTAs (each computing a partial sum over its K-slice), and reduces them by
**atomic add into an f32 workspace**:

$$
C_{ij} \;=\; \sum_{z=0}^{\texttt{split\_k}-1}\ \underbrace{\sum_{k\in\text{slice }z} A_{ik}B_{jk}}_{\text{one CTA's partial}} ,
\qquad \texttt{split\_k}=1\ \text{keeps the single-pass body byte-identical}.
$$

The atomic add makes the slice order irrelevant, so the kernel stays embarrassingly
parallel. The degree is chosen from **per-slice K-depth** (≈ $K/512$, snapped to a
factor that evenly slices $K$): $K=4096 \to \texttt{split\_k}=8$ (slice 512),
$K=2048 \to 4$. The heuristic engages *only* when the base grid leaves the device
idle ($\texttt{base\_grid} < \texttt{target\_ctas}/2$); a square/prefill grid that
already fills the device keeps `split_k=1`, so the lever is default-safe.

**Why this does not contradict the "split-K doesn't help" ceiling note.** That
note is about a *single config already streaming at the sustained-bf16 wall* —
for it, splitting $K$ only replicates $B$ traffic. Split-K here addresses the
orthogonal failure mode: the decode grid never reached that wall because it was
device-idle. The two statements describe different operating points (saturated vs
idle).

---

## 8. The whole kernel in pseudo-code

```
# Grid = (n_tiles, m_tiles, split_k);  m_tiles ≈ 1 for decode
for each CTA (n_tile, m_tile, z):
    acc = 0                                        # 16x16 f32, per warp, in registers
    for k in range(z*ks, (z+1)*ks, tile_k):        # ks = K/split_k; full K if split_k==1
        A_slab = load A[m_tile, k:k+tile_k]         # -> LDS (DTLA or round-trip)
        B_slab = load B[n_tile, k:k+tile_k]         # -> LDS; B is 99.9% of traffic
        for sub in range(tile_k // 32):             # 32 = atom K
            acc += MFMA_16x16x32_bf16(A_frag[sub], B_frag[sub])
    if split_k == 1:
        C[m_tile, n_tile] = bf16(acc)               # cshuffle epilogue
    else:
        atomic_add(Cf32[m_tile, n_tile], acc)       # f32 workspace; cast to bf16 after
```

Every README lever changes only *how this streams onto the hardware*:
`tile_n`/`tile_m` set the grid and the skinny-M padding (§§3–4); `tile_k` +
DTLA set the K-loop depth and the load path (§5, §7.1); the pipeline/scheduler
knobs reorder issue and are neutral on this bound (§6); chiplet swizzle steers
$A$'s L2 residency (§7.2); split-K fills the idle CUs (§7.3). The math of §1 is
fixed; the engineering is all in feeding $B$ past an otherwise-idle matrix unit.

---

## 9. Where the algorithm ends and tuning begins

The sum of §1 is exact and invariant; the README's 24 steps change only the
*schedule* — which never alters what is computed (every variant is correctness-
gated bit-exact, or `rel < 5e-2` for the split-K f32-workspace path, before any
timing). The honest result is that the tuned single-pass kernel reaches the
part's sustained-bf16 ceiling (~40 % HBM, on par with rocBLAS), and split-K is the
one structural change that extends past a saturated single config by recruiting
the idle CUs the decode grid leaves behind.

## 10. Where to go next

- [`README.md`](README.md) — the step-by-step optimization case study: the static
  occupancy probe, the hygiene-disciplined sweeps, the rocprof ground-truth
  reading, every negative result, and the DTLA / chiplet / split-K wins.
- `scripts/02_sweep_bench.py` — the tile/pipeline sweep this document's §§3–6
  predict the outcome of.
- `scripts/21_chiplet.py` — the §7.2 chiplet-swizzle win.
- `scripts/23_splitk_universal.py` / `scripts/24_splitk_dispatch.py` — the §7.3
  split-K body and its dispatcher heuristic.
