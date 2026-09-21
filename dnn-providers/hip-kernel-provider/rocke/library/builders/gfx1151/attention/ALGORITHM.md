# The winning kernel, from the math up

A step-by-step derivation of the algorithm the **single-wave WMMA flash-attention
forward** kernel (`fmha_singlewave.py`, and its software-pipelined sibling
`fmha_pipelined.py`) actually computes — the campaign winner at ~11 TF on gfx1151.

This is written for a reader with a math / CS background who has **not** seen
flash attention before. We start from the definition of attention, derive the
online-softmax recurrence that makes it tile-able, and then map every line of
that recurrence onto what one wave32 does per loop iteration. Every step is given
in math first, then in "what the kernel does."

> If you just want to *run* the kernel, see [`README.md`](README.md). This file
> explains *why the code is shaped the way it is*.

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
| $B_k$ | scalar | key-tile width, $B_k = 16$ (the WMMA $K$ dimension) |

A subscript like $Q_i$ denotes row $i$; $S_{i,:}$ denotes row $i$ of a score
matrix. $\odot$ is elementwise multiply, $\exp$ and $\max$ act rowwise.

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
query $i$ "attends to" key $j$. That is the entire mathematical object. The rest
of this document is about computing it **without ever materializing the
$S_q \times S_k$ score matrix** — which for long sequences does not fit in
registers or even LDS.

---

## 2. The softmax stability trick (a one-line lemma we will need twice)

The softmax is invariant to subtracting any constant $c$ from every score,
because the constant factors out of numerator and denominator:

$$
\frac{e^{\,s_j}}{\sum_{j'} e^{\,s_{j'}}}
= \frac{e^{\,s_j - c}}{\sum_{j'} e^{\,s_{j'} - c}}
\qquad\text{for any } c .
$$

In finite precision $e^{s_j}$ overflows for $s_j \gtrsim 88$ (f32). The standard
fix is to choose $c = \max_j s_j$, so the largest exponent is $e^0 = 1$ and
nothing overflows. **The whole difficulty of flash attention is that we want
this max, but we are processing keys in tiles and do not know the global max
until we have seen every tile.** Section 3 resolves that.

---

## 3. Online (streaming) softmax — the heart of the algorithm

We process the $S_k$ keys in **tiles** of width $B_k = 16$. Let the tiles be
$t = 1, 2, \dots, N$ with $N = S_k / B_k$. For a fixed query row, write the
score sub-vector of tile $t$ as $s^{(t)} \in \mathbb{R}^{B_k}$.

We maintain three running quantities, updated tile by tile:

- $m^{(t)}$ — the running **max** of all scores seen through tile $t$,
- $\ell^{(t)}$ — the running **denominator** $\sum e^{s - m^{(t)}}$ (rescaled to the current max),
- $\mathbf{o}^{(t)} \in \mathbb{R}^{d}$ — the running **un-normalized output** $\sum e^{s - m^{(t)}}\, V$.

**Initial state** ($t=0$, before any tile): $m^{(0)} = -\infty$,
$\ell^{(0)} = 0$, $\mathbf{o}^{(0)} = \mathbf{0}$.

**Update for tile $t$.** First extend the running max with this tile's local max:

$$
m^{(t)} \;=\; \max\!\Big(m^{(t-1)},\; \max_j s^{(t)}_j\Big).
$$

Because the max may have *grown* ($m^{(t)} > m^{(t-1)}$), everything we
accumulated under the old max is now scaled by the wrong constant. The
correction factor is

$$
\boxed{\;\alpha^{(t)} \;=\; e^{\,m^{(t-1)} - m^{(t)}}\;}\qquad (0 < \alpha^{(t)} \le 1),
$$

and we apply it to both the denominator and the output accumulator before adding
this tile's contribution:

$$
\begin{aligned}
p^{(t)}_j &= e^{\,s^{(t)}_j - m^{(t)}} & &\text{(this tile's weights, current max)}\\[2pt]
\ell^{(t)} &= \alpha^{(t)}\,\ell^{(t-1)} \;+\; \textstyle\sum_j p^{(t)}_j & &\text{(rescale old sum, add new)}\\[2pt]
\mathbf{o}^{(t)} &= \alpha^{(t)}\,\mathbf{o}^{(t-1)} \;+\; \textstyle\sum_j p^{(t)}_j\, V^{(t)}_j & &\text{(rescale old output, add new)}
\end{aligned}
$$

**Final normalization** after the last tile:

$$
O_i \;=\; \frac{\mathbf{o}^{(N)}}{\ell^{(N)}} .
$$

### Why this is exactly the softmax (induction)

**Claim.** For every $t$,
$\displaystyle \mathbf{o}^{(t)} = \sum_{j \in \text{tiles} \le t} e^{\,s_j - m^{(t)}}\, V_j$
and $\displaystyle \ell^{(t)} = \sum_{j \in \text{tiles} \le t} e^{\,s_j - m^{(t)}}$.

*Base case* $t=0$: both sums are empty, $=0$. ✓

*Inductive step:* assume the claim at $t-1$. Multiplying by
$\alpha^{(t)} = e^{m^{(t-1)} - m^{(t)}}$ re-bases every stored term from "$-m^{(t-1)}$" to "$-m^{(t)}$":

$$
\alpha^{(t)} \sum_{j\le t-1} e^{\,s_j - m^{(t-1)}} V_j
= \sum_{j \le t-1} e^{\,s_j - m^{(t)}} V_j ,
$$

and adding $\sum_j p^{(t)}_j V^{(t)}_j = \sum_{j \in t} e^{s_j - m^{(t)}} V_j$
extends the sum to include tile $t$. Same argument for $\ell^{(t)}$. ∎

So at $t=N$, $m^{(N)}$ **is** the global max, and
$\mathbf{o}^{(N)}/\ell^{(N)} = \sum_j \frac{e^{s_j - m^{(N)}}}{\sum_{j'} e^{s_{j'} - m^{(N)}}} V_j$,
which by the Section-2 lemma equals the true softmax output. The streaming
computation is **exact**, not an approximation — it never holds more than one
$B_k$-wide tile of scores at a time, yet produces the identical result to the
dense formula.

---

## 4. From recurrence to kernel: who computes what

The kernel parallelizes the recurrence across the GPU like this:

- **Grid (independent work).** One workgroup (CTA) per
  $(\text{q-tile}, \text{head}, \text{batch})$. A q-tile is
  $16\,B_m$ **query rows** (`q_rows_per_cta = 16 * bm_tiles`); the single-wave
  winner uses `bm_tiles = 1`, so 16 rows (BM amplification was swept but
  always regressed — see the README). Different q-rows, heads,
  and batches share nothing, so they run fully independently — this is the
  "embarrassingly parallel" outer structure of attention.
- **Block (one wave).** Each CTA is a **single wave32** (32 lanes). The 16
  query rows × 16 keys of a tile map onto a $16\times16$ WMMA fragment; the wave
  computes the whole recurrence for its 16 rows.
- **The K-loop is the recurrence.** The `scf.for kt in 0..N` loop *is* the
  tile index $t$ from Section 3. The running state
  $\big(m^{(t)}, \ell^{(t)}, \mathbf{o}^{(t)}\big)$ is carried across iterations
  as **loop-carried values** (`iter_args`): `m` and `l` (one each per accumulator
  row-slot) and `acc` (the output accumulator, $\mathbf{o}$).

The matrix unit on gfx1151 is the WMMA instruction `wmma_f32_16x16x16_f16`: it
multiplies two $16\times16$ f16 fragments and accumulates into a $16\times16$ f32
result, $C \mathrel{+}= A B^{\top}$. **Note the transpose** — the hardware
computes $A B^{\top}$, not $AB$. This single fact dictates how $K$ and $V$ are
laid out below.

---

## 5. One K-loop iteration, step by step

Fix a CTA (16 query rows $Q \in \mathbb{R}^{16\times d}$) and a tile $t$ with
keys $K^{(t)}, V^{(t)} \in \mathbb{R}^{16\times d}$. Here is the body of the
loop, each step in math then in code.

### Step 5.1 — Scores $S = \tau\, Q (K^{(t)})^{\top}$ (the QK matmul)

$$
S \;=\; \tau\, Q\,(K^{(t)})^{\top} \;\in\; \mathbb{R}^{16\times 16}.
$$

Because WMMA natively computes $A B^{\top}$, setting $A = Q$ and $B = K^{(t)}$
gives exactly $Q (K^{(t)})^{\top}$ with **no transpose gymnastics** — this is the
one place the hardware's $B^{\top}$ convention is free. The head dimension $d$
exceeds the WMMA $K=16$, so we sum over $d/16$ sub-matmuls (`n_dk = head_size //
16`), accumulating into one f32 score fragment:

```python
score = WmmaTensor.zero_acc(b, atom, arch=arch)            # <8 x f32> C-fragment tile
for d in range(n_dk):                                      # contract over head dim, 16 at a time
    q_tile = load_wmma_tile(b, qwin, atom, lane, role="a", k_offset=d*16, lead=[c0])  # <16 x f16>
    k_tile = load_wmma_tile(b, kwin, atom, lane, role="b", k_offset=d*16, lead=[c0])  # <16 x f16>
    score  = wmma_mma(b, q_tile, k_tile, score)            # C += A B^T
```

The $\tau$ factor is folded in *after* the matmul (Step 5.2), not before, to keep
the matmul inputs in their natural range.

### Step 5.2 — Scale and mask

$$
s^{(t)}_j \;\leftarrow\; \tau\, S_{:,j}, \qquad
s^{(t)}_j \;\leftarrow\; \begin{cases} s^{(t)}_j & \text{key } j \text{ allowed}\\ -\infty & \text{key } j \text{ masked}\end{cases}
$$

Setting a masked score to $-\infty$ makes its weight $e^{-\infty} = 0$, i.e. the
key is dropped from the convex combination. (See Section 7 for what "allowed"
means under causal masking.)

```python
s_r = b.fmul(score.slot(b, r), scale_log2)          # tau folded into scale_log2
s_r = apply_attention_mask(b, s_r, mask_mode=cfg.mask_mode, k_idx=..., query_pos=...)
```

### Step 5.3 — Running max and the rescale factor $\alpha$

$$
m^{(t)} = \max\!\big(m^{(t-1)}, \operatorname{rowmax}_j s^{(t)}_j\big), \qquad
\alpha^{(t)} = e^{\,m^{(t-1)} - m^{(t)}} .
$$

The rowmax is a **reduction across the 16 lanes** that hold one score row
(`wave_reduce_max(..., lanes_per_row=16)` — a 4-stage XOR butterfly, masks
1,2,4,8). The same `lanes_per_row=16` reduction runs byte-for-byte on wave64
CDNA and wave32 WMMA: identical stage count and masks; the masks always stay
inside one wave half, so the only difference is the wave width, not the
reduction.

```python
row_max = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
m_new   = b.fmax(ms[t][r], row_max)
alpha   = b.exp2(b.fsub(ms[t][r], m_new))           # alpha = 2^(m_old - m_new)
```

### Step 5.4 — Tile weights $p$ and running denominator $\ell$

$$
p^{(t)}_j = e^{\,s^{(t)}_j - m^{(t)}}, \qquad
\ell^{(t)} = \alpha^{(t)}\,\ell^{(t-1)} + \textstyle\sum_j p^{(t)}_j .
$$

```python
p_r     = b.exp2(b.fsub(s_r, m_new))                # 2^(s - m_new)
row_sum = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
new_l   = b.fadd(b.fmul(ls[t][r], alpha), row_sum)  # l*alpha + rowsum
```

### Step 5.5 — Rescale the output accumulator

$$
\mathbf{o}^{(t)} \;\leftarrow\; \alpha^{(t)}\,\mathbf{o}^{(t-1)} \quad(\text{before adding tile } t).
$$

This is one vector multiply per head-dim sub-block (`WmmaTensor.scale`, a single
`v_mul`), with $\alpha$ packed into a `<8 x f32>` vector (`alpha_vec`, one slot
per accumulator row) and broadcast across the row's accumulator lanes:

```python
for d in range(n_dk):
    new_accs[t][d] = new_accs[t][d].scale(b, alpha_vec)
```

### Step 5.6 — Add this tile's contribution $\mathbf{o} \mathrel{+}= P^{(t)} V^{(t)}$ (the PV matmul)

$$
\mathbf{o}^{(t)} \;\mathrel{+}=\; P^{(t)} V^{(t)}, \qquad P^{(t)} \in \mathbb{R}^{16\times16},\; V^{(t)} \in \mathbb{R}^{16\times d}.
$$

Two layout problems, both caused by WMMA computing $A B^{\top}$:

1. **$P$ is in the wrong layout.** $P^{(t)}$ was produced in Step 5.1 as the WMMA
   *accumulator* (C) distribution, but to feed the next matmul it must be an *A*
   operand. The kernel transposes it through a small $16\times16$ **LDS** buffer:
   each lane scatters its score slots to their $(\text{row}, \text{col})$ cell, a
   barrier, then reads back its A-operand row as two contiguous `ds_read_b128`s +
   a concat. (`_transpose_p`.)
2. **$V$ must be presented as $V^{\top}$.** We want $P V$, but WMMA gives
   $A B^{\top}$; with $A = P$ we need $B$ such that $B^{\top} = V^{(t)}$, i.e.
   $B = (V^{(t)})^{\top}$ of shape $d \times 16$. For this lane's $d$-column the
   B-fragment is the column $V^{(t)}[\,:,\,d_{\text{col}}]$ — a **row-strided
   gather** down the 16 keys of the tile (`_load_v_b` walks $k=0..15$ at the
   per-token stride `stride_v_token`, holding the head-dim column $d_{\text{col}}$
   fixed). On this cache-resident APU that gather is cheaper than staging $V$
   through LDS (the central finding of the campaign — see the README; the
   `v_mode="lds_t"` transpose-staging path exists but measured a regression).

```python
p_a = _transpose_p(...)                       # C-dist -> A-operand, via LDS
p_tiles = [WmmaTensor(atom, "a", pa, arch) for pa in p_a]
for d in range(n_dk):
    v_b = _load_v_b(...)                       # column gather of V (the B^T trick)
    v_tile = WmmaTensor(atom, "b", v_b, arch)
    new_accs[t][d] = wmma_mma(b, p_tiles[t], v_tile, new_accs[t][d])   # o += P V
```

### Step 5.7 — Carry the state to the next iteration

The updated $\big(m^{(t)}, \ell^{(t)}, \mathbf{o}^{(t)}\big)$ are `scf_yield`-ed
as the loop-carried values for tile $t+1$. That closes the recurrence.

---

## 6. The epilogue — final normalization

After the last tile, divide the accumulated output by the denominator (Section 3
final step), guarding the empty-row case $\ell = 0$ (a fully-masked row → output
$0$ rather than $0/0$):

$$
O_i \;=\; \frac{\mathbf{o}^{(N)}_i}{\ell^{(N)}_i}, \qquad
\text{with } 1/\ell \text{ defined as } 0 \text{ when } \ell = 0.
$$

The reciprocal is computed once per row (hoisted out of the head-dim loop) and
passed as a `transform` to `store_wmma_tile`, which writes the f16 result back
through the O `TileWindow`:

```python
inv_l = [ b.select(l == 0, 0, b.rcp(l))  for each row-slot r ]
def _rescale(bld, val, slot, row, colv): return bld.fmul(val, inv_l[slot])
for d in range(n_dk):
    store_wmma_tile(b, owin, accs_f[t][d], lane,
                    col_offset=d * 16, lead=[c0], align=2, transform=_rescale)
```

---

## 7. Causal masking and the early-exit (the one ~2× win)

**Causal** attention forbids a query from attending to *future* keys: query at
position $q$ may only use keys $k \le q$. In Step 5.2 this means

$$
s_j \leftarrow -\infty \quad\text{whenever}\quad k_j > q_i .
$$

Naively you still *compute* every score and then mask the upper triangle — doing
~2× the matmul work the math needs. The **early-exit** observation: a CTA owns
query rows $[\,r_0,\; r_0 + 16\!\cdot\!B_m - 1\,]$, and tile $t$ covers keys
$[\,16t,\; 16t+15\,]$. Every key in tile $t$ is in the future of *every* query
in the CTA exactly when $16t > r_0 + 16 B_m - 1$, i.e. the tile is fully masked
for $t \ge r_0/16 + B_m$. So we simply **stop the K-loop early**:

$$
N_{\text{causal}} \;=\; \min\!\Big(\tfrac{S_k}{16},\; \tfrac{r_0}{16} + B_m\Big).
$$

```python
loop_stop = seqlen_k // 16
if causal:
    causal_stop = group_row0 // 16 + BM      # last needed tile + 1
    loop_stop = select(causal_stop < loop_stop, causal_stop, loop_stop)
```

This skips entire fully-masked tiles (about half of them on average), and is the
single largest speedup in the whole campaign (~2× on causal shapes) — precisely
because it is **algorithmic** (it removes matmuls whose output is zero) rather
than a microarchitectural lever. The boundary tile that straddles the diagonal is
still masked elementwise by Step 5.2.

---

## 8. Two implementation details worth the math

### 8.1 Base-2 everything ($\exp_2$ instead of $\exp$)

The GPU has a hardware $2^x$ (`v_exp_f32`) but no native $e^x$. Using the
identity $e^x = 2^{x \log_2 e}$, we fold $\log_2 e$ into the scale **once, on the
host**:

$$
e^{\,\tau\, S} = 2^{\,(\tau \log_2 e)\, S}, \qquad
\texttt{scale\_log2} = \tau \log_2 e = \frac{\log_2 e}{\sqrt d}.
$$

So every `exp2(s - m)` in Steps 5.3–5.4 computes the true $e^{\tau S - \cdot}$
with a single hardware instruction and no runtime $\ln$/$\log$. The max and the
rescale $\alpha$ live in the same base-2 domain, so the Section-3 algebra is
unchanged.

### 8.2 Why $m$, $\ell$, $\mathbf{o}$ live in registers, not memory

The entire point of flash attention is that the running state is **tiny** —
$m, \ell$ are one scalar per query row, $\mathbf{o}$ is $d$ values per row — so it
fits in registers and rides the `scf.for` as loop-carried values. The
$S_q \times S_k$ score matrix is **never materialized**; only one $16\times16$
tile exists at a time. That is what makes attention $O(S_k)$ memory instead of
$O(S_k^2)$, and it is why the recurrence of Section 3 (rather than the dense
formula of Section 1) is the thing the kernel implements.

---

## 9. Putting it together — the whole kernel in pseudo-math

For each $(\text{q-tile}, \text{head}, \text{batch})$, one wave runs:

$$
\begin{aligned}
&m \leftarrow -\infty,\quad \ell \leftarrow 0,\quad \mathbf{o} \leftarrow \mathbf{0} \\
&\textbf{for } t = 0 \dots N_{\text{causal}}-1: \\
&\quad S \leftarrow \tau\, Q (K^{(t)})^{\top} & &\text{(WMMA }A B^\top\text{)} \\
&\quad S \leftarrow \text{mask}(S) \\
&\quad m' \leftarrow \max(m, \operatorname{rowmax} S),\quad \alpha \leftarrow 2^{\,m - m'} \\
&\quad P \leftarrow 2^{\,S - m'} \\
&\quad \ell \leftarrow \alpha\,\ell + \operatorname{rowsum} P \\
&\quad \mathbf{o} \leftarrow \alpha\,\mathbf{o} + P\,V^{(t)} & &\text{(transpose }P\text{; WMMA }A B^\top\text{)} \\
&\quad m \leftarrow m' \\
&O \leftarrow \mathbf{o} / \ell & &\text{(guard }\ell=0\text{)}
\end{aligned}
$$

Every line above is one of Steps 5.1–5.7 / Section 6. The whole campaign in the
[README](README.md) is about the *implementation* of two lines — the
$Q(K^{(t)})^\top$ and $P V^{(t)}$ matmuls and how their operands are fed — because
on gfx1151 those, not the FLOPs, are the bottleneck. The math here is fixed and
exact; the engineering is all in how cheaply you can stream the operands past the
matrix unit.

---

## 10. Where to go next

- [`README.md`](./README.md) — the optimization case study: why single-wave gather
  wins, why every LDS-staging rewrite loses, the roofline, and the iteration
  ledger.
- `fmha_singlewave.py` — the kernel this document describes (D128 winner). Each step
  above maps to a labelled section in `build_wmma_fmha_singlewave`.
- `fmha_pipelined.py` — the same algorithm with the QK of tile $t+1$ hoisted to
  overlap with the softmax of tile $t$ (software pipelining); the D64 winner. The
  *math* is identical — only the *order* of the independent operations changes.
