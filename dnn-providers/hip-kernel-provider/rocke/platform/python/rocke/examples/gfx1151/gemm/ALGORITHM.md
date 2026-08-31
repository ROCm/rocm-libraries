# The WMMA GEMM and its quantization ladder, from the math up

A from-the-math-up derivation of what the **wave32 WMMA GEMM** kernels on this
page actually compute, and of the three precision rungs built on the same
skeleton: f16, int8-storage / f16-compute ("Path B"), and true int8 → f16
("Path A").

This is written for a reader who knows what a matrix multiply is but has **not**
seen the gfx11 WMMA instruction or its lane/fragment ABI. We start from the GEMM
spec, derive the one-wave-per-16×16-tile mapping the kernels use, then layer the
three quantization schemes onto it — each as a small, exact change to one of the
GEMM's three stages (load, contract, store).

> If you just want to *run* the kernels and read the optimization/throughput
> story, see [`README.md`](README.md). This file explains *what* each rung
> computes and *why* the lane mapping is shaped the way it is.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $A$ | $M \times K$ | left operand, **row-major** |
| $B$ | $N \times K$ | right operand, **row-major** (note: $N \times K$, not $K \times N$) |
| $C$ | $M \times N$ | output |
| $M_t, N_t, K_t$ | scalar | the WMMA tile dims, all $= 16$ here |
| $l$ | scalar | wave-relative lane index, $0 \le l < 32$ |
| $\tau_a, \tau_b$ | scalar | per-tensor symmetric dequant scales (int8 rungs) |

The layout is **RCR**: $A$ Row-major, $B$ "Column-of-the-product" stored
row-major as $N \times K$, $C$ Row-major. The product the hardware and the numpy
reference both compute is

$$
C \;=\; A\,B^{\top}, \qquad C_{mn} \;=\; \sum_{k=0}^{K-1} A_{mk}\,B_{nk}.
$$

The $B^{\top}$ is not a transpose you pay for — it is the **native** WMMA
convention (§3), and storing $B$ as $N\times K$ is exactly what makes the
contraction index $k$ run contiguously down each operand's row.

---

## 1. What a GEMM computes (the specification)

A general matrix multiply is the dense triple sum above: every output element
$C_{mn}$ is the dot product of row $m$ of $A$ with row $n$ of $B$, contracted over
the shared dimension $K$. That is the entire mathematical object. The rest of this
document is about computing it on a GPU **one $16\times16$ output tile at a time**,
with each tile owned by a single wave of 32 lanes — and then about quantizing the
operands without changing the result.

---

## 2. The matrix unit on this part: gfx11 WMMA

gfx1151 (RDNA3.5 / Strix Halo) has **no MFMA** (the CDNA matrix instruction).
Its matrix unit is the RDNA3 **WMMA** instruction, and the only f16 atom is

$$
\texttt{wmma\_f32\_16x16x16\_f16} : \quad C \mathrel{+}= A\,B^{\top},
$$

which multiplies two $16\times16$ f16 fragments and accumulates into a
$16\times16$ **f32** result. Two facts about it drive the whole kernel:

1. **It computes $A B^{\top}$, not $AB$.** With $A$ a tile of the left operand and
   $B$ a tile of the (row-major $N\times K$) right operand, $A B^{\top}$ is
   *exactly* the $C = AB^{\top}$ the spec wants — the transpose is free, and
   $B$'s $N\times K$ storage is what makes that so. This is why the layout is RCR.
2. **It runs on a single wave32** (32 lanes), with a fixed lane→data ABI. The
   tile dims are fixed at $16\times16\times16$ (`_WMMA_M = _WMMA_N = _WMMA_K = 16`
   in `_wmma_common.py`); there is no `16x16x32`, no `32x32x*`. That fixed,
   single-shape, wave32 matrix unit is the structural fact every rung is built on.

---

## 3. From spec to kernel: who computes what

The kernels parallelize the GEMM across the GPU like this:

- **Grid (independent work).** One wave (CTA of 32 threads) per $16\times16$
  output tile. The grid is $(\lceil M/16\rceil,\ \lceil N/16\rceil)$
  (`wmma16_grid`), with `block_id.x` → the M-tile and `block_id.y` → the N-tile
  (`grid_order="MN"` in the f16 baseline). Different tiles share nothing, so they
  run fully independently — the embarrassingly-parallel outer structure of GEMM.
- **Block (one wave).** Each tile's $16\times16 = 256$ output elements are
  produced by the wave's 32 lanes: each lane owns **8 of the 256** outputs,
  carried as the WMMA accumulator `<8 x f32>`.
- **The K-loop is the contraction.** The `scf.for k0 in 0..K step 16` loop walks
  the shared dimension $K$ in 16-wide steps. The running accumulator (the
  `<8 x f32>` C-fragment) is carried across iterations as a loop-carried value
  (`iter_args`), so the dot product of §1 is built up tile-by-tile in registers,
  never spilled.

There is **no LDS and no cross-lane shuffle** — the WMMA fragment ABI does all
the lane distribution. That makes these minimal *correctness-first reference*
kernels: the point is to prove the wave32 WMMA ABI exactly, not to hit peak.

### 3.1 The lane → data map (the one ABI fact to internalize)

Decode the wave-relative lane into two coordinates (identical in every rung):

```python
lane = thread_id_x() % 32
frag = lane % 16     # A-frag row, B-frag row (= math column of B), and output column
half = lane / 16     # 0 or 1: selects the even/odd output rows this lane writes
```

The hardware-verified gfx1151 layout for this atom is:

- **Operand load.** For tile bases $m_0 = 16\cdot\texttt{block\_id.x}$,
  $n_0 = 16\cdot\texttt{block\_id.y}$, lane $l$ loads a `<16 x half>` **A-row
  fragment** $A[m_0 + (l\bmod 16),\ k_0{:}k_0{+}16]$ and a `<16 x half>` **B-row
  fragment** $B[n_0 + (l\bmod 16),\ k_0{:}k_0{+}16]$ — one contiguous 16-wide
  slice down each operand's row, gathered straight from global memory (no LDS).
  Because $A$ and $B$ are row-major with $K$ as the inner dim, those 16 elements
  are contiguous: one `global_load_vN(..., 16)`.
- **Accumulator scatter.** After the K-loop, slot $i$ of lane $l$'s `<8 x f32>`
  accumulator maps to output

  $$
  \big(\text{row} = m_0 + 2i + \lfloor l/16\rfloor,\quad \text{col} = n_0 + (l \bmod 16)\big),
  \qquad i = 0\dots 7.
  $$

  So the 32 lanes × 8 slots tile the $16\times16$ output: the 16 columns come from
  `lane%16`, and the 16 rows come from the 8 slots × the 2 `lane/16` halves.

That is the complete kernel skeleton. Each rung below changes exactly one of the
three stages — **load**, **contract**, or **store** — and nothing else.

---

## 4. Rung 01 — the f16 baseline (`wmma_gemm.py`)

The skeleton verbatim. Per K-step:

$$
\texttt{acc} \mathrel{+}= A_{\text{frag}}\, B_{\text{frag}}^{\top}
\qquad (\text{f16}\times\text{f16}\to\text{f32 accumulate}).
$$

```python
acc = zero_vec_f32(8)
for k0 in range(0, K, 16):                       # the contraction
    a_frag = global_load_vN_f16(A, a_base + k0, 16)   # <16 x half>
    b_frag = global_load_vN_f16(B, b_base + k0, 16)   # <16 x half>
    acc    = wmma_f32_16x16x16_f16(a_frag, b_frag, acc)   # C += A B^T
# epilogue: trunc each of the 8 f32 slots -> f16, scatter to C (the §3.1 map)
```

This establishes the load/contract/store stages the other rungs reuse.

**A note on exactness.** WMMA accumulates in f32 in a **different summation order**
than numpy's f32 reference, so the two agree only to within an f32 rounding
tolerance, not bit-for-bit — which is why the verify gate uses a tolerance
(`--tol 1e-2`) rather than demanding zero. On small integer inputs the agreement
is essentially exact (`max_abs ≈ 1.8e-5`); see the README's int4 tolerance note
for the one shape where a single f16 output ULP trips an *absolute* tolerance.

---

## 5. The quantization lemma (used by both int8 rungs)

Per-tensor symmetric quantization stores each operand as int8 with one scalar
scale: a stored byte $q$ represents the real value $q\cdot\tau$. The dot product
over a row then **dequantizes once, at the end**, because the scales are scalars:

$$
\sum_{k} (a_k\,\tau_a)(b_k\,\tau_b)
\;=\; \tau_a\,\tau_b \sum_{k} a_k\, b_k .
$$

So the matrix unit can contract the **raw int8 values** and the f32/i32
accumulator is multiplied by the single combined scale $\tau_a\tau_b$ **in the
epilogue** — never per element. Both int8 rungs fold the scale exactly this way
(`scale = scale_a * scale_b`, applied once per output element). This is the same
"dequant at the block boundary" idea the fp8 MoE kernel uses, specialized to one
scale for the whole tensor.

The two rungs differ only in *what the contraction runs on*:

- **Path B** dequantizes the int8 to f16 **before** the matmul and reuses the f16
  WMMA — so the sum above is computed as the §4 f16 dot product. (int8 with
  $|x|\le 127$ is exact in f16, so this conversion is lossless.)
- **Path A** keeps int8 all the way into a dedicated **int8→int32** WMMA atom, so
  the sum $\sum a_k b_k$ is an *exact integer* accumulation; the scale is applied
  after converting the i32 result to f32.

---

## 6. Rung 03 — int8 storage / f16 compute, "Path B" (`wmma_gemm_int8.py`)

Change **only the load stage**: read int8 fragments, sign-extend each element to
f16, then run the *identical* f16 WMMA of §4.

$$
A_{\text{frag}}^{(\text{f16})} = \operatorname{sitofp}_{\text{f16}}(A_{\text{frag}}^{(\text{i8})}),
\qquad
\texttt{acc} \mathrel{+}= A_{\text{frag}}^{(\text{f16})}\, (B_{\text{frag}}^{(\text{f16})})^{\top}.
$$

```python
a_i8   = global_load_vN(A, a_base + k0, I8, 16)        # <16 x i8>
a_frag = vec_pack([i8_to_f16(a_i8, i) for i in range(16)], F16)   # sext -> sitofp -> f16
...    = wmma_f32_16x16x16_f16(a_frag, b_frag, acc)    # the SAME f16 atom
# epilogue: elem = acc[i] * (scale_a*scale_b); trunc -> f16; scatter (§3.1 map)
```

The win here is **storage / memory bandwidth** (int8 operands are half the bytes),
**not compute** — the tensor-core throughput is identical to the f16 kernel
because the matmul *is* the f16 matmul. The decisive property is that this needs
**no DSL core change**: it reuses the proven f16 atom unchanged. The combined
dequant scale is folded into the epilogue exactly as the §5 lemma prescribes.

---

## 7. Rung 04 — true int8 / f16 dequant out, "Path A" (`wmma_gemm_iu8_dequant.py`)

Change **the contract stage and the operand ABI**: run the hardware
`wmma_i32_16x16x16_iu8` atom — int8×int8 → **int32** accumulate — then dequantize
the i32 accumulator to f16 in the epilogue.

$$
\texttt{acc}^{(\text{i32})} \mathrel{+}= A_{\text{frag}}^{(\text{i8})}\, (B_{\text{frag}}^{(\text{i8})})^{\top},
\qquad
C_{mn} = \operatorname{trunc}_{\text{f16}}\!\big(\operatorname{sitofp}_{\text{f32}}(\texttt{acc}^{(\text{i32})})\cdot \tau_a\tau_b\big).
$$

Two ABI changes follow from the int8 tensor core (everything else is the §3.1
skeleton):

1. **Operands are passed packed as i32.** A/B pointers are `i32`, with 4 int8
   values packed per i32 (slot $j$ holds $K=[4j..4j{+}3]$) — the iu8 fragment
   format. So each lane's WMMA operand fragment is `<4 x i32>` (= 16 int8), and
   the K-loop strides 16 int8 = 4 i32 columns per step (`k4 = K/4`). The bytes on
   the device are the **same bytes** Path B uploads — only the read width differs.
2. **The accumulator is `<8 x i32>`** (loop-carried), with the identical slot→
   output map of §3.1.

```python
acc = zero_vec(I32, 8)
for k0 in range(0, K/4, 4):                       # i32 columns
    a_frag = global_load_vN(A, a_base + k0, I32, 4)   # <4 x i32> = 16 int8
    b_frag = global_load_vN(B, b_base + k0, I32, 4)
    acc    = mma(wmma_i32_16x16x16_iu8, a_frag, b_frag, acc)   # int32 accumulate
# epilogue: deq = sitofp(acc[i]) * (scale_a*scale_b); trunc -> f16; scatter (§3.1 map)
```

Because the i32 accumulation of $\sum a_k b_k$ is **exact** (integers, no rounding
until the final f16 store), this rung is the most numerically faithful of the
three — the committed verify (`data/04_iu8_dequant_verify.json`, 256×128×256)
reports `max_abs_diff ≈ 7.7e-7` with `bad=0`, i.e. error only from the final f16
store, not the contraction. And because
the int8 tensor core runs at ~**2× the f16 rate** on this part, it is the
throughput winner of the A-vs-B comparison (README rung 05). The
`wmma_i32_16x16x16_iu8` atom is upstream's native-int landing; this rung is the
**f16-dequant-output** sibling of upstream's raw-i32-output `wmma_gemm_iu8.py`,
making the A-vs-B comparison apples-to-apples (both output f16).

---

## 8. Rung 02 — int4 weight-only, W4A16 (`matmul_nbits`, `large_n` family)

The fourth quantization point on the ladder is **weight-only int4**: fp16
activations $A$ times **packed-int4** weights $B$, with one fp16 scale per group
of `group_size` (here 32) weight elements. The dequant happens **on load**,
per group, then the same f16 WMMA contracts the result:

$$
C \;=\; A \cdot \operatorname{dequant}(B,\ \text{scales})^{\top},
\qquad \operatorname{dequant}(B)_{nk} = b_{nk}^{(\text{i4})}\cdot s_{n,\lfloor k/32\rfloor}.
$$

Unlike rungs 01/03/04 (one wave per $16\times16$ tile, no LDS), `matmul_nbits`
`large_n` is a genuinely **tiled** body: a `64×128×16` tile with a `2×2` warp grid
over the same `wmma_f32_16x16x16_f16` atom (`_large_n_spec` in the script). It is
the production shape for a group-of-32 weight-quantized projection, included here
as the int4 rung of the ladder. The math is still the §1 GEMM with the §5 lemma
specialized to **per-group** (not per-tensor) scales applied at the group
boundary along $K$.

(See the README's int4 tolerance note for why the default `M=128 N=4096 K=4096`
shape reports `max_abs=0.125` against an *absolute* `1e-2` gate — it is one f16
output ULP from the WMMA accumulation order, ~0.08 % relative, not an int4 bug;
the same kernel at a smaller `K` passes well under the gate.)

---

## 9. The three stages, three rungs — the whole ladder in one table

Every rung is the §3 skeleton with exactly one stage swapped:

| stage | 01 f16 | 03 Path B (int8→f16) | 04 Path A (true int8) |
|---|---|---|---|
| **load** | `<16×f16>` frag | `<16×i8>` → sext → f16 frag | `<4×i32>` (= 16 int8) frag |
| **contract** | `wmma_f32_16x16x16_f16` | **same** f16 atom | `wmma_i32_16x16x16_iu8` (i32 acc) |
| **store** | trunc f32 → f16 | `acc·τ_aτ_b` → f16 | `sitofp(i32)·τ_aτ_b` → f16 |
| **acc** | `<8×f32>` | `<8×f32>` | `<8×i32>` |
| **core change?** | — | none (reuses f16 atom) | iu8 atom (upstream's) |

The grid, the lane→data map (§3.1), the K-loop structure, and the epilogue
scatter are **identical** across all three. That shared skeleton is the point of
the example: it makes the quantization choice a one-stage edit, and it makes the
A-vs-B throughput comparison (same bytes in, same f16 out, same ABI and grid) a
clean measurement of *compute path alone*.

`matmul_nbits` (rung 02) departs from the skeleton (it is a multi-warp tiled body)
because it is a production weight-only shape rather than a minimal reference — but
it sits on the same WMMA atom and the same dequant lemma.

---

## 10. Where the algorithm ends and tuning begins

The math above is fixed and exact: all four rungs compute $C = A B^{\top}$ (with
the appropriate dequant), and the only approximation anywhere is f16/f32 rounding
in the accumulator/store. What the README explores is **throughput**, not
correctness:

- these are no-LDS, one-tile-per-wave *reference* kernels far from peak (the
  committed §05 int8 suite lands at ~4–11 % of the gfx1151 ceilings); the
  headline is the **A/B ratio**, not the absolute rate;
- the central result is that **K, not arithmetic intensity, drives Path A's win**
  (K-heavy shapes amortize the fixed per-wave overhead, where Path A's true-int8
  tensor core gets to run at its ~2× ceiling), surfaced only by the §05 perf
  suite's hardened, regime-tagged timing.

Read [`README.md`](README.md) for the full A-vs-B study, the roofline pins, the
hardened-timing methodology, and the per-shape results.
