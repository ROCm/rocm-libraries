# CK Tile Distribution Encoding - Direct convolution

## 1. Introduction

A GPU kernel launches a grid of threads. Each thread must know:
1. **Which element(s) of a tile it is responsible for** (its "slice" of the work)
2. **Where those elements live in memory** (coordinates in the tensor)

`tile_distribution_encoding` is a pure compile-time type that encodes this mapping. It answers: *given a thread ID decomposed into (warp_id, lane_id), and a per-thread element index Y, what are the tensor coordinates?*

This document contains a self-contained description of the tile distribution encoding in the extent that it is relevant for direct convolutions. 

---

## 2. The Three Spaces

```
Hardware IDs          Logical space           Tensor coordinates
──────────────        ─────────────           ──────────────────
warp_id   ──┐         ┌── R (replicated)  ──→ (ignored / multi-buffer)
lane_id   ──┼── P ──→ ┤
             │         └── H (hidden)     ──→ X[0], X[1], ..., X[NDimX-1]
             │               ↑
             └── Y ──────────┘ (per-thread iteration)
```

- **P dimensions** (NDimP): hardware partition IDs. Typically `P[0] = warp_id`, `P[1] = lane_id`. Each thread has a *unique* combination of P values.
- **Y dimensions** (NDimY): loop counters within a single thread. A thread iterates over all Y combinations, loading/storing multiple elements.
- **R dimensions** (NDimR): replication. Multiple threads share the *same* R index — useful for multi-buffering (e.g. LDS double buffer) or broadcast. Usually empty.
- **H dimensions** (hidden, per X): the factored representation of each tensor dimension X[i].
- **X dimensions** (NDimX): the bottom-level tensor indices (rows, columns, channel groups, etc.).

The key principle: **each X[i] dimension is decomposed into a product of H factors**. Some H factors are owned by P (determined by thread ID), others are owned by Y (iterated per thread), and the rest are trivially size-1.

The tile distribution is characterized by the template parameters that make it compile-time constant and allow the compiler to optimize the data access

```cpp
tile_distribution_encoding<
    RsLengths,       // sequence<...>
    HsLengthss,      // tuple<sequence<...>, ...>
    Ps2RHssMajor,    // tuple<sequence<...>, ...>
    Ps2RHssMinor,    // tuple<sequence<...>, ...>
    Ys2RHsMajor,     // sequence<...>
    Ys2RHsMinor      // sequence<...>
>
```

**Naming convention**: the double "ss" follows the CK Tile convention of pluralizing with "s" twice when the type is a *tuple of sequences*. `HsLengths` would be the factor lengths for a single X dimension (e.g. `seq<4, 8>`); `HsLengthss` is the collection of those, one per X dimension (e.g. `tuple<seq<1>, seq<4,8>, seq<16>>`). The same pattern applies to `Ps2RHssMajor` / `Ps2RHssMinor` (tuple of sequences, one per P dimension), while `Ys2RHsMajor` / `Ys2RHsMinor` are plain sequences (one entry per Y dimension).

---

## 3. The RH Indexing System

The encoding uses a two-level (major, minor) index called **RH** to address all factor dimensions uniformly:

```
RH_major = 0         →  R dimensions (replication)
RH_major = 1         →  H sub-dimensions of X[0]  ← HsLengthss[0]
RH_major = 2         →  H sub-dimensions of X[1]  ← HsLengthss[1]
...
RH_major = NDimX     →  H sub-dimensions of X[NDimX-1]
```

Within a given `RH_major`, `RH_minor` is the index into the inner sequence of that group:

- When `RH_major = 0`: `RH_minor` indexes into `RsLengths` (the R factor sizes).
- When `RH_major = i+1`: `RH_minor` indexes into `HsLengthss[i]` (the H factor sizes for X[i]).

So `(RH_major, RH_minor)` uniquely identifies a single scalar factor, and its size is:

```
size(RH_major=0,   RH_minor=j)  =  RsLengths[j]
size(RH_major=i+1, RH_minor=j)  =  HsLengthss[i][j]
```

Example: `HsLengthss = tuple<seq<1>, seq<4,8>, seq<16>>`

```
(RH_major=1, RH_minor=0)  →  HsLengthss[0][0] = 1   (only factor of X[0], trivial)
(RH_major=2, RH_minor=0)  →  HsLengthss[1][0] = 4   (outer factor of X[1])
(RH_major=2, RH_minor=1)  →  HsLengthss[1][1] = 8   (inner factor of X[1])
(RH_major=3, RH_minor=0)  →  HsLengthss[2][0] = 16  (only factor of X[2])
```

**The X coordinate** is reconstructed from its H factors by a mixed-radix sum (see Section 4.2 for the precise formula), where each factor's stride is the product of the *sizes* of all inner factors from `HsLengthss[i]`.

Note: the number factors can be different for each `i`. It is determined by `HsLengthss[i].size()`.

---

## 4. Template Parameters In Depth

Let's consider again the `tile_distribution_encoding` template parameters

```cpp
tile_distribution_encoding<
    RsLengths,       // sequence<...>
    HsLengthss,      // tuple<sequence<...>, ...>
    Ps2RHssMajor,    // tuple<sequence<...>, ...>
    Ps2RHssMinor,    // tuple<sequence<...>, ...>
    Ys2RHsMajor,     // sequence<...>
    Ys2RHsMinor      // sequence<...>
>
```

### 4.1 `RsLengths` — Replication Dimensions

`sequence<r0, r1, ...>` — lengths of R dimensions. Usually `sequence<>` (empty) for most kernels. When non-empty, multiple threads share the same H/X coordinates but differ in R. 
NOn-trivial replication is useful for multi-buffered LDS where different warps operate on different LDS banks.

### 4.2 `HsLengthss` — Hidden Factor Lengths per X Dimension

`tuple<seq<a0,a1,...>, seq<b0,b1,...>, ...>` — one inner sequence per X dimension. Each inner sequence lists the factor *sizes* (lengths) for that X dimension. The number of factors can differ across X dimensions.

**Critical rule**: the X coordinate is recovered by a mixed-radix unmerge, with the **first factor being outermost (largest stride)** and the **last factor being innermost (unit stride)**. The stride for each factor is the product of the *sizes* of all inner factors as given by `HsLengthss[i]`:

```
X[i] = H[i][0] * stride[0]
      + H[i][1] * stride[1]
      + ...
      + H[i][last]

where stride[j] = HsLengthss[i][j+1] * HsLengthss[i][j+2] * ... * HsLengthss[i][last]
  and stride[last] = 1
```

Here `H[i][j]` are *coordinate values* (runtime indices in `[0, HsLengthss[i][j])`), while the strides are determined entirely by the *sizes* in `HsLengthss[i]`.

Example with `HsLengthss[1] = seq<4, 8>`:
```
stride[0] = 8,  stride[1] = 1
X[1] = H[1][0] * 8 + H[1][1]     H[1][0] ∈ [0,4),  H[1][1] ∈ [0,8)
```
X[1] ranges over `[0, 32)`, covering all `4 × 8 = 32` values.

### 4.3 `Ps2RHssMajor` and `Ps2RHssMinor` — P → RH Mapping (RH indexing)

These are parallel structures: one entry per P dimension, each holding a sequence of (major, minor) pairs.

`Ps2RHssMajor[p]` and `Ps2RHssMinor[p]` together define which RH sub-dimensions P[p] controls.

The P value is **unmerged** (using the product-of-sizes rule) into indices for each of those RH sub-dimensions, in the order listed. The **first listed is the outer (highest-weight) factor**, last is inner.

Example — P[0] controls two H factors, both belonging to X[1], where `HsLengthss[1] = seq<4, 8>`:

```
Ps2RHssMajor[0] = seq<2, 2>
Ps2RHssMinor[0] = seq<0, 1>
```

Reading the mapping:
- Entry 0: `(RH_major=2, RH_minor=0)` → from Section 3, RH_major=2 corresponds to X[1], and RH_minor=0 is the first (outer) factor, with size `HsLengthss[1][0] = 4`.
- Entry 1: `(RH_major=2, RH_minor=1)` → also X[1], RH_minor=1 is the second (inner) factor, with size `HsLengthss[1][1] = 8`.

Entry 0 is listed first so it is the **outer** factor; P[0] is unmerged outer→inner using the inner factor's size (8) as the divisor:

```
// P[0] ∈ [0, 32):
H[1][0] = P[0] / 8     (outer factor, size 4,  range [0,4))
H[1][1] = P[0] % 8     (inner factor, size 8,  range [0,8))
```

The connection to X[1] comes entirely from `RH_major = 2`: the RH system maps `RH_major = i+1` to `X[i]`, so `RH_major=2` always means X[1], regardless of which P or Y dimension references it.

Note that in the current example`P[0] ≠ warp_id`, because it is from range `P[0] ∈ [0, 32)`.

### 4.4 `Ys2RHsMajor` and `Ys2RHsMinor` — Y → RH Mapping

`sequence<m0, m1, ...>` and `sequence<n0, n1, ...>` — parallel sequences, one element per Y dimension. 
Each Y dimension maps to exactly one RH sub-dimension:

```
Y[i]  controls  RH[ Ys2RHsMajor[i] ][ Ys2RHsMinor[i] ]
```

The Y index directly *is* the coordinate in that H factor (no unmerging — each Y owns exactly one factor).

Example with `HsLengthss = tuple<seq<1>, seq<4,8>, seq<16>>` and two Y dimensions:

```
Ys2RHsMajor = sequence<2, 3>
Ys2RHsMinor = sequence<1, 0>
```

- Y[0] → `(RH_major=2, RH_minor=1)` → `HsLengthss[1][1]` = 8 (inner factor of X[1])
- Y[1] → `(RH_major=3, RH_minor=0)` → `HsLengthss[2][0]` = 16 (only factor of X[2])

Each thread iterates Y[0] ∈ [0,8) and Y[1] ∈ [0,16), operating on 8×16 = 128 elements in total.

---

## 5. How a Thread Computes Its Tensor Coordinates

Given `thread_id = warp_id * 64 + lane_id`, decomposed into P values:

**Step 1** — Assign hardware IDs to P values (by convention: `P[0]=warp_id`, `P[1]=lane_id`).

**Step 2** — For each P[p], unmerge P[p] into the RH factors it controls. The `(RH_major, RH_minor)` pairs for the k controlled factors come directly from the template parameters:

```
mj = Ps2RHssMajor[p][j]      (RH_major of the j-th controlled factor)
nj = Ps2RHssMinor[p][j]      (RH_minor of the j-th controlled factor)
```

The factors are listed outer→inner (j=0 is outermost), with sizes `s_j = HsLengthss[mj-1][nj]`.
Define the suffix products:

```
S[j] = s_{j+1} * s_{j+2} * ... * s_{k-1}     (S[k-1] = 1)
```

Then each H factor coordinate is extracted by:

```
H[m0-1][n0]     = (P[p] / S[0]) % s0     (outermost factor)
H[m1-1][n1]     = (P[p] / S[1]) % s1
...
H[m_{k-1}-1][n_{k-1}] = P[p]  % s_{k-1} (innermost factor)
```

The total range of P[p] must equal `s0 * s1 * ... * s_{k-1}`.

*Example using the Input DRAM distribution from Section 6:*

**P[0] = warp_id** controls one factor: `(m0,n0) = (2,0)`, size `s0 = NUM_WAVES`, `S[0] = 1`:
```
H[1][0] = warp_id % NUM_WAVES = warp_id       (warp_id ∈ [0, NUM_WAVES))
```

**P[1] = lane_id** controls two factors: `(m0,n0)=(2,1)` [outer, size `LANES_PER_ROW`] and `(m1,n1)=(3,0)` [inner, size `BLOCK_C8`], with `S[0]=BLOCK_C8`, `S[1]=1`:
```
H[1][1] = (lane_id / BLOCK_C8) % LANES_PER_ROW = lane_id / BLOCK_C8
H[2][0] = (lane_id / 1)        % BLOCK_C8      = lane_id % BLOCK_C8
```
Total range of lane_id must be `LANES_PER_ROW * BLOCK_C8 = 64`.

**Step 3** — For each X dimension, collect all its H sub-dimension indices. The X coordinate is the mixed-radix sum where strides come from `HsLengthss[i]`:

```
X[i] = H[i][0] * stride[0] + H[i][1] * stride[1] + ... + H[i][last]

where 

stride[j] = HsLengthss[i][j+1] * ... * HsLengthss[i][last],  stride[last] = 1

and 

last = HsLengthss[i].size() - 1
```

**Step 4** — For each Y[j], the thread iterates over `0 .. L[j]` where, following the RH lookup from Section 3:

```
L[j] = HsLengthss[Ys2RHsMajor[j] - 1][Ys2RHsMinor[j]]     (when Ys2RHsMajor[j] > 0)
L[j] = RsLengths[Ys2RHsMinor[j]]                            (when Ys2RHsMajor[j] = 0)
```

For each Y combination, the H (or R) factor it controls takes the current Y value, and the full X coordinate is recomputed.

---

## 6. Concrete Example: Input DRAM Distribution

From `grouped_conv_descriptors.hpp`, `Input::MakeDramReadTileDistribution()`:

```cpp
tile_distribution_encoding<
    sequence<>,                                     // No replication
    tuple<
        sequence<1>,                                // X[0]: trivial (size 1)
        sequence<TC::NUM_WAVES, TC::LANES_PER_ROW>, // X[1]: 2 factors → NUM_WAVES × LANES_PER_ROW
        sequence<TC::BLOCK_C8>,                     // X[2]: channel groups
        sequence<8>                                 // X[3]: 8 fp16 per load
    >,
    tuple<sequence<2>, sequence<2, 3>>,             // Ps2RHssMajor
    tuple<sequence<0>, sequence<1, 0>>,             // Ps2RHssMinor
    sequence<1, 4>,                                 // Ys2RHsMajor
    sequence<0, 0>                                  // Ys2RHsMinor
>
```

### RH Map

```
RH_major=0 (R):   empty
RH_major=1 (X[0]): minor=0 → length 1
RH_major=2 (X[1]): minor=0 → length NUM_WAVES
                   minor=1 → length LANES_PER_ROW
RH_major=3 (X[2]): minor=0 → length BLOCK_C8
RH_major=4 (X[3]): minor=0 → length 8
```

### P Mapping

```
P[0] = warp_id:
    Ps2RHssMajor[0] = seq<2>    Ps2RHssMinor[0] = seq<0>
    → controls (RH_major=2, RH_minor=0) = H[X[1]][0]  (length=NUM_WAVES)
    → H[X[1]][0] = warp_id

P[1] = lane_id:
    Ps2RHssMajor[1] = seq<2, 3>    Ps2RHssMinor[1] = seq<1, 0>
    → controls (RH_major=2, RH_minor=1) [outer] and (RH_major=3, RH_minor=0) [inner]
    → factor sizes: LANES_PER_ROW (outer) and BLOCK_C8 (inner)
    → H[X[1]][1] = lane_id / BLOCK_C8
    → H[X[2]][0] = lane_id % BLOCK_C8
```

### Y Mapping

```
Y[0]:  Ys2RHsMajor[0]=1, Ys2RHsMinor[0]=0  →  H[X[0]][0]  (length=1, always 0)
Y[1]:  Ys2RHsMajor[1]=4, Ys2RHsMinor[1]=0  →  H[X[3]][0]  (length=8, fp16 sub-element)
```

### Coordinate Reconstruction

For thread `(warp_id=w, lane_id=l)` at `Y[1]=y`:

```
X[0] = 0                                    (trivial)
X[1] = w * LANES_PER_ROW + (l / BLOCK_C8)  (spatial column)
X[2] = l % BLOCK_C8                         (channel group)
X[3] = y                                    (fp16 sub-element, 0..7)
```

The LDS offset is: `X[1] * BLOCK_C8 * 8 + X[2] * 8 + X[3]`

which simplifies to `(w * LANES_PER_ROW * BLOCK_C8 + l) * 8 + y` — each lane's base is at `lane_flat_idx * 8`, giving perfectly coalesced `global_load_lds` writes.

---

## 7. Concrete Example: Weight DRAM Distribution

From `grouped_conv_descriptors.hpp`, `Weight::MakeDramReadTileDistribution()`:

```cpp
tile_distribution_encoding<
    sequence<>,
    tuple<
        sequence<TC::NUM_WAVES, 64>,   // X[0]: NUM_WAVES × 64 (flat thread index = row)
        sequence<8>                    // X[1]: 8 fp16 sub-elements per load
    >,
    tuple<sequence<1>, sequence<1>>,   // Ps2RHssMajor
    tuple<sequence<0>, sequence<1>>,   // Ps2RHssMinor
    sequence<2>,                       // Ys2RHsMajor
    sequence<0>                        // Ys2RHsMinor
>
```

```
P[0] = warp_id → H[X[0]][0] (length=NUM_WAVES, outer factor)
P[1] = lane_id → H[X[0]][1] (length=64, inner factor)
Y[0]           → H[X[1]][0] (length=8, sub-element)
```

```
X[0] = warp_id * 64 + lane_id   (row = flat thread index)
X[1] = Y[0] ∈ {0..7}            (fp16 sub-element)
```

Each thread owns one contiguous 128-bit (8×fp16) row. Sub-elements are iterated by Y — perfectly coalesced.

---

## 8. Design Principles for Optimal Memory Transactions

### 8.1 Align the innermost H factor with the load granularity

The last (innermost) factor of each X dimension should match the hardware load width. For fp16 with 128-bit loads: **inner factor = 8**. This ensures Y[last] strides over contiguous elements with no gaps.

### 8.2 Map lane_id to contiguous memory

For coalesced global loads, `lane_id` should control the unit-stride (innermost) dimension. `global_load_lds` writes lane L at `M0 + L * data_size_bytes`. The H factors controlled by `P[1] (lane_id)` must produce offsets that form the linear sequence `0, 1, ..., 63`.

**Rule**: the total product of all H factor lengths controlled by lane_id must equal 64 (the wavefront size). The unmerge goes outer→inner in the order listed in `Ps2RHss`.

### 8.3 Y dimensions iterate, P dimensions partition

- P selects a unique starting position per thread.
- Y drives per-thread iteration. If an H factor is covered by Y, every thread visits every value of that factor — effectively unrolling a loop over those elements.
- If a factor appears in `HsLengthss` but nothing in `Ps2RHss` or `Ys2RHs` points to it, it is unreachable (silent bug — data is silently skipped).

### 8.4 Factor coverage must be complete

Every H factor must be owned by exactly one of: a P dimension, a Y dimension, or be size-1. **Verify**:
- `product(all H factors of X[i]) == logical size of X[i]`
- `product(all H factors owned by P[1]) == 64` (wavefront size)

### 8.5 Factor order in `Ps2RHss` determines the unmerge stride

`Ps2RHssMajor[p] = seq<m0, m1>` with `Ps2RHssMinor[p] = seq<n0, n1>`:
- `(m0, n0)` is the **outer** factor (P[p] / size(m1,n1))
- `(m1, n1)` is the **inner** factor (P[p] % size(m1,n1))

Getting this order wrong produces incorrect data or bank conflicts. For example, if lane_id controls factors of sizes `{LANES_PER_ROW, BLOCK_C8}` listed in that order:
```
H[outer] = lane_id / BLOCK_C8       range: [0, LANES_PER_ROW)
H[inner] = lane_id % BLOCK_C8       range: [0, BLOCK_C8)
```
Swapping the listing order would swap which factor gets which range.

### 8.6 Warp-level vs lane-level control

You can control factors at different hardware granularities:
- Map one RH factor to `P[0]` (warp granularity) for inter-warp distribution.
- Map another to `P[1]` (lane granularity) for intra-warp distribution.
- Or fold both into a single P entry by listing multiple RH targets under one P dimension — the P value is then unmerged into both factors simultaneously.

### 8.7 LDS bank conflict avoidance

If the innermost stride lands multiple lanes on the same LDS bank (every 4 bytes = 1 bank on CDNA), consider using XOR or CyclicShift swizzle on the DRAM descriptor. The tile distribution itself remains unchanged — only the descriptor changes.

---

## 9. Summary Table

| Parameter | Type | Controls | Key Convention |
|---|---|---|---|
| `RsLengths` | `sequence<...>` | Replication factor lengths | Usually `sequence<>` (empty) |
| `HsLengthss` | `tuple<seq<...>, ...>` | Factor lengths per X dimension | Outer factor first in each seq |
| `Ps2RHssMajor` | `tuple<seq<...>, ...>` | RH major indices owned by each P | One inner seq per P dimension |
| `Ps2RHssMinor` | `tuple<seq<...>, ...>` | RH minor indices owned by each P | Parallel to Major; outer factor listed first |
| `Ys2RHsMajor` | `sequence<...>` | RH major index owned by each Y | One element per Y dimension |
| `Ys2RHsMinor` | `sequence<...>` | RH minor index owned by each Y | Parallel to Major |

---

## 10. Quick Construction Checklist

1. **Define X dimensions**: identify the tensor's logical shape and decide how many X dims it has (often: rows, cols, channel_groups, sub_elements).
2. **Factor each X**: fill `HsLengthss`, putting the warp-level factor outermost and the sub-element factor innermost. Every factor must be a power of two.
3. **Assign P[0] (warp_id)**: which factor(s) does the warp index select? Fill the first entry of `Ps2RHssMajor/Minor`.
4. **Assign P[1] (lane_id)**: which factor(s) does the lane index select? The product of those factor lengths must equal 64. Fill the second entry outer→inner.
5. **Assign Y**: remaining non-unit factors become Y dimensions. Each Y points to exactly one (major, minor). Fill `Ys2RHsMajor/Minor`.
6. **Verify completeness**: for each X[i], `product(H[i][*]) == X[i].size`.
7. **Verify lane coverage**: `product(factors owned by P[1]) == 64`.
8. **Verify inner factor**: the innermost factor of the unit-stride X dimension equals the load granularity (8 for 128-bit fp16 loads).
