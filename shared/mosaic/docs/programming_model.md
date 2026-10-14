# Mosaic Programming Model

Mosaic is a CuTe-inspired layout abstraction for AMD Instinct GPUs. Use it to
express thread-value mappings, matrix layouts, workgroup schedules, and NUMA-ware tensor
allocations in a unified layout abstraction.

**References:**
- **Math:** Cecka, *"CuTe Layout Representation and Algebra"*, [arXiv:2603.02298](https://arxiv.org/abs/2603.02298) — the formal layout-algebra system mosaic implements a subset of.
- **API vocabulary:** NVIDIA CuTe. mosaic is an independent AMD implementation; no CUTLASS source is vendored or derived from.

## Layout

A `Layout` is a function from a coordinate to an offset. It has two
parts:

- **shape** — the logical extents (what coordinates are valid)
- **stride** — the per-axis step (how a coord becomes an offset)

```cpp
#include <mosaic/mosaic.hpp>

auto L = mosaic::make_layout(
    mosaic::make_shape (4, 8),    // 4 rows, 8 columns
    mosaic::make_stride(1, 4));   // column-major: row stride 1, col stride 4

// The layout function does a dot product between coordinate and stride.
L(2, 3);   // = 2*1 + 3*4 = 14
L(0, 0);   // = 0
L(3, 7);   // = 31
```

Read the layout notation `(4,8):(1,4)` as **`shape:stride`**.

```
L = (shape):(stride)
L(coord) = sum_i(coord[i] * stride[i])
```

### Compile-time vs. runtime values

Stride and shape entries can be plain `int`s (runtime values) or `mosaic::Int<v>`
(compile-time constants):

```cpp
auto L_runtime = mosaic::make_layout(mosaic::make_shape(M, N));
auto L_static  = mosaic::make_layout(mosaic::make_shape(mosaic::Int<4>{},
                                                        mosaic::Int<8>{}));
```

`Int<v>` lets the compiler fold strides and indexing math at compile time.

---

## Building layouts

There are four primary ways to build a Layout:

### 1. Shape only — defaults to compact column-major

```cpp
auto L = mosaic::make_layout(mosaic::make_shape(4, 8));
// → Layout (4,8):(1,4) — leftmost coord varies fastest
```

If you want row-major, pass the stride
explicitly (next).

### 2. Shape + explicit stride

```cpp
auto rm = mosaic::make_layout(
    mosaic::make_shape (4, 8),
    mosaic::make_stride(8, 1);
// → (4,8):(8,1) — rightmost coord varies fastest

auto cm = mosaic::make_layout(
    mosaic::make_shape (4, 8),
    mosaic::make_stride(1, 4);
// → (4,8):(1,4) — same as default
```

### 3. Convenience helpers

```cpp
auto L1 = mosaic::row_major(4, 8);     // (4,8):(8,1)
auto L2 = mosaic::column_major(4, 8);  // (4,8):(1,4)
```

These return Layouts directly.

---

## Coordinates and the natural enumeration

Mosaic distinguishes between two kinds of coordinates for a `Shape`:

- A **natural coordinate** is a tuple of integers that mirrors the shape's
  structure. For `Shape (4, 8)`, a natural coord is `(i, j)` with
  `i ∈ [0, 4)` and `j ∈ [0, 8)`. For nested `Shape ((2, 2), 8)` it's
  `((i0, i1), j)`. The structure of the coord matches the structure of the
  shape — element-by-element.

- A **flat index** is a single integer in `[0, size(shape))`.

Every Shape `S` has a canonical bijection between flat indices and natural
coordinates. Mosaic uses **column-major (colexicographic) enumeration**:
the leftmost coordinate varies fastest. For shape `(4, 8)`:

| flat idx | natural coord |
|---|---|
| 0 | (0, 0) |
| 1 | (1, 0) |
| 2 | (2, 0) |
| 3 | (3, 0) |
| 4 | (0, 1) |
| 5 | (1, 1) |
| ... | ... |
| 31 | (3, 7) |

The two directions of the bijection are exposed as free functions in
`mosaic/stride.hpp`:

```cpp
auto shape = mosaic::make_shape(4, 8);

mosaic::idx2crd(5, shape);                          // returns (1, 1)
mosaic::crd2idx(mosaic::make_coord(1, 1), shape);   // returns 5
```

Note this bijection is **shape-only** — no stride is involved.
It essentially defines an implicit "traversal order" or "indxing order" of a layout.
It's the canonical "how do I enumerate every position in this shape" map. Round-trip
holds for every `i` in `[0, size(shape))`:

```
crd2idx(idx2crd(i, shape), shape) == i
```

### Compact vs. non-compact layouts

A layout is **compact** when its strides minimally cover the codomain — every
integer in `[0, size(L))` is hit exactly once, with no gaps and no overlaps.
For shape `(4, 8)`, both `(1, 4)` and `(8, 1)` are compact — they're just
column-major and row-major orderings of the same 32 positions. Equivalently:
a layout is compact when `size(L) == cosize(L)`.

A **non-compact** layout has gaps (or overlaps) in its image. The shape still
has 32 positions, but the strides spread them across a larger codomain:

```cpp
// Compact column-major: shape (4,8), stride (1,4). cosize == size == 32.
auto L_compact = mosaic::make_layout(mosaic::make_shape(4, 8));
mosaic::cosize(L_compact);   // 32 — no gaps

// Non-compact: column stride is 8 instead of 4. cosize == 60.
auto L_padded = mosaic::make_layout(mosaic::make_shape(4, 8),
                                    mosaic::make_stride(1, 8));
mosaic::cosize(L_padded);    // 60 — the layout reaches offset 59 but only
                             //      occupies 32 of those 60 positions
```

Non-compact strides are useful when the data you're indexing has padding,
when you want to interleave with another layout in the same buffer, or when
you're slicing into a larger tensor.

### Putting it together: `L(idx)`

A `Layout<Shape, Stride>` builds on the natural enumeration: the natural
coord stays the same, but the output is computed using the layout's actual
stride (which may or may not be compact). When you call `L(idx)` with a
single integer, mosaic expands it via the natural enumeration to a coord,
then evaluates with the layout's stride:

```cpp
// Compact column-major (matches the natural enumeration). L(idx) == idx.
auto L_compact = mosaic::make_layout(mosaic::make_shape(4, 8));    // stride (1, 4)
L_compact(5);    // → natural coord (1, 1), offset = 1*1 + 1*4 = 5

// Non-compact stride: same shape, same enumeration order, but offsets are
// spread across a larger range (column stride 8 instead of 4 → gaps).
auto L_padded  = mosaic::make_layout(mosaic::make_shape(4, 8),
                                     mosaic::make_stride(1, 8));
L_padded(5);     // → natural coord (1, 1), offset = 1*1 + 1*8 = 9
```

So `L(idx)` is mechanically `crd2idx(idx2crd(idx, shape), shape, stride)` —
"decompose flat index into natural coord using the canonical enumeration,
then re-evaluate at that coord using the layout's stride." This is what
makes layouts useful as **permutations**: the input is just a flat integer,
the output is a permuted offset, and the layout's stride defines the
permutation.

---

## Using a layout

```cpp
auto L = mosaic::row_major(4, 8);  // (4,8):(8,1)

// Coordinate evaluation (the hot path):
L(2, 3);                             // = 2*8 + 3*1 = 19
L(mosaic::make_coord(2, 3));         // same, via tuple coord

// Flat-index evaluation (uses the natural enumeration internally):
L(19);                               // → natural coord (3, 2), offset = 3*8 + 2*1 = 26

mosaic::size(L);     // 32 — number of elements (domain of the layout function)
mosaic::cosize(L);   // 32 — max offset + 1 (codomain footprint)
mosaic::shape(L);    // (4, 8) tuple
mosaic::stride(L);   // (8, 1) tuple
mosaic::rank(L);     // Int<2>{}

// Bijection between flat index and natural coord (shape-only, no stride):
auto c = mosaic::idx2crd(19, L.shape());   // = (3, 2) tuple
```

`L(idx)` is most useful when you have a flat workgroup ID or a flat lane
index and want the layout to remap it according to its permutation —
this is exactly what the GEMM grid scheduling example below does.

---

## Tensor: layout + data

A `Layout` is a function `coord → offset`. A `Tensor` is the next step: it
pairs a Layout with an **engine** (the actual data — typically a raw pointer)
so that evaluation reads/writes memory directly.

```
Layout:  coord  →  offset
Tensor:  coord  →  offset  →  *data
```

```cpp
#include <mosaic/mosaic.hpp>

float buf[32];
auto T = mosaic::make_tensor(buf, mosaic::make_shape(4, 8));
//   T is a Tensor<float*, Layout<(4,8):(1,4)>>

T(2, 3);          // = buf[ L(2, 3) ] = buf[14]
T(2, 3) = 7.0f;   // writes into buf[14]
```

The engine is duck-typed: anything that supports `engine[offset]` works.
Raw pointers (`float*`, `int*`, etc.) are the common case. The Tensor's
`operator()` mirrors Layout's — it accepts variadic coords, tuple coords,
and flat indices, and dispatches into the engine through the layout.

Three factories:

```cpp
mosaic::make_tensor(ptr, shape);                  // default col-major stride
mosaic::make_tensor(ptr, shape, stride);          // explicit stride
mosaic::make_tensor(ptr, layout);                 // layout you've already built
```

Tensor accessors mirror Layout's: `T.layout()`, `T.engine()`, `T.size()`.
Free-function `mosaic::size(T.layout())`, `mosaic::shape(T.layout())`, etc.,
work as expected on the layout component.

---

## Slicing

Use `_` (underscore) as a placeholder to keep a dimension free; pass an
integer to fix that dimension. The result is a sub-object — what kind
depends on whether you slice a Layout or a Tensor.

### Slicing a Tensor

Slicing a Tensor handles the base-offset arithmetic for you: it returns a
sub-Tensor whose engine pointer has been shifted to absorb the bound coords'
offsets. The sub-Tensor's `operator()` indexes from 0 and reads the right
memory directly.

```cpp
using mosaic::_;

float buf[32];   // imagine it's already initialized
auto T = mosaic::make_tensor(buf, mosaic::make_shape(4, 8));   // col-major

auto T_col3 = T(_, 3);    // sub-Tensor for column 3
T_col3(0);                // = buf[12]  — the pointer has been shifted
T_col3(1);                // = buf[13]
T_col3(2) = 99.0f;        // writes to buf[14]
```

Conceptually:
```
T(_, 3) = Tensor{ engine + base_offset_of_(_, 3), slice(_, 3, layout) }
        = Tensor{ buf + 12, sub_layout (4):(1) }
```

This is what makes Tensor slicing the natural operation: the Layout-level
slice is the primitive used under the hood, but you almost never want to
call it directly unless you're building your own data abstraction on top.


See [examples/cpp/slice/](../examples/cpp/slice/) for a runnable demo.

---

## Worked example: tiled GEMM grid scheduling

For a GEMM kernel on AMD Instinct GPUs (like MI300/MI350), you typically want workgroup
IDs grouped onto the same XCD (chiplet) for L2 cache locality, then arranged
within each XCD to maximize tile reuse. Mosaic provides two pieces:

```cpp
#include <mosaic/mosaic.hpp>
#include <mosaic/algorithms.hpp>   // chiplet_transform_*, gemm_grid_transform

__device__ void kernel(int wgid, int grid_h, int grid_w, int num_xcds,
                         int& pid_m, int& pid_n) {
    constexpr int L2_TILE_Y = 8;
    constexpr int L2_TILE_X = 4;

    // Step 1: remap WG IDs so consecutive IDs land on the same XCD.
    wgid = mosaic::chiplet_transform_chunk(
        wgid, grid_h * grid_w, /*chunk=*/32, num_xcds);

    // Step 2: define a hierarchical tile schedule and apply it.
    auto block = mosaic::row_major(L2_TILE_Y, L2_TILE_X);
    auto tiler = mosaic::column_major(grid_h / L2_TILE_Y, grid_w / L2_TILE_X);
    auto schedule = mosaic::blocked_product(block, tiler);

    int idx = static_cast<int>(schedule(wgid));
    pid_m = idx / grid_w;
    pid_n = idx % grid_w;
}
```

**Key operators:**
- `chiplet_transform_chunk(wgid, total, chunk, num_xcds)` — chiplet-aware
  ID remapping. AMD-specific, lives in `mosaic/algorithms.hpp`.
- `blocked_product(block, tiler)` — "place a copy of block at every position
  of tiler." This is the named operator for "tile a small block over a
  grid" patterns.
- `gemm_grid_transform(idx, layout, grid_h, grid_w, qy, qx)` — handles the
  case where the grid isn't perfectly divisible by the tile dimensions
  (boundary strips get default row-major mapping).

Host-side schedule inspection: [examples/python/pymosaic/gemm_grid/](../examples/python/pymosaic/gemm_grid/).

---

## Worked example: MFMA register layout

An MFMA instruction distributes matrix elements across the 64 lanes of a
wavefront and across each lane's VGPRs. You can encode that distribution as
a Layout that maps `(lane, item) → matrix coord`:

```cpp
#include <mosaic/mosaic.hpp>

// CDNA3 V_MFMA_F32_16x16x8_xf32 D output:
//   M=N=16, H=4, B_I=1, M_I=4, G=1
const auto L_register = mosaic::make_layout(
    mosaic::column_major(/*B_I=*/1, /*B/B_I=*/1),
    mosaic::column_major(/*M_I=*/4, /*G=*/1),
    mosaic::column_major(/*N=*/16, /*H=*/4));
// = ((1,1),(4,1),(16,4)):((1,1),(1,4),(1,16))

// L_matrix = inverse(L_register) — maps (lane, item) → flat matrix offset
const auto L_matrix = mosaic::right_inverse(L_register);
```

Pair this with a hardware layout `row_major(64, ITEMS)` to map flat
positions to `(lane, item)`, and a logical layout `row_major(B, M, N)` to
map flat matrix offsets to `(block, row, col)`. The end-to-end function
`(lane, item) → (block, row, col)` is just composition of these layouts'
evaluations.

Full example with the full CDNA3 MFMA derivation:
[examples/cpp/mfma/](../examples/cpp/mfma/).
