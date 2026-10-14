# Layout Tutorial

## Resources

- [CUTLASS CuTe Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md)
    - This document covers similar ground to the tutorials in the CUTLASS repo and parallels their structure. The linked folder also contains other really useful markdown files for showcasing how layouts are used for MMA and TMA programming, and other higher level algorithms.


## Introduction

Modern GPU programming for tensor computations requires complex mappings of multidimensional data to multiple levels of memory hierarchy and multiple levels of nested parallelism.
Matrix core instructions, for example, require the logical elements of the input/output tensors to be mapped specific vector lanes and registers.
Avoiding bank conflicts while accessing LDS requires a careful consideration of how data is placed physically in the programmable memory space and the order of how it's accessed.

As the architectures evolve, these type of optimizations become more important, present at each new level of the hiearchy.
L2 dissagregation on MI300 and MI350 require a particular scheduling of workgroups to the hardware to maximize for L2 cache utilization and reuse.
Even how physical addresses are mapped to HBM resources and how our compute accesses that data can change the overall performance due to NUMA effects.

Each of these optimization problems can be described by a *layout*, a mapping from logical coordinates (e.g row/col of a matrix) to physical resources (e.g physical memory location, hardware lane, register index, CU, XCD, etc).
A *layout abstraction* is a framework/specification for representing these layout mappings using algebraic expressions.
This gives you a richer ability to express complex layouts and partitioning patterns required to reach peak performance on GPU hardware.
Furthermore, it allows for a seperation of concerns, where physical layouts (e.g in memory or across compute units) can be declared and reasoned about seperately from algorithmic logic.
Additionally, a rich and formal algebra over layouts allows for *compositional reasoning*.
Each hardware hiearchy requires layout optimizations within that hierarchy.
These layouts can be composed, again allowing for modular reasoning, but also facilitating easier co-optimization across layers.
This algebra also allows for static analysis and further discovery of optimizations.

Prior works have already demonstrated the utility of layout abstractions for GPU tensor programming, namely CuTe (Cecka) and Linear Layouts (Zhou, Lezcano-Casado et al.).
This document will focus on the CuTe layout abstraction, showcasing it's application to AMD hardware.

## Layouts

A *layout* is defined by a *shape* and *stride* (we will typically write layouts like `layout` = `shape` : `stride`)
We will focus on *flat* layouts, where the shape and stride are simple tuples of integers.

The most familiar type of layout in tensor programming is the memory layout describing how a tensor is laid out over memory offset.
For example, a 3 by 4 "row-major" matrix lays the elements of each row out contiguously, whereas a "column-major" matrix lays the elements of each column out contiguously.

Focusing on the row major layout, by enumerating how each logical coordinate of the matrix maps to an offset, we can see that "row-major" describes which *mode* (i.e dimension) is physically contiguous in memory:

```
L_RowMajor(0, 0) = 0
L_RowMajor(1, 0) = 4
L_RowMajor(2, 0) = 8
L_RowMajor(0, 1) = 1
L_RowMajor(1, 1) = 5
...
L_RowMajor(2, 3) = 11
```

As the second ("row") mode changes, the offset increases by 1. As the first mode ("column") changes, we increment the offset by the size of the row mode.
How the offset changes depending on the coordinate is dependent on a layout's **stride**.
The shape and stride for the above row major layout is $(3, 4) : (4, 1)$.
You can compute the offset for a given coordinate in the layout by simply computing the *dot product* of the coordinate with the stride:

```
L_RowMajor(2, 1) = 2*4 + 1*1 = 8 + 1 = 9
```

The *layout function* is simply applying the dot product of a coordinate with the stride, mapping a coordinate to an offset.

We say that the above layout has a **size** of 12 (product of it's shape modes) and a **rank** of 2.
This layout would also be considered **compact**, because the range of output offsets are in $[0, \mathrm{size(L)})$ and every offset appears exactly once.

An important concept that we will expand on further is that a layout defines multiple ways to index into it's layout function.
Even though the layouts described above are rank 2, we can still index into the layout with a 1D integer coordinate via `idx2crd` and `crd2idx`, which describe a one-to-one mapping between any coordinate and a 1D index.
This mapping is defined by a **colexicographical ordering** over coordinates.

A **natural coordinate** for a layout is just a tuple whose structure mirrors the layout's shape — one integer per mode, each in the valid range for that mode. For our $(3, 4)$ layout, a natural coordinate is any pair $(m, n)$ with $m \in [0, 3)$ and $n \in [0, 4)$. The 1D-index ↔ natural-coordinate bijection enumerates these 12 coordinates in colex order (leftmost mode varies fastest), shown below alongside the offset each one produces under the row-major stride:

| 1D index `i` | Natural coordinate `(m, n)` | Offset `L(m, n) = m*4 + n*1` |
| :---: | :---: | :---: |
|  0 | (0, 0) |  0 |
|  1 | (1, 0) |  4 |
|  2 | (2, 0) |  8 |
|  3 | (0, 1) |  1 |
|  4 | (1, 1) |  5 |
|  5 | (2, 1) |  9 |
|  6 | (0, 2) |  2 |
|  7 | (1, 2) |  6 |
|  8 | (2, 2) | 10 |
|  9 | (0, 3) |  3 |
| 10 | (1, 3) |  7 |
| 11 | (2, 3) | 11 |

Notice that the leftmost mode (`m`) varies fastest — colexicographic ordering walks `m = 0, 1, 2` for each fixed `n`, then advances `n`. This is what `crd2idx(coord, shape)` and its inverse `idx2crd(i, shape)` compute. Notice also that the offset column reads `0, 4, 8, 1, 5, 9, ...` rather than `0, 1, 2, ...`: the layout's stride `(4, 1)` makes the contiguous-in-memory dimension the *second* mode, so scanning the coordinate space in colex order jumps across the matrix in stride-4 hops.

Fundamentally, a layout can also just be viewed as a function from integers to integers (again, coordinates can be implicitly converted to an integer index via `crd2idx`).
In this view, the $\mathrm{size}$ of a layout $L$ can be viewed as it's *domain*.
Following this, we can also define the *cosize* of a layout, which is the maximum offset value produced by a layout function over it's defined input.

For example, you could define a layout `L = (5) : (8)`, where `size(L) == 5` and `cosize(L) = 4*8 = 32`.


### Hierarchical Layouts

This simple `shape : stride` representation with flat tuples of integers already allows for the expression of a rich variety of layouts.
However, we may want to express a layout that a flat shape and stride cannot capture. Consider a $4 \times 4$ matrix arranged as a $2 \times 2$ grid of $2 \times 2$ tiles, where each tile is column-major internally and the tiles themselves are laid out row-major across the grid. Can we express this with a flat shape:stride? Look at the offsets along the first column: $0, 1, 8, 9$. That sequence is *not* an arithmetic progression — there's no single integer stride $d_m$ such that `m * d_m` produces $0, 1, 8, 9$ as $m = 0, 1, 2, 3$. **Flat `(M, N) : (d_m, d_n)` cannot encode this layout.** We need more expressive structure.

#### Nested shape and stride

The key extension: let each entry of the shape (and the matching entry of the stride) be itself a tuple. The shape tree and stride tree must be *congruent* — same nesting, same per-level rank — so every leaf in shape has a partner leaf in stride. For that tiling:

$$
((2, 2), (2, 2)) : ((1, 8), (2, 4))
$$

Reading this as `((m_inner, m_outer), (n_inner, n_outer)) : ((1, 8), (2, 4))`:

- The m-axis decomposes into an inner mode (size 2, stride 1 — adjacent rows within a tile) and an outer mode (size 2, stride 8 — jumping to the next tile-row).
- The n-axis decomposes into an inner mode (size 2, stride 2 — adjacent cols within a tile) and an outer mode (size 2, stride 4 — jumping to the next tile-col).

#### Computing the layout function

The layout function on a *natural coordinate* — a tuple congruent to the shape — is just the dot product over the **leaves** of (coord, stride):

$$
L\bigl(((m_i, m_o), (n_i, n_o))\bigr) = m_i \cdot 1 + m_o \cdot 8 + n_i \cdot 2 + n_o \cdot 4
$$

For instance, the matrix cell at row $m = 3$, col $n = 2$ decomposes (via colex on each sub-shape) to $((m_i = 1, m_o = 1), (n_i = 0, n_o = 1))$, so $L = 1 + 8 + 0 + 4 = 13$ — matching the figure.

#### Colexicographic ordering over nested layouts

The 1D-index ↔ natural-coord bijection from the previous section generalizes naturally: walk the shape tree depth-first, and at each level the **leftmost** leaf varies fastest. For shape `((2,2),(2,2))` this enumerates the 16 natural coordinates in the order

$$
((0,0),(0,0)), \; ((1,0),(0,0)), \; ((0,1),(0,0)), \; ((1,1),(0,0)), \; ((0,0),(1,0)), \; \dots
$$

The above layout has a **depth** of 2. Every flat layout has a depth of 1.

This simple representation of a layout as a pair of nested `shape` and `stride` allows us to express a wide variety of useful layouts for GPU tensor programming.
From this fundamental primitive, we can define an algebra over layouts, and further define useful primitives to make the construction and composition of layouts more intuitive.


## Thread-Value (TV) Layouts

Matrix core instructions expect a particular mapping of threads and registers to logical matrix elements.
We can represent this mapping with a *Thread-Value* (TV) layout, who's input is a `(thread_id, reg_idx)` pair and who's output is an offset representing a logical element of a matrix.

To enforce these layouts in practice we index-specialize on the `threadIdx` and do index math:

```C++
    a_reg = *reinterpret_cast<const fp16x4_t*>(
        A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));

    for (int i = 0; i < 4; i++) {
        b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        *(D + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
    }
```


With a TV layout, we can define these mappings once for each instruction (or define an algorithm for generating TV layouts based on datatypes/arch):

```C++
    using namespace mosaic;

    auto const A_tv = Layout(make_shape (make_shape(16u, 4u), 4u),
                             make_stride(make_stride(16u, 4u), 1u));
    auto const B_tv = Layout(make_shape (make_shape(16u, 4u), 4u),
                             make_stride(make_stride(1u, 64u), 16u));
    auto const D_tv = Layout(make_shape (make_shape(16u, 4u), 4u),
                             make_stride(make_stride(1u, 64u), 16u));

    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    unsigned const tid = threadIdx.x;

    #pragma unroll
    for (int v = 0; v < 4; ++v) a_reg[v] = A[A_tv(tid, v)];
    #pragma unroll
    for (int v = 0; v < 4; ++v) b_reg[v] = B[B_tv(tid, v)];

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    #pragma unroll
    for (int v = 0; v < 4; ++v) D[D_tv(tid, v)] = c_reg[v];
```

[This tutorial](https://github.com/NVIDIA/cutlass/blob/f74fea9ce35868d3ae9f8d1dce1969d7250d3f90/media/docs/cpp/cute/0t_mma_atom.md) gives some more examples of how having a TV layout is useful.
This layout can be *composed* with a layout representing data access into LDS, or can be treated as an atom layout that get's hierarhically tiled into larger tiles via layout operations.
See [mfma_example.cpp](../examples/cpp/mfma/mfma_example.cpp) for a concrete code example as well.







---

## References

- [cute-paper]: Cecka, *CuTe Layout Representation and Algebra*, arXiv:2603.02298, 2026. <https://arxiv.org/abs/2603.02298>
- [linearlayouts-paper]: Zhou, Lezcano-Casado et al., *Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2*, ASPLOS '26, arXiv:2505.23819. <https://arxiv.org/abs/2505.23819>

[cute-paper]: https://arxiv.org/abs/2603.02298
[linearlayouts-paper]: https://arxiv.org/abs/2505.23819

