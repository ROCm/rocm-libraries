# Memory, Layout, And Coordinate Transforms

The DSL has two related but distinct memory-description layers. Use the transform DAG when the logical index space is not a simple linear layout. Use tensor views and tile windows when the access pattern is a regular tile over a known view. They can be bridged: transform descriptors can produce offsets and validity predicates consumed by loaders or views.

```text
ck_dsl.transforms                  # algebraic coordinate-transform DAG over SSA values
ck_dsl.helpers.tensor_view         # CK Tile-like TensorDescriptor / TensorView / TileWindow
ck_dsl.helpers.distribution        # TileDistributionEncoding / StaticDistributedTensor
ck_dsl.helpers.layouts             # LdsLayout, TransposeLdsReader
```

## Transform DAG (`transforms.py`)

Core object:

```text
TensorDescriptor.naive(name, lengths, dtype=..., coord_names=...)
TensorDescriptor.transform(t1, t2, ...) -> TensorDescriptor
TensorDescriptor.offset(b, **upper_values) -> (i32 offset, optional i1 valid)
```

Transform constructors:

```text
pass_through(name, into=None)
pad(name, lo, hi)                    # bounds check; coord value unchanged
pad_dynamic(name, lo=None, hi=None)  # lo/hi may be SSA values
embed(upper, into, strides, offset, lo, hi)
                                      # affine map + bounds check
unmerge(upper, into, dims)            # split flat coord into N
merge(upper, into, dims)              # inverse of unmerge for packed shapes
indirect(upper, into, table, base)    # consume coord; produce lower via i32 table lookup
```

### Worked Example: conv input

```python
desc = (
    TensorDescriptor
    .naive("A_nhwc", lengths=[N, Hi, Wi, C],
           coord_names=["n", "hi", "wi", "c"])
    .transform(
        unmerge("m",  into=["n", "ho", "wo"], dims=[N, Ho, Wo]),
        embed(["ho", "r"], "hi", strides=[sH, dH], offset=-pH, lo=0, hi=Hi),
        embed(["wo", "s"], "wi", strides=[sW, dW], offset=-pW, lo=0, hi=Wi),
        unmerge("k",  into=["r", "s", "c"], dims=[R, S, C]),
        pad("r", lo=0, hi=R),
        pad("s", lo=0, hi=S),
    )
)
off, valid = desc.offset(b, m=m_val, k=k_val)
```

The DAG emits SSA arithmetic and an `i1` validity predicate at IR-build time. The author writes algebra, not hand-expanded offset expressions.

### `pass_through`

Identity or rename. Use when a coordinate should pass through the DAG unchanged but naming needs to align with another transform.

### `pad`

Adds a static bounds predicate:

```text
valid &= (lo <= coord) && (coord < hi)
```

The coordinate value is not changed. Use for:

- convolution input H/W padding,
- partial K-tile R/S guards,
- output tail masks when the descriptor participates in store masking.

### `pad_dynamic`

Like `pad`, but `lo` and/or `hi` are runtime SSA values. Use for:

- attention sequence-length tails (`cur_batch_q_len`);
- per-batch dynamic windows;
- dynamic causal / sliding bounds.

### `embed`

Affine map from one or more upper coordinates into one lower coordinate:

```text
lower = sum(upper_i * stride_i) + offset
valid &= (lo <= lower) && (lower < hi)
```

Convolution example:

```text
hi = ho * stride_h + r * dilation_h - pad_h
wi = wo * stride_w + s * dilation_w - pad_w
```

### `unmerge`

Splits a flat coordinate into multiple. Convolution implicit-GEMM:

```text
n  = m / (Ho * Wo)
ho = (m / Wo) % Ho
wo = m % Wo
```

Also used for `k -> r, s, c`, attention `token/dim` decomposition, and tile-local lane decomposition.

### `merge`

Flattens multiple coordinates into one. Inverse of `unmerge` for simple packed shapes.

### `indirect`

Loads an i32 table entry as part of coordinate mapping. The paged-KV-cache primitive in attention:

```text
physical_block = block_tables[seq_base + logical_block]
```

Use when a logical coordinate must pass through a page table, group map, or other runtime index table before computing a physical memory offset.

## Tensor Views (`helpers/tensor_view.py`)

Primary objects:

```text
TensorDescriptor         # shape + strides + dtype (strides may be int or SSA)
TensorView               # pointer + descriptor + addr space ("global" / "lds" / "buffer")
TileWindow               # moveable origin + extents into a TensorView
TensorCoordinate         # cached (index, offset) pair with incremental move
BufferResource           # 128-bit AMDGPU buffer descriptor + soffset wrapper
```

Constructors:

```text
make_global_view(ptr, shape, dtype, strides=None)
make_lds_view(b, dtype=..., shape=...)
make_buffer_resource(b, ptr, num_bytes=...)
make_buffer_view(rsrc, shape, dtype, strides=None)
make_naive_tensor_descriptor_packed(shape, dtype)
make_naive_tensor_view_packed(ptr, shape, dtype)
make_tile_window(view, lengths, origin)
make_tensor_coordinate(b, desc, idx)
move_tensor_coordinate(b, coord, step)
view_from_transforms_descriptor(ptr, rich_desc)   # bridge from transforms.TensorDescriptor
```

`TensorView` dispatches loads / stores by address space. The buffer-view API:

```text
view.load_vec_at(b, elem_off, n, mask=...)
view.load_scalar_at(b, elem_off, mask=...)
view.store_vec_at(b, elem_off, value, n, mask=...)
view.store_scalar_at(b, elem_off, value, mask=...)
view.async_load_lds_at(b, lds_ptr, elem_off, dwords=..., mask=...)
```

`mask=` works by replacing a False lane's byte offset with `INT32_MAX`; the bounds-checked buffer descriptor turns the access into a silent OOB. No software branch needed.

## TileWindow

A `TileWindow` is a view plus tile lengths plus an origin:

```python
window = make_tile_window(view, lengths=(tile_m, tile_n), origin=(block_m, block_n))
v = window.load_vec(b, n=8, ...)
window.store_vec(b, value, n=8, ...)
window.move_to(b, new_origin)
window.shift_by(b, deltas)
```

Use it when a kernel repeatedly loads / stores a tile from a logical origin. The CK Tile parity table in `helpers/README.md` maps `make_tile_window(view, lengths, origin)` to CK Tile's `make_tile_window` and `tile.move_to(*origin)` / `tile.shift_by(b, *deltas)` to `tile_window.set_window_origin(origin)`.

## Distribution Layer (`helpers/distribution.py`)

Objects:

```text
TileDistributionEncoding(Rs, Hs, Ps2RHs_major, Ps2RHs_minor, Ys2RHs_major, Ys2RHs_minor)
TileDistribution
StaticDistributedTensor
LoadStoreTraits
```

Constructors:

```text
make_static_tile_distribution(encoding)
make_load_store_traits(distribution, max_vec=8)
make_static_distributed_tensor(distribution, dtype=...)
load_tile(b, window, distribution=..., ps=[[tid]], traits=...)
store_tile(b, window, distributed_tensor, ps=[[tid]], traits=...)
```

Feature matrix (from `helpers/README.md`):

```text
Rs == () (no replication)               done
Rs != () (replication)                  done
1D X                                    done
2D X (2-contributor P)                  done
3D+ X                                   encoding accepts, untested in shipped demos
LoadStoreTraits smart picker            done (scans Y dims for stride-1 X dim)
LoadStoreTraits scalar fallback         done
Snake traversal                         done
Row-major (non-snake) traversal         done (iterate_accesses(snake=False))
TileWindow.load/store(distribution=...) done
Validity / mask threading through load_tile   not yet
```

Use the distribution layer when:

- a tile has a non-trivial `(Y, P)` decomposition (multi-warp tiles, multi-dim Y space, replicated workloads);
- you want vector dim and width picked automatically;
- you want snake traversal order without hand-coding it.

For simple row-wise kernels, `sweep_row_chunks` from `helpers/sweep.py` is easier.

## LdsLayout (`helpers/layouts.py`)

`LdsLayout` centralizes LDS shape and access rules:

- per-axis lengths;
- K padding (`lds_k_pad`; default `+8` on sync paths when `block_k >= 16`, `+0` on async paths);
- packed async constraints;
- swizzle support;
- guardrails for invalid async layouts.

Key rule:

```text
Async DRAM-to-LDS writes lane-contiguous bytes.
```

Therefore a packed-async LDS destination cannot itself be per-lane swizzled. If a consumer needs a swizzle to avoid bank conflicts, put it in the LDS *read* address arithmetic.

`xor_swizzle_bytes(addr, swizzle_size)` produces the XOR-style swizzle used by `TransposeLdsReader`.

## TransposeLdsReader

Provides formulas for transpose-style LDS reads, including `ds_read_b64_tr_b16`-like patterns. Used by attention's PV matmul where V is in LDS row-major and we want the MFMA B-operand layout directly (one wave64 cross-lane transpose, replacing four strided `ds_read_u16` instructions).

`b.ds_read_tr16_b64(smem, *indices, dtype=F16)` is the underlying IR op.

## OOB And Tail Patterns

Preferred buffer path:

```python
off, valid = desc.offset(b, ...)              # element offset + i1 valid
off_bytes = b.mul(off, b.const_i32(sizeof_elem))
safe = b.select(valid, off_bytes, b.const_i32((1 << 31) - 1))
v = b.buffer_load_vN_f16(rsrc, safe, c0, dwords=4)
```

For stores via the higher-level helpers:

```python
view.store_vec_at(b, elem_off, value, n=8, mask=valid)
```

For normal pointer loads:

```python
safe_idx = b.select(valid, real_idx, b.const_i32(0))
v = b.global_load_f16(ptr, safe_idx)
v = b.select(valid, v, zero)
```

Do not issue a global load from an invalid pointer and then select the value away. The load can fault before the select matters.

## Choosing The Right Abstraction

```text
Plain GEMM A/B/C, regular strides              -> TensorView + TileWindow
Conv input/weight/output (NHWC/KRSC/NHWK)       -> transform DAG descriptors
Paged-KV attention addressing                  -> transform DAG with `indirect`
Small row-wise ops (norm, reduce, elementwise) -> packed view + sweep_row_chunks
Tails / padding / masks                         -> buffer resources + descriptor.valid
Complex distributed tile load/store             -> TileDistributionEncoding
Specialized one-off pattern                     -> raw IR; document why helpers don't fit
```

## Debug Checklist

When addressing is wrong:

1. Print descriptor upper names (`desc.upper_names`) — these are the coords you supply.
2. Test a tiny shape: compare `desc.offset(b, ...)` SSA output against a Python reference for a handful of (coord) inputs.
3. Check `valid` semantics for padding and tails (`pad` adds to valid; `unmerge` doesn't).
4. Confirm offsets are in *elements* or *bytes* as required by the consumer (the buffer path needs bytes; multiply by `sizeof(dtype)`).
5. Confirm vector widths do not cross invalid boundaries (a `<8 x half>` load that spans valid + invalid lanes can still fault on the buffer path if `num_records` is sized exactly to valid bytes).
6. Confirm output epilogue lane-to-output mapping (`MfmaAtom.lane_to_output`) matches the atom that produced the accumulator.
7. For buffer ops, verify the sentinel offset is in bytes and the `*_bytes` ABI arg matches the buffer descriptor's `num_records`.
8. For async copies, verify LDS byte layout matches consumer read formulas (`smem_load_vN_f16` indexing and any `xor_swizzle_bytes` in the reader).
