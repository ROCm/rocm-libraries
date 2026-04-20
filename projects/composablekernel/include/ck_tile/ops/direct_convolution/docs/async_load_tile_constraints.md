# Why `async_load_tile` Cannot Be Used with XOR-Swizzled LDS

## Context

The v2 grouped convolution kernel (`grouped_4c_fp16_tile_conv_impl_v2.hpp`) loads
input tiles from global memory into LDS using asynchronous 128-bit loads
(`global_load_lds` / `__builtin_amdgcn_raw_ptr_buffer_load_lds`).  The data in
LDS is stored in an XOR-swizzled layout so that the subsequent MFMA reads are
free of LDS bank conflicts.

This document explains why CK Tile's `async_load_tile` API cannot express the
XOR-swizzled global-to-LDS transfer, what constraints the hardware imposes, and
what alternatives exist.

## Background: the two async load APIs

CK Tile provides two levels of abstraction for the `global_load_lds` hardware
instruction:

| API | Abstraction level | LDS address control |
|-----|-------------------|---------------------|
| `tensor_view::async_get_vectorized_elements` | Per-thread. Each thread specifies its own global source coordinate and LDS destination pointer. | Caller supplies the LDS pointer directly. |
| `async_load_tile(lds_window, dram_window)` | Tile-level. A tile distribution maps `(warp_id, lane_id)` to logical tile coordinates. CK Tile computes both global and LDS addresses from the distribution. | Computed from the LDS window's descriptor using the **warp-level** coordinate only. |

The kernel currently uses the per-thread API (`async_get_vectorized_elements`).
This document analyses why the tile-level API (`async_load_tile`) does not work
for our use case.

## The hardware constraint: wave-collective LDS writes

The `global_load_lds` instruction is **wave-collective**: all 64 lanes in a warp
participate in a single transfer.  The instruction has two address components:

```
__builtin_amdgcn_raw_ptr_buffer_load_lds(
    rsrc,           // buffer descriptor (global base + bounds)
    smem,           // LDS destination (wave-level, written to M0 register)
    bytes,          // 4, 12, or 16 bytes per lane
    v_offset,       // per-lane offset into the global buffer
    wave_offset,    // wave-level offset into the global buffer
    imm, coherence);
```

The critical detail is that **`smem` is a wave-level value**: it is loaded into
the M0 register once for the entire warp.  The hardware then writes each lane's
data to a contiguous block of LDS:

```
LDS[smem + lane_id * bytes .. smem + lane_id * bytes + bytes - 1]
    = GLOBAL[rsrc + wave_offset + v_offset[lane_id] .. + bytes - 1]
```

This means:
- **Global reads** are fully flexible: each lane reads from an independent
  address (via per-lane `v_offset`).
- **LDS writes** are rigid: lane 0's data always goes to `smem`, lane 1's
  to `smem + bytes`, lane 2's to `smem + 2*bytes`, etc.  There is no way
  for different lanes to write to non-contiguous or reordered LDS locations.
  Hence, the tile distribution representing LDS write must be linear.

## How `async_load_tile` computes the LDS address

In `tile_window.hpp`, `async_load_with_offset` computes the LDS destination:

```cpp
// tile_window.hpp:625-632
auto lds_bottom_tensor_thread_idx =
    window_origin + window_adaptor_warp_coord.get_bottom_index();
const auto lds_coord =
    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);
CK_TILE_LDS_ADDR LdsDataType* smem =
    lds_base_ptr + (lds_coord.get_offset() + lds_ys_offset) / PackedSize;
```

The `smem` pointer is derived from `window_adaptor_warp_coord` -- the **P0
(warp)** component of the tile distribution.  There is no per-lane (P1)
contribution to the LDS address.  This matches the hardware: `smem` is written
to M0 and is the same for all lanes.

The lane-level (P1) coordinate only affects the **global read** address
(`bottom_tensor_thread_coord`).  Within a warp, each lane reads from a
different global location, but they all write to the same contiguous LDS block.

## Why XOR-swizzled LDS doesn't work

The XOR swizzle reorders the channel index based on the spatial position:

```
LDS_offset(x, c8) = x * C8 + (c8 ^ (x % C8))
```

This means that two lanes loading the same column `x` but different channel
groups `c8_a` and `c8_b` must write to LDS offsets that differ by
`(c8_a ^ (x % C8)) - (c8_b ^ (x % C8))`, which is NOT simply `c8_a - c8_b`.
The XOR creates a non-linear, position-dependent permutation of the channel
slots within each row.

For `async_load_tile` to produce an XOR-swizzled LDS layout, the LDS
descriptor would need to apply the XOR transform.  But the hardware writes
lanes contiguously starting from `smem` -- the LDS descriptor can only affect
where the **warp's block** starts, not how individual lanes are ordered within
it.

Concretely, if we set up a tile distribution where lane `l` in warp `w` maps to
logical coordinate `(col, c8) = (w * LANES_PER_ROW + l/C8, l%C8)`:

- Lane `l` **reads** from `GLOBAL[row, col, c8 * 8]` -- correct.
- Lane `l` **writes** to `LDS[smem + l * 16_bytes]` -- the hardware forces
  this sequential placement.  No XOR permutation is possible.

The resulting LDS layout is linear `[BLOCK_W, BLOCK_C]` row-major regardless
of what LDS descriptor we provide.

## Why the distribution dimensions also don't fit

There is a second, independent problem.  The tile distribution encodes how
threads map to a tile of fixed dimensions.  For our kernel:

- **Tile width** `BLOCK_W = block_q + kw - 1`.  For a typical config
  `(waves_c64=2, waves_q4=8)`: `BLOCK_W = 8*4 + 2 = 34`.
- **Threads per row** `LANES_PER_ROW = 64 / BLOCK_C8`.  For `BLOCK_C8 = 16`:
  `LANES_PER_ROW = 4`.
- **Total rows covered** = `LANES_PER_ROW * NUM_WAVES = 4 * 16 = 64`.

The distribution produces 64 spatial positions, but the tile only has 34.
The H-encoding for the spatial dimension must have factors that multiply to
`BLOCK_W`, but `BLOCK_W = 34 = 2 * 17` is not decomposable into the available
P factors (4 lanes/row and 16 warps).

CK Tile's `tile_distribution_encoding` requires exact factorisation of each
tile dimension into P (thread), Y (iteration), and R (replicate) factors.
When `BLOCK_W` is not a product of the available hardware decomposition, there
is no valid encoding.

For `async_load_tile`, threads with coordinates beyond `BLOCK_W` would produce
out-of-bounds LDS writes and corrupt adjacent data.  Unlike `load_tile` (which
goes through VGPRs and can mask invalid lanes), `async_load_tile` writes
directly to LDS through the hardware instruction and cannot selectively skip
lanes within a warp.

## How the current per-thread approach works

The per-thread `InputLoader` avoids both problems:

1. **XOR swizzle**: Each thread `tid` computes its column and channel via the
   XOR-inverse mapping (`Swizzle::x(tid)`, `Swizzle::c8(tid)`).  The thread
   reads from the XOR-permuted global address and writes sequentially to
   `input_lds[tid]`.  This places data in XOR-swizzled order in LDS.

2. **Arbitrary BLOCK_W**: Threads with `tid >= BLOCK_W * BLOCK_C8` simply skip
   the load (`load_active = false`).  No distribution factorisation needed.

3. **Pad transform for OOB**: The global descriptor uses `make_pad_transform`
   on the W dimension.  Columns outside `[0, wi)` are automatically flagged as
   invalid and the hardware writes zeros.  This replaces the manual
   `input_valid` check from v1.

## Alternatives for using `async_load_tile`

### Option A: Drop the XOR swizzle (linear LDS layout)

**Approach**: Use `async_load_tile` with a simple linear LDS descriptor (no
XOR).  The MFMA read path would also use linear offsets.

**Trade-offs**:
- (+) Full CK Tile abstraction: descriptors, tile windows, `move_tile_window`.
- (+) Code reuse: same pattern as `image_to_column` and FMHA kernels.
- (+) Easier to reason about correctness.
- (-) LDS bank conflicts during MFMA reads.  With `mfma_f32_4x4x4f16`, 16
      lanes in the same reduction group read from the same channel positions.
      Without XOR swizzle, these map to the same LDS bank, causing 16-way
      conflicts and serialization.
- (-) Performance regression: bank conflicts stall the MFMA pipeline, likely
      dominating the kernel's execution time.  The kernel is already
      bandwidth-limited, so adding LDS latency is costly.
- (-) **BLOCK_W factorisation problem remains**: the tile distribution still
      requires `BLOCK_W` to decompose into P and Y factors that match the warp
      count and lane decomposition.  This would require padding `BLOCK_W` up to
      a suitable multiple (e.g., 36 or 64) and wasting LDS.

**Verdict**: Functionally correct but likely unacceptable performance loss.  The
BLOCK_W factorisation problem adds additional LDS waste.  Not recommended
unless bank conflict impact is measured and found acceptable.

### Option B: XOR-scramble global reads via the DRAM descriptor

**Approach**: Apply `make_xor_transform` to the DRAM descriptor so that each
thread's global read address is XOR-permuted.  The LDS descriptor stays linear.
The hardware writes contiguously, but because global reads are XOR-scrambled,
the data arrives in LDS in XOR order.

**Trade-offs**:
- (+) Maintains XOR-swizzled LDS layout for bank-conflict-free MFMA reads.
- (+) Uses `async_load_tile` for the transfer.
- (-) The XOR transform must be applied at the **tile-local** coordinate level,
      not at the global level.  CK Tile's `make_xor_transform` applies to the
      descriptor's full coordinate space.  If the global descriptor has
      coordinates `(row, col_global, c_global)`, the XOR is
      `c8 ^ (col_global % C8)` -- this uses the absolute column, not the
      tile-local column.  Two tiles at different `block_q` positions would get
      different XOR patterns, breaking the fixed LDS layout that the MFMA read
      path expects.
- (-) Working around this would require the XOR transform to operate on
      tile-local coordinates, which CK Tile's descriptor system doesn't
      currently support (transforms operate on descriptor-level indices, not
      window-relative indices).
- (-) **BLOCK_W factorisation problem remains**.

**Verdict**: Architecturally incompatible with CK Tile's current descriptor
system.  Would require framework-level changes to support tile-local transforms.

### Option C: Use `load_tile` (VGPR path) instead of `async_load_tile`

**Approach**: Use `load_tile` to load from global into VGPRs, then
`store_tile` to write from VGPRs to LDS with an XOR-swizzled LDS descriptor.

This is how `image_to_column` works: `load_tile(image_tile)` loads data into a
register tile, then `store_tile(gemm_tile, loaded_tile)` writes it out.

**Trade-offs**:
- (+) Full per-lane control of both global read and LDS write addresses.
- (+) XOR swizzle is naturally expressed in the LDS store descriptor.
- (+) Full CK Tile abstraction for both global and LDS paths.
- (+) Pad transform works correctly (per-lane validity).
- (-) Two-step process: global → VGPR → LDS instead of global → LDS.
      Consumes VGPR budget: each thread needs 4 VGPRs for one 128-bit load.
      With `BLOCK_W * BLOCK_C8 / block_size` loads per thread, this can be
      significant.
- (-) Cannot overlap the global load with MFMA compute the same way that
      `async_get_vectorized_elements` can (the async instruction returns
      immediately and the data arrives in LDS without VGPR involvement).
- (-) **BLOCK_W factorisation problem remains** for the distribution.

**Verdict**: Functionally correct and composable, but loses the key performance
benefit of async global→LDS transfers.  VGPR pressure and loss of
async-compute overlap make this worse than the current per-thread approach.

### Option D: Keep per-thread loads with padded descriptor (current approach)

**Approach**: Use `tensor_view::async_get_vectorized_elements` directly for
each thread.  Each thread computes its XOR-swizzled coordinates, reads from
the padded global descriptor, and writes to its LDS slot.

**Trade-offs**:
- (+) Preserves XOR swizzle for bank-conflict-free MFMA reads.
- (+) Preserves async global→LDS (no VGPR detour).
- (+) Handles arbitrary `BLOCK_W` via `load_active` check.
- (+) Pad transform provides automatic OOB zero-fill (replaces manual check).
- (-) Manual coordinate computation (though encapsulated in `InputLoader`).
- (-) Manual row advancement (pass `y+1` to `prefetch` instead of
      `move_tile_window`).
- (-) Does not benefit from CK Tile's tile window / distribution abstractions
      for the input loading path.

**Verdict**: The current choice.  Best performance, acceptable code complexity.

## Summary

| Criterion | Option A (no XOR) | Option B (DRAM XOR) | Option C (VGPR path) | Option D (current) |
|-----------|--------------------|---------------------|----------------------|---------------------|
| Uses `async_load_tile` | Yes | Yes | No (`load_tile`) | No (per-thread) |
| XOR bank-conflict avoidance | No | Yes | Yes | Yes |
| Async global→LDS | Yes | Yes | No | Yes |
| BLOCK_W flexibility | Needs padding | Needs padding | Needs padding | Any size |
| Code reuse (CK Tile patterns) | High | Medium | High | Low |
| Framework changes needed | None | Tile-local transforms | None | None |
| Expected performance | Poor (bank conflicts) | N/A (infeasible) | Moderate | Best |

The fundamental tension is between CK Tile's tile-level abstractions (which
assume power-of-2 tile shapes and uniform thread mappings) and this kernel's
requirements (non-power-of-2 tile width, XOR-swizzled LDS, async hardware
transfers).  The per-thread approach sacrifices code reuse for performance.

## Future directions

If CK Tile were extended with:

1. **Tile-local coordinate transforms** -- a way to apply transforms relative
   to the tile window origin rather than the global descriptor origin.

2. **Partial-tile distributions** -- distributions that can declare some
   threads as inactive when the tile dimension doesn't perfectly decompose.

3. **Per-lane LDS write control in `async_load_tile`** -- decoupling the LDS
   destination from the wave-collective M0 model (unlikely without hardware
   changes).

Then Option B could become viable.  Items 1 and 2 are software-only
changes to the CK Tile framework.  Item 3 would require new GPU hardware
instructions.

## Appendix: BLOCK_W factorisation analysis

### No existing configuration fits the distribution

The tile distribution requires the spatial dimension's H-encoding factors to
multiply exactly to the tile width.  For all configurations in this kernel:

```
total_spatial = LANES_PER_ROW * NUM_WAVES
              = (64 / BLOCK_C8) * (waves_c64 * waves_q4)
              = (8 / waves_c64) * waves_c64 * waves_q4
              = 8 * waves_q4

BLOCK_W       = block_q + kw - 1
              = 4 * waves_q4 + (kw - 1)
              = 4 * waves_q4 + 2          (for kw = 3)
```

| waves_c64 | waves_q4 | BLOCK_C8 | LANES_PER_ROW | NUM_WAVES | total_spatial | BLOCK_W | Ratio |
|-----------|----------|----------|---------------|-----------|---------------|---------|-------|
| 2 | 8 | 16 | 4 | 16 | 64 | 34 | 1.88 |
| 2 | 4 | 16 | 4 | 8 | 32 | 18 | 1.78 |
| 2 | 2 | 16 | 4 | 4 | 16 | 10 | 1.60 |
| 2 | 1 | 16 | 4 | 2 | 8 | 6 | 1.33 |
| 1 | 1 | 8 | 8 | 1 | 8 | 6 | 1.33 |

For `total_spatial == BLOCK_W`, we would need `8 * waves_q4 = 4 * waves_q4 + 2`,
i.e., `waves_q4 = 0.5`.  Not an integer.  The `+2` filter halo (from `kw - 1`)
breaks alignment for **every** configuration.

Even for `kw = 1` (1x1 convolution), `BLOCK_W = 4 * waves_q4` while
`total_spatial = 8 * waves_q4`, giving a fixed 2x mismatch.  The mismatch is
structural: each lane covers one spatial position and one channel group, and
the lane count (64) always over-provisions the spatial dimension relative to
the tile width.

### Can surplus threads be treated as no-ops?

Yes.  The `global_load_lds` hardware instruction already supports this.

When a thread's tile distribution coordinate maps to a position outside the
global tensor bounds, CK Tile's validity check
(`coordinate_has_valid_offset_assuming_top_index_is_valid`) returns `false`.
The `amd_async_buffer_load` function then sets `v_offset = 0x7fffffff`,
which causes an out-of-bounds global read.  The hardware behaviour for
OOB async loads is:

- **The LDS write for that lane is suppressed** (no data written).
- **The lane's LDS slot is still reserved** in the contiguous block
  (subsequent lanes' slots do not shift).
- No exception or memory violation occurs.

This means a distribution with `total_spatial = 64` and tile width
`BLOCK_W = 34` is safe: threads at spatial positions 34-63 will attempt
to read from global coordinates beyond the tile, the pad transform or
tensor bounds will flag them as invalid, and the hardware suppresses
their LDS writes.  The MFMA read path only accesses positions 0-33,
so the dead LDS slots at positions 34-63 are never read.

CK Tile imposes no static assertion or runtime check requiring the
distribution span to match the tile window dimensions.  The
`make_tile_window` constructor simply computes the per-thread tensor
coordinate as `window_origin + distribution_offset` and relies on the
tensor descriptor's bounds checking for validity.

### LDS overhead from surplus slots

The LDS allocation must accommodate the full distribution span, not just
the actual tile width.  For the main config `(waves_c64=2, waves_q4=8)`:

```
Actual tile LDS per buffer  = BLOCK_W * BLOCK_C8 = 34 * 16 =  544 uint4 =  8,704 bytes
Padded LDS per buffer       = 64 * 16            =          = 1024 uint4 = 16,384 bytes
Double-buffered actual      = 2 * 544            =          = 1088 uint4 = 17,408 bytes
Double-buffered padded      = 2 * 1024           =          = 2048 uint4 = 32,768 bytes

Weight/output LDS           = max(576, 512)      =            576 uint4 =  9,216 bytes

Total LDS (current)         = 17,408 + 9,216 = 26,624 bytes  (41% of 64 KB)
Total LDS (padded)          = 32,768 + 9,216 = 41,984 bytes  (64% of 64 KB)
```

The padded approach fits within the 64 KB LDS limit on gfx950 but reduces
headroom for occupancy.

### Global bandwidth waste from surplus loads

Surplus threads whose spatial coordinate falls within the (padded) image
bounds will load real data from global memory that is never used.  For
the main config:

```
Useful loads per row    = BLOCK_W * BLOCK_C8 = 34 * 16 = 544
Total loads per row     = total_spatial * BLOCK_C8 = 64 * 16 = 1024
Surplus loads per row   = 1024 - 544 = 480  (47% wasted)
```

For blocks near the right edge of the image, the pad transform flags
surplus columns as OOB, suppressing those loads.  But for interior
blocks, all 480 surplus loads fetch valid (but unused) data from global
memory.  This wastes nearly half the global memory bandwidth per input
row load.

### Summary of the surplus-thread approach

| Factor | Value (main config) |
|--------|---------------------|
| LDS overhead | +88% per input buffer (+15 KB total) |
| Global bandwidth waste | ~47% of loads are surplus |
| Occupancy impact | LDS usage increases from 41% to 64% of budget |
| Correctness | Correct (OOB suppressed, surplus slots never read) |
| Code structure | Full CK Tile tile distribution + `async_load_tile` |

The approach is **functionally correct** and would allow using `async_load_tile`
with a linear LDS layout (dropping XOR swizzle, as in Option A).  However,
the combination of LDS waste and bandwidth waste makes it unattractive for
a performance-sensitive kernel.  The per-thread `InputLoader` (Option D)
avoids both overheads by loading exactly `BLOCK_W * BLOCK_C8` elements with
no surplus.
