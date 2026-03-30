# Example 03 — `rocm_ck` Vector Add (Full Tuning Surface)

Demonstrates CK Tile's **Signature + Algorithm** pattern for distributing
pre-compiled GPU kernels with full control over the ElementWise tuning surface.
Supports mixed input/output types and scaled addition (`c = alpha * a + beta * b`).

## Key Concepts

### Signature / Algorithm Separation

Kernel configuration is split into two orthogonal concerns:

- **Signature** (`Signature`): *What* the kernel computes — a directed compute
  graph of typed operators referencing named tensors. For vector add, the graph
  is a single `AddOp`. `resolve()` flattens the signature into concrete tensor
  descriptors at compile time.
- **Algorithm** (`ElementwiseAlgorithm`): *How* the kernel executes — tile geometry,
  wavefront count, vector width, padding.

This separation lets the same operation (vector add) be compiled with many
different tuning configurations, each producing a distinct `.hsaco` binary.

### Operator-Centric Signature

The signature describes the compute graph using typed operator structs:

```cpp
// Same-type: all FP32
Signature{.dtype = DataType::FP32,
          .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// Widening: fp16 inputs, fp32 output
Signature{.dtype = DataType::FP16,
          .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
          .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// Narrowing: fp32 inputs, fp16 output
Signature{.dtype = DataType::FP32,
          .tensors = {Tensor{.name = "C", .dtype = DataType::FP16}},
          .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}
```

`Signature::dtype` sets the default for all tensors. Explicit `Tensor` entries
override specific tensors by name. For mixed types, `kVectorM` is constrained
by the wider type (fewer elements per 128-bit register), validated at compile
time by `make_spec`.

### Scaled Addition

The kernel always computes `c = alpha * a + beta * b`. Scalar parameters
`alpha` and `beta` are passed via the generic `Args` struct's scalar slots
(`scalars[0].f32` and `scalars[1].f32`), matching the BLAS convention where
scalar parameters are always present. For plain addition, pass
`alpha = 1.0f` and `beta = 1.0f`.

### Custom CK Tile Kernel

The device kernel (`run`) uses CK Tile tile primitives directly
(`load_tile`, `sweep_tile_span`, `store_tile`) rather than the stock
`ElementWiseKernel`. This enables passing scalar parameters and provides
a more instructive example of building custom kernels from CK Tile building
blocks.

### Tuning Surface

The algorithm exposes four independent knobs matching CK Tile's `ElementWiseShape`:

| Parameter | Field | Description |
|-----------|-------|-------------|
| BlockTile | `block_tile` | Elements processed per workgroup |
| BlockWarps | `block_waves` | Wavefronts per workgroup (must be power of 2) |
| WarpTile | `wave_tile` | Controls vector load width (`kVectorM`) |
| Pad | `pad` | Enable padding for unaligned problem sizes |

Derived quantities (validated at compile time by `make_spec`):
- `kVectorM = min(128 / max_type_bits, wave_tile / 64)` — vector elements per work-item
- `kRepeatM = block_tile / (block_waves × kVectorM × 64)` — iterations per wavefront
- `workgroup_size = 64 × block_waves` — work-items per workgroup

### Variant Registry

`vector_add_variants.hpp` is the single source of truth for all variant specs,
shared by both device `.hip` files and the host `main.cpp`:
- `vector_add_variants[]` — constexpr table of all compiled kernel variants
- `vector_add_variant_spec("name")` — `consteval` lookup for device code (typo = compile error)
- `findVariant(in_dtype, out_dtype, problem_size)` — runtime selection: largest
  `block_tile` that divides `problem_size` cleanly, or padded fallback

## Compiled Variants

| Variant | In | Out | BlockTile | BlockWarps | WarpTile | Threads |
|---------|----|-----|-----------|------------|----------|---------|
| `fp32_b256` | FP32 | FP32 | 256 | 1 | 256 | 64 |
| `fp32_b512` | FP32 | FP32 | 512 | 1 | 512 | 64 |
| `fp32_b1024` | FP32 | FP32 | 1024 | 1 | 1024 | 64 |
| `fp16_b512` | FP16 | FP16 | 512 | 1 | 512 | 64 |
| `fp16_b1024` | FP16 | FP16 | 1024 | 1 | 1024 | 64 |
| `bf16_b512` | BF16 | BF16 | 512 | 1 | 512 | 64 |
| `bf16_b1024` | BF16 | BF16 | 1024 | 1 | 1024 | 64 |
| `fp32_b2048_w8` | FP32 | FP32 | 2048 | 8 | 64 | 512 |
| `fp16_b1024_w2` | FP16 | FP16 | 1024 | 2 | 512 | 128 |
| `fp16_fp32_b1024` | FP16 | FP32 | 1024 | 1 | 1024 | 64 |
| `fp32_fp16_b1024` | FP32 | FP16 | 1024 | 1 | 1024 | 64 |
| `bf16_fp32_b1024` | BF16 | FP32 | 1024 | 1 | 1024 | 64 |

The `_w8`/`_w2` suffixes indicate multi-wave variants.

## File Roles

| File | Compiled by | Purpose |
|------|-------------|---------|
| `elementwise_spec.hpp` | Both (`include/rocm_ck/`) | **Structural types** — `ElementwiseSpec`, `ElementwiseAlgorithm`, `consteval` factory (`make_spec`). No runtime code. |
| `vector_add_variants.hpp` | Both (g++ and hipcc) | **Variant registry** — constexpr table of all kernel configurations, `consteval vector_add_variant_spec()` lookup, `findVariant()` selection. Single source of truth for device and host code. |
| `main.cpp` | Host only | Host loader — variant selection demo, verify-all, scaled-add test |
| `vector_add_*.hip` | Device only | Variant instantiations (~12 lines each) — include `vector_add_variants.hpp` and `elementwise_dev.hpp` |
| `elementwise_dev.hpp` | Device only (`include/rocm_ck/`) | **CK Tile bridge** — `run<S>`, `CkTypeMap`, `__device__` functions. Guards with `#error` on host compilation. |
| `pack.py` | — | Archive packer with variant metadata |
| `CMakeLists.txt` | — | Build system (variant × arch nested loop) |

### Compilation Boundary

```text
    <rocm_ck/elementwise_spec.hpp>      (structural types)
                    |
       vector_add_variants.hpp          (example variant table)
                 /        \
          main.cpp      vector_add_*.hip
           (host)           |
                    <rocm_ck/elementwise_dev.hpp>  (device bridge: run<S>)
```

`elementwise_spec.hpp` defines structural types (`ElementwiseSpec`,
`ElementwiseAlgorithm`). `vector_add_variants.hpp` builds the variant table on
top of those types — included by both host and device code, ensuring specs are
defined exactly once.

`.hip` files are compiled with `--cuda-device-only` and include both
`vector_add_variants.hpp` (for the spec) and `elementwise_dev.hpp` (for
`run<S>`). `main.cpp` includes only `vector_add_variants.hpp` to access variant
metadata and selection.

## Build

```bash
cd experimental/rocm_ck/examples/03_rocm_ck_vector_add
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"
ninja -C build
```

## Run

```bash
./build/kpack_rocm_ck_vector_add build/kernels.kpack
```

Expected output:
```
Opened build/kernels.kpack — architectures: gfx90a, gfx942

Variant selection for N=4096:
  FP32 -> vector_add_fp32_b2048_w8 (tile=2048, waves=8)
  FP16 -> vector_add_fp16_b1024 (tile=1024, waves=1)
  BF16 -> vector_add_bf16_b1024 (tile=1024, waves=1)

Running all 12 variants (alpha=1, beta=1):
  vector_add_fp32_b256: tile=256, waves=1, work_items=64, N=4096 (aligned)
  vector_add_fp32_b256 (grid=16, block=64): PASSED
  ...
  vector_add_fp16_fp32_b1024: tile=1024, waves=1, work_items=64, N=4096 (aligned)
  vector_add_fp16_fp32_b1024 (grid=4, block=64): PASSED
  vector_add_fp32_fp16_b1024: tile=1024, waves=1, work_items=64, N=4096 (aligned)
  vector_add_fp32_fp16_b1024 (grid=4, block=64): PASSED

Scaled-add test (alpha=2, beta=0.5):
  vector_add_fp32_b2048_w8: tile=2048, waves=8, work_items=512, N=4096 (aligned)
  vector_add_fp32_b2048_w8 (grid=2, block=512): PASSED
```

## Design Notes

**C++20 required.** Struct NTTPs and `consteval` validation are C++20 features.

**`workgroup_size = 64 × block_waves`, not `block_tile`.** The number of
GPU threads equals the wavefront count times the wavefront size (64 on AMD CDNA).
Each wavefront iterates `kRepeatM` times to cover its share of the block tile.

**Archive metadata.** `pack.py` writes a `variant_metadata` section in the
kpack TOC containing each variant's tuning parameters (in_dtype, out_dtype,
block_tile, block_waves, wave_tile, pad). This is ignored by the kpack reader
but available for tooling that inspects archives.

**Unified Signature.** All operations share the same `Signature` struct. The
operation type is determined by the operator structs in the `ops` array
(`AddOp` for elementwise, `GemmOp` for GEMM, etc.). `make_spec()` pattern-
matches the ops to select the appropriate kernel path and validation.