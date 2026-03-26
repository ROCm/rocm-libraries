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
  warp count, vector width, padding.

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
time by `make_kernel`.

### Scaled Addition

The kernel always computes `c = alpha * a + beta * b`. Scalar parameters
`alpha` and `beta` are passed via the generic `Args` struct's scalar slots
(`scalars[0].f32` and `scalars[1].f32`), matching the BLAS convention where
scalar parameters are always present. For plain addition, pass
`alpha = 1.0f` and `beta = 1.0f`.

### Custom CK Tile Kernel

The device kernel (`runVectorAdd`) uses CK Tile tile primitives directly
(`load_tile`, `sweep_tile_span`, `store_tile`) rather than the stock
`ElementWiseKernel`. This enables passing scalar parameters and provides
a more instructive example of building custom kernels from CK Tile building
blocks.

### Tuning Surface

The algorithm exposes four independent knobs matching CK Tile's `ElementWiseShape`:

| Parameter | Field | Description |
|-----------|-------|-------------|
| BlockTile | `block_tile` | Elements processed per thread block |
| BlockWarps | `block_warps` | Warps per thread block (must be power of 2) |
| WarpTile | `warp_tile` | Controls vector load width (`kVectorM`) |
| Pad | `pad` | Enable padding for unaligned problem sizes |

Derived quantities (validated at compile time by `make_kernel`):
- `kVectorM = min(128 / max_type_bits, warp_tile / 64)` — vector elements per thread
- `kRepeatM = block_tile / (block_warps × kVectorM × 64)` — iterations per warp
- `thread_block_size = 64 × block_warps` — actual threads launched

### Variant Registry

`rocm_vector_add_registry.hpp` provides programmatic variant selection:
- `ALL_VARIANTS[]` — constexpr table of all compiled kernel variants
- `findVariant(in_dtype, out_dtype, problem_size)` — selects the best variant:
  largest `block_tile` that divides `problem_size` cleanly, or padded fallback

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

The `_w8`/`_w2` suffixes indicate multi-warp variants.

## File Roles

Each example uses a three-file compilation boundary that separates
metaprogramming, host runtime, and device code:

| File | Compiled by | Purpose |
|------|-------------|---------|
| `rocm_vector_add_kernel.hpp` | Both (g++ and hipcc) | **Metaprogramming** — structural types (`VectorAddKernel`, `ElementwiseAlgorithm`), `consteval` factory (`make_kernel`), compile-time `static_assert` tests. No runtime code — everything evaluates at compile time. |
| `rocm_vector_add_api.hpp` | Host only (g++) | **Host runtime** — arg assembly, launch helpers, runtime validation. Guards with `#error` on device compilation. Includes `_kernel.hpp`. |
| `rocm_vector_add_dev.hpp` | Device only (hipcc `--cuda-device-only`) | **Device code** — CK Tile bridge (`runVectorAdd<K>`), `__device__` functions. Guards with `#error` on host compilation. Includes `_kernel.hpp`. |
| `rocm_vector_add_registry.hpp` | Host only | Variant table (`ALL_VARIANTS[]`) and `findVariant` selection |
| `vector_add_*.hip` | Device only | Variant instantiations (~10 lines each) — include only `_dev.hpp` |
| `main.cpp` | Host only | Host loader — variant selection demo, verify-all, scaled-add test |
| `pack.py` | — | Archive packer with variant metadata |
| `CMakeLists.txt` | — | Build system (variant × arch nested loop) |

### Compilation Boundary

```text
                    _kernel.hpp (metaprogramming)
                   /            \
          _api.hpp               _dev.hpp
         (host only)            (device only)
             |                       |
         main.cpp              *.hip files
       registry.hpp
```

`_kernel.hpp` is pure metaprogramming: `consteval` factories, `constexpr` structural
types, and `static_assert` tests. The compiler evaluates it all and produces constants.
No runtime code is generated on either side.

`.hip` files are compiled with `--cuda-device-only` and include only `_dev.hpp`
(which transitively includes `_kernel.hpp`). `main.cpp` is compiled as plain C++
and includes `_api.hpp` (via the registry). The `#error` guards enforce these
boundaries — including the wrong header produces a clear compile error.

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
  FP32 -> vector_add_fp32_b2048_w8 (tile=2048, warps=8)
  FP16 -> vector_add_fp16_b1024 (tile=1024, warps=1)
  BF16 -> vector_add_bf16_b1024 (tile=1024, warps=1)

Running all 12 variants (alpha=1, beta=1):
  vector_add_fp32_b256: tile=256, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp32_b256 (grid=16, block=64): PASSED
  ...
  vector_add_fp16_fp32_b1024: tile=1024, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp16_fp32_b1024 (grid=4, block=64): PASSED
  vector_add_fp32_fp16_b1024: tile=1024, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp32_fp16_b1024 (grid=4, block=64): PASSED

Scaled-add test (alpha=2, beta=0.5):
  vector_add_fp32_b2048_w8: tile=2048, warps=8, threads=512, N=4096 (aligned)
  vector_add_fp32_b2048_w8 (grid=2, block=512): PASSED
```

## Design Notes

**C++20 required.** Struct NTTPs and `consteval` validation are C++20 features.

**`thread_block_size = 64 × block_warps`, not `block_tile`.** The number of
GPU threads equals the warp count times the wavefront size (64 on AMD CDNA).
Each warp iterates `kRepeatM` times to cover its share of the block tile.

**Archive metadata.** `pack.py` writes a `variant_metadata` section in the
kpack TOC containing each variant's tuning parameters (in_dtype, out_dtype,
block_tile, block_warps, warp_tile, pad). This is ignored by the kpack reader
but available for tooling that inspects archives.

**Unified Signature.** All operations share the same `Signature` struct. The
operation type is determined by the operator structs in the `ops` array
(`AddOp` for elementwise, `GemmOp` for GEMM, etc.). `make_kernel()` pattern-
matches the ops to select the appropriate kernel path and validation.