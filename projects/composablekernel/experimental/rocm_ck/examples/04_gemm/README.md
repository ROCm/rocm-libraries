# Example 04 — GEMM (Multi-Type, Parameterized Tile Geometry, Fused Epilogue)

Demonstrates kpack distribution of a **GEMM kernel** with multiple data type
variants, parameterized tile geometry, and fused epilogues. Extends example 03's
`{.signature, .algorithm}` pattern to GEMM's multi-type domain.

## Key Concepts

### Signature — the "WHAT" of a GEMM

The GEMM operation is described by an operator-centric `Signature` — a directed
compute graph where `GemmOp` defines the matrix multiply and optional epilogue
operators (`AddOp`, `ReluOp`, etc.) compose the post-GEMM fusion:

```cpp
// Plain GEMM (acc_dtype defaults to FP32)
Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// GEMM + bias + ReLU (operator composition)
Signature{.dtype = DataType::FP16,
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                  ReluOp{.in = "D", .out = "E"}}}
```

`Signature::dtype` sets the default for all tensors. Explicit `Tensor` entries
override specific tensors by name. `GemmOp::acc_dtype` defaults to FP32
independently — every practical GEMM accumulates in fp32.

### GemmAlgorithm — the "HOW" of a GEMM

Describes tile geometry through three `Dim3{m, n, k}` fields:

```cpp
struct Dim3 { int m, n, k; };

struct GemmAlgorithm {
    Dim3 block_tile;   // Elements per thread block {M, N, K}
    Dim3 block_warps;  // Warp distribution {M, N, K}
    Dim3 warp_tile;    // Elements per warp per MFMA step {M, N, K}
};
```

Independent of data types — paired with `Signature` in the `make_kernel()` call:

```cpp
make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{.block_tile  = {128, 128, 32},
                  .block_warps = {2, 2, 1},
                  .warp_tile   = {32, 32, 16}})
```

### Consteval Validation

`make_kernel()` performs compile-time validation:

- **Warp tile validity**: Checks against CK Tile's `WarpGemmDispatcher` table
  via `is_valid_warp_gemm()`. For example, FP32 supports 32×32×{4,8} but not
  32×32×16, while FP16 supports 32×32×{8,16}.
- **Tile divisibility**: `block_tile.m` must be divisible by
  `block_warps.m × warp_tile.m` (and similarly for N and K).
- **CShuffleEpilogue constraint**: `block_warps.k` must be 1.
- **Thread block size**: Derived as `block_warps.m × block_warps.n × 64`.

Invalid configurations produce compile errors — no runtime surprises.

### GemmKernel — Structural NTTP

`make_kernel()` produces a `GemmKernel` struct with all types, layouts, and
tile geometry resolved. All members are structural types (enums, ints,
aggregates), so `GemmKernel` works as a C++20 non-type template parameter.

### CK Tile Type Wiring (gemm_dev.hpp)

`CkTypeMap` and `CkLayoutMap` map our schema enums to CK Tile's C++ types and
layout tags. `runGemm<K>` wires the 7-type CK Tile GEMM stack (shape, traits,
problem, pipeline, partitioner, epilogue, kernel) from a `GemmKernel` NTTP.
Tile geometry flows from `K.block_tile`, `K.block_warps`, and `K.warp_tile`.

Each `.hip` variant file is ~20 lines:

```cpp
static constexpr rocm_ck::GemmKernel kernel = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                            .block_warps = {2, 2, 1},
                            .warp_tile   = {16, 16, 16}});

extern "C" __global__ void gemm_fp16(rocm_ck::Args args) {
    rocm_ck::runGemm<kernel>(args);
}
```

### Mixed-Precision Epilogue

All variants accumulate in fp32. For fp16 and bf16, the CShuffleEpilogue
handles the fp32 → fp16/bf16 conversion when storing to global memory. This is
CK Tile's standard mixed-precision path.

### Fused Epilogue (Operator Composition)

Post-GEMM fusion is expressed as operators in the signature's compute graph.
Each epilogue step is a typed operator that references tensors by name:

```cpp
// GEMM + bias addition
make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{...})

// GEMM + bias + ReLU
make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{...})
```

`make_kernel()` pattern-matches the ops sequence to select the CK Tile epilogue:
- `[GemmOp]` — plain GEMM (tensors 0-2: A, B, C)
- `[GemmOp, AddOp]` — fused bias addition (tensors 0-3: A, B, E, D0)
- `[GemmOp, AddOp, ReluOp]` — fused bias + activation (tensors 0-3: A, B, E, D0)

All variants use the generic `Args` struct. Tensor slot count varies by
epilogue — `runGemm` branches on `EpilogueTypes<K>::NumDTensors` (derived
from `K.num_physical_tensors - 3`) at compile time on the device side.
On the host side, `k.num_physical_tensors > 3` determines whether D
tensor slots need populating.

The "bias" tensor's dtype cascades from `Signature::dtype`. Explicit `Tensor`
entries can override it for mixed-precision epilogues.

## Compiled Variants

| Variant | A | B | Out | Acc | Epilogue | BlockTile | WarpTile | Threads |
|---------|---|---|-----|-----|----------|-----------|----------|---------|
| `gemm_fp32` | FP32 | FP32 | FP32 | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16` | FP16 | FP16 | FP16 | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_bf16` | BF16 | BF16 | BF16 | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_w32` | FP16 | FP16 | FP16 | FP32 | — | 128×128×32 | 32×32×16 | 256 |
| `gemm_fp16_add` | FP16 | FP16 | FP16 | FP32 | `+bias` | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_add_relu` | FP16 | FP16 | FP16 | FP32 | `+bias+relu` | 128×128×32 | 16×16×16 | 256 |

All variants use 128×128×32 block tile with 2×2×1 warp layout (4 warps =
256 threads). `gemm_fp16_w32` demonstrates a wider 32×32 warp tile.
`gemm_fp16_add` demonstrates fused bias addition. `gemm_fp16_add_relu`
demonstrates composed epilogue (bias + activation).

## File Roles

Each example uses a three-file compilation boundary that separates
metaprogramming, host runtime, and device code:

| File | Compiled by | Purpose |
|------|-------------|---------|
| `gemm_kernel.hpp` | Both (g++ and hipcc) | **Metaprogramming** — structural types (`GemmKernel`, `GemmAlgorithm`, `Dim3`, `EpilogueOp`), `consteval` factories (`make_kernel`, `is_valid_warp_gemm`), named accessors (`tensor`, `slot`, `dtype`, `layout`), compile-time `static_assert` tests. No runtime code. |
| `gemm_api.hpp` | Host only (g++) | **Host runtime** — arg assembly, launch helpers, runtime validation. Guards with `#error` on device compilation. Includes `_kernel.hpp`. |
| `gemm_dev.hpp` | Device only (hipcc `--cuda-device-only`) | **Device code** — CK Tile bridge (`runGemm<K>`), `CkTypeMap`, `CkLayoutMap`, `EpilogueTypes`, `ComposedCDEOp`. Guards with `#error` on host compilation. Includes `_kernel.hpp`. |
| `gemm_*.hip` | Device only | Variant instantiations (~20 lines each) — include only `_dev.hpp` |
| `cpu_ref.{hpp,cpp}` | Host only | CPU reference GEMM for verification |
| `main.cpp` | Host only | Host loader — multi-variant loop with D tensor support, CPU reference verification |
| `pack.py` | — | Archive packer with per-variant dtype, epilogue, and tile metadata |
| `CMakeLists.txt` | — | Build system (variant × arch nested loop) |

### Compilation Boundary

```text
                    gemm_kernel.hpp (metaprogramming)
                   /                \
         gemm_api.hpp            gemm_dev.hpp
         (host only)             (device only)
             |                       |
         main.cpp                *.hip files
```

`gemm_kernel.hpp` is pure metaprogramming: `consteval` factories, `constexpr`
structural types, and `static_assert` tests. The compiler evaluates it all and
produces constants. No runtime code is generated on either side.

`.hip` files are compiled with `--cuda-device-only` and include only `gemm_dev.hpp`
(which transitively includes `gemm_kernel.hpp`). `main.cpp` is compiled as plain C++
and includes `gemm_api.hpp`. The `#error` guards enforce these boundaries —
including the wrong header produces a clear compile error.

## Build

```bash
cd experimental/rocm_ck/examples/04_gemm
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"
ninja -C build
```

Requires C++20 for struct NTTPs and `consteval` validation.

## Run

```bash
./build/kpack_gemm build/gemm.kpack
```

Expected output:
```
Opened build/gemm.kpack — architectures: gfx90a, gfx942
Detected GPU: gfx90a
gemm_fp32: M=512, N=512, K=256, grid=16, block=256
gemm_fp32: PASSED
gemm_fp16: M=512, N=512, K=256, grid=16, block=256
gemm_fp16: PASSED
gemm_bf16: M=512, N=512, K=256, grid=16, block=256
gemm_bf16: PASSED
gemm_fp16_w32: M=512, N=512, K=256, grid=16, block=256
gemm_fp16_w32: PASSED
gemm_fp16_add: M=512, N=512, K=256, grid=16, block=256
gemm_fp16_add: PASSED
gemm_fp16_add_relu: M=512, N=512, K=256, grid=16, block=256
gemm_fp16_add_relu: PASSED
```

## Design Notes

**Unified Signature with operator composition.** Both GEMM and elementwise
use the same `Signature` type. The operation is determined by the `ops`
array — `GemmOp` for GEMM, `AddOp` for elementwise. Epilogues are expressed
as additional operators in the graph rather than enum flags.

**Layouts are non-optional.** Unlike data types (where defaults reduce
boilerplate), layouts have universally sensible BLAS defaults: A=Row, B=Col,
C=Row. Making them optional would add complexity without reducing verbosity.

**Signature and Algorithm are independent.** The same tile geometry works
across fp32, fp16, and bf16 (with appropriate warp tiles). Separating "what
types" from "how to tile" lets each vary independently — different types can
share tile configs, and the same type can use different tile configs
(`gemm_fp16` vs `gemm_fp16_w32`).

**Consteval catches mistakes at compile time.** Invalid warp tile / dtype
combinations (e.g., fp32 with 32×32×16 warp tile) and bad tile divisibility
produce compile errors, not runtime crashes. The `is_valid_warp_gemm()`
lookup table mirrors CK Tile's WarpGemmDispatcher specializations.

**Test values kept small.** Input values are `i % 8` (max 7). Worst-case
accumulation: 256 × (7 × 7) = 12544, within fp16's exact integer range.
This ensures the fp32 CPU reference matches GPU output for all four variants.
