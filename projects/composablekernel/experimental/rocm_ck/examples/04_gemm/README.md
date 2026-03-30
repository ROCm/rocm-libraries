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
    Dim3 block_tile;   // Elements per workgroup {M, N, K}
    Dim3 block_waves;  // Wavefront layout within workgroup {M, N, K}
    Dim3 mfma_tile;    // MFMA instruction tile {M, N, K}
    int k_batch = 1;   // Split-K factor (1 = no split)
};
```

Independent of data types — paired with `Signature` in the `make_spec()` call:

```cpp
make_spec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{.block_tile  = {128, 128, 32},
                  .block_waves = {2, 2, 1},
                  .mfma_tile   = {32, 32, 16}})
```

### Consteval Validation

`make_spec()` performs compile-time validation:

- **Warp tile validity**: Checks against CK Tile's `WarpGemmDispatcher` table
  via `is_valid_mfma()`. For example, FP32 supports 32×32×{4,8} but not
  32×32×16, while FP16 supports 32×32×{8,16}.
- **Tile divisibility**: `block_tile.m` must be divisible by
  `block_waves.m × mfma_tile.m` (and similarly for N and K).
- **CShuffleEpilogue constraint**: `block_waves.k` must be 1.
- **Workgroup size**: Derived as `block_waves.m × block_waves.n × wavefront_size`.

Invalid configurations produce compile errors — no runtime surprises.

### GemmSpec — Structural NTTP

`make_spec()` produces a `GemmSpec` struct with all types, layouts, and
tile geometry resolved. All members are structural types (enums, ints,
aggregates), so `GemmSpec` works as a C++20 non-type template parameter.

### CK Tile Type Wiring (gemm_dev.hpp)

`CkTypeMap` and `CkLayoutMap` map our schema enums to CK Tile's C++ types and
layout tags. `run<S>` wires the 7-type CK Tile GEMM stack (shape, traits,
problem, pipeline, partitioner, epilogue, kernel) from a `GemmSpec` NTTP.
Tile geometry flows from `S.block_tile`, `S.block_waves`, and `S.mfma_tile`.

### Variant Table (gemm_variants.hpp)

All variant specs are defined once in `gemm_variants.hpp` — a single source of
truth shared by both device `.hip` files and the host `main.cpp`. Each `.hip`
file looks up its spec by name at compile time:

```cpp
#include "gemm_variants.hpp"
#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::gemm_variant_spec("gemm_fp16");

extern "C" __global__ void gemm_fp16(rocm_ck::Args args) {
    rocm_ck::run<spec>(args);
}
```

`gemm_variant_spec()` is `consteval` — a typo in the variant name produces a
compile error, not a runtime failure.

### Mixed-Precision Epilogue

All variants accumulate in fp32. For fp16 and bf16, the CShuffleEpilogue
handles the fp32 → fp16/bf16 conversion when storing to global memory. This is
CK Tile's standard mixed-precision path.

### Fused Epilogue (Operator Composition)

Post-GEMM fusion is expressed as operators in the signature's compute graph.
Each epilogue step is a typed operator that references tensors by name:

```cpp
// GEMM + bias addition
make_spec(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{...})

// GEMM + bias + ReLU
make_spec(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{...})
```

`make_spec()` pattern-matches the ops sequence to select the CK Tile epilogue:
- `[GemmOp]` — plain GEMM (tensors 0-2: A, B, C)
- `[GemmOp, AddOp]` — fused bias addition (tensors 0-3: A, B, E, D0)
- `[GemmOp, AddOp, ReluOp]` — fused bias + activation (tensors 0-3: A, B, E, D0)

All variants use the generic `Args` struct. Tensor slot count varies by
epilogue — `run` branches on `EpilogueTypes<S>::NumDTensors` (derived
from `S.num_physical_tensors - 3`) at compile time on the device side.
On the host side, `spec.num_physical_tensors > 3` determines whether D
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
| `gemm_fp16_rr` | FP16 (R) | FP16 (R) | FP16 (R) | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_cr` | FP16 (C) | FP16 (R) | FP16 (R) | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_cc` | FP16 (C) | FP16 (C) | FP16 (R) | FP32 | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_splitk` | FP16 | FP16 | FP16 | FP32 | — | 128×128×32 | 16×16×16 | 256 | k_batch=4 |

All variants use 128×128×32 block tile with 2×2×1 wavefront layout (4 waves =
256 work-items). `gemm_fp16_w32` demonstrates a wider 32×32 MFMA tile.
`gemm_fp16_add` demonstrates fused bias addition. `gemm_fp16_add_relu`
demonstrates composed epilogue (bias + activation). Layout variants
(`_rr`, `_cr`, `_cc`) override GemmOp's BLAS-convention defaults (A=Row,
B=Col) via explicit `Tensor` entries — R = RowMajor, C = ColumnMajor.
`gemm_fp16_splitk` demonstrates Split-K scheduling — the K dimension is
partitioned across `k_batch` workgroups (blockIdx.z), with partial
results accumulated via atomic addition.

## File Roles

| File | Compiled by | Purpose |
|------|-------------|---------|
| `gemm_spec.hpp` | Both (`include/rocm_ck/`) | **Structural types** — `GemmSpec`, `GemmAlgorithm`, `Dim3`, `EpilogueOp`, `consteval` factories (`make_spec`, `is_valid_mfma`). No runtime code. |
| `gemm_variants.hpp` | Both (g++ and hipcc) | **Variant registry** — constexpr table of all kernel configurations, `consteval gemm_variant_spec()` lookup. Single source of truth for device and host code. |
| `main.cpp` | Host only | Host loader — iterates `gemm_variants[]`, launches each, verifies against CPU reference |
| `gemm_*.hip` | Device only | Variant instantiations (~12 lines each) — include `gemm_variants.hpp` and `gemm_dev.hpp` |
| `gemm_dev.hpp` | Device only (`include/rocm_ck/`) | **CK Tile bridge** — `run<S>`, `CkTypeMap`, `CkLayoutMap`, `EpilogueTypes`, `ComposedCDEOp`. Guards with `#error` on host compilation. |
| `cpu_ref.{hpp,cpp}` | Host only | CPU reference GEMM for verification |
| `pack.py` | — | Archive packer with per-variant dtype, epilogue, and tile metadata |
| `CMakeLists.txt` | — | Build system (variant × arch nested loop) |

### Compilation Boundary

```text
        <rocm_ck/gemm_spec.hpp>      (structural types)
                    |
           gemm_variants.hpp         (example variant table)
                 /        \
          main.cpp      gemm_*.hip
           (host)           |
                    <rocm_ck/gemm_dev.hpp>  (device bridge: run<S>)
```

`gemm_spec.hpp` defines structural types (`GemmSpec`, `GemmAlgorithm`).
`gemm_variants.hpp` builds the variant table on top of those types — included
by both host and device code, ensuring specs are defined exactly once.

`.hip` files are compiled with `--cuda-device-only` and include both
`gemm_variants.hpp` (for the spec) and `gemm_dev.hpp` (for `run<S>`).
`main.cpp` includes only `gemm_variants.hpp` to iterate variant metadata.

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
gemm_fp32: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp32: PASSED
gemm_fp16: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16: PASSED
gemm_bf16: M=512, N=512, K=256, grid=16x1, block=256
gemm_bf16: PASSED
gemm_fp16_w32: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_w32: PASSED
gemm_fp16_add: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_add: PASSED
gemm_fp16_add_relu: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_add_relu: PASSED
gemm_fp16_rr: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_rr: PASSED
gemm_fp16_cr: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_cr: PASSED
gemm_fp16_cc: M=512, N=512, K=256, grid=16x1, block=256
gemm_fp16_cc: PASSED
gemm_fp16_splitk: M=512, N=512, K=256, grid=16x4, block=256
gemm_fp16_splitk: PASSED
```

## Design Notes

**Unified Signature with operator composition.** Both GEMM and elementwise
use the same `Signature` type. The operation is determined by the `ops`
array — `GemmOp` for GEMM, `AddOp` for elementwise. Epilogues are expressed
as additional operators in the graph rather than enum flags.

**Layouts default to BLAS convention, with per-tensor overrides.** `GemmOp`
implies A=Row, B=Col, C=Row (standard BLAS convention). Explicit `Tensor`
entries override individual tensors: `Tensor{.name = "B", .layout = Layout::Row}`
switches B to RowMajor. No schema changes needed — the same resolve() override
mechanism used for dtypes handles layouts.

**Signature and Algorithm are independent.** The same tile geometry works
across fp32, fp16, and bf16 (with appropriate MFMA tiles). Separating "what
types" from "how to tile" lets each vary independently — different types can
share tile configs, and the same type can use different tile configs
(`gemm_fp16` vs `gemm_fp16_w32`).

**Split-K via k_batch.** When `k_batch > 1`, the K dimension is partitioned
across `blockIdx.z`. Each Z-slice computes a partial sum over `K/k_batch`
elements, and results are accumulated via atomic addition. The output buffer
must be pre-zeroed. CK Tile's `UniversalGemmKernel` handles this
automatically — no code changes needed beyond setting the field and
launching with `grid_z = k_batch`.

**Consteval catches mistakes at compile time.** Invalid MFMA tile / dtype
combinations (e.g., fp32 with 32×32×16 MFMA tile) and bad tile divisibility
produce compile errors, not runtime crashes. The `is_valid_mfma()`
lookup table mirrors CK Tile's WarpGemmDispatcher specializations.

**Test values kept small.** Input values are `i % 8` (max 7). Worst-case
accumulation: 256 × (7 × 7) = 12544, within fp16's exact integer range.
This ensures the fp32 CPU reference matches GPU output for all four variants.
