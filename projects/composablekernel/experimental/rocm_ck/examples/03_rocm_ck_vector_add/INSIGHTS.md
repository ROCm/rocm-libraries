# Insights: Prototyping the rocm_ck API

This example is the design prototype for the production rocm_ck API. It validates config validation, variant management, type dispatch, and host/device separation patterns that will scale to GEMM, convolution, and FMHA.

## 1. Config -> consteval -> Kernel: The Three-Layer Pattern

The user-facing `Signature` + `ElementwiseAlgorithm` (using `std::optional`, `std::variant`, not valid as NTTPs) are transformed by `consteval make_spec()` into `ElementwiseSpec` (a structural type, valid as an NTTP). This separation is fundamental:

- **Signature + Algorithm** — flexible, user-facing. Uses `std::optional` fields and operator composition so users specify only what differs from defaults.
- **consteval** — validates and transforms. Throws at compile time if the config is invalid. Invalid configs never produce binaries.
- **Kernel** — structural, concrete. Every field has a definite value. Used as a C++20 NTTP to instantiate device templates.

```cpp
// User writes:
constexpr auto kernel = make_spec(
    Signature{.dtype = DataType::FP32, .ops = {AddOp{}}},
    ElementwiseAlgorithm{.block_tile = 1024, .block_waves = 1, .wave_tile = 1024, .pad = true});

// make_spec returns a ElementwiseSpec with all fields resolved and validated.
// This value is used directly as a template parameter: run<kernel>(args)
```

> **Production implication**: Every rocm_ck operation will follow this pattern. The Signature defines WHAT; the Algorithm defines HOW. `consteval` is the bridge that enforces invariants.

## 2. Dtype Resolution via Operator-Centric Signature

The `Signature` uses a two-level dtype cascade resolved by `consteval resolve()`:

```
Signature::dtype         (kernel-level default for all tensors)
+-- Tensor::dtype        (per-tensor override by name)
```

`Signature::dtype` sets the default for every tensor discovered by walking the operator graph. Explicit `Tensor` entries in `sig.tensors[]` override specific tensors by name. This is the same model for all operations — elementwise, GEMM, FMHA.

```cpp
// Homogeneous: all FP32
Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// Widening: fp16 inputs, fp32 output
Signature{.dtype = DataType::FP16,
          .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
          .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}
```

`resolve()` flattens the cascade at compile time. `static_assert` tests verify all resolution paths — homogeneous, widening, narrowing, and per-tensor overrides.

> **Production implication**: The dtype cascade is operation-independent — one model for all operations. Per-tensor overrides via `Tensor::dtype` handle any mixed-precision pattern without operation-specific fields.

## 3. consteval for Immediate Feedback

Throwing in a `consteval` function produces a compile error at the call site with the throw message as the diagnostic. `make_spec` validates 7 constraints:

- Input types must match (`a_dtype == b_dtype` for vector add)
- Tile dimensions must be positive
- `block_waves` must be a power of 2
- `kVectorM` must be >= 1 (validates type/tile compatibility)
- `block_tile` must be divisible by `(block_waves * kVectorM * wavefront_size)`
- `kRepeatM` must be >= 1

Invalid configs fail at compile time with readable messages like `"block_tile must be divisible by (block_waves * kVectorM * wavefront_size)"`. This is dramatically better than runtime validation — broken kernel configs never produce `.hsaco` files.

## 4. Signature / Algorithm Separation

"What" (data types + operation) vs "How" (tile geometry) are orthogonal concerns:

- **Signature**: `{dtype, tensors[], scalars[], ops[]}` — specifies the compute graph and types
- **Algorithm**: `{block_tile, block_waves, wave_tile, pad}` — specifies the tile geometry

The same signature can be paired with different algorithms to explore the tuning surface. The same algorithm can run on FP32, FP16, or BF16 (subject to `kVectorM` constraints validated by `make_spec`).

> **Production implication**: Tuning tools generate Algorithm variants; the Signature comes from the user's problem specification. This composition is how rocm_ck will expose its tuning surface.

## 5. _spec / _variants / _dev Interface Model

Three layers, strict dependency boundary:

| Header | CK Tile dependency | Used by |
|--------|-------------------|---------|
| `elementwise_spec.hpp` (shared) | None | Host, device, tests |
| `vector_add_variants.hpp` (example-local) | None | Host, device |
| `elementwise_dev.hpp` (shared) | Yes (CK Tile) | Device only |

The spec header (in `include/rocm_ck/`) contains config types, `make_spec`, `ElementwiseSpec`, and all compile-time validation. It is testable with `constexpr`/`consteval` without any GPU. The variants header defines the variant table and consteval lookup, shared by both device `.hip` files and host `main.cpp`. The generic `Args` struct (from `args.hpp`) replaces per-operation args structs.

The device header contains the CK Tile kernel implementation, `CkTypeMap`, and `run<S>`. It is compiled only with `--cuda-device-only` and never included in host code.

> **Production implication**: This is the model for all rocm_ck operation headers. Shared `_spec` and `_dev` headers live in `include/rocm_ck/`. Example-local `_variants` headers define concrete configurations.

## 6. Generic Args ABI

All operations share a single `Args` struct (defined in `include/rocm_ck/args.hpp`):

```cpp
struct TensorArg {
    const void* ptr;              //  8 bytes
    index_t lengths[kMaxRank];    // 24 bytes (int32)
    int64_t strides[kMaxRank];    // 48 bytes (int64)
};  // 80 bytes total

union ScalarValue { float f32; int32_t i32; uint32_t u32; double f64; };  // 8 bytes

struct Args {
    std::array<TensorArg, kMaxTensors> tensors;   // 1280 bytes
    std::array<ScalarValue, kMaxScalars> scalars;  //  128 bytes
};  // 1408 bytes total (34% of 4KB kernarg budget)
```

Tensor slots carry their own shape — problem dimensions (N, M, K) are derivable from tensor lengths rather than redundant scalar fields. Pointers are `const void*` because the actual type depends on the variant — type dispatch happens inside the device kernel via `CkTypeMap`. Output tensors use `const_cast` on the device side.

ABI is locked with `static_assert` checks for size, alignment, standard layout, and trivial copyability. The GPU only loads fields it reads via `s_load` — unused slots cost nothing.

> **Production implication**: One Args struct for all operations eliminates per-operation ABI structs. Adding a new operation requires no new args type — just new slot conventions documented in the device header.

## 7. One .hip File Per Variant

Variant specs are defined once in `vector_add_variants.hpp`. Each `.hip` file
looks up its spec by name and wraps it in a thin kernel:

```cpp
#include "vector_add_variants.hpp"
#include <rocm_ck/elementwise_dev.hpp>

static constexpr rocm_ck::ElementwiseSpec spec =
    rocm_ck::vector_add_variant_spec("vector_add_fp32_b1024");

extern "C" __global__ void vector_add_fp32_b1024(rocm_ck::Args args) {
    rocm_ck::run<spec>(args);
}
```

Separate files enable parallel compilation and clear traceability — each `.hip` produces one `.hsaco` per architecture. CMake loops over `(variant x arch)` to produce `N x M` code objects.

> **Production implication**: The `.hip` file is pure boilerplate — include, lookup, wrapper. At ~20+ variants, a code generator producing these files from the variant table is a natural next step.

## 8. CkTypeMap for Enum -> Type Dispatch

Maps the runtime `DataType` enum to CK Tile types on the device side:

```cpp
template <DataType> struct CkTypeMap;  // primary template undefined

template <> struct CkTypeMap<DataType::FP32> { using type = float; };
template <> struct CkTypeMap<DataType::FP16> { using type = ck_tile::half_t; };
template <> struct CkTypeMap<DataType::BF16> { using type = ck_tile::bf16_t; };
```

The primary template is intentionally undefined — only valid specializations compile. An unsupported `DataType` enum value produces a compile error, not a runtime failure.

This is the device-side bridge between the enum-based API (shared with the host) and CK Tile's template system (device-only).

## 9. Custom CK Tile Kernel Using Primitives

`run` uses `load_tile`, `sweep_tile_span`, `store_tile` directly instead of the stock `ElementWiseKernel`:

```cpp
auto a_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(args.a)));
auto b_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(args.b)));

auto y_tile = ck_tile::make_static_distributed_tensor<Y>(a_tile.get_tile_distribution());

ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto idx) {
    const float a_val = ck_tile::type_convert<float>(a_tile(make_tuple(idx)));
    const float b_val = ck_tile::type_convert<float>(b_tile(make_tuple(idx)));
    y_tile(make_tuple(idx)) = ck_tile::type_convert<Y>(alpha * a_val + beta * b_val);
});

ck_tile::store_tile(y_window, y_tile);
```

The stock `ElementWiseKernel` default-constructs its functor — no path to pass runtime `alpha`/`beta`. Building from primitives enables scalar parameters and mixed-type compute (convert inputs to float, compute, convert output).

> **Production implication**: Production kernels will be built from CK Tile building blocks, not from stock kernels. The primitives (`load_tile`, `sweep_tile_span`, `store_tile`, tile distributions) are the actual API surface of CK Tile for rocm_ck purposes.

## 10. kVectorM: Wider Type Governs Mixed-Type Vector Loads

For mixed-type operations, the wider type constrains vector width:

```
kVectorM = min(128 / max_type_bits, wave_tile / wavefront_size)
```

FP16+FP32 mixed: `max_type_bits = 32`, so `kVectorM_a = 4` (vs 8 for homogeneous FP16). The constraint comes from the 128-bit vector register width — wider elements mean fewer elements per vector load/store.

This is validated at `consteval` time. A tile configuration that works for FP32 may not work for FP8 (which allows 16 elements per vector but requires larger minimum block tiles).

## 11. workgroup_size = wavefront_size * block_waves, NOT block_tile

Common confusion caught during development. Block tile determines elements per block; threads are determined by wavefront count:

```
workgroup_size = wavefront_size * block_waves    (threads launched per block)
block_tile        = workgroup_size * kRepeatM * kVectorM  (elements processed per block)
```

A `block_tile = 1024` with `block_waves = 1` still launches only 64 threads — each wavefront iterates `kRepeatM` times, processing `kVectorM` elements per iteration.

## 12. Variant Registry and Selection

`vector_add_variants[]` is a constexpr table of `{name, spec}` pairs defined in `vector_add_variants.hpp`. `findVariant()` selects at runtime:

1. Filter by matching `in_dtype` and `out_dtype`
2. Among matches, prefer the largest `block_tile` that divides the problem size evenly (aligned)
3. Fall back to the largest padded variant if no aligned variant exists

This is a simple but effective model. Production will need more dimensions (architecture-specific variants, memory layout preferences), but the basic pattern — constexpr table + runtime selection function — scales naturally.

## 13. Archive Metadata Mirrors Source

`pack.py` embeds tuning parameters in the kpack TOC's `variant_metadata` section: block tile, block waves, wave tile, pad flag, input/output dtypes. This enables tooling to inspect archives without source code — a profiler can read the metadata to understand what kernel configuration produced a particular performance result.

Basic readers ignore the metadata; advanced tools consume it. The kpack format supports this naturally because MessagePack maps are extensible.

## 14. Shared Utility Headers as the Beginning of rocm_ck

`include/rocm_ck/` contains `datatype_utils.hpp`, `gpu_arch.hpp`, `hip_check.hpp` — independent of CK Tile, reusable across operations. These follow CK's own convention (`include/ck/`, `include/ck_tile/`) and are the seed of the rocm_ck public include library.

These headers reduce boilerplate and establish common patterns (error checking, type conversion, architecture detection) that all rocm_ck operations will share.