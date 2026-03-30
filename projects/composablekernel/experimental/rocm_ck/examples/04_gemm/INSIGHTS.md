# Insights: GEMM through kpack — Validating the Schema

This example stress-tests the rocm_ck schema against a real 2D tiled kernel. GEMM is where complexity arrives: 3D tile geometry, hardware-constrained MFMA configurations, accumulator types independent of storage types, and fused epilogues that change both the computation and the ABI. Every insight here was earned by a failure or a design decision during implementation.

## 1. Bridge Pattern Extends to GEMM

The same bridge pattern from example 03 works for GEMM. Variant specs are defined once in `gemm_variants.hpp`. Each `.hip` file looks up its spec by name and wraps it in a thin `extern "C"` kernel:

```cpp
#include "gemm_variants.hpp"
#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::gemm_variant_spec("gemm_fp16");

extern "C" __global__ void gemm_fp16(rocm_ck::Args args) {
    rocm_ck::run<spec>(args);
}
```

The 7 CK Tile types (TileGemmShape, TileGemmTraits, PipelineProblem, GemmPipeline, TilePartitioner, CShuffleEpilogue, GemmSpec) are assembled in `gemm_dev.hpp`'s `run<S>` from the `GemmSpec` NTTP fields. Each `.hip` file is ~12 lines of lookup + kernel wrapper. Host code never includes CK Tile headers.

> **Key difference from elementwise**: GEMM's bridge assembles 7 types (vs 4 for elementwise) and wires a 2D tile partitioner. But the pattern shape is identical — the bridge complexity scales with CK Tile's template depth, not with our schema.

## 2. Two-Axis Config: Signature × Algorithm

The `Signature` (what to compute) and `GemmAlgorithm` (how to compute it) are independent:

- **Signature**: data types, layouts, epilogue via operator composition
- **Algorithm**: `block_tile {M,N,K}`, `block_waves {M,N,K}`, `mfma_tile {M,N,K}`

The same algorithm works across fp32, fp16, and bf16. The same type can use different tile configs (gemm_fp16 vs gemm_fp16_w32). This was proven by having 6 variants that mix types and tile shapes — no coupling between the axes.

> **Implication**: Tuning is purely an algorithm-space search. Given a signature, sweep tile configs and let `make_spec` reject invalid combinations at compile time. The search space is defined by `is_valid_mfma` × divisibility constraints.

## 3. Dtype Cascade — Uniform Across Operations

All operations use the same two-level dtype cascade: `Signature::dtype` sets a kernel-level default for all tensors; explicit `Tensor` entries in `sig.tensors[]` override specific tensors by name. After `resolve()` flattens the cascade, `make_spec()` stores each physical tensor's dtype in the `physical_tensors[]` table. `GemmOp::acc_dtype` defaults to FP32 independently.

```cpp
// Mixed precision: fp16 compute, fp32 output
Signature{.dtype = DataType::FP16,
          .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}
```

> **Design principle**: One dtype cascade model for all operations. Per-tensor `Tensor::dtype` overrides handle any mixed-precision pattern — no operation-specific dtype fields needed.

## 4. Accumulator Defaults Independently

`acc_dtype` defaults to FP32 regardless of `dtype`. Every practical GEMM accumulates in fp32 — even fp16×fp16 GEMM uses fp32 accumulation to avoid catastrophic precision loss over K iterations. CK Tile's `CShuffleEpilogue` handles the acc→output type conversion automatically.

This was flagged as medium risk during planning (would CShuffleEpilogue handle fp16 output from fp32 acc?) but worked on the first attempt across all three architectures.

## 5. Dim3: Tile Geometry as Structural Aggregates

`Dim3{m, n, k}` groups the natural M/N/K triplet:

```cpp
struct Dim3 { int m, n, k; };

GemmAlgorithm{
    .block_tile  = {128, 128, 32},
    .block_waves = {2, 2, 1},
    .mfma_tile   = {16, 16, 16}
};
```

`Dim3` is a structural type — it works as an NTTP member. `S.block_tile.m` compiles cleanly in template arguments for `ck_tile::sequence<>`. Designated initializers make configs self-documenting.

> **Why not `ck_tile::sequence<M,N,K>`**: Our schema is CK-independent. The _dev header maps `Dim3` fields to `ck_tile::sequence<>` in one place. If CK Tile's API changes, only `gemm_dev.hpp` needs updating.

## 6. Hardware Constraints Flow Through consteval

The valid set of (dtype, mfma_tile) combinations is hardware-specific and non-obvious. `is_valid_mfma` is a consteval lookup table that mirrors CK Tile's `WarpGemmDispatcher` specializations for gfx9 (MFMA):

| dtype | 16×16 K values | 32×32 K values |
|-------|----------------|----------------|
| FP32  | 4, 8, 16       | 4, 8           |
| FP16  | 16, 32         | 8, 16          |
| BF16  | 16, 32         | 8, 16          |

`make_spec` checks five constraints in sequence: positive dimensions, `block_waves.k == 1`, MFMA tile validity, tile divisibility, and derives `workgroup_size`. An invalid config throws a readable message at compile time.

> **Hard-won lesson**: The initial attempt used 256×256×64 block tile with 32×32×16 warp tile (copied from CK's `gemm_basic_invoker.hpp`). Failed on ALL architectures. Root causes: (1) no WarpGemmDispatcher for fp32 32×32×16, (2) 256×256 block tile requires 128KB LDS, exceeding 64KB limit. The fix — 128×128×32 with 16×16×16 — came from cross-referencing the dispatcher table with CK's own validated configs.

## 7. Generic Args Replaces Per-Epilogue Structs

All variants now use the generic `Args` struct (1408 bytes) from `include/rocm_ck/args.hpp`. Tensor slots carry their own shape — M, N, K are derivable from tensor lengths rather than redundant scalar fields.

| Physical tensor | Args slot | Shape |
|-----------------|-----------|-------|
| `physical_tensors[0]` (A) | 0 | `[M, K]` with layout-dependent strides |
| `physical_tensors[1]` (B) | 1 | `[K, N]` with layout-dependent strides |
| `physical_tensors[2]` (output) | 2 | `[M, N]` with layout-dependent strides |
| `physical_tensors[3]` (D0, optional) | 3 | `[M, N]` — present for fused epilogue |

`run<S>` unpacks tensors and extracts leading dimension strides based on layout (RowMajor → `strides[0]`, ColMajor → `strides[1]`). The device code branches on `EpilogueTypes<S>::NumDTensors` (derived from `S.num_physical_tensors - 3`) with `if constexpr` — each branch is compiled away. On the host side, `spec.num_physical_tensors > 3` determines whether D tensor slots need populating.

> **Design evolution**: The original per-epilogue structs (`GemmArgs`, `GemmArgs1D`) kept the ABI minimal but required new structs for each D-tensor count. The generic Args eliminates this — adding D tensors is just populating more slots. The GPU only loads fields it reads, so the larger struct costs nothing at runtime.

## 8. Epilogue Is Signature, Not Algorithm

Epilogue changes WHAT is computed (C = A×B vs E = A×B + D0 vs E = ReLU(A×B + D0)). It does NOT change HOW the matmul executes — CShuffleEpilogue handles all epilogue variants with the same shuffle strategy.

The operator-centric model expresses this naturally: operators in the signature's ops array compose the compute graph:

```cpp
// Plain GEMM
Signature{.dtype = FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// GEMM + bias
Signature{.dtype = FP16,
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}}

// GEMM + bias + ReLU
Signature{.dtype = FP16,
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                  ReluOp{.in = "D", .out = "E"}}}
```

`make_spec` pattern-matches the ops array to build the `epilogue_ops[]` chain in GemmSpec — a composable sequence of `EpilogueOp` values (Add, Mul, Relu, etc.). `ComposedCDEOp<S>` applies the chain with `if constexpr` unrolling, delegating activation math to CK Tile's optimized implementations.

> **Why a composable chain**: A fixed `CombineOp × Activation` enum model requires O(N×M) enum combinations. The epilogue_ops array is O(N+M) — add one `EpilogueOp` value and one `apply_op` clause. The representation mirrors the Signature's operator chain, minus the string names (which aren't structural types for NTTP).

## 9. CShuffleEpilogue Handles Everything

CK Tile's `CShuffleEpilogue` is the universal epilogue for GEMM:
- Handles acc→output type conversion (fp32 acc → fp16 output)
- Handles D tensor fusion (bias addition, scaling)
- Handles composed operations via a single functor parameter

`ComposedCDEOp<Combine, Act>` is a thin functor that converts to float, applies combine (fold over D tensor pack), applies activation (delegates to CK Tile's optimized implementations), and casts to output type. All epilogue variation is captured in one template — no `if constexpr` branching in `run`.

## 10. What Generalized from Elementwise

| Pattern | Elementwise | GEMM | Verdict |
|---------|------------|------|---------|
| Config → consteval → Kernel | `Signature + ElementwiseAlgorithm` → `ElementwiseSpec` | `Signature + GemmAlgorithm` → `GemmSpec` | Generalizes |
| Dtype cascade | 2 levels (Signature::dtype→Tensor::dtype) | 2 levels (Signature::dtype→Tensor::dtype) | Identical model |
| `_api` / `_dev` header split | Yes | Yes | Generalizes |
| `CkTypeMap` enum→type dispatch | Yes | Yes + `CkLayoutMap` | Generalizes (adds layout) |
| Generic Args struct | 40 bytes (old) | 1408 bytes (shared) | Generalizes — one struct for all ops |
| consteval validation in `make_spec` | 6 checks | 5 checks | Generalizes (different checks) |
| One `.hip` file per variant | ~12 lines each | ~12 lines each | Generalizes |
| `Signature` with operator composition | `AddOp{}` only | `GemmOp` + epilogue ops | Generalizes |

## 11. What Didn't Generalize

- **Tile validation**: Elementwise validates 1D `kVectorM`; GEMM validates 3D tile divisibility + MFMA dispatch. The validation logic is operation-specific.
- **Scalar parameters**: Elementwise uses `scalars[0]`/`scalars[1]` for `alpha`/`beta`. GEMM's D-tensor fusion is data-driven (D tensor passed as a pointer in `tensors[3]`). Both fit in the generic Args — different slot types for different use cases.
- **Grid calculation**: Elementwise uses `ceil(N / block_tile)`; GEMM uses `ceil(M/M_tile) × ceil(N/N_tile)` flattened to 1D via `GemmTile1DPartitioner`. More complex but CK Tile handles it.

## 12. Per-Tensor Layouts Need No Schema Changes

The Signature already supports per-tensor layout overrides via explicit `Tensor` entries. `GemmOp` provides BLAS convention defaults (lhs→Row, rhs→Col, out→Row), but any tensor can be overridden:

```cpp
// A=Row (default), B=Row (override Col default) → R×R layout
Signature{.dtype = DataType::FP16,
          .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}
```

The mechanism is `resolve()` Phase 3: explicit `Tensor` entries override operator-implied defaults. `gemm_dev.hpp` already maps both `Layout::Row` and `Layout::Col` via `CkLayoutMap`, and stride extraction handles both with `PT.layout == Layout::Row ? strides[0] : strides[1]`.

This was validated with 3 layout variants (R×R, C×R, C×C) alongside the existing R×C default. All compile and pass correctness tests. **No schema changes were needed** — the existing override mechanism handles arbitrary layout combinations.

> **Design principle**: Operator-implied defaults + explicit overrides handle the common case (BLAS convention) and the general case (arbitrary layouts) with one mechanism. No layout enum in `GemmAlgorithm` needed.

## 13. Split-K Is a One-Field Extension

Adding Split-K scheduling required adding a single `int k_batch = 1` field to `GemmAlgorithm` and `GemmSpec`. CK Tile's `UniversalGemmKernel` already supports Split-K via `blockIdx.z` — when `k_batch > 1`, each Z-slice processes a K partition and partial results are accumulated via atomic addition.

The host side sets `grid_z = spec.k_batch` in the launch call. The device side passes `S.k_batch` to CK Tile's `KernelArgs`. The output buffer must be pre-zeroed (already done per-variant). No epilogue changes needed — the atomic accumulation is handled inside the CK Tile epilogue.

The default `k_batch = 1` preserves backward compatibility — all existing variants continue to work with no changes.

> **Design principle**: When the underlying library already supports a feature, exposing it through the schema should be a single field with a backward-compatible default. Split-K demonstrates this: one field in the schema, two lines in the device bridge, one line in the host launcher.

## 14. Open Design Questions

1. **Runtime epilogue parameters** — `ScaleOp` with a float scale can't be a consteval field. Needs a different args-passing mechanism (scalar in args struct? separate buffer?).
2. **Composed activations** — CK Tile doesn't have built-in Add+GELU composition. Current model requires custom functors for each combination.
3. **Broadcast D tensors** — 1×N bias via stride=0 works at the tensor descriptor level but is untested in CK Tile examples.
4. **Architecture-dependent tile configs** — Optimal configs differ across gfx90a/gfx942/gfx950. kpack archives can include multiple variants per arch, but the selection mechanism isn't defined yet.
5. **Pipeline selection** — All variants use `GemmPipelineAGmemBGmemCRegV1` (simplest). Production needs `GemmPipelineAGmemBGmemCRegV3` (prefetching) and possibly async pipelines. This is a future `GemmAlgorithm` field.
