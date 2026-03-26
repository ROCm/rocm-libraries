// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// VectorAddKernel — structural NTTP descriptor and consteval factory for
// elementwise template instantiation.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types and consteval make_kernel() factory. No runtime code.
//
// Compilation boundary:
//   _kernel.hpp (this) — schema types + consteval factory (both passes)
//   _api.hpp           — host-only helpers (host pass only, #error on device)
//   _dev.hpp           — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>
#include <rocm_ck/types.hpp>

namespace rocm_ck {

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// Independent of data types — paired with Signature in make_kernel().
struct ElementwiseAlgorithm
{
    int block_tile;  // Elements processed per thread block (BlockTile)
    int block_warps; // Number of warps per thread block (BlockWarps)
    int warp_tile;   // Warp tile size for vector width calculation (WarpTile)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Validated kernel descriptor used as NTTP and host launch info.
/// All members are structural types (no std::optional, no pointers, etc.).
struct VectorAddKernel
{
    int block_tile;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (= warp_size * block_warps)
    DataType in_dtype;     // Input storage type (a, b)
    DataType out_dtype;    // Output storage type (c)
    int block_warps;       // Warps per block
    int warp_tile;         // Warp tile size
    bool pad;              // Padding enabled
};

/// Check if problem size N is aligned to a variant's block_tile (no padding needed).
constexpr bool isAligned(VectorAddKernel k, int n) { return n > 0 && n % k.block_tile == 0; }

// ============================================================================
// make_kernel: operator-centric Signature -> VectorAddKernel
// ============================================================================

/// Resolve and validate a vector add using the operator-centric Signature.
///
/// Pattern-matches: first op must be AddOp. Tensors get rank=1,
/// Layout::Contiguous as elementwise defaults.
///
/// Validates:
///   - Input types match (a_dtype == b_dtype)
///   - block_tile, block_warps, warp_tile are positive
///   - block_warps is power of 2 (CK Tile reduce_on_sequence requirement)
///   - warp_tile >= warp_size (64)
///   - kVectorM >= 1 and block_tile divisibility
consteval VectorAddKernel make_kernel(Signature sig, ElementwiseAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be AddOp
    if(!std::holds_alternative<AddOp>(sig.ops[0]))
        throw "vector add make_kernel requires AddOp as first operator";
    const AddOp& add = std::get<AddOp>(sig.ops[0]);

    // Remaining ops must be empty
    for(int i = 1; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "vector add make_kernel only supports a single AddOp";

    TensorDesc a_td   = resolved.tensor(add.lhs);
    TensorDesc b_td   = resolved.tensor(add.rhs);
    TensorDesc out_td = resolved.tensor(add.out);

    if(a_td.dtype != b_td.dtype)
        throw "vector add requires matching input dtypes";

    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_warps <= 0)
        throw "block_warps must be positive";
    if((algo.block_warps & (algo.block_warps - 1)) != 0)
        throw "block_warps must be a power of 2 (required by CK Tile reduce_on_sequence)";
    if(algo.warp_tile <= 0)
        throw "warp_tile must be positive";
    if(algo.warp_tile < warp_size)
        throw "warp_tile must be >= warp_size (64)";

    int in_bits    = data_type_bits(a_td.dtype);
    int out_bits   = data_type_bits(out_td.dtype);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;
    int kVectorM_b = algo.warp_tile / warp_size;
    int kVectorM   = kVectorM_a < kVectorM_b ? kVectorM_a : kVectorM_b;

    if(kVectorM < 1)
        throw "computed kVectorM must be >= 1 (warp_tile too small for type)";

    int elements_per_iter = algo.block_warps * kVectorM * warp_size;
    if(algo.block_tile % elements_per_iter != 0)
        throw "block_tile must be divisible by (block_warps * kVectorM * warp_size)";

    int kRepeatM = algo.block_tile / elements_per_iter;
    if(kRepeatM < 1)
        throw "computed kRepeatM must be >= 1 (block_tile too small for given warps)";

    return {algo.block_tile,
            warp_size * algo.block_warps,
            a_td.dtype,
            out_td.dtype,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

// ============================================================================
// Compile-time tests
// ============================================================================

// --- Homogeneous: dtype sets all tensors ---
static_assert(make_kernel(Signature{.dtype = DataType::FP32,
                                    .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          ElementwiseAlgorithm{1024, 1, 1024, true})
                  .in_dtype == DataType::FP32);
static_assert(make_kernel(Signature{.dtype = DataType::FP32,
                                    .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          ElementwiseAlgorithm{1024, 1, 1024, true})
                  .out_dtype == DataType::FP32);

// --- Mixed types via Tensor override ---
static_assert(make_kernel(Signature{.dtype   = DataType::FP16,
                                    .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                                    .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          ElementwiseAlgorithm{1024, 1, 1024, true})
                  .in_dtype == DataType::FP16);
static_assert(make_kernel(Signature{.dtype   = DataType::FP16,
                                    .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                                    .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          ElementwiseAlgorithm{1024, 1, 1024, true})
                  .out_dtype == DataType::FP32);

} // namespace rocm_ck
