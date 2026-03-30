// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// ElementwiseSpec — structural NTTP descriptor and consteval factory for
// elementwise template instantiation.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types and consteval make_spec() factory. No runtime code.
//
// Compilation boundary:
//   _spec.hpp (this) — schema types + consteval factory (both passes)
//   _api.hpp           — host-only helpers (host pass only, #error on device)
//   _dev.hpp           — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/physical_tensor.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>
#include <rocm_ck/types.hpp>

namespace rocm_ck {

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// Independent of data types — paired with Signature in make_spec().
struct ElementwiseAlgorithm
{
    int block_tile;  // Elements processed per workgroup
    int block_waves; // Wavefronts per workgroup (must be power of 2)
    int wave_tile;   // Elements per wavefront (vector width control)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Validated kernel descriptor used as NTTP and host launch info.
/// All members are structural types (no std::optional, no pointers, etc.).
///
/// Physical tensor table layout (ordered by args_slot):
///   [0] = lhs (left input operand — name is user-chosen, e.g., "A")
///   [1] = rhs (right input operand — name is user-chosen, e.g., "B")
///   [2] = output (result — name is user-chosen, e.g., "C")
struct ElementwiseSpec
{
    // Physical tensor table — the kernel's view of Args::tensors[]
    int num_physical_tensors;
    std::array<PhysicalTensor, kMaxPhysicalTensors> physical_tensors;

    // Tile geometry
    int block_tile;     // Elements per workgroup (for grid calculation)
    int workgroup_size; // Work-items per workgroup (= wavefront_size * block_waves)
    int block_waves;    // Wavefronts per workgroup
    int wave_tile;      // Elements per wavefront
    bool pad;           // Padding enabled

    /// Left-hand input operand (position 0 in the physical tensor table).
    constexpr PhysicalTensor lhs() const { return physical_tensors[0]; }

    /// Right-hand input operand (position 1 in the physical tensor table).
    constexpr PhysicalTensor rhs() const { return physical_tensors[1]; }

    /// Output tensor (position 2 in the physical tensor table).
    constexpr PhysicalTensor output() const { return physical_tensors[2]; }
};

/// Check if problem size N is aligned to a variant's block_tile (no padding needed).
constexpr bool isAligned(ElementwiseSpec k, int n) { return n > 0 && n % k.block_tile == 0; }

// ============================================================================
// make_spec: operator-centric Signature -> ElementwiseSpec
// ============================================================================

/// Resolve and validate a vector add using the operator-centric Signature.
///
/// Pattern-matches: first op must be AddOp. Tensors get rank=1,
/// Layout::Contiguous as elementwise defaults.
///
/// Validates:
///   - Input types match (a_dtype == b_dtype)
///   - block_tile, block_waves, wave_tile are positive
///   - block_waves is power of 2 (CK Tile reduce_on_sequence requirement)
///   - wave_tile >= wavefront_size (64)
///   - kVectorM >= 1 and block_tile divisibility
consteval ElementwiseSpec make_spec(Signature sig, ElementwiseAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be AddOp
    if(!std::holds_alternative<AddOp>(sig.ops[0]))
        throw "vector add make_spec requires AddOp as first operator";
    const AddOp& add = std::get<AddOp>(sig.ops[0]);

    // Remaining ops must be empty
    for(int i = 1; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "vector add make_spec only supports a single AddOp";

    TensorDesc a_td   = resolved.tensor(add.lhs);
    TensorDesc b_td   = resolved.tensor(add.rhs);
    TensorDesc out_td = resolved.tensor(add.out);

    if(a_td.dtype != b_td.dtype)
        throw "vector add requires matching input dtypes";

    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_waves <= 0)
        throw "block_waves must be positive";
    if((algo.block_waves & (algo.block_waves - 1)) != 0)
        throw "block_waves must be a power of 2 (CK Tile reduce_on_sequence requirement)";
    if(algo.wave_tile <= 0)
        throw "wave_tile must be positive";
    if(algo.wave_tile < wavefront_size)
        throw "wave_tile must be >= wavefront_size";

    int in_bits    = data_type_bits(a_td.dtype);
    int out_bits   = data_type_bits(out_td.dtype);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;
    int kVectorM_b = algo.wave_tile / wavefront_size;
    int kVectorM   = kVectorM_a < kVectorM_b ? kVectorM_a : kVectorM_b;

    if(kVectorM < 1)
        throw "computed kVectorM must be >= 1 (wave_tile too small for this dtype)";

    int elements_per_iter = algo.block_waves * kVectorM * wavefront_size;
    if(algo.block_tile % elements_per_iter != 0)
        throw "block_tile must be divisible by (block_waves * kVectorM * wavefront_size)";

    int kRepeatM = algo.block_tile / elements_per_iter;
    if(kRepeatM < 1)
        throw "computed kRepeatM must be >= 1 (block_tile too small for given wave count)";

    // Build physical tensor table
    std::array<PhysicalTensor, kMaxPhysicalTensors> phys{};
    phys[0] = {add.lhs, a_td.dtype, Layout::Contiguous, 0};
    phys[1] = {add.rhs, b_td.dtype, Layout::Contiguous, 1};
    phys[2] = {add.out, out_td.dtype, Layout::Contiguous, 2};

    return {3,
            phys,
            algo.block_tile,
            wavefront_size * algo.block_waves,
            algo.block_waves,
            algo.wave_tile,
            algo.pad};
}

} // namespace rocm_ck
