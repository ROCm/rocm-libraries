// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GemmSpec — structural NTTP descriptor and consteval factory for GEMM
// template instantiation.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval make_spec() factory, named accessors,
// and MFMA tile validation. No runtime code.
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

#include <string_view>

namespace rocm_ck {

// ============================================================================
// Epilogue operations (composable chain)
// ============================================================================

/// Epilogue operations applied after the GEMM matmul result.
///
/// Binary ops (Add, Mul) fold over D tensors via parameter pack:
///   Add — result += D0 [+ D1]     (bias addition)
///   Mul — result *= D0 [* D1]     (scaling)
///
/// Unary ops transform the accumulator in place:
///   Relu     — max(0, x)
///   FastGelu — approximate GELU: x * sigmoid(1.702 * x)
///   Gelu     — exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
///   Silu     — x * sigmoid(x)  (aka Swish with beta=1)
///   Sigmoid  — 1 / (1 + exp(-x))
///
/// Operations compose as an ordered sequence in GemmSpec::epilogue_ops[].
/// The Signature's operator chain (AddOp -> ReluOp) maps directly to this
/// array, minus the string names (which aren't structural types for NTTP).
enum class EpilogueOp
{
    Add,
    Mul,
    Relu,
    FastGelu,
    Gelu,
    Silu,
    Sigmoid
};

/// Maximum epilogue ops in a chain. 4 covers combine + activation + future ops.
inline constexpr int kMaxEpilogueOps = 4;

// ============================================================================
// Tile geometry types
// ============================================================================

/// Pipeline implementation strategy for the GEMM kernel.
///
/// V1: Simple pipeline — A/B from global memory, C in registers.
///     Uses GemmPipelineProblem + GemmPipelineAGmemBGmemCRegV1.
///
/// V3: Compute-optimized pipeline — software-pipelined loads.
///     Uses UniversalGemmPipelineProblem + GemmPipelineAgBgCrCompV3.
///     Better compute utilization through overlapped memory/compute.
///
/// Preshuffle: Weight preshuffle pipeline — B matrix pre-rearranged for
///     optimal LDS loads. Uses WeightPreshufflePipelineAGmemBGmemCRegV2.
///     Requires A=RowMajor, B=ColumnMajor. Host must call preshuffle on B
///     before kernel launch.
enum class Pipeline
{
    V1,
    V3,
    Preshuffle
};

/// Tile-to-CU work distribution strategy.
///
/// TileBased: Standard 1D tile partitioner. Each tile mapped to one block.
///     Grid: ceil(M/TileM) * ceil(N/TileN) x 1 x k_batch.
///     Uses GemmKernel<GemmTile1DPartitioner, Pipeline, Epilogue>.
///
/// StreamK: Work-balanced scheduling across CUs. Tiles streamed across blocks.
///     Grid: num_CUs * occupancy (not tile-based).
///     Uses StreamKKernel<StreamKTilePartitioner, Pipeline, Epilogue>.
enum class Scheduling
{
    TileBased,
    StreamK
};

/// M x N x K dimension triple for tile geometry specification.
struct Dim3
{
    int m, n, k;
};

/// Algorithm: describes HOW a GEMM executes (tile geometry, scheduling).
/// Independent of data types — paired with Signature in make_spec().
struct GemmAlgorithm
{
    Dim3 block_tile;                               // Elements per workgroup {M, N, K}
    Dim3 block_waves;                              // Wavefront layout within workgroup {M, N, K}
    Dim3 mfma_tile;                                // MFMA instruction tile {M, N, K}
    int k_batch           = 1;                     // Split-K factor: partitions K across blockIdx.z
    Pipeline pipeline     = Pipeline::V1;          // Pipeline implementation strategy
    Scheduling scheduling = Scheduling::TileBased; // Work distribution strategy
};

// ============================================================================
// GemmSpec — structural NTTP for template instantiation
// ============================================================================

/// Validated kernel descriptor with all types, layouts, and tile geometry resolved.
/// All members are structural types (enums, ints, aggregates) so this works as NTTP.
///
/// Physical tensor table layout (ordered by args_slot):
///   [0] = lhs (GEMM left operand — name is user-chosen, e.g., "A", "Q")
///   [1] = rhs (GEMM right operand — name is user-chosen, e.g., "B", "K")
///   [2] = output (final output — name varies by epilogue chain)
///   [3] = D0 (optional — first auxiliary tensor, e.g., "bias")
///   [4] = D1 (optional — second auxiliary tensor)
struct GemmSpec
{
    // Physical tensor table — the kernel's view of Args::tensors[]
    int num_physical_tensors;
    std::array<PhysicalTensor, kMaxPhysicalTensors> physical_tensors;

    // Accumulator type (register-only, not a physical tensor)
    DataType acc_dtype;

    // Tile geometry
    Dim3 block_tile;
    Dim3 block_waves;
    Dim3 mfma_tile;
    int workgroup_size;
    int k_batch; // Split-K factor (1 = no split)

    // Pipeline implementation strategy
    Pipeline pipeline;

    // Work distribution strategy
    Scheduling scheduling;

    // Epilogue: composable op chain applied after matmul
    int num_epilogue_ops;
    std::array<EpilogueOp, kMaxEpilogueOps> epilogue_ops;

    /// Check if the epilogue chain contains a specific op.
    constexpr bool has_epilogue_op(EpilogueOp op) const
    {
        for(int i = 0; i < num_epilogue_ops; ++i)
            if(epilogue_ops[i] == op)
                return true;
        return false;
    }

    /// GEMM left operand (position 0 in the physical tensor table).
    constexpr PhysicalTensor lhs() const { return physical_tensors[0]; }

    /// GEMM right operand (position 1 in the physical tensor table).
    constexpr PhysicalTensor rhs() const { return physical_tensors[1]; }

    /// GEMM output tensor (position 2 in the physical tensor table).
    /// Name varies by epilogue chain: "C" (plain), "D" (with combine), "E" (with activation).
    constexpr PhysicalTensor output() const { return physical_tensors[2]; }

    /// First auxiliary tensor D0 (position 3 — e.g., bias for AddOp).
    /// Only valid when num_physical_tensors > 3.
    constexpr PhysicalTensor d0() const { return physical_tensors[3]; }

    /// Second auxiliary tensor D1 (position 4 — e.g., second bias/scale).
    /// Only valid when num_physical_tensors > 4.
    constexpr PhysicalTensor d1() const { return physical_tensors[4]; }
};

// ============================================================================
// Named tensor accessors (consteval — compile-time only)
// ============================================================================

/// Lookup a physical tensor by name. consteval — compile-time only.
/// Used in static_asserts and consteval make_spec() result inspection.
/// For runtime access, use GemmSpec::output() or physical_tensors[] directly.
consteval PhysicalTensor tensor(GemmSpec k, std::string_view name)
{
    for(int i = 0; i < k.num_physical_tensors; ++i)
        if(k.physical_tensors[i].name == name)
            return k.physical_tensors[i];
    throw "tensor is not a physical slot in this kernel";
}

/// Slot index lookup by name. consteval — compile-time only.
consteval int slot(GemmSpec k, std::string_view name) { return tensor(k, name).args_slot; }

/// Dtype lookup by name. consteval — compile-time only.
consteval DataType dtype(GemmSpec k, std::string_view name) { return tensor(k, name).dtype; }

/// Layout lookup by name. consteval — compile-time only.
consteval Layout layout(GemmSpec k, std::string_view name) { return tensor(k, name).layout; }

// ============================================================================
// MFMA tile validation
// ============================================================================

/// Check if (a_dtype, m, n, k) maps to a valid MFMA instruction shape.
/// Based on CK Tile's WarpGemmDispatcher specializations for gfx9.
/// Only covers standard symmetric tile shapes.
consteval bool is_valid_mfma(DataType a_dtype, int m, int n, int k)
{
    if(a_dtype == DataType::FP32)
    {
        if(m == 16 && n == 16 && (k == 4 || k == 8 || k == 16))
            return true;
        if(m == 32 && n == 32 && (k == 4 || k == 8))
            return true;
    }
    if(a_dtype == DataType::FP16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }
    if(a_dtype == DataType::BF16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }
    return false;
}

// ============================================================================
// make_spec: operator-centric Signature -> GemmSpec
// ============================================================================

/// Resolve and validate a GEMM using the operator-centric Signature.
///
/// Pattern-matches the ops array to build the epilogue_ops chain:
///   {GemmOp}                          -> epilogue_ops = {}
///   {GemmOp, AddOp}                   -> epilogue_ops = {Add}
///   {GemmOp, AddOp, ReluOp}           -> epilogue_ops = {Add, Relu}
///   {GemmOp, MulOp}                   -> epilogue_ops = {Mul}
///   {GemmOp, ReluOp}                  -> epilogue_ops = {Relu}
///
/// The D tensor is the rhs of the AddOp/MulOp (the "bias" or "scale" operand).
/// c_dtype comes from the GemmOp output tensor.
///
/// Validates:
///   - All tile dimensions are positive
///   - block_waves.k == 1 (CShuffleEpilogue requires waves_m x waves_n layout)
///   - MFMA tile matches instruction table for the input dtype
///   - Block tile is divisible by (block_waves x mfma_tile) in each dimension
///
/// Derives workgroup_size = block_waves.m x block_waves.n x block_waves.k x wavefront_size.
consteval GemmSpec make_spec(Signature sig, GemmAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be GemmOp
    if(!std::holds_alternative<GemmOp>(sig.ops[0]))
        throw "GEMM make_spec requires GemmOp as first operator";
    const GemmOp& gemm = std::get<GemmOp>(sig.ops[0]);

    TensorDesc a_td = resolved.tensor(gemm.lhs);
    TensorDesc b_td = resolved.tensor(gemm.rhs);
    TensorDesc c_td = resolved.tensor(gemm.out);
    DataType acc    = gemm.acc_dtype;

    // Build epilogue op chain from remaining ops after GemmOp.
    // Track the final output name (varies by epilogue chain) and D tensor names.
    int num_epi_ops = 0;
    std::array<EpilogueOp, kMaxEpilogueOps> epi_ops{};
    std::string_view final_output = gemm.out; // "C" by default
    int num_d_tensors             = 0;
    std::string_view d0_name;
    DataType d0_dtype = DataType::FP32;
    Layout d0_layout  = Layout::Row;
    std::string_view d1_name;
    DataType d1_dtype = DataType::FP32;
    Layout d1_layout  = Layout::Row;

    int next_op = 1;

    // Binary ops: AddOp or MulOp — consumes a D tensor
    if(next_op < kMaxOps && std::holds_alternative<AddOp>(sig.ops[next_op]))
    {
        const AddOp& add       = std::get<AddOp>(sig.ops[next_op]);
        epi_ops[num_epi_ops++] = EpilogueOp::Add;
        num_d_tensors          = 1;
        d0_name                = add.rhs;
        TensorDesc d0_td       = resolved.tensor(d0_name);
        d0_dtype               = d0_td.dtype;
        d0_layout              = d0_td.layout != Layout::Auto ? d0_td.layout : Layout::Row;
        final_output           = add.out;
        next_op++;

        // Second consecutive AddOp — second D tensor (Add+Add: result += D0 + D1).
        // CK Tile's ComposedCDEOp folds D tensors via parameter pack:
        //   ((result += convert(ds)), ...) applies one Add to ALL D tensors.
        // Two consecutive AddOps with one EpilogueOp::Add gives correct behavior.
        if(next_op < kMaxOps && std::holds_alternative<AddOp>(sig.ops[next_op]))
        {
            const AddOp& add2 = std::get<AddOp>(sig.ops[next_op]);
            num_d_tensors     = 2;
            d1_name           = add2.rhs;
            TensorDesc d1_td  = resolved.tensor(d1_name);
            d1_dtype          = d1_td.dtype;
            d1_layout         = d1_td.layout != Layout::Auto ? d1_td.layout : Layout::Row;
            final_output      = add2.out;
            next_op++;
        }
    }
    else if(next_op < kMaxOps && std::holds_alternative<MulOp>(sig.ops[next_op]))
    {
        const MulOp& mul       = std::get<MulOp>(sig.ops[next_op]);
        epi_ops[num_epi_ops++] = EpilogueOp::Mul;
        num_d_tensors          = 1;
        d0_name                = mul.rhs;
        TensorDesc d0_td       = resolved.tensor(d0_name);
        d0_dtype               = d0_td.dtype;
        d0_layout              = d0_td.layout != Layout::Auto ? d0_td.layout : Layout::Row;
        final_output           = mul.out;
        next_op++;
    }

    // Unary ops: activations applied after binary combine
    if(next_op < kMaxOps)
    {
        if(std::holds_alternative<ReluOp>(sig.ops[next_op]))
        {
            final_output           = std::get<ReluOp>(sig.ops[next_op]).out;
            epi_ops[num_epi_ops++] = EpilogueOp::Relu;
            next_op++;
        }
        else if(std::holds_alternative<FastGeluOp>(sig.ops[next_op]))
        {
            final_output           = std::get<FastGeluOp>(sig.ops[next_op]).out;
            epi_ops[num_epi_ops++] = EpilogueOp::FastGelu;
            next_op++;
        }
        else if(std::holds_alternative<GeluOp>(sig.ops[next_op]))
        {
            final_output           = std::get<GeluOp>(sig.ops[next_op]).out;
            epi_ops[num_epi_ops++] = EpilogueOp::Gelu;
            next_op++;
        }
        else if(std::holds_alternative<SiluOp>(sig.ops[next_op]))
        {
            final_output           = std::get<SiluOp>(sig.ops[next_op]).out;
            epi_ops[num_epi_ops++] = EpilogueOp::Silu;
            next_op++;
        }
        else if(std::holds_alternative<SigmoidOp>(sig.ops[next_op]))
        {
            final_output           = std::get<SigmoidOp>(sig.ops[next_op]).out;
            epi_ops[num_epi_ops++] = EpilogueOp::Sigmoid;
            next_op++;
        }
    }

    // Remaining ops must be empty
    for(int i = next_op; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "unexpected operator after GEMM epilogue chain";

    // Tile validation
    if(algo.block_tile.m <= 0 || algo.block_tile.n <= 0 || algo.block_tile.k <= 0)
        throw "block_tile dimensions must be positive";
    if(algo.block_waves.m <= 0 || algo.block_waves.n <= 0 || algo.block_waves.k <= 0)
        throw "block_waves dimensions must be positive";
    if(algo.mfma_tile.m <= 0 || algo.mfma_tile.n <= 0 || algo.mfma_tile.k <= 0)
        throw "mfma_tile dimensions must be positive";

    if(algo.k_batch <= 0)
        throw "k_batch must be positive";

    if(algo.block_waves.k != 1)
        throw "block_waves.k must be 1 (CShuffleEpilogue constraint)";

    // Pipeline-specific constraints
    if(algo.pipeline == Pipeline::Preshuffle)
    {
        if(a_td.layout != Layout::Row)
            throw "Preshuffle pipeline requires A layout = Row";
        if(b_td.layout != Layout::Col)
            throw "Preshuffle pipeline requires B layout = Col";
    }

    // Scheduling constraints
    if(algo.scheduling == Scheduling::StreamK && algo.k_batch > 1)
        throw "Stream-K scheduling is incompatible with split-K (k_batch > 1)";

    if(!is_valid_mfma(a_td.dtype, algo.mfma_tile.m, algo.mfma_tile.n, algo.mfma_tile.k))
        throw "mfma_tile is not a valid MFMA instruction shape for this dtype";

    if(algo.block_tile.m % (algo.block_waves.m * algo.mfma_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_waves.m * mfma_tile.m)";
    if(algo.block_tile.n % (algo.block_waves.n * algo.mfma_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_waves.n * mfma_tile.n)";
    if(algo.block_tile.k % (algo.block_waves.k * algo.mfma_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_waves.k * mfma_tile.k)";

    int workgroup_size =
        algo.block_waves.m * algo.block_waves.n * algo.block_waves.k * wavefront_size;

    // Build physical tensor table
    int num_phys = 3;
    std::array<PhysicalTensor, kMaxPhysicalTensors> phys{};
    phys[0] = {gemm.lhs, a_td.dtype, a_td.layout, 0};     // A
    phys[1] = {gemm.rhs, b_td.dtype, b_td.layout, 1};     // B
    phys[2] = {final_output, c_td.dtype, c_td.layout, 2}; // output (C, D, or E)

    if(num_d_tensors >= 1)
    {
        phys[num_phys] = {d0_name, d0_dtype, d0_layout, num_phys};
        num_phys++;
    }
    if(num_d_tensors >= 2)
    {
        phys[num_phys] = {d1_name, d1_dtype, d1_layout, num_phys};
        num_phys++;
    }

    return {num_phys,
            phys,
            acc,
            algo.block_tile,
            algo.block_waves,
            algo.mfma_tile,
            workgroup_size,
            algo.k_batch,
            algo.pipeline,
            algo.scheduling,
            num_epi_ops,
            epi_ops};
}

} // namespace rocm_ck
