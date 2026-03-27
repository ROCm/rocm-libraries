// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GemmSpec — structural NTTP descriptor and consteval factory for GEMM
// template instantiation.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval make_kernel() factory, named accessors,
// and warp tile validation. No runtime code.
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

/// M x N x K dimension triple for tile geometry specification.
struct Dim3
{
    int m, n, k;
};

/// Algorithm: describes HOW a GEMM executes (tile geometry).
/// Independent of data types — paired with Signature in make_kernel().
struct GemmAlgorithm
{
    Dim3 block_tile;  // Elements per thread block {M, N, K}
    Dim3 block_warps; // Warp distribution {M, N, K}
    Dim3 warp_tile;   // Elements per warp per MFMA step {M, N, K}
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
    Dim3 block_warps;
    Dim3 warp_tile;
    int thread_block_size;

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
};

// ============================================================================
// Named tensor accessors (consteval — compile-time only)
// ============================================================================

/// Lookup a physical tensor by name. consteval — compile-time only.
/// Used in static_asserts and consteval make_kernel() result inspection.
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
// Warp tile validation
// ============================================================================

/// Check if (a_dtype, warp_m, warp_n, warp_k) is a valid MFMA warp gemm
/// configuration. Based on CK Tile's WarpGemmDispatcher specializations
/// for gfx9 (MFMA). Only covers standard symmetric tile shapes.
consteval bool is_valid_warp_gemm(DataType a_dtype, int m, int n, int k)
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
// make_kernel: operator-centric Signature -> GemmSpec
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
///   - block_warps.k == 1 (CShuffleEpilogue requires warps_m x warps_n block size)
///   - Warp tile matches MFMA dispatcher table for the input dtype
///   - Block tile is divisible by (block_warps x warp_tile) in each dimension
///
/// Derives thread_block_size = block_warps.m x block_warps.n x block_warps.k x 64.
consteval GemmSpec make_kernel(Signature sig, GemmAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be GemmOp
    if(!std::holds_alternative<GemmOp>(sig.ops[0]))
        throw "GEMM make_kernel requires GemmOp as first operator";
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
    if(algo.block_warps.m <= 0 || algo.block_warps.n <= 0 || algo.block_warps.k <= 0)
        throw "block_warps dimensions must be positive";
    if(algo.warp_tile.m <= 0 || algo.warp_tile.n <= 0 || algo.warp_tile.k <= 0)
        throw "warp_tile dimensions must be positive";

    if(algo.block_warps.k != 1)
        throw "block_warps.k must be 1 (CShuffleEpilogue constraint)";

    if(!is_valid_warp_gemm(a_td.dtype, algo.warp_tile.m, algo.warp_tile.n, algo.warp_tile.k))
        throw "warp_tile is not a valid MFMA configuration for this dtype";

    if(algo.block_tile.m % (algo.block_warps.m * algo.warp_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_warps.m * warp_tile.m)";
    if(algo.block_tile.n % (algo.block_warps.n * algo.warp_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_warps.n * warp_tile.n)";
    if(algo.block_tile.k % (algo.block_warps.k * algo.warp_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_warps.k * warp_tile.k)";

    int thread_block_size =
        algo.block_warps.m * algo.block_warps.n * algo.block_warps.k * warp_size;

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

    return {num_phys,
            phys,
            acc,
            algo.block_tile,
            algo.block_warps,
            algo.warp_tile,
            thread_block_size,
            num_epi_ops,
            epi_ops};
}

// ============================================================================
// Compile-time tests
// ============================================================================

// --- is_valid_warp_gemm ---

// FP32: supports 16x16x{4,8,16} and 32x32x{4,8}
static_assert(is_valid_warp_gemm(DataType::FP32, 16, 16, 4));
static_assert(is_valid_warp_gemm(DataType::FP32, 16, 16, 16));
static_assert(is_valid_warp_gemm(DataType::FP32, 32, 32, 8));
static_assert(!is_valid_warp_gemm(DataType::FP32, 32, 32, 16)); // fp32 only k in {4,8} at 32x32

// FP16: supports 16x16x{16,32} and 32x32x{8,16}
static_assert(is_valid_warp_gemm(DataType::FP16, 16, 16, 16));
static_assert(is_valid_warp_gemm(DataType::FP16, 32, 32, 16));
static_assert(!is_valid_warp_gemm(DataType::FP16, 32, 32, 4)); // fp16 only k in {8,16} at 32x32

// BF16: same valid set as FP16
static_assert(is_valid_warp_gemm(DataType::BF16, 16, 16, 16));
static_assert(is_valid_warp_gemm(DataType::BF16, 32, 32, 16));

// --- make_kernel: plain GEMM — physical tensor table ---

static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .num_physical_tensors == 3);
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "A") == 0);
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "B") == 1);
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "C") == 2);
static_assert(dtype(make_kernel(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                    "A") == DataType::FP16);
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .thread_block_size == 256);
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .num_epilogue_ops == 0);

// --- make_kernel: GEMM + Add — output is "D", D0 is "bias" ---

static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .epilogue_ops[0] == EpilogueOp::Add);
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .num_physical_tensors == 4);
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "D") == 2);
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "bias") == 3);
static_assert(dtype(make_kernel(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                    "bias") == DataType::FP16);

// --- make_kernel: GEMM + Add + Relu — output is "E" ---

static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                              ReluOp{.in = "D", .out = "E"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .num_epilogue_ops == 2);
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                              ReluOp{.in = "D", .out = "E"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .has_epilogue_op(EpilogueOp::Add));
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                              ReluOp{.in = "D", .out = "E"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .has_epilogue_op(EpilogueOp::Relu));
static_assert(slot(make_kernel(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                   ReluOp{.in = "D", .out = "E"}}},
                               GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                   "E") == 2);
static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                              ReluOp{.in = "D", .out = "E"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}})
                  .num_physical_tensors == 4);

// --- make_kernel: fp16 32x32x16 warp tile ---

static_assert(make_kernel(Signature{.dtype = DataType::FP16,
                                    .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                          GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}})
                  .thread_block_size == 256);

// --- Layout defaults (A=Row, B=Col, output=Row) ---

static_assert(layout(make_kernel(Signature{.dtype = DataType::FP32,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                     "A") == Layout::Row);
static_assert(layout(make_kernel(Signature{.dtype = DataType::FP32,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}),
                     "B") == Layout::Col);

} // namespace rocm_ck
