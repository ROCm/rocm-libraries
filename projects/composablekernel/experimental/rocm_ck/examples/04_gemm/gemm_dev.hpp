// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side CK Tile GEMM type wiring for kpack.
//
// DEVICE ONLY: compiled via --cuda-device-only in .hip files.
// This header must NOT be included from host-only .cpp files.
//
// Maps GemmKernel (our schema) -> CK Tile template stack (7 types).
// Each .hip variant file includes this header and instantiates runGemm<K>
// with a specific constexpr GemmKernel.
//
// CkLayoutMap: Layout enum   -> CK Tile layout tag (RowMajor, ColumnMajor)
// runGemm<K>:  wires the full CK Tile GEMM pipeline from K's types/layouts
//
// Tile geometry is parameterized through GemmKernel fields, validated by
// make_kernel() against CK Tile's WarpGemmDispatcher table.
//
// Compilation boundary:
//   _kernel.hpp — schema types + consteval factory (both passes)
//   _api.hpp    — host-only helpers (host pass only, #error on device)
//   _dev.hpp (this) — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "gemm_dev.hpp requires device compilation. Host code should include gemm_api.hpp."
#endif

#include "gemm_kernel.hpp"

#include <rocm_ck/args.hpp>

#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace rocm_ck {

// ============================================================================
// CkLayoutMap: Layout → CK Tile layout tag
// ============================================================================

template <Layout>
struct CkLayoutMap;

template <>
struct CkLayoutMap<Layout::Row>
{
    using type = ck_tile::tensor_layout::gemm::RowMajor;
};

template <>
struct CkLayoutMap<Layout::Col>
{
    using type = ck_tile::tensor_layout::gemm::ColumnMajor;
};

// ============================================================================
// ComposedCDEOp: epilogue op chain → single CK Tile elementwise functor
// ============================================================================

/// Composed epilogue functor: applies K's epilogue_ops chain in sequence.
///
/// Binary ops (Add, Mul) fold D tensors into the accumulator via parameter pack.
/// Unary ops (Relu, etc.) transform the accumulator in place, delegating to
/// CK Tile's optimized implementations.
///
/// All arithmetic is in float. The 4 if-constexpr lines are bounded by
/// kMaxEpilogueOps and fully resolved at compile time — zero runtime overhead.
template <GemmKernel K>
struct ComposedCDEOp
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const Ds&... ds) const
    {
        float result = ck_tile::type_convert<float>(c);
        if constexpr(K.num_epilogue_ops > 0)
            apply_op<K.epilogue_ops[0]>(result, ds...);
        if constexpr(K.num_epilogue_ops > 1)
            apply_op<K.epilogue_ops[1]>(result, ds...);
        if constexpr(K.num_epilogue_ops > 2)
            apply_op<K.epilogue_ops[2]>(result, ds...);
        if constexpr(K.num_epilogue_ops > 3)
            apply_op<K.epilogue_ops[3]>(result, ds...);
        e = ck_tile::type_convert<E>(result);
    }

    private:
    template <EpilogueOp Op, typename... Ds>
    CK_TILE_HOST_DEVICE static void apply_op(float& result, const Ds&... ds)
    {
        if constexpr(Op == EpilogueOp::Add)
            ((result += ck_tile::type_convert<float>(ds)), ...);
        else if constexpr(Op == EpilogueOp::Mul)
            ((result *= ck_tile::type_convert<float>(ds)), ...);
        else if constexpr(Op == EpilogueOp::Relu)
            ck_tile::element_wise::Relu{}(result, result);
        else if constexpr(Op == EpilogueOp::FastGelu)
            ck_tile::element_wise::FastGelu{}(result, result);
        else if constexpr(Op == EpilogueOp::Gelu)
            ck_tile::element_wise::Gelu{}(result, result);
        else if constexpr(Op == EpilogueOp::Silu)
            ck_tile::element_wise::Silu{}(result, result);
        else if constexpr(Op == EpilogueOp::Sigmoid)
            ck_tile::element_wise::Sigmoid{}(result, result);
    }
};

// ============================================================================
// EpilogueTypes<K>: derive DsDataType, DsLayout, and Op tuples from GemmKernel
// ============================================================================

template <GemmKernel K>
struct EpilogueTypes
{
    using Op = ComposedCDEOp<K>;

    // D tensor count: physical tensors beyond lhs(0), rhs(1), output(2)
    static constexpr int NumDTensors = K.num_physical_tensors - 3;

    // D0/D1 types from physical tensor table (indices 3 and 4).
    // Always resolved — std::conditional_t below selects which go into tuples.
    using D0Type   = typename CkTypeMap<K.physical_tensors[3].dtype>::type;
    using D1Type   = typename CkTypeMap<K.physical_tensors[4].dtype>::type;
    using D0Layout = typename CkLayoutMap<K.physical_tensors[3].layout>::type;
    using D1Layout = typename CkLayoutMap<K.physical_tensors[4].layout>::type;

    using DsDataType = std::conditional_t<NumDTensors == 0,
                                          ck_tile::tuple<>,
                                          std::conditional_t<NumDTensors == 1,
                                                             ck_tile::tuple<D0Type>,
                                                             ck_tile::tuple<D0Type, D1Type>>>;

    using DsLayout = std::conditional_t<NumDTensors == 0,
                                        ck_tile::tuple<>,
                                        std::conditional_t<NumDTensors == 1,
                                                           ck_tile::tuple<D0Layout>,
                                                           ck_tile::tuple<D0Layout, D1Layout>>>;
};

// ============================================================================
// runGemm<K>: assemble CK Tile GEMM pipeline from GemmKernel descriptor
// ============================================================================

/// Device-side GEMM bridge: Args → CK Tile template stack → ck_tile::GemmKernel.
///
/// Tensor slot mapping comes from K's role-based accessors:
///   lhs() = left operand,  rhs() = right operand,  output() = final output,  [3] = D0 (optional)
///   lengths[0] = first dim, lengths[1] = second dim
///   strides follow dimension order (RowMajor: strides[0]=ld, ColMajor: strides[1]=ld)
///
/// Wires the 7-type CK Tile GEMM stack (shape, traits, problem, pipeline,
/// partitioner, epilogue, kernel) with types and layouts from the GemmKernel
/// NTTP. Tile geometry comes from GemmKernel fields (validated at compile time).
template <GemmKernel K>
__device__ void runGemm(Args args)
{
    // Device-side validation — catches invalid manual construction.
    static_assert(K.num_physical_tensors >= 3,
                  "kernel must have at least lhs, rhs, and output tensors");
    static_assert(K.thread_block_size > 0, "thread_block_size must be positive");
    static_assert(EpilogueTypes<K>::NumDTensors <= 1,
                  "at most 1 D tensor supported in this example");

    // Physical tensor table: role-based access (compile-time constants)
    static constexpr auto PT_LHS = K.lhs();    // GEMM left operand
    static constexpr auto PT_RHS = K.rhs();    // GEMM right operand
    static constexpr auto PT_OUT = K.output(); // final output

    // --- Map schema types to CK Tile types ---
    using AType   = typename CkTypeMap<PT_LHS.dtype>::type;
    using BType   = typename CkTypeMap<PT_RHS.dtype>::type;
    using AccType = typename CkTypeMap<K.acc_dtype>::type;
    using OType   = typename CkTypeMap<PT_OUT.dtype>::type;

    using ALayout = typename CkLayoutMap<PT_LHS.layout>::type;
    using BLayout = typename CkLayoutMap<PT_RHS.layout>::type;
    using CLayout = typename CkLayoutMap<PT_OUT.layout>::type;

    // --- Unpack generic Args — compiler generates s_load at fixed offsets ---
    const TensorArg& t_a = args.tensors[PT_LHS.args_slot];
    const TensorArg& t_b = args.tensors[PT_RHS.args_slot];
    const TensorArg& t_c = args.tensors[PT_OUT.args_slot];

    index_t M     = t_a.lengths[0];
    index_t N     = t_b.lengths[1];
    index_t K_dim = t_a.lengths[1];

    // Leading dimension stride depends on layout:
    //   RowMajor → strides[0],  ColMajor → strides[1]
    index_t stride_A =
        static_cast<index_t>(PT_LHS.layout == Layout::Row ? t_a.strides[0] : t_a.strides[1]);
    index_t stride_B =
        static_cast<index_t>(PT_RHS.layout == Layout::Row ? t_b.strides[0] : t_b.strides[1]);
    index_t stride_C =
        static_cast<index_t>(PT_OUT.layout == Layout::Row ? t_c.strides[0] : t_c.strides[1]);

    // --- Step 1: Tile geometry (from GemmKernel, validated by make_kernel) ---
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<K.block_tile.m, K.block_tile.n, K.block_tile.k>,
                               ck_tile::sequence<K.block_warps.m, K.block_warps.n, K.block_warps.k>,
                               ck_tile::sequence<K.warp_tile.m, K.warp_tile.n, K.warp_tile.k>>;

    // --- Step 2: Traits (no padding, layouts from kernel descriptor) ---
    using GemmTraits = ck_tile::TileGemmTraits<false, false, false, ALayout, BLayout, CLayout>;

    // --- Step 3: Pipeline problem (types from kernel descriptor) ---
    using PipelineProblem =
        ck_tile::GemmPipelineProblem<AType, BType, AccType, GemmShape, GemmTraits>;

    // --- Step 4: Pipeline (simplest: A/B from global memory, C in registers) ---
    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;

    // --- Step 5: Partitioner (1D blockIdx → 2D tile coordinates) ---
    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    // --- Epilogue types from kernel descriptor ---
    using EpiOp      = typename EpilogueTypes<K>::Op;
    using DsDataType = typename EpilogueTypes<K>::DsDataType;
    using DsLayout   = typename EpilogueTypes<K>::DsLayout;

    // --- Step 6: Epilogue (shuffle accumulator through LDS, store to global) ---
    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AType,
                                                                   BType,
                                                                   DsDataType,
                                                                   AccType,
                                                                   OType,
                                                                   DsLayout,
                                                                   CLayout,
                                                                   EpiOp,
                                                                   TilePartitioner::MPerBlock,
                                                                   TilePartitioner::NPerBlock,
                                                                   K.block_warps.m,
                                                                   K.block_warps.n,
                                                                   K.warp_tile.m,
                                                                   K.warp_tile.n,
                                                                   K.warp_tile.k,
                                                                   PipelineProblem::TransposeC>>;

    // --- Step 7: Kernel (ties everything together) ---
    using CkKernel   = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    using KernelArgs = typename CkKernel::UniversalGemmKernel::KernelArgs;

    // Convert generic Args to CK Tile's internal args.
    // Branch on D tensor count to construct the right KernelArgs initializer.
    static constexpr int NumDTensors = EpilogueTypes<K>::NumDTensors;

    if constexpr(NumDTensors == 0)
    {
        const KernelArgs kargs{{t_a.ptr},                  // as_ptr
                               {t_b.ptr},                  // bs_ptr
                               {},                         // ds_ptr  (empty — no D tensors)
                               const_cast<void*>(t_c.ptr), // e_ptr
                               M,                          // M
                               N,                          // N
                               K_dim,                      // K
                               {stride_A},                 // stride_As
                               {stride_B},                 // stride_Bs
                               {},                         // stride_Ds (empty)
                               stride_C,                   // stride_E
                               1};                         // k_batch (no split-K)
        CkKernel{}(kargs);
    }
    else if constexpr(NumDTensors == 1)
    {
        static constexpr auto PT_D0 = K.physical_tensors[3];
        const TensorArg& t_d0       = args.tensors[PT_D0.args_slot];
        index_t stride_D0 =
            static_cast<index_t>(PT_D0.layout == Layout::Row ? t_d0.strides[0] : t_d0.strides[1]);

        const KernelArgs kargs{{t_a.ptr},                  // as_ptr
                               {t_b.ptr},                  // bs_ptr
                               {t_d0.ptr},                 // ds_ptr (1 D tensor)
                               const_cast<void*>(t_c.ptr), // e_ptr
                               M,                          // M
                               N,                          // N
                               K_dim,                      // K
                               {stride_A},                 // stride_As
                               {stride_B},                 // stride_Bs
                               {stride_D0},                // stride_Ds (1 stride)
                               stride_C,                   // stride_E
                               1};                         // k_batch (no split-K)
        CkKernel{}(kargs);
    }
}

} // namespace rocm_ck
