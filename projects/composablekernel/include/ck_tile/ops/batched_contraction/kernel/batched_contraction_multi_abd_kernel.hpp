// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_multi_abd_problem.hpp"
#include "ck_tile/ops/batched_contraction/utils/tensor_descriptor_utils.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {

/// @brief Host arguments for batched tensor contraction with multiple A, B, and D tensors.
///
/// @tparam NumATensor Number of A input tensors (fused via elementwise op)
/// @tparam NumBTensor Number of B input tensors (fused via elementwise op)
/// @tparam NumDTensor Number of D auxiliary tensors (fused in epilogue)
template <ck_tile::index_t NumATensor, ck_tile::index_t NumBTensor, ck_tile::index_t NumDTensor>
struct BatchedContractionMultiABDHostArgs
{
    std::array<const void*, NumATensor> as_ptr;
    std::array<const void*, NumBTensor> bs_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;

    // Per-tensor dims: A[i] = [G0,..,M0,..,K0,..], B[j] = [G0,..,N0,..,K0,..]
    std::array<std::vector<ck_tile::index_t>, NumATensor> As_dims;
    std::array<std::vector<ck_tile::index_t>, NumBTensor> Bs_dims;
    std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_dims;
    std::vector<ck_tile::index_t> E_dims;

    // Per-tensor strides
    std::array<std::vector<ck_tile::index_t>, NumATensor> As_strides;
    std::array<std::vector<ck_tile::index_t>, NumBTensor> Bs_strides;
    std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_strides;
    std::vector<ck_tile::index_t> E_strides;
};

/// @brief Kernel arguments for batched contraction with multiple A, B, and D tensors.
template <ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          ck_tile::index_t NumATensor,
          ck_tile::index_t NumBTensor,
          ck_tile::index_t NumDTensor,
          ck_tile::index_t VectorSizeA = 1,
          ck_tile::index_t VectorSizeB = 1,
          ck_tile::index_t VectorSizeE = 1>
struct BatchedContractionMultiABDKernelArgs
{
    std::array<const void*, NumATensor> as_ptr;
    std::array<const void*, NumBTensor> bs_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;

    ck_tile::index_t M_dims[NumDimM];
    ck_tile::index_t N_dims[NumDimN];
    ck_tile::index_t K_dims[NumDimK];
    ck_tile::index_t G_dims[NumDimG];

    std::array<ck_tile::index_t, NumATensor> batch_stride_As;
    std::array<ck_tile::index_t, NumBTensor> batch_stride_Bs;
    ck_tile::index_t batch_stride_E;
    std::array<ck_tile::index_t, NumDTensor> batch_stride_Ds;

    ck_tile::index_t G_total;
    ck_tile::index_t M_total;
    ck_tile::index_t N_total;
    ck_tile::index_t K_total;

    // Per-tensor descriptors (type depends on NumDimG/M/N/K + VectorSize, not runtime strides)
    using AGridDesc_M_K_ =
        decltype(TensorDescriptorUtils<NumDimG,
                                       NumDimM,
                                       NumDimN,
                                       NumDimK,
                                       VectorSizeA,
                                       VectorSizeB,
                                       VectorSizeE>::Make_A_GridDescriptor_M_K({}, {}));
    using BGridDesc_N_K_ =
        decltype(TensorDescriptorUtils<NumDimG,
                                       NumDimM,
                                       NumDimN,
                                       NumDimK,
                                       VectorSizeA,
                                       VectorSizeB,
                                       VectorSizeE>::Make_B_GridDescriptor_N_K({}, {}));
    using EGridDesc_M_N_ =
        decltype(TensorDescriptorUtils<NumDimG,
                                       NumDimM,
                                       NumDimN,
                                       NumDimK,
                                       VectorSizeA,
                                       VectorSizeB,
                                       VectorSizeE>::Make_E_GridDescriptor_M_N({}, {}));

    std::array<AGridDesc_M_K_, NumATensor> as_grid_desc_m_k;
    std::array<BGridDesc_N_K_, NumBTensor> bs_grid_desc_n_k;
    EGridDesc_M_N_ e_grid_desc_m_n;
    std::array<EGridDesc_M_N_, NumDTensor> ds_grid_desc_m_n;

    // Simple strides for IsSupportedArguments delegation
    std::array<ck_tile::index_t, NumATensor> stride_As;
    std::array<ck_tile::index_t, NumBTensor> stride_Bs;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;
};

/// @brief GPU kernel for batched tensor contraction with multiple A, B, and D tensors.
///
/// @par Overview
///     Combines BatchedContractionKernel's descriptor-based multi-dim contraction logic
///     with GemmKernelMultiABD's multi-tensor support. Builds per-tensor descriptor-based
///     tile windows as tuples for the pipeline's load_tile_with_elementwise path.
///
/// @par Mathematical Formulation
///     fused_A[G,M,K] = a_element_op(A0[G,M,K], A1[G,M,K], ...)
///     fused_B[G,N,K] = b_element_op(B0[G,N,K], B1[G,N,K], ...)
///     C[G,M,N] = sum_K fused_A[G,M,K] * fused_B[G,N,K]
///     E[G,M,N] = cde_element_op(C[G,M,N], D0[G,M,N], D1[G,M,N], ...)
template <typename Problem_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct BatchedContractionMultiABDKernel
{
    using Problem          = ck_tile::remove_cvref_t<Problem_>;
    using TilePartitioner  = ck_tile::remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = ck_tile::remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = ck_tile::remove_cvref_t<EpiloguePipeline_>;

    using AsDataType = ck_tile::remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType = ck_tile::remove_cvref_t<typename Problem::BsDataType>;
    using DsDataType = ck_tile::remove_cvref_t<typename Problem::DsDataType>;
    using EDataType  = ck_tile::remove_cvref_t<typename Problem::EDataType>;

    static constexpr ck_tile::index_t NumATensor = Problem::NumATensor;
    static constexpr ck_tile::index_t NumBTensor = Problem::NumBTensor;
    static constexpr ck_tile::index_t NumDTensor = Problem::NumDTensor;

    static constexpr ck_tile::index_t NumDimG = Problem::NumDimG;
    static constexpr ck_tile::index_t NumDimM = Problem::NumDimM;
    static constexpr ck_tile::index_t NumDimN = Problem::NumDimN;
    static constexpr ck_tile::index_t NumDimK = Problem::NumDimK;

    using ADataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using UniversalGemmKernel =
        ck_tile::UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr ck_tile::index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using DescriptorUtils = TensorDescriptorUtils<NumDimG,
                                                  NumDimM,
                                                  NumDimN,
                                                  NumDimK,
                                                  GemmPipeline::GetVectorSizeA(),
                                                  GemmPipeline::GetVectorSizeB(),
                                                  EpiloguePipeline::GetVectorSizeC()>;

    using KernelArgs = BatchedContractionMultiABDKernelArgs<NumDimG,
                                                            NumDimM,
                                                            NumDimN,
                                                            NumDimK,
                                                            NumATensor,
                                                            NumBTensor,
                                                            NumDTensor,
                                                            GemmPipeline::GetVectorSizeA(),
                                                            GemmPipeline::GetVectorSizeB(),
                                                            EpiloguePipeline::GetVectorSizeC()>;

    CK_TILE_HOST static constexpr auto GetKernelName()
    {
        return "batched_contraction_multi_abd_kernel";
    }

    CK_TILE_HOST static constexpr bool IsSupportedArguments(const KernelArgs& kargs)
    {
        if(kargs.k_batch > 1)
        {
            return false;
        }
        if(kargs.G_total <= 0)
        {
            return false;
        }

        typename UniversalGemmKernel::KernelArgs gemm_kargs{kargs.as_ptr,
                                                            kargs.bs_ptr,
                                                            kargs.ds_ptr,
                                                            kargs.e_ptr,
                                                            kargs.M_total,
                                                            kargs.N_total,
                                                            kargs.K_total,
                                                            kargs.stride_As,
                                                            kargs.stride_Bs,
                                                            kargs.stride_Ds,
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        return UniversalGemmKernel::IsSupportedArgument(gemm_kargs);
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return UniversalGemmKernel::GetSmemSize();
    }

    CK_TILE_HOST static auto GetBlockSize()
    {
        // Delegate to UniversalGemmKernel::BlockSize() so that the launch uses the
        // correct number of threads on both wave64 (CDNA) and wave32 (RDNA) targets.
        // kBlockSize is computed assuming wave64; on wave32 the actual thread count
        // must be halved to keep the same number of wavefronts (warps) per block.
        return UniversalGemmKernel::BlockSize();
    }

    CK_TILE_HOST static constexpr auto GridSize(const KernelArgs& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.M_total, kargs.N_total), kargs.G_total, kargs.k_batch);
    }

    /// @brief Core GEMM computation with tuple-based descriptor windows for multi-ABD.
    CK_TILE_DEVICE static void RunGemm(const std::array<const void*, NumATensor>& as_ptr,
                                       const std::array<const void*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const index_t k_size,
                                       const index_t i_m,
                                       const index_t i_n)
    {
        // Build tuple of A tensor tile windows (one per A tensor, using per-tensor descriptors)
        const auto as_block_windows = generate_tuple(
            [&](auto idx) {
                using AiDataType =
                    ck_tile::remove_cvref_t<std::tuple_element_t<idx.value, AsDataType>>;
                const AiDataType* ai_ptr = static_cast<const AiDataType*>(as_ptr[idx]);

                auto ai_tensor_view = make_tensor_view<address_space_enum::global>(
                    ai_ptr, kargs.as_grid_desc_m_k[idx]);

                auto ai_pad_view = pad_tensor_view(ai_tensor_view,
                                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                                              number<TilePartitioner::KPerBlock>{}),
                                                   sequence<false, GemmPipeline::kPadK>{});

                return make_tile_window(ai_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            },
            number<NumATensor>{});

        // Build tuple of B tensor tile windows
        const auto bs_block_windows = generate_tuple(
            [&](auto idx) {
                using BiDataType =
                    ck_tile::remove_cvref_t<std::tuple_element_t<idx.value, BsDataType>>;
                const BiDataType* bi_ptr = static_cast<const BiDataType*>(bs_ptr[idx]);

                auto bi_tensor_view = make_tensor_view<address_space_enum::global>(
                    bi_ptr, kargs.bs_grid_desc_n_k[idx]);

                auto bi_pad_view = pad_tensor_view(bi_tensor_view,
                                                   make_tuple(number<TilePartitioner::NPerBlock>{},
                                                              number<TilePartitioner::KPerBlock>{}),
                                                   sequence<false, GemmPipeline::kPadK>{});

                return make_tile_window(bi_pad_view,
                                        make_tuple(number<TilePartitioner::NPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_n, 0});
            },
            number<NumBTensor>{});

        const index_t num_loop =
            __builtin_amdgcn_readfirstlane(TilePartitioner::GetLoopNum(k_size));

        using AElementWise = ck_tile::remove_cvref_t<typename GemmPipeline::AElementWise>;
        using BElementWise = ck_tile::remove_cvref_t<typename GemmPipeline::BElementWise>;

        const auto& c_block_tile = GemmPipeline{}(
            as_block_windows, AElementWise{}, bs_block_windows, BElementWise{}, num_loop, smem_ptr);

        // Build E window
        auto e_tensor_view =
            make_tensor_view<address_space_enum::global>(e_ptr, kargs.e_grid_desc_m_n);

        auto e_pad_view = pad_tensor_view(
            e_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<false, GemmPipeline::kPadN>{});

        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        // Build D windows
        auto ds_block_windows = generate_tuple(
            [&](auto i) {
                using DDataType =
                    ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                const DDataType* d_ptr = static_cast<const DDataType*>(ds_ptr[i]);

                auto d_tensor_view =
                    make_tensor_view<address_space_enum::global>(d_ptr, kargs.ds_grid_desc_m_n[i]);

                return make_tile_window(d_tensor_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            },
            number<NumDTensor>{});

        EpiloguePipeline{}(e_block_window, c_block_tile, ds_block_windows, smem_ptr);
    }

    CK_TILE_HOST static constexpr KernelArgs MakeKernelArgs(
        const BatchedContractionMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor>& host_args)
    {
        const auto expected_A_dims = NumDimG + NumDimM + NumDimK;
        const auto expected_B_dims = NumDimG + NumDimN + NumDimK;
        const auto expected_E_dims = NumDimG + NumDimM + NumDimN;

        // Validate all A tensor dimensions
        for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        {
            if(host_args.As_dims[a].size() != expected_A_dims ||
               host_args.As_strides[a].size() != expected_A_dims)
            {
                throw std::invalid_argument("A[" + std::to_string(a) + "] dimension size mismatch");
            }
        }

        // Validate all B tensor dimensions
        for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        {
            if(host_args.Bs_dims[b].size() != expected_B_dims ||
               host_args.Bs_strides[b].size() != expected_B_dims)
            {
                throw std::invalid_argument("B[" + std::to_string(b) + "] dimension size mismatch");
            }
        }

        if(host_args.E_dims.size() != expected_E_dims ||
           host_args.E_strides.size() != expected_E_dims)
        {
            throw std::invalid_argument("E dimension size mismatch");
        }

        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            if(host_args.Ds_dims[d].size() != expected_E_dims ||
               host_args.Ds_strides[d].size() != expected_E_dims)
            {
                throw std::invalid_argument("D[" + std::to_string(d) + "] dimension size mismatch");
            }
        }

        KernelArgs kargs;
        kargs.as_ptr  = host_args.as_ptr;
        kargs.bs_ptr  = host_args.bs_ptr;
        kargs.ds_ptr  = host_args.ds_ptr;
        kargs.e_ptr   = host_args.e_ptr;
        kargs.k_batch = host_args.k_batch;

        // Validate G dimensions are identical across all tensors (use first A tensor as reference)
        for(ck_tile::index_t i = 0; i < NumDimG; ++i)
        {
            kargs.G_dims[i] = host_args.As_dims[0][i];

            for(ck_tile::index_t a = 1; a < NumATensor; ++a)
            {
                if(host_args.As_dims[a][i] != kargs.G_dims[i])
                    throw std::invalid_argument("G dimensions must match across all A tensors");
            }
            for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
            {
                if(host_args.Bs_dims[b][i] != kargs.G_dims[i])
                    throw std::invalid_argument("G dimensions must match across all B tensors");
            }
            if(host_args.E_dims[i] != kargs.G_dims[i])
                throw std::invalid_argument("G dimensions must match between A and E tensors");
        }

        // Validate and set M dimensions (from first A tensor, must match E)
        for(ck_tile::index_t i = 0; i < NumDimM; ++i)
        {
            kargs.M_dims[i] = host_args.As_dims[0][NumDimG + i];
            if(kargs.M_dims[i] != host_args.E_dims[NumDimG + i])
                throw std::invalid_argument("M dimension mismatch between A and E");

            for(ck_tile::index_t a = 1; a < NumATensor; ++a)
            {
                if(host_args.As_dims[a][NumDimG + i] != kargs.M_dims[i])
                    throw std::invalid_argument("M dimensions must match across all A tensors");
            }
        }

        // Validate and set N dimensions (from first B tensor, must match E)
        for(ck_tile::index_t i = 0; i < NumDimN; ++i)
        {
            kargs.N_dims[i] = host_args.Bs_dims[0][NumDimG + i];
            if(kargs.N_dims[i] != host_args.E_dims[NumDimG + NumDimM + i])
                throw std::invalid_argument("N dimension mismatch between B and E");

            for(ck_tile::index_t b = 1; b < NumBTensor; ++b)
            {
                if(host_args.Bs_dims[b][NumDimG + i] != kargs.N_dims[i])
                    throw std::invalid_argument("N dimensions must match across all B tensors");
            }
        }

        // Validate and set K dimensions (from first A, must match all A and all B tensors)
        for(ck_tile::index_t i = 0; i < NumDimK; ++i)
        {
            kargs.K_dims[i] = host_args.As_dims[0][NumDimG + NumDimM + i];

            for(ck_tile::index_t a = 1; a < NumATensor; ++a)
            {
                if(host_args.As_dims[a][NumDimG + NumDimM + i] != kargs.K_dims[i])
                    throw std::invalid_argument("K dimensions must match across all A tensors");
            }
            for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
            {
                if(host_args.Bs_dims[b][NumDimG + NumDimN + i] != kargs.K_dims[i])
                    throw std::invalid_argument("K dimensions must match between A and B tensors");
            }
        }

        kargs.G_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimG; ++i)
            kargs.G_total *= kargs.G_dims[i];

        kargs.M_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimM; ++i)
            kargs.M_total *= kargs.M_dims[i];

        kargs.N_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimN; ++i)
            kargs.N_total *= kargs.N_dims[i];

        kargs.K_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimK; ++i)
            kargs.K_total *= kargs.K_dims[i];

        // Build per-tensor descriptors
        for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        {
            kargs.as_grid_desc_m_k[a] = DescriptorUtils::Make_A_GridDescriptor_M_K(
                host_args.As_dims[a], host_args.As_strides[a]);
            kargs.batch_stride_As[a] = host_args.As_strides[a][NumDimG - 1];
            kargs.stride_As[a]       = kargs.K_total;
        }

        for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        {
            kargs.bs_grid_desc_n_k[b] = DescriptorUtils::Make_B_GridDescriptor_N_K(
                host_args.Bs_dims[b], host_args.Bs_strides[b]);
            kargs.batch_stride_Bs[b] = host_args.Bs_strides[b][NumDimG - 1];
            kargs.stride_Bs[b]       = kargs.K_total;
        }

        kargs.e_grid_desc_m_n =
            DescriptorUtils::Make_E_GridDescriptor_M_N(host_args.E_dims, host_args.E_strides);
        kargs.batch_stride_E = host_args.E_strides[NumDimG - 1];
        kargs.stride_E       = kargs.N_total;

        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            for(ck_tile::index_t i = 0; i < NumDimG; ++i)
            {
                if(host_args.Ds_dims[d][i] != kargs.G_dims[i])
                    throw std::invalid_argument("D tensor G dimensions must match");
            }
            kargs.ds_grid_desc_m_n[d] = DescriptorUtils::Make_E_GridDescriptor_M_N(
                host_args.Ds_dims[d], host_args.Ds_strides[d]);
            kargs.batch_stride_Ds[d] = host_args.Ds_strides[d][NumDimG - 1];
            kargs.stride_Ds[d]       = kargs.N_total;
        }

        return kargs;
    }

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {
        const auto [iM, iN] =
            TilePartitioner{kargs.M_total, kargs.N_total}.GetOutputTileIndex(blockIdx.x);
        const ck_tile::index_t i_m =
            __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const ck_tile::index_t i_n =
            __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto i_batch_flat = __builtin_amdgcn_readfirstlane(blockIdx.y);

        // Per-tensor batch offsets with correct per-tensor data types
        std::array<const void*, NumATensor> as_batch_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using AiDataType = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, AsDataType>>;
            const auto batch_offset = i_batch_flat * kargs.batch_stride_As[i];
            as_batch_ptr[i] = static_cast<const AiDataType*>(kargs.as_ptr[i]) + batch_offset;
        });

        std::array<const void*, NumBTensor> bs_batch_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BiDataType = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, BsDataType>>;
            const auto batch_offset = i_batch_flat * kargs.batch_stride_Bs[i];
            bs_batch_ptr[i] = static_cast<const BiDataType*>(kargs.bs_ptr[i]) + batch_offset;
        });

        EDataType* e_ptr =
            static_cast<EDataType*>(kargs.e_ptr) + i_batch_flat * kargs.batch_stride_E;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DiDataType = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
            const auto batch_offset = i_batch_flat * kargs.batch_stride_Ds[i];
            ds_batch_ptr[i] = static_cast<const DiDataType*>(kargs.ds_ptr[i]) + batch_offset;
        });

        __shared__ char smem_ptr[GetSmemSize()];

        RunGemm(as_batch_ptr,
                bs_batch_ptr,
                ds_batch_ptr,
                e_ptr,
                smem_ptr,
                kargs,
                kargs.K_total,
                i_m,
                i_n);
    }
};

} // namespace ck_tile
