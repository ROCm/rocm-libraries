// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"

namespace ck_tile {

template <typename ScaleM    = MXScalePointer<e8m0_t, -1>,
          typename ScaleN    = MXScalePointer<e8m0_t, -1>,
          index_t NumATensor = 1,
          index_t NumBTensor = 1,
          index_t NumDTensor = 0>
struct MXGemmKernelArgs : UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    using Base = UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST MXGemmKernelArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                  const std::array<const void*, NumBTensor>& bs_ptr_,
                                  const std::array<const void*, NumDTensor>& ds_ptr_,
                                  void* e_ptr_,
                                  index_t k_batch_,
                                  index_t M_,
                                  index_t N_,
                                  index_t K_,
                                  const std::array<index_t, NumATensor>& stride_As_,
                                  const std::array<index_t, NumBTensor>& stride_Bs_,
                                  const std::array<index_t, NumDTensor>& stride_Ds_,
                                  index_t stride_E_,
                                  ScaleM scale_m_ptr_,
                                  ScaleN scale_n_ptr_)
        : Base{as_ptr_,
               bs_ptr_,
               ds_ptr_,
               e_ptr_,
               M_,
               N_,
               K_,
               stride_As_,
               stride_Bs_,
               stride_Ds_,
               stride_E_,
               k_batch_},
          scale_m_ptr(scale_m_ptr_),
          scale_n_ptr(scale_n_ptr_)
    {
    }

    ScaleM scale_m_ptr;
    ScaleN scale_n_ptr;
};

template <typename TilePartitioner_, typename MXGemmPipeline_, typename EpiloguePipeline_>
struct MXGemmKernel : UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>
{
    using Underlying = UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXGemmPipeline   = remove_cvref_t<MXGemmPipeline_>;
    using BlockGemmShape   = remove_cvref_t<typename MXGemmPipeline::BlockGemmShape>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename MXGemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename MXGemmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename MXGemmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = MXGemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = MXGemmPipeline::UsePersistentKernel;

    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static constexpr index_t NumATensor = Underlying::AsDataType::size();
    static constexpr index_t NumBTensor = Underlying::BsDataType::size();
    static constexpr index_t NumDTensor = Underlying::DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::BsDataType>>;

    static constexpr auto MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr auto NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr auto KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr auto APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr auto BPackedSize = numeric_traits<BDataType>::PackedSize;

    // These are fixed by the MX preshuffle pipeline design. Keep them local to the kernel so
    // the baseline comp_async path does not need to surface preshuffle-only constants.
    static constexpr index_t MXdlPack = 2;
    static constexpr index_t NXdlPack = 2;
    static constexpr index_t KXdlPack = 2;

    static constexpr int kBlockPerCu = 1;

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mx_gemm", gemm_prec_str<ADataType, BDataType>, MXGemmPipeline::GetName());
        // clang-format on
    }

    template <typename ScaleM, typename ScaleN>
    using KernelArgs = MXGemmKernelArgs<ScaleM, ScaleN, NumATensor, NumBTensor, NumDTensor>;

    template <typename ScaleM, typename ScaleN>
    CK_TILE_HOST static auto MakeKernelArgs(const std::array<const void*, NumATensor>& as_ptr,
                                            const std::array<const void*, NumBTensor>& bs_ptr,
                                            const std::array<const void*, NumDTensor>& ds_ptr,
                                            void* e_ptr,
                                            index_t k_batch,
                                            index_t M,
                                            index_t N,
                                            index_t K,
                                            const std::array<index_t, NumATensor>& stride_As,
                                            const std::array<index_t, NumBTensor>& stride_Bs,
                                            const std::array<index_t, NumDTensor>& stride_Ds,
                                            index_t stride_E,
                                            ScaleM scale_m_ptr,
                                            ScaleN scale_n_ptr)
    {
        return KernelArgs<ScaleM, ScaleN>(as_ptr,
                                          bs_ptr,
                                          ds_ptr,
                                          e_ptr,
                                          k_batch,
                                          M,
                                          N,
                                          K,
                                          stride_As,
                                          stride_Bs,
                                          stride_Ds,
                                          stride_E,
                                          scale_m_ptr,
                                          scale_n_ptr);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto GridSize(const KernelArgs<ScaleM, ScaleN>& kargs)
    {
        const int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
                throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                         hipGetErrorName(hipGetLastError()));

            if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                   &maxActiveBlocksPerCU,
                   reinterpret_cast<void*>(
                       kentry<1, MXGemmKernel, remove_cvref_t<decltype(kargs)>>),
                   KernelBlockSize,
                   dync_smem_size) != hipSuccess)
                throw std::runtime_error(
                    std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                    hipGetErrorName(hipGetLastError()));

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int actual_grid_size      = min(persistent_block_size, total_work_tile_cnt);

            return dim3(actual_grid_size, 1, 1);
        }
        else
        {
            // Non-persistent: use full grid size based on number of tiles
            return dim3(total_work_tile_cnt, 1, 1);
        }
    }

    using SplitKBatchOffset = typename Underlying::SplitKBatchOffset;

    // Create C block window following UniversalGemmKernel pattern
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set,
              typename ScaleM,
              typename ScaleN>
    CK_TILE_DEVICE static auto MakeCBlockWindows(EDataType* e_ptr,
                                                 const KernelArgs<ScaleM, ScaleN>& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Create tensor view for E/C tensor
        constexpr index_t vector_size = EpiloguePipeline::GetVectorSizeC();
        const auto& e_tensor_view     = [&]() -> auto {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<vector_size>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_E),
                    number<1>{},
                    number<vector_size>{});
            }
        }();

        // Create padded view
        const auto& e_pad_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, false>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, false>{});
            }
        }();

        // Create block window
        auto c_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return c_block_window;
    }

    // Create scale A block windows following the pattern of MakeABlockWindows
    template <typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto MakeScaleABlockWindows(const KernelArgs<ScaleM, ScaleN>& kargs,
                                                      const index_t i_m)
    {
        auto scale_a                        = kargs.scale_m_ptr;
        static constexpr int BlockScaleSize = ScaleM::GranularityK;

        if constexpr(MXGemmPipeline::Preshuffle)
        {
            // Rebuild the logical A-scale view from the byte-shuffled preshuffle buffer using
            // the same lane grouping as the original mx_flatmm path.
            const auto scale_packs_m = integer_divide_ceil(kargs.M, (MXdlPack * MThreadPerXdl));
            const auto scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

            const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_m, scale_packs_k, KThreadPerXdl, MThreadPerXdl));
            const auto scale_a_desc = transform_tensor_descriptor(
                scale_a_naive_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const auto scale_a_tensor_view = make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_a.ptr), scale_a_desc);

            return make_tile_window(
                scale_a_tensor_view,
                make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                           number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
                {i_m / MXdlPack, 0});
        }
        else
        {
            const auto scale_k_size = kargs.K / BlockScaleSize;

            // Baseline MX GEMM consumes raw e8m0_t scales with shape [M, K/32].
            const auto scale_a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                reinterpret_cast<const e8m0_t*>(scale_a.ptr),
                make_tuple(kargs.M, scale_k_size),
                make_tuple(scale_k_size, 1));

            return make_tile_window(
                scale_a_tensor_view,
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number<TilePartitioner::KPerBlock / BlockScaleSize>{}),
                {i_m, 0});
        }
    }

    template <typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto
    MakeBFlatBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                          const KernelArgs<ScaleM, ScaleN>& kargs,
                          const index_t i_n)
    {
        static_assert(NumBTensor == 1, "MX GEMM preshuffle currently supports one B tensor");

        constexpr index_t kKPerBlock    = MXGemmPipeline::kKPerBlock;
        constexpr index_t kNWarpTile    = BlockGemmShape::WarpTile::at(I1);
        constexpr index_t flatKPerBlock = kKPerBlock * kNWarpTile;
        const index_t kFlatKBlocks      = kargs.K / kKPerBlock;
        const index_t kFlatN            = kargs.N / kNWarpTile;

        auto b_flat_tensor_view = [&]() {
            static_assert(flatKPerBlock % MXGemmPipeline::GetVectorSizeB() == 0,
                          "wrong! vector size for preshuffled B tensor");
            auto naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(kFlatN, kFlatKBlocks, number<flatKPerBlock>{}));
            auto desc = transform_tensor_descriptor(
                naive_desc,
                make_tuple(make_pass_through_transform(kFlatN),
                           make_merge_transform_v3_division_mod(
                               make_tuple(kFlatKBlocks, number<flatKPerBlock>{}))),
                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return make_tensor_view<address_space_enum::global>(bs_ptr[number<0>{}], desc);
        }();

        return generate_tuple(
            [&](auto) {
                return make_tile_window(
                    b_flat_tensor_view,
                    make_tuple(number<MXGemmPipeline::flatNPerWarp>{},
                               number<MXGemmPipeline::flatKPerWarp>{}),
                    {static_cast<int>(i_n / BlockGemmShape::WarpTile::at(I1)), 0});
            },
            number<NumBTensor>{});
    }

    // Create scale B block windows following the pattern of MakeBBlockWindows
    template <typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto MakeScaleBBlockWindows(const KernelArgs<ScaleM, ScaleN>& kargs,
                                                      const index_t i_n)
    {
        auto scale_b                        = kargs.scale_n_ptr;
        static constexpr int BlockScaleSize = ScaleN::GranularityK;

        if constexpr(MXGemmPipeline::Preshuffle)
        {
            const auto scale_packs_n = integer_divide_ceil(kargs.N, (NXdlPack * NThreadPerXdl));
            const auto scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

            const auto scale_b_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_n, scale_packs_k, KThreadPerXdl, NThreadPerXdl));
            const auto scale_b_desc = transform_tensor_descriptor(
                scale_b_naive_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const auto scale_b_tensor_view = make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_b.ptr), scale_b_desc);

            return make_tile_window(
                scale_b_tensor_view,
                make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                           number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
                {i_n / NXdlPack, 0});
        }
        else
        {
            const auto scale_k_size = kargs.K / BlockScaleSize;

            // Baseline MX GEMM consumes raw e8m0_t scales; B is accessed logically as [N, K/32].
            const auto scale_b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                reinterpret_cast<const e8m0_t*>(scale_b.ptr),
                make_tuple(kargs.N, scale_k_size),
                make_tuple(scale_k_size, 1));

            return make_tile_window(
                scale_b_tensor_view,
                make_tuple(number<TilePartitioner::NPerBlock>{},
                           number<TilePartitioner::KPerBlock / BlockScaleSize>{}),
                {i_n, 0});
        }
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE static void RunMxGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                         const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                         const std::array<const void*, NumDTensor>& ds_ptr,
                                         EDataType* e_ptr,
                                         void* smem_ptr_ping,
                                         void* smem_ptr_pong,
                                         const KernelArgs<ScaleM, ScaleN>& kargs,
                                         const SplitKBatchOffset& splitk_batch_offset,
                                         const index_t i_m,
                                         const index_t i_n)
    {
        // Create block windows directly, following the new pattern from UniversalGemmKernel
        // i_m and i_n are element offsets (iM * MPerBlock, iN * NPerBlock), not tile indices
        const auto& a_block_window =
            Underlying::MakeABlockWindows(as_ptr, kargs, splitk_batch_offset.splitted_k, i_m);
        const auto& b_block_window = [&]() {
            if constexpr(MXGemmPipeline::Preshuffle)
            {
                // Do not reuse UniversalGemmKernel::MakeBBlockWindows here. The preshuffled B
                // buffer is already flattened, so this path must build the flat view directly.
                return MakeBFlatBlockWindows(bs_ptr, kargs, i_n);
            }
            else
            {
                return Underlying::MakeBBlockWindows(
                    bs_ptr, kargs, splitk_batch_offset.splitted_k, i_n);
            }
        }();
        const auto& d_block_window = Underlying::MakeDBlockWindows(ds_ptr, kargs, i_m, i_n);

        // Create scale block windows using our new functions
        const auto& scale_a_block_window = MakeScaleABlockWindows(kargs, i_m);
        const auto& scale_b_block_window = MakeScaleBBlockWindows(kargs, i_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");

        const auto& c_block_tile = MXGemmPipeline{}(a_block_window[number<0>{}],
                                                    b_block_window[number<0>{}],
                                                    scale_a_block_window,
                                                    scale_b_block_window,
                                                    num_loop,
                                                    smem_ptr_ping,
                                                    smem_ptr_pong);

        // Run Epilogue Pipeline - create C block window directly
        auto c_block_window = MakeCBlockWindows(e_ptr, kargs, i_m, i_n);
        EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_ping);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(MXGemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return MXGemmPipeline::GetSmemSize();
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(KernelArgs<ScaleM, ScaleN> kargs,
                                   int partition_idx = get_block_id()) const
    {
        const int total_work_tile_cnt =
            amd_wave_read_first_lane(TilePartitioner::GridSize(kargs.M, kargs.N));

        // Allocate shared memory for ping pong buffers
        __shared__ char smem_ptr_ping[GetSmemPingSize()];
        __shared__ char smem_ptr_pong[GetSmemPongSize()];

        // Support both persistent and non-persistent modes
        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            // Cast to base class for SplitKBatchOffset construction
            const SplitKBatchOffset splitk_batch_offset(
                static_cast<const typename Underlying::KernelArgs&>(kargs));
            // options
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // options
            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i] / APackedSize;
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i] / BPackedSize;
            });

            RunMxGemm<ScaleM, ScaleN>(as_ptr,
                                      bs_ptr,
                                      kargs.ds_ptr,
                                      e_ptr,
                                      smem_ptr_ping,
                                      smem_ptr_pong,
                                      kargs,
                                      splitk_batch_offset,
                                      i_m,
                                      i_n);
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
