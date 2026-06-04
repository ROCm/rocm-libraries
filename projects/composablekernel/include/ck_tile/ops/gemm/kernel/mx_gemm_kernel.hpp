// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct MxGemmHostArgs : public UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>
{
    using BaseHostArgs = UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST explicit MxGemmHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                         const std::array<const void*, NumATensor>& as_scale_ptr_,
                                         const std::array<const void*, NumBTensor>& bs_ptr_,
                                         const std::array<const void*, NumBTensor>& bs_scale_ptr_,
                                         const std::array<const void*, NumDTensor>& ds_ptr_,
                                         void* e_ptr_,
                                         index_t k_batch_,
                                         index_t M_,
                                         index_t N_,
                                         index_t K_,
                                         const std::array<index_t, NumATensor>& stride_As_,
                                         const std::array<index_t, NumBTensor>& stride_Bs_,
                                         const std::array<index_t, NumDTensor>& stride_Ds_,
                                         index_t stride_E_)
        : BaseHostArgs(as_ptr_,
                       bs_ptr_,
                       ds_ptr_,
                       e_ptr_,
                       k_batch_,
                       M_,
                       N_,
                       K_,
                       stride_As_,
                       stride_Bs_,
                       stride_Ds_,
                       stride_E_),
          as_scale_ptr(as_scale_ptr_),
          bs_scale_ptr(bs_scale_ptr_)
    {
    }

    const std::array<const void*, NumATensor> as_scale_ptr;
    const std::array<const void*, NumBTensor> bs_scale_ptr;
};

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct MxGemmKernelArgs : public UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    const std::array<const void*, NumATensor> as_scale_ptr;
    const std::array<const void*, NumBTensor> bs_scale_ptr;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct MxGemmKernel
    : public UniversalGemmKernel<TilePartitioner_,
                                 GemmPipeline_,
                                 EpiloguePipeline_,
                                 MxGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>
{
    using BaseKernel =
        UniversalGemmKernel<TilePartitioner_,
                            GemmPipeline_,
                            EpiloguePipeline_,
                            MxGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MxGemmPipeline   = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using BaseKernel::PersistentKernel;
    using typename BaseKernel::AsLayout;
    using typename BaseKernel::BsLayout;
    using typename BaseKernel::DsLayout;

    using typename BaseKernel::ADataType;
    using typename BaseKernel::BDataType;
    using typename BaseKernel::EDataType;

    using BaseKernel::NumATensor;
    using BaseKernel::NumBTensor;
    using BaseKernel::NumDTensor;

    using BaseKernel::GetBlockId;
    using BaseKernel::GetGridSize;
    using BaseKernel::GetNumTiles;
    using BaseKernel::GetSmemSize;
    using typename BaseKernel::SplitKBatchOffset;

    using BaseKernel::APackedSize;
    using BaseKernel::BPackedSize;

    using BaseKernel::I1;

    using AElementWise = remove_cvref_t<typename MxGemmPipeline::AElementWise>;
    using BElementWise = remove_cvref_t<typename MxGemmPipeline::BElementWise>;

    using BlockGemmShape = remove_cvref_t<typename MxGemmPipeline::BlockGemmShape>;

    static constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});

    static constexpr int BlockScaleSize = MxGemmPipeline::ScaleBlockSize;
    using ScalePtrType                 = const int32_t*;
#ifdef __gfx950__
    static_assert(BlockScaleSize == 32, "unsupported BlockScaleSize");
    // Effective pack sizes: fall back to 1 when dimension is too small
    using BlockWarps_                      = typename BlockGemmShape::BlockWarps;
    static constexpr index_t MPerBlock_    = BlockGemmShape::kM;
    static constexpr index_t NPerBlock_    = BlockGemmShape::kN;
    static constexpr index_t KPerBlock_    = BlockGemmShape::kK;
    static constexpr index_t MWarp_        = BlockWarps_::at(number<0>{});
    static constexpr index_t NWarp_        = BlockWarps_::at(number<1>{});
    static constexpr index_t KPerXdl_      = BlockGemmShape::WarpTile::at(number<2>{});
    static constexpr index_t MIterPerWarp_ = MPerBlock_ / (MWarp_ * MThreadPerXdl);
    static constexpr index_t NIterPerWarp_ = NPerBlock_ / (NWarp_ * NThreadPerXdl);
    static constexpr index_t KIterPerWarp_ = KPerBlock_ / KPerXdl_;

    static constexpr index_t MXdlPack = 2;
    static constexpr index_t NXdlPack = 2;
    static constexpr index_t KXdlPack = 2;

    static constexpr index_t MXdlPackEff =
        (MIterPerWarp_ >= MXdlPack && MIterPerWarp_ % MXdlPack == 0) ? MXdlPack : 1;
    static constexpr index_t NXdlPackEff =
        (NIterPerWarp_ >= NXdlPack && NIterPerWarp_ % NXdlPack == 0) ? NXdlPack : 1;
    static constexpr index_t KXdlPackEff =
        (KIterPerWarp_ >= KXdlPack && KIterPerWarp_ % KXdlPack == 0) ? KXdlPack : 1;
#else
    static_assert(BlockScaleSize == 16 || BlockScaleSize == 32, "unsupported BlockScaleSize");
    // Scale tensor element type is always int32_t (4 packed e8m0 bytes).
    // For scale16, each thread needs 8 bytes = 2 int32_t elements.
    // For scale32, each thread needs 4 bytes = 1 int32_t element.
    static constexpr index_t MXdlPackEff = 1;
    static constexpr index_t NXdlPackEff = 1;
    static constexpr index_t KXdlPackEff = 4; // 4 is because scale tensor is
                                              // int32_t data type, each int32_t
                                              // exists 4 fp8 scale values
#endif

    using KernelArgs = MxGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const MxGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs)
    {
        return KernelArgs{{hostArgs.as_ptr,
                           hostArgs.bs_ptr,
                           hostArgs.ds_ptr,
                           hostArgs.e_ptr,
                           hostArgs.M,
                           hostArgs.N,
                           hostArgs.K,
                           hostArgs.stride_As,
                           hostArgs.stride_Bs,
                           hostArgs.stride_Ds,
                           hostArgs.stride_E,
                           hostArgs.k_batch,
                           hostArgs.async_input_scheduler},
                          hostArgs.as_scale_ptr,
                          hostArgs.bs_scale_ptr};
    }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if(kargs.k_batch != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("SplitK (k_batch > 1) is not supported for MX GEMM!");
            }
            return false;
        }
        return BaseKernel::IsSupportedArgument(kargs);
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeScaleABlockWindow(const std::array<ScalePtrType, NumATensor>& as_scale_ptr,
                          const KernelArgs& kargs,
                          index_t block_idx_m)
    {
        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, MThreadPerXdl * MXdlPackEff);
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / KXdlPackEff;

        // Scale16: descriptor order [packs_m, MThreadPerXdl, packs_k] -- K contiguous per M-row,
        //          no pre-shuffle needed (natural row-major layout matches).
        // Scale32: descriptor order [packs_m, packs_k, MThreadPerXdl] -- original layout,
        //          requires pre-shuffle to match.
        const auto scale_a_naive_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_m, MThreadPerXdl, scale_packs_k));
            else
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_m, scale_packs_k, MThreadPerXdl));
        }();
        const auto scale_a_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return transform_tensor_descriptor(
                    scale_a_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            else
                return transform_tensor_descriptor(
                    scale_a_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
        }();
        const auto& scale_a_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(as_scale_ptr[i], scale_a_desc);
            },
            number<NumATensor>{});
        const auto& scale_a_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(
                    scale_a_tensor_view[i],
                    make_tuple(
                        number<TilePartitioner::MPerBlock / MXdlPackEff>{},
                        number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPackEff)>{}),
                    {block_idx_m / MXdlPackEff, 0});
            },
            number<NumATensor>{});

        return scale_a_block_window;
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeScaleBBlockWindow(const std::array<ScalePtrType, NumBTensor>& bs_scale_ptr,
                          const KernelArgs& kargs,
                          index_t block_idx_n)
    {
        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, NThreadPerXdl * NXdlPackEff);
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / KXdlPackEff;

        const auto scale_b_naive_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_n, NThreadPerXdl, scale_packs_k));
            else
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_n, scale_packs_k, NThreadPerXdl));
        }();
        const auto scale_b_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return transform_tensor_descriptor(
                    scale_b_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            else
                return transform_tensor_descriptor(
                    scale_b_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
        }();
        const auto& scale_b_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(bs_scale_ptr[i], scale_b_desc);
            },
            number<NumBTensor>{});
        const auto& scale_b_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(
                    scale_b_tensor_view[i],
                    make_tuple(
                        number<TilePartitioner::NPerBlock / NXdlPackEff>{},
                        number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPackEff)>{}),
                    {block_idx_n / NXdlPackEff, 0});
            },
            number<NumBTensor>{});
        return scale_b_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeBFlatBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                          const KernelArgs& kargs,
                          const index_t i_n)
    {
        static_assert(NumBTensor == 1, "MX GEMM preshuffle currently supports one B tensor");

        constexpr index_t kKPerBlock    = MxGemmPipeline::kKPerBlock;
        constexpr index_t kNWarpTile    = BlockGemmShape::WarpTile::at(I1);
        constexpr index_t flatKPerBlock = kKPerBlock * kNWarpTile;
        const index_t kFlatKBlocks      = kargs.K / kKPerBlock;
        const index_t kFlatN            = kargs.N / kNWarpTile;

        auto b_flat_tensor_view = [&]() {
            static_assert(flatKPerBlock % MxGemmPipeline::GetVectorSizeB() == 0,
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
                    make_tuple(number<MxGemmPipeline::flatNPerWarp>{},
                               number<MxGemmPipeline::flatKPerWarp>{}),
                    {static_cast<int>(i_n / BlockGemmShape::WarpTile::at(I1)), 0});
            },
            number<NumBTensor>{});
    }

    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        std::array<ScalePtrType, NumATensor> as_scale_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.as_scale_ptr[i]);
        });

        std::array<ScalePtrType, NumBTensor> bs_scale_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.bs_scale_ptr[i]);
        });

        // cluster launch pads grid to cluster boundaries; skip out-of-bound blocks
        if constexpr(BaseKernel::ClusterLaunch)
        {
            if(block_idx_m >= kargs.M || block_idx_n >= kargs.N)
                return;
        }

        const auto& as_block_window = BaseKernel::MakeABlockWindows(
            as_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& bs_block_window = [&]() {
            if constexpr(MxGemmPipeline::Preshuffle)
            {
                return MakeBFlatBlockWindows(bs_ptr, kargs, block_idx_n);
            }
            else
            {
                return BaseKernel::MakeBBlockWindows(
                    bs_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
            }
        }();
        const auto& ds_block_window =
            BaseKernel::MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);
        const auto& scale_a_block_window = MakeScaleABlockWindow(as_scale_ptr, kargs, block_idx_m);
        const auto& scale_b_block_window = MakeScaleBBlockWindow(bs_scale_ptr, kargs, block_idx_n);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        const auto& c_block_tile = MxGemmPipeline{}.template operator()(as_block_window,
                                                                        AElementWise{},
                                                                        bs_block_window,
                                                                        BElementWise{},
                                                                        scale_a_block_window,
                                                                        scale_b_block_window,
                                                                        num_loop,
                                                                        smem_ptr);

        auto c_block_window = BaseKernel::template MakeCBlockWindows<memory_operation_enum::set>(
            e_ptr, kargs, block_idx_m, block_idx_n);
        EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
    }
};

} // namespace ck_tile
