// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {

namespace impl {
template <typename T, typename = void>
struct universal_gemm_kernel_of
{
    using type = T;
};
// GemmKernel and GemmKernelMultiD wrap a UniversalGemmKernel; run that one.
template <typename T>
struct universal_gemm_kernel_of<T, std::void_t<typename T::UniversalGemmKernel>>
{
    using type = typename T::UniversalGemmKernel;
};
} // namespace impl

/// @brief A GEMM as one megakernel phase: work item w is output tile (w % tiles) of split
/// (w / tiles). Counter slots are M tiles, so a consumer may wait on just the rows it reads
/// (after_rows).
///
/// Tiles are ordered M-major (all N tiles of M tile 0 first), so the first slots complete first.
/// The kernel's TilePartitioner mapping is not used: RunGemm takes the tile origin, and
/// GemmTile1DPartitioner keeps N in a static device variable that every kernel with the same
/// partitioner type shares, which two GEMM phases running at once would race on.
///
/// @tparam GemmKernel_ a UniversalGemmKernel, or a GemmKernel / GemmKernelMultiD wrapping one.
/// Args are that kernel's KernelArgs, made by its MakeKernelArgs.
template <typename GemmKernel_>
struct GemmPhase
{
    using Kernel           = typename impl::universal_gemm_kernel_of<GemmKernel_>::type;
    using GemmKernelImpl   = typename Kernel::SelfType;
    using Args             = typename Kernel::KernelArgs;
    using TilePartitioner  = typename Kernel::TilePartitioner;
    using GemmPipeline     = typename Kernel::GemmPipeline;
    using ADataType        = typename Kernel::ADataType;
    using BDataType        = typename Kernel::BDataType;
    using EDataType        = typename Kernel::EDataType;
    using barrier_pipeline = typename Kernel::barrier_pipeline;

    static_assert(!Kernel::ClusterLaunch, "GemmPhase: cluster launch is not supported yet");

    static constexpr index_t kBlockSize = Kernel::kBlockSize;
    static constexpr index_t kSlotRows  = TilePartitioner::MPerBlock;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Kernel::GetSmemSize(); }

    CK_TILE_HOST_DEVICE static index_t TilesM(const Args& a)
    {
        return integer_divide_ceil(a.M, TilePartitioner::MPerBlock);
    }
    CK_TILE_HOST_DEVICE static index_t TilesN(const Args& a)
    {
        return integer_divide_ceil(a.N, TilePartitioner::NPerBlock);
    }
    CK_TILE_HOST_DEVICE static index_t NumTiles(const Args& a) { return TilesM(a) * TilesN(a); }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return NumTiles(a) * a.k_batch; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return TilesM(a); }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t /*slot*/)
    {
        return TilesN(a) * a.k_batch;
    }

    CK_TILE_HOST_DEVICE static tuple<index_t, index_t> TileIndex(const Args& a, index_t work)
    {
        const index_t tile = work % NumTiles(a);
        const index_t i_m  = tile / TilesN(a);
        return make_tuple(i_m, tile - i_m * TilesN(a));
    }

    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return amd_wave_read_first_lane((work % NumTiles(a)) / TilesN(a));
    }

    /// Rows of A this item reads, which is also the rows of E it writes.
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t r0 = SlotOf(a, work) * TilePartitioner::MPerBlock;
        return make_tuple(r0, min(a.M, r0 + TilePartitioner::MPerBlock));
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return Kernel::IsSupportedArgument(a);
    }

    CK_TILE_DEVICE static auto InitBarriers() { return Kernel::InitBarriers(); }

    template <typename BarrierToken>
    CK_TILE_DEVICE static void
    Run(const Args& kargs, index_t work, void* smem, const BarrierToken& barriers)
    {
        const index_t tiles = NumTiles(kargs);
        const index_t k_id  = amd_wave_read_first_lane(work / tiles);
        const auto [iM, iN] = TileIndex(kargs, work);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
        const auto split    = typename Kernel::SplitKBatchOffset(kargs, k_id);

        std::array<const ADataType*, Kernel::NumATensor> as_ptr;
        static_for<0, Kernel::NumATensor, 1>{}([&](auto i) {
            as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                        split.as_k_split_offset[i] / Kernel::APackedSize;
        });

        std::array<const BDataType*, Kernel::NumBTensor> bs_ptr;
        static_for<0, Kernel::NumBTensor, 1>{}([&](auto i) {
            if constexpr(GemmPipeline::Preshuffle)
            {
                // The flat-B window applies the split's K offset itself (see the kernel).
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]);
            }
            else
            {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            split.bs_k_split_offset[i] / Kernel::BPackedSize;
            }
        });

        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
        if constexpr(Kernel::has_tile_partitioner_output_offset)
        {
            e_ptr += TilePartitioner::GetOutputOffset(kargs, k_id);
        }

        if constexpr(std::is_void_v<barrier_pipeline>)
        {
            ignore = barriers;
            GemmKernelImpl::RunGemm(
                as_ptr, bs_ptr, kargs.ds_ptr, e_ptr, smem, kargs, split, i_m, i_n);
        }
        else
        {
            GemmKernelImpl::RunGemm(
                as_ptr, bs_ptr, kargs.ds_ptr, e_ptr, smem, kargs, split, i_m, i_n, barriers);
        }
    }
};

} // namespace ck_tile
