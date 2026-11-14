// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled.hpp"

namespace ck_tile {

/**
 * @brief Problem of MxDirectStoreEpilogue: the pipeline whose C block tile it stores and the store
 *        width in elements, by default the pipeline's MxTdmScheduledGemmTraits. E is row-major and
 *        there are no Ds tensors.
 */
template <typename Pipeline_, index_t StoreWidth_ = remove_cvref_t<Pipeline_>::Traits::kStoreWidth>
struct MxDirectStoreEpilogueProblem
{
    using Pipeline   = remove_cvref_t<Pipeline_>;
    using ODataType  = typename Pipeline::ODataType;
    using DsDataType = tuple<>;
    using DsLayout   = tuple<>;
    using ELayout    = tensor_layout::gemm::RowMajor;

    static constexpr index_t kStoreWidth = StoreWidth_;
};

/**
 * @brief Epilogue of MxGemmPipelineAgBgCrCompTDMScheduled: E = acc, stored
 *        from the accumulators without LDS.
 *
 * The C block tile is shuffled to the store distribution (N repeat innermost), so each thread
 * stores vectors of StoreWidth adjacent columns through E's window, whose padding masks partial
 * tiles. FP8 saturates as hipBLASLt (+-inf to NaN, NaN kept, finite values clamped to +-448);
 * FP16 and BF16 round to nearest even without clamping.
 */
template <typename Problem_>
struct MxDirectStoreEpilogue
{
    using Problem    = remove_cvref_t<Problem_>;
    using Pipeline   = typename Problem::Pipeline;
    using ODataType  = typename Problem::ODataType;
    using DsDataType = typename Problem::DsDataType;
    using DsLayout   = typename Problem::DsLayout;
    using ELayout    = typename Problem::ELayout;

    static constexpr index_t kStoreWidth = Problem::kStoreWidth;

    /// E replaces its tile: k_batch > 1 is not supported.
    static constexpr bool SupportsSplitK = false;
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return kStoreWidth; }
    /// The stores go from the accumulators straight to E.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }
    using PipelinePolicy               = typename Pipeline::Policy;
    using PipelineProblem              = typename Pipeline::Problem;
    static constexpr bool kFp8         = std::is_same_v<typename Pipeline::ODataType, fp8_t>;
    static constexpr bool kFp16        = std::is_same_v<typename Pipeline::ODataType, half_t>;
    static constexpr index_t kOutBytes = PipelinePolicy::template GetOutBytes<PipelineProblem>();
    static constexpr index_t kFp8Width =
        PipelinePolicy::template GetFp8ConvertWidth<PipelineProblem>();
    /// Rows past M lie past the buffer, which drops or zeroes them; the 64-bit global path and
    /// columns past N (inside the next row) need the windows' checks.
    static constexpr bool kCheckBounds =
        Pipeline::kPadN || (Pipeline::LargeTensors && Pipeline::kPadM);
    /// Static Y moves offset each access from the thread's first one, whose coordinate alone is
    /// checked: only right while the accesses move in M alone (one vector holds the thread's
    /// N repeat) and rows past M are dropped by the buffer.
    static constexpr bool kStaticMoveYs = (!Pipeline::kPadN || kStoreWidth == Pipeline::kRn) &&
                                          !(Pipeline::LargeTensors && Pipeline::kPadM);

    static_assert(
        kStoreWidth >= 1 &&
            kStoreWidth <= PipelinePolicy::template GetMaxStoreWidth<PipelineProblem>() &&
            Pipeline::kRn % kStoreWidth == 0 && (kStoreWidth & (kStoreWidth - 1)) == 0,
        "comp_tdm_scheduled: the E store width must be a power of 2 dividing the N repeat, "
        "<= 16 B");

    using Storage = std::conditional_t<kOutBytes == 1, uint8_t, uint16_t>;
    using SVec = std::conditional_t<kStoreWidth == 1, Storage, ext_vector_t<Storage, kStoreWidth>>;

    CK_TILE_HOST_DEVICE static constexpr auto MakeStoreDistribution()
    {
        return Pipeline::Policy::template MakeCStoreTileDistribution<typename Pipeline::Problem>();
    }

    /// kFp8Width values to fp8 e4m3 (no saturation) in kFp8Width bytes of `words`.
    CK_TILE_DEVICE static void Fp8Convert([[maybe_unused]] const float* v,
                                          [[maybe_unused]] uint32_t* words)
    {
#if defined(__gfx125__)
        if constexpr(kFp8Width == 8)
        {
            const fp32x8_t x = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
            const auto packed =
                bit_cast<uint32x2_t>(fp32x8_to_fp8x8<fp8_rounding_mode::standard>(x, 1.0f));
            words[0] = static_cast<uint32_t>(packed[0]);
            words[1] = static_cast<uint32_t>(packed[1]);
        }
        else
        {
            const auto lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], 0, false);
            words[0] = static_cast<uint32_t>(__builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], lo, true));
        }
#endif
    }

    /// kStoreWidth values as the bits of E's type.
    CK_TILE_DEVICE static SVec Pack(const float (&v)[kStoreWidth])
    {
        SVec out{};
        if constexpr(kFp8)
        {
            constexpr index_t chunks = (kStoreWidth + kFp8Width - 1) / kFp8Width;
            uint32_t words[chunks * kFp8Width / 4];
            static_for<0, chunks, 1>{}([&](auto chunk) {
                constexpr index_t first = decltype(chunk)::value * kFp8Width;
                float part[kFp8Width];
                // Fewer values than one convert: pad with zeros, keep the first.
                static_for<0, kFp8Width, 1>{}([&](auto i) {
                    constexpr index_t at = first + decltype(i)::value;
                    part[i]              = at < kStoreWidth ? v[at < kStoreWidth ? at : 0] : 0.0f;
                });
                Fp8Convert(part, words + first / 4);
            });
            __builtin_memcpy(&out, words, sizeof(SVec));
        }
        else
        {
            Storage bits[kStoreWidth];
            static_for<0, kStoreWidth, 1>{}([&](auto i) {
                if constexpr(kFp16)
                    bits[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(v[i]));
                else
                    bits[i] = __builtin_bit_cast(uint16_t, static_cast<__bf16>(v[i]));
            });
            __builtin_memcpy(&out, bits, sizeof(SVec));
        }
        return out;
    }

    /// Stores `acc` through `out_window` one access at a time, each vector's elements v becoming
    /// E = Pack(v), saturated for FP8; each access's accumulators die at its store.
    template <typename OutWindow, typename AccTile>
    CK_TILE_DEVICE static void StoreByAccess(const OutWindow& out_window, const AccTile& acc)
    {
        using Traits                  = typename OutWindow::Traits;
        using SFC_Ys                  = typename Traits::SFC_Ys;
        constexpr index_t kVector     = Traits::ScalarPerVector;
        constexpr index_t kVectorDimY = Traits::VectorDimY;
        constexpr auto ys_to_d        = MakeStoreDistribution().get_ys_to_d_descriptor();
        static_assert(kVector % kStoreWidth == 0, "E's vectors hold whole store groups");
        using OVec = array<ODataType, kStoreWidth>;

        auto out = make_static_distributed_tensor<ODataType>(MakeStoreDistribution());
        static_for<0, Traits::NumAccess, 1>{}([&](auto i_access) {
            constexpr auto idx_ys_start = SFC_Ys::get_index(i_access);
            static_for<0, kVector / kStoreWidth, 1>{}([&](auto group) {
                float part[kStoreWidth];
                constexpr index_t first = decltype(group)::value * kStoreWidth;
                static_for<0, kStoreWidth, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == kVectorDimY ? (idx_ys_start[jj] + first + j)
                                                     : idx_ys_start[jj];
                        },
                        number<AccTile::StaticTileDistribution::get_num_of_dimension_y()>{});
                    constexpr index_t at = ys_to_d.calculate_offset(idx_ys);
                    const float v        = acc.get_thread_buffer()[number<at>{}];
                    if constexpr(kFp8)
                        part[j] = element_wise::SaturateFp8E4M3::apply(v);
                    else
                        part[j] = v;
                });
                constexpr auto idx_ys_group = generate_tuple(
                    [&](auto jj) {
                        return jj == kVectorDimY ? (idx_ys_start[jj] + first) : idx_ys_start[jj];
                    },
                    number<AccTile::StaticTileDistribution::get_num_of_dimension_y()>{});
                constexpr index_t at_group = ys_to_d.calculate_offset(idx_ys_group);
                static_assert(at_group % kStoreWidth == 0);
                out.get_thread_buffer().template set_as<OVec>(number<at_group / kStoreWidth>{},
                                                              bit_cast<OVec>(Pack(part)));
            });
            out_window.store(
                out, i_access, bool_constant<kCheckBounds>{}, bool_constant<kStaticMoveYs>{});
        });
    }

    /// Stores the C block tile through E's tile window.
    template <typename ODramWindow, typename DsDramWindows>
    CK_TILE_DEVICE void operator()(ODramWindow& out_dram_window,
                                   const typename Pipeline::CBlockTile& c_block_tile,
                                   const DsDramWindows& /*ds_dram_windows*/,
                                   void* /*p_smem*/) const
    {
        static_assert(std::tuple_size_v<DsDramWindows> == 0, "no Ds tensors");
        auto d = make_static_distributed_tensor<float>(MakeStoreDistribution());
        shuffle_tile(d, c_block_tile);
        StoreByAccess(make_tile_window(out_dram_window, MakeStoreDistribution()), d);
    }
};

/**
 * @brief TdmEpilogue for MxGemmPipelineAgBgCrCompTDMScheduled, the alternative to
 *        MxDirectStoreEpilogue: E = acc staged in LDS (kTileM * kTileN * sizeof(E) bytes) and
 *        stored by TDM. FP8 saturates as MxDirectStoreEpilogue. The pipeline's
 *        MxTdmScheduledGemmTraits should take StoreWidth 1: TDM stores have no vector width.
 */
template <typename Pipeline_,
          typename Pipeline = remove_cvref_t<Pipeline_>,
          typename Shape    = typename Pipeline::BlockGemmShape>
using MxTdmStoreEpilogue =
    TdmEpilogue<CShuffleEpilogueProblem<typename Pipeline::ADataType,
                                        typename Pipeline::BDataType,
                                        tuple<>,
                                        float,
                                        typename Pipeline::ODataType,
                                        tuple<>,
                                        tensor_layout::gemm::RowMajor,
                                        element_wise::PassThrough,
                                        Shape::kM,
                                        Shape::kN,
                                        Shape::BlockWarps::at(number<0>{}),
                                        Shape::BlockWarps::at(number<1>{}),
                                        Shape::WarpTile::at(number<0>{}),
                                        Shape::WarpTile::at(number<1>{}),
                                        Shape::WarpTile::at(number<2>{}),
                                        /*isCTransposed=*/false>,
                std::conditional_t<std::is_same_v<typename Pipeline::ODataType, fp8_t>,
                                   element_wise::SaturateFp8E4M3,
                                   element_wise::PassThrough>>;

} // namespace ck_tile
