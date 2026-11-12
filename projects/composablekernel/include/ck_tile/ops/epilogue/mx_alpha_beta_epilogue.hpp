// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled.hpp"

namespace ck_tile {

/**
 * @brief Problem of MxAlphaBetaEpilogue: the pipeline whose C block tile it stores and the D store
 *        width in elements, by default the pipeline's MxTdmScheduledGemmTraits. C is the single D
 * tensor, row-major and of D's type.
 */
template <typename Pipeline_, index_t StoreWidth_ = remove_cvref_t<Pipeline_>::Traits::kStoreWidth>
struct MxAlphaBetaEpilogueProblem
{
    using Pipeline   = remove_cvref_t<Pipeline_>;
    using ODataType  = typename Pipeline::ODataType;
    using DsDataType = tuple<ODataType>;
    using DsLayout   = tuple<tensor_layout::gemm::RowMajor>;
    using ELayout    = tensor_layout::gemm::RowMajor;

    static constexpr index_t kStoreWidth = StoreWidth_;
};

/**
 * @brief Epilogue of MxGemmPipelineAgBgCrCompTDMScheduled after rocKE's mx_gemm: D = alpha * acc +
 * beta * C with C the single Ds tensor; the beta == 0 path never reads C.
 *
 * The C block tile is shuffled to the store distribution (N repeat innermost), so each thread
 * stores and loads vectors of StoreWidth adjacent columns through the D and C windows; their
 * padding masks partial tiles. FP8 saturates as hipBLASLt (+-inf to NaN, NaN kept, finite values
 * clamped to +-448); FP16 and BF16 round to nearest even without clamping.
 */
template <typename Problem_>
struct MxAlphaBetaEpilogue
{
    using Problem    = remove_cvref_t<Problem_>;
    using Pipeline   = typename Problem::Pipeline;
    using ODataType  = typename Problem::ODataType;
    using DsDataType = typename Problem::DsDataType;
    using DsLayout   = typename Problem::DsLayout;
    using ELayout    = typename Problem::ELayout;

    static constexpr index_t kStoreWidth = Problem::kStoreWidth;

    /// D replaces its tile: k_batch > 1 is not supported.
    static constexpr bool SupportsSplitK = false;
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return kStoreWidth; }
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I>)
    {
        return kStoreWidth;
    }
    /// The stores go from the accumulators straight to D.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }
    static constexpr MxTdmScheduledSpec kSpec =
        Pipeline::Policy::template GetSpec<typename Pipeline::Problem>();
    static constexpr bool kFp8         = kSpec.out == MxTdmScheduledOutDtype::Fp8E4M3;
    static constexpr bool kFp16        = kSpec.out == MxTdmScheduledOutDtype::Fp16;
    static constexpr index_t kOutBytes = kSpec.out_bytes();
    static constexpr index_t kFp8Width = kSpec.fp8_convert_width();
    static constexpr float kFp8Max     = 448.0f;
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
        kStoreWidth >= 1 && kStoreWidth <= kSpec.max_store_width() &&
            Pipeline::kRn % kStoreWidth == 0 && (kStoreWidth & (kStoreWidth - 1)) == 0,
        "comp_tdm_scheduled: the D store width must be a power of 2 dividing the N repeat, "
        "<= 16 B");

    using Storage = std::conditional_t<kOutBytes == 1, uint8_t, uint16_t>;
    using SVec = std::conditional_t<kStoreWidth == 1, Storage, ext_vector_t<Storage, kStoreWidth>>;

    CK_TILE_HOST_DEVICE static constexpr auto MakeStoreDistribution()
    {
        return Pipeline::Policy::template MakeCStoreTileDistribution<typename Pipeline::Problem>();
    }

    CK_TILE_DEVICE static float Saturate(float v)
    {
        // fma(v, 0, v) turns +-inf into NaN, which the NaN-propagating clamp keeps.
        v = __builtin_fmaf(v, 0.0f, v);
        return __builtin_elementwise_minimum(__builtin_elementwise_maximum(v, -kFp8Max), kFp8Max);
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

    /// kStoreWidth values as the bits of D's type.
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
    /// D = Pack(op(v, i)) with i the element's index in the thread buffer, after prepare(access)
    /// has loaded what op reads; each access's accumulators die at its store.
    template <typename OutWindow, typename AccTile, typename Prepare, typename Op>
    CK_TILE_DEVICE static void
    StoreByAccess(const OutWindow& out_window, const AccTile& acc, Prepare prepare, Op op)
    {
        using Traits                  = typename OutWindow::Traits;
        using SFC_Ys                  = typename Traits::SFC_Ys;
        constexpr index_t kVector     = Traits::ScalarPerVector;
        constexpr index_t kVectorDimY = Traits::VectorDimY;
        constexpr auto ys_to_d        = MakeStoreDistribution().get_ys_to_d_descriptor();
        static_assert(kVector % kStoreWidth == 0, "D's vectors hold whole store groups");
        using OVec = array<ODataType, kStoreWidth>;

        auto out = make_static_distributed_tensor<ODataType>(MakeStoreDistribution());
        static_for<0, Traits::NumAccess, 1>{}([&](auto i_access) {
            prepare(i_access);
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
                    part[j]              = op(acc.get_thread_buffer()[number<at>{}], number<at>{});
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

    /// D = alpha * acc + beta * C; MxGemmKernel sets them from its arguments.
    float alpha                      = 1.0f;
    float beta                       = 0.0f;
    static constexpr bool kAlphaBeta = true;

    /// Stores the C block tile through D's tile window, reading C (Ds[0]) only for beta != 0.
    template <typename ODramWindow, typename DsDramWindows>
    CK_TILE_DEVICE void operator()(ODramWindow& out_dram_window,
                                   const typename Pipeline::CBlockTile& c_block_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* /*p_smem*/) const
    {
        // alpha * acc + beta * C rounds twice, as rocKE's (and the host reference's) fmul, fadd.
#pragma clang fp contract(off)
        static_assert(std::tuple_size_v<DsDramWindows> == 1, "C is the single Ds tensor");
        auto d = make_static_distributed_tensor<float>(MakeStoreDistribution());
        shuffle_tile(d, c_block_tile);
        const float a         = alpha;
        const float b         = beta;
        const auto out_window = make_tile_window(out_dram_window, MakeStoreDistribution());
        const auto finish     = [](float v) {
            if constexpr(kFp8)
                v = Saturate(v);
            return v;
        };
        // GW_B0 / GW_B1.
        if(b == 0.0f)
        {
            StoreByAccess(out_window, d, [](auto) {}, [&](float v, auto) { return finish(a * v); });
        }
        else
        {
            const auto c_window =
                make_tile_window(ds_dram_windows[number<0>{}], MakeStoreDistribution());
            static_assert(decltype(c_window)::Traits::NumAccess ==
                              decltype(out_window)::Traits::NumAccess,
                          "C loads as D stores");
            auto c = make_static_distributed_tensor<ODataType>(MakeStoreDistribution());
            StoreByAccess(
                out_window,
                d,
                [&](auto i_access) {
                    c_window.load(
                        c, i_access, bool_constant<kCheckBounds>{}, bool_constant<kStaticMoveYs>{});
                },
                [&](float v, auto at) {
                    return finish(a * v + b * type_convert<float>(c.get_thread_buffer()[at]));
                });
        }
    }
};

} // namespace ck_tile
