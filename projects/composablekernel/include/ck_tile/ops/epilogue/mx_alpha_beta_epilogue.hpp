// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4.hpp"

// D and C are stored and loaded as kStoreWidth-element vectors through their element pointers.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"

namespace ck_tile {

/**
 * @brief Problem of MxAlphaBetaEpilogue: the pipeline whose accumulators it stores, the D store
 *        width in elements and the shape traits of the stores (EdgeM / EdgeN mask partial tiles,
 *        WideD addresses rows in i64), by default the pipeline's MxTdmV4GemmTraits. C is the
 *        single D tensor, row-major and of D's type.
 */
template <typename Pipeline_,
          index_t StoreWidth_ = remove_cvref_t<Pipeline_>::Traits::kStoreWidth,
          bool EdgeM_         = remove_cvref_t<Pipeline_>::Traits::kEdgeM,
          bool EdgeN_         = remove_cvref_t<Pipeline_>::Traits::kEdgeN,
          bool WideD_         = remove_cvref_t<Pipeline_>::Traits::kWideD>
struct MxAlphaBetaEpilogueProblem
{
    using Pipeline   = remove_cvref_t<Pipeline_>;
    using ODataType  = typename Pipeline::ODataType;
    using DsDataType = tuple<ODataType>;
    using DsLayout   = tuple<tensor_layout::gemm::RowMajor>;
    using ELayout    = tensor_layout::gemm::RowMajor;

    static constexpr index_t kStoreWidth = StoreWidth_;
    static constexpr bool kEdgeM         = EdgeM_;
    static constexpr bool kEdgeN         = EdgeN_;
    static constexpr bool kWideD         = WideD_;
};

/**
 * @brief Epilogue of GemmPipelineAgBgCrCompTDMV4 after rocKE's mx_gemm: D = alpha * acc + beta * C
 *        stored as one vector of StoreWidth columns per accumulator slot; the beta == 0 path never
 *        reads C.
 *
 * FP8 saturates as hipBLASLt (+-inf to NaN, NaN kept, finite values clamped to +-448); FP16 and
 * BF16 round to nearest even without clamping. C and D are M x N row-major. With EdgeM / EdgeN
 * each vector is read and written only if its row and columns are inside D (StoreWidth divides N,
 * so a vector is wholly inside or outside); WideD addresses rows in i64.
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
    using CVec       = typename Pipeline::CVec;
    /// The C block tile's floats: slot `slot` of repeat (rm, rn) at (rm * kRn + rn) * kCFragLen +
    /// slot.
    using CBuffer = remove_cvref_t<decltype(typename Pipeline::CBlockTile{}.get_thread_buffer())>;

    static constexpr index_t kStoreWidth = Problem::kStoreWidth;
    static constexpr bool kEdgeM         = Problem::kEdgeM;
    static constexpr bool kEdgeN         = Problem::kEdgeN;
    static constexpr bool kWideD         = Problem::kWideD;

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
    static constexpr index_t kRm       = Pipeline::kRm;
    static constexpr index_t kRn       = Pipeline::kRn;
    static constexpr index_t kCFragLen = Pipeline::kCFragLen;
    static constexpr MxTdmV4Spec kSpec = Pipeline::kSpec;
    static constexpr bool kFp8         = kSpec.out == MxTdmV4OutDtype::Fp8E4M3;
    static constexpr bool kFp16        = kSpec.out == MxTdmV4OutDtype::Fp16;
    static constexpr index_t kOutBytes = kSpec.out_bytes();
    static constexpr index_t kFp8Width = kSpec.fp8_convert_width();
    static constexpr float kFp8Max     = 448.0f;

    static_assert(kStoreWidth >= 1 && kStoreWidth <= kSpec.max_store_width() &&
                      kRn % kStoreWidth == 0 && (kStoreWidth & (kStoreWidth - 1)) == 0,
                  "comp_tdm_v4: the D store width must be a power of 2 dividing the N repeat, "
                  "<= 16 B");

    using Storage = std::conditional_t<kOutBytes == 1, uint8_t, uint16_t>;
    using SVec = std::conditional_t<kStoreWidth == 1, Storage, ext_vector_t<Storage, kStoreWidth>>;

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

    CK_TILE_DEVICE static float ToFloat(Storage bits)
    {
        if constexpr(kFp8)
            return type_convert<float>(bit_cast<fp8_t>(bits));
        else if constexpr(kFp16)
            return static_cast<float>(__builtin_bit_cast(_Float16, bits));
        else
            return static_cast<float>(__builtin_bit_cast(__bf16, bits));
    }

    CK_TILE_DEVICE static Storage Lane(const SVec& v, index_t i)
    {
        if constexpr(kStoreWidth == 1)
            return v;
        else
            return v[i];
    }

    template <bool ReadC>
    CK_TILE_DEVICE static void Store(const Storage* c,
                                     Storage* d,
                                     index_t M,
                                     index_t N,
                                     float alpha,
                                     float beta,
                                     const mx_tdm_v4::Coords& co,
                                     const CBuffer& acc)
    {
        // alpha * acc + beta * C rounds twice, as rocKE's (and the host reference's) fmul, fadd.
#pragma clang fp contract(off)
        const index_t col = co.block_n + Pipeline::NRow(co, 0, co.lane_col);
        static_for<0, kRm, 1>{}([&](auto rm_) {
            constexpr index_t rm = decltype(rm_)::value;
            static_for<0, kCFragLen, 1>{}([&](auto slot_) {
                constexpr index_t slot = decltype(slot_)::value;
                const index_t m_local  = (slot / 8) * 16 + slot % 8 + co.lane_half * 8;
                const index_t row      = co.block_m + Pipeline::MRow(co, rm, m_local);
                const Storage* c_row   = c;
                Storage* d_row         = d;
                index_t index;
                if constexpr(kWideD)
                {
                    // Row base in i64, the column offset in i32.
                    const uint64_t row_bytes = static_cast<uint64_t>(static_cast<uint32_t>(row)) *
                                               static_cast<uint32_t>(N * kOutBytes);
                    d_row = reinterpret_cast<Storage*>(reinterpret_cast<uint8_t*>(d) + row_bytes);
                    if constexpr(ReadC)
                        c_row = reinterpret_cast<const Storage*>(
                            reinterpret_cast<const uint8_t*>(c) + row_bytes);
                    index = col;
                }
                else
                {
                    index = row * N + col;
                }
                if constexpr(kEdgeM || kEdgeN)
                {
                    StoreMasked<ReadC, rm, slot>(
                        c_row, d_row, index, row, col, M, N, alpha, beta, acc);
                }
                else
                {
                    float values[kRn];
                    static_for<0, kRn, 1>{}([&](auto rn) {
                        values[rn] = alpha * acc[(rm * kRn + rn) * kCFragLen + slot];
                    });
                    if constexpr(ReadC)
                    {
                        SVec words[kRn / kStoreWidth];
                        static_for<0, kRn / kStoreWidth, 1>{}([&](auto chunk) {
                            words[chunk] = *reinterpret_cast<const SVec*>(
                                c_row + index + decltype(chunk)::value * kStoreWidth);
                        });
                        static_for<0, kRn, 1>{}([&](auto i) {
                            values[i] = values[i] + beta * ToFloat(Lane(words[i / kStoreWidth],
                                                                        i % kStoreWidth));
                        });
                    }
                    if constexpr(kFp8)
                        static_for<0, kRn, 1>{}([&](auto i) { values[i] = Saturate(values[i]); });
                    static_for<0, kRn / kStoreWidth, 1>{}([&](auto chunk) {
                        constexpr index_t begin = decltype(chunk)::value * kStoreWidth;
                        float part[kStoreWidth];
                        static_for<0, kStoreWidth, 1>{}(
                            [&](auto i) { part[i] = values[begin + decltype(i)::value]; });
                        *reinterpret_cast<SVec*>(d_row + index + begin) = Pack(part);
                    });
                }
            });
        });
    }

    /// One accumulator slot of a partial tile: each vector of kStoreWidth columns is read and
    /// written only if its row and columns are in D.
    template <bool ReadC, index_t Rm, index_t Slot>
    CK_TILE_DEVICE static void StoreMasked(const Storage* c_row,
                                           Storage* d_row,
                                           index_t index,
                                           index_t row,
                                           index_t col,
                                           index_t M,
                                           index_t N,
                                           float alpha,
                                           float beta,
                                           const CBuffer& acc)
    {
#pragma clang fp contract(off)
        static_for<0, kRn / kStoreWidth, 1>{}([&](auto chunk) {
            constexpr index_t begin = decltype(chunk)::value * kStoreWidth;
            bool inside             = true;
            if constexpr(kEdgeM)
                inside = row < M;
            if constexpr(kEdgeN)
                inside = kEdgeM ? (inside && col + begin < N) : col + begin < N;
            if(inside)
            {
                const index_t at = index + begin;
                float values[kStoreWidth];
                static_for<0, kStoreWidth, 1>{}([&](auto i) {
                    values[i] =
                        alpha * acc[(Rm * kRn + begin + decltype(i)::value) * kCFragLen + Slot];
                });
                if constexpr(ReadC)
                {
                    const SVec words = *reinterpret_cast<const SVec*>(c_row + at);
                    static_for<0, kStoreWidth, 1>{}(
                        [&](auto i) { values[i] = values[i] + beta * ToFloat(Lane(words, i)); });
                }
                if constexpr(kFp8)
                    static_for<0, kStoreWidth, 1>{}(
                        [&](auto i) { values[i] = Saturate(values[i]); });
                *reinterpret_cast<SVec*>(d_row + at) = Pack(values);
            }
        });
    }

    /// D = alpha * acc + beta * C; MxGemmKernel sets them from its arguments.
    float alpha                      = 1.0f;
    float beta                       = 0.0f;
    static constexpr bool kAlphaBeta = true;

    /// Stores the C block tile into D's tile window, reading C (Ds[0]) only for beta != 0. The
    /// windows give the pointers, the unpadded M and N and the tile origin.
    template <typename ODramWindow, typename DsDramWindows>
    CK_TILE_DEVICE void operator()(ODramWindow& out_dram_window,
                                   const typename Pipeline::CBlockTile& c_block_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* /*p_smem*/) const
    {
        static_assert(std::tuple_size_v<DsDramWindows> == 1, "C is the single Ds tensor");
        const auto& out_view       = out_dram_window.get_bottom_tensor_view();
        const auto lengths         = detail::tdm_real_lengths(out_view.get_tensor_descriptor());
        const index_t M            = lengths[number<0>{}];
        const index_t N            = lengths[number<1>{}];
        const auto origin          = out_dram_window.get_window_origin();
        const mx_tdm_v4::Coords co = Pipeline::MakeCoords(origin[number<0>{}], origin[number<1>{}]);
        const CBuffer& acc         = c_block_tile.get_thread_buffer();
        const auto* c_bits         = reinterpret_cast<const Storage*>(
            ds_dram_windows[number<0>{}].get_bottom_tensor_view().get_buffer_view().p_data_);
        auto* d_bits = reinterpret_cast<Storage*>(out_view.get_buffer_view().p_data_);
        // GW_B0 / GW_B1.
        if(beta == 0.0f)
            Store<false>(c_bits, d_bits, M, N, alpha, beta, co, acc);
        else
            Store<true>(c_bits, d_bits, M, N, alpha, beta, co, acc);
    }
};

} // namespace ck_tile

#pragma clang diagnostic pop
