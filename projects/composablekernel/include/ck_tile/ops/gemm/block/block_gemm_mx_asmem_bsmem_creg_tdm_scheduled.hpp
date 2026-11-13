// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

/// The block GEMM of MxGemmPipelineAgBgCrCompTDMScheduled: one wave's MX WMMAs on A and B tiles
/// it reads itself from the pipeline's LDS buffers (A and B in shared memory, C in registers).
///
/// The pipeline calls it per K step: LocalPrefetchA and LocalPrefetchB read a K step's A
/// fragments and scales and a range of its B fragments into registers, and operator() issues
/// the WMMAs of a range of B fragments. A hook runs at a fixed WMMA of the range, so that the
/// pipeline's cluster signal or TDM refill lands between two given WMMAs.
///
/// Policy is MxGemmPipelineAgBgCrCompTDMScheduledPolicy: the LDS layout, the read distributions
/// and the C distribution are its.
template <typename Problem_, typename Policy_>
struct BlockGemmMxAsmemBsmemCRegTDMScheduled
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = typename Problem::ADataType;
    using BDataType      = typename Problem::BDataType;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kRm          = Policy::template GetWaveRepeatM<Problem>();
    static constexpr index_t kRn          = Policy::template GetWaveRepeatN<Problem>();
    static constexpr index_t kNumAcc      = kRm * kRn;
    static constexpr index_t kAtomM       = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr index_t kAtomN       = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr bool kMma32          = kAtomM == 32;
    static constexpr index_t kCFragLen    = Policy::template GetCFragLen<Problem>();
    static constexpr index_t kStepBytes   = Policy::template GetStepBytes<Problem>();
    static constexpr index_t kChunks      = Policy::template GetChunksPerLane<Problem>();
    static constexpr index_t kRowWords    = kChunks * 4;
    static constexpr index_t kLdsRowBytes = Policy::template GetLdsRowBytes<Problem>();

    using AFrag   = int32x16_t;
    using BFrag   = std::conditional_t<kMma32, int32x8_t, int32x16_t>;
    using RowFrag = std::conditional_t<kRowWords == 8, int32x8_t, int32x16_t>;
    using CVec    = std::conditional_t<kMma32, fp32x16_t, fp32x8_t>;
    /// The accumulators: one CVec per M and N repeat, index rm * kRn + rn.
    using CBlockTile = decltype(make_static_distributed_tensor<float>(
        Policy::template MakeCBlockTileDistribution<Problem>()));
    using AccRegs    = thread_buffer<CVec, kNumAcc>;
    static_assert(CBlockTile::get_thread_buffer_size() == kNumAcc * kCFragLen,
                  "comp_tdm_scheduled: the C block tile holds the C fragments of every repeat");

    /// A K step's A fragments and the A and B scales.
    struct ATile
    {
        AFrag a[kRm];
        int32_t as[kRm];
        int32_t bs[kRn];
    };
    /// A K step's B fragments: all of them, or two halves with half-B.
    struct BTile
    {
        BFrag b[kRn];
    };

    /// The hook of an operator() that has none.
    struct NoHook
    {
        CK_TILE_DEVICE void operator()() const {}
    };

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile() { return CBlockTile{}; }

    /// Whether A's or B's fragments are read through a tile window (see
    /// MxGemmPipelineAgBgCrCompTDMScheduledPolicy::IsLdsReadLinear); otherwise RowFragment reads
    /// them.
    static constexpr bool kTileReadA = Policy::template IsLdsReadLinear<Problem, true>();
    static constexpr bool kTileReadB = Policy::template IsLdsReadLinear<Problem, false>();

    /// Wave 0's window on A's or B's fragment of one K step at repeat 0 of buffer 0. The wave
    /// stays out of the window's per-lane coordinate so that its offset remains scalar.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsWindow(CK_TILE_LDS_ADDR uint8_t* smem)
    {
        constexpr index_t rows = Policy::template GetFragmentHalves<Problem, IsA>() * 16;
        const auto view        = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsBlockDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<rows>{}, number<kStepBytes / 4>{}),
                                make_multi_index(0, 0),
                                Policy::template MakeLdsReadDistribution<Problem, IsA>());
    }
    using ALdsWindow = decltype(MakeLdsWindow<true>(nullptr));
    using BLdsWindow = decltype(MakeLdsWindow<false>(nullptr));

    /// Whether A's or B's scales are read through a tile window (see
    /// MxGemmPipelineAgBgCrCompTDMScheduledPolicy::IsLdsScaleReadLinear); otherwise ScaleWords
    /// reads them.
    static constexpr bool kTileReadAScale = Policy::template IsLdsScaleReadLinear<Problem, true>();
    static constexpr bool kTileReadBScale = Policy::template IsLdsScaleReadLinear<Problem, false>();

    /// A wave's window on A's or B's scales of K step 0 in buffer 0.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsScaleWindow(CK_TILE_LDS_ADDR uint8_t* smem, index_t wave)
    {
        constexpr index_t atom    = IsA ? kAtomM : kAtomN;
        constexpr index_t repeats = IsA ? kRm : kRn;
        constexpr index_t lanes   = Policy::template GetScaleLanes<Problem, IsA>();
        const auto view           = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsScaleDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<1>{}, number<lanes>{}, number<repeats>{}),
                                make_multi_index(0, wave * atom, 0),
                                Policy::template MakeLdsScaleReadDistribution<Problem, IsA>());
    }
    using AScaleWindow = decltype(MakeLdsScaleWindow<true>(nullptr, 0));
    using BScaleWindow = decltype(MakeLdsScaleWindow<false>(nullptr, 0));

    /// A fragment as an operand of the WMMA traits: the fragment itself, or the 8 data words of
    /// an fp4 one padded to 16 (the traits pad it again).
    template <typename Operand, typename Frag>
    CK_TILE_DEVICE static Operand MmaOperand(const Frag& f)
    {
        if constexpr(sizeof(Operand) == sizeof(Frag))
            return bit_cast<Operand>(f);
        else
            return bit_cast<Operand>(__builtin_shufflevector(f, f, 0, 1, 2, 3, 4, 5, 6, 7));
    }

    CK_TILE_DEVICE static void Mma(CVec& c, const AFrag& a, const BFrag& b, int32_t as, int32_t bs)
    {
#if defined(__gfx125__)
        using WarpGemm =
            WarpGemmDispatcher<ADataType, BDataType, float, kAtomM, kAtomN, Policy::kAtomK, false>;
        using AVec        = typename WarpGemm::WarpGemmAttribute::AVecType;
        using BVec        = typename WarpGemm::WarpGemmAttribute::BVecType;
        constexpr auto I0 = number<0>{};
        typename WarpGemm::AWarpTensor a_warp;
        typename WarpGemm::BWarpTensor b_warp;
        typename WarpGemm::CWarpTensor c_warp;
        a_warp.get_thread_buffer().template set_as<AVec>(I0, MmaOperand<AVec>(a));
        b_warp.get_thread_buffer().template set_as<BVec>(I0, MmaOperand<BVec>(b));
        c_warp.get_thread_buffer().template set_as<CVec>(I0, c);
        WarpGemm{}(c_warp, a_warp, b_warp, as, bs);
        c = c_warp.get_thread_buffer().template get_as<CVec>()[I0];
#else
        ignore = c;
        ignore = a;
        ignore = b;
        ignore = as;
        ignore = bs;
#endif
    }

    CK_TILE_DEVICE constexpr BlockGemmMxAsmemBsmemCRegTDMScheduled() = default;

    /// `smem` is the pipeline's LDS; the lane and the wave's M and N position are the pipeline's.
    CK_TILE_DEVICE BlockGemmMxAsmemBsmemCRegTDMScheduled(CK_TILE_LDS_ADDR uint8_t* smem_,
                                                         index_t lane_,
                                                         index_t wave_m_,
                                                         index_t wave_n_)
        : smem(smem_),
          lane(lane_),
          lane_col(integer_modulo_nonnegative<16>(lane_)),
          lane_half(integer_divide_nonnegative<16>(lane_)),
          wave_m(wave_m_),
          wave_n(wave_n_),
          a_window(MakeLdsWindow<true>(smem_)),
          b_window(MakeLdsWindow<false>(smem_)),
          as_window(MakeLdsScaleWindow<true>(smem_, wave_m_)),
          bs_window(MakeLdsScaleWindow<false>(smem_, wave_n_))
    {
    }

    /// A K step's scales and A fragments in LDS buffer `buffer` (a byte offset).
    CK_TILE_DEVICE ATile LocalPrefetchA(index_t buffer, index_t step) const
    {
        ATile h;
        if constexpr(kTileReadAScale)
        {
            LdsScaleWords<true>(h.as, buffer, step);
        }
        else
        {
            constexpr auto as_lds = Policy::template GetLdsOperand<Problem, true, true>();
            const index_t a_lane  = kMma32 ? lane : lane_col;
            ScaleWords<kRm>(
                h.as,
                buffer + static_cast<index_t>(Policy::template GetLdsScaleByteOffset<uint32_t>(
                             as_lds,
                             static_cast<uint32_t>(step),
                             static_cast<uint32_t>(MRow(0, a_lane)))));
        }
        if constexpr(kTileReadBScale)
        {
            LdsScaleWords<false>(h.bs, buffer, step);
        }
        else
        {
            constexpr auto bs_lds = Policy::template GetLdsOperand<Problem, false, true>();
            ScaleWords<kRn>(
                h.bs,
                buffer + static_cast<index_t>(Policy::template GetLdsScaleByteOffset<uint32_t>(
                             bs_lds,
                             static_cast<uint32_t>(step),
                             static_cast<uint32_t>(NRow(0, lane_col)))));
        }
        static_for<0, kRm, 1>{}([&](auto rm) { h.a[rm] = AFragment(buffer, rm, step); });
        return h;
    }

    /// B fragments First .. First + Count - 1 of a K step into b[First ..].
    template <index_t First, index_t Count>
    CK_TILE_DEVICE void LocalPrefetchB(BTile& tile, index_t buffer, index_t step) const
    {
        static_for<First, First + Count, 1>{}(
            [&](auto rn) { tile.b[rn] = BFragment(buffer, rn, step); });
    }

    /// The WMMAs of B fragments [First, First + Count): half-B order (rn outer, rm inner) with
    /// BOuter, else rm outer. `hook` runs before WMMA HookAt of the range, or after its last one
    /// when HookAt is the WMMA count.
    template <bool BOuter,
              index_t First,
              index_t Count,
              index_t HookAt = -1,
              typename Hook  = NoHook>
    CK_TILE_DEVICE void
    operator()(AccRegs& acc, const ATile& a, const BTile& b, const Hook& hook = {}) const
    {
        constexpr index_t total = kRm * Count;
        static_for<0, total, 1>{}([&](auto i) {
            constexpr index_t issued = decltype(i)::value;
            constexpr index_t rm     = BOuter ? issued % kRm : issued / Count;
            constexpr index_t rn     = First + (BOuter ? issued / kRm : issued % Count);
            if constexpr(issued == HookAt)
                hook();
            Mma(acc[rm * kRn + rn], a.a[rm], b.b[rn], a.as[rm], a.bs[rn]);
        });
        if constexpr(total == HookAt)
            hook();
    }

    private:
    /// RowMaps: WMMA positions to tile rows. A lane's M repeats are adjacent A rows, so its A
    /// scales of a K step are contiguous; its N repeats are adjacent D columns, so one
    /// accumulator slot is one vector store.
    CK_TILE_DEVICE index_t MRow(index_t rm, index_t m_local) const
    {
        return wave_m * (kRm * kAtomM) + m_local * kRm + rm;
    }
    CK_TILE_DEVICE index_t NRow(index_t rn, index_t n_local) const
    {
        return wave_n * (kRn * kAtomN) + n_local * kRn + rn;
    }

    CK_TILE_DEVICE int32x4_t Load4(index_t offset) const
    {
        return *reinterpret_cast<const CK_TILE_LDS_ADDR int32x4_t*>(smem + offset);
    }

    CK_TILE_DEVICE index_t LaneK(index_t step) const
    {
        return step * kStepBytes + lane_half * Policy::kChunkBytes;
    }

    CK_TILE_DEVICE RowFrag RowFragment(index_t buffer, bool is_a, index_t row, index_t lane_k) const
    {
        constexpr auto a_lds = Policy::template GetLdsOperand<Problem, true>();
        constexpr auto b_lds = Policy::template GetLdsOperand<Problem, false>();
        const uint32_t urow  = static_cast<uint32_t>(row);
        const index_t base =
            buffer +
            static_cast<index_t>(
                is_a ? Policy::template GetLdsRowByteOffset<uint32_t>(a_lds, kLdsRowBytes, urow)
                     : Policy::template GetLdsRowByteOffset<uint32_t>(b_lds, kLdsRowBytes, urow)) +
            lane_k;
        constexpr index_t o0 = Policy::template GetChunkOffset<Problem>(0);
        constexpr index_t o1 = Policy::template GetChunkOffset<Problem>(1);
        const int32x4_t c0   = Load4(base + o0);
        const int32x4_t c1   = Load4(base + o1);
        if constexpr(kChunks == 2)
        {
            return __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        else
        {
            constexpr index_t o2 = Policy::template GetChunkOffset<Problem>(2);
            constexpr index_t o3 = Policy::template GetChunkOffset<Problem>(3);
            const int32x4_t c2   = Load4(base + o2);
            const int32x4_t c3   = Load4(base + o3);
            const int32x8_t lo   = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            const int32x8_t hi   = __builtin_shufflevector(c2, c3, 0, 1, 2, 3, 4, 5, 6, 7);
            return __builtin_shufflevector(
                lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        }
    }

    /// One 8-word row fragment zero-padded to the 16 words of the 16x16x128 operand.
    CK_TILE_DEVICE static int32x16_t Pad16(const int32x8_t& x)
    {
        return __builtin_shufflevector(
            x, int32x8_t{}, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    CK_TILE_DEVICE static int32x16_t Pad16(const int32x16_t& x) { return x; }

    CK_TILE_DEVICE AFrag AFragment(index_t buffer, index_t rm, index_t step) const
    {
        if constexpr(kTileReadA)
        {
            return Pad16(LdsFragment<true>(buffer, rm, step));
        }
        else if constexpr(kMma32)
        {
            const index_t lane_k = LaneK(step);
            const int32x8_t r0   = RowFragment(buffer, true, MRow(rm, lane_col), lane_k);
            const int32x8_t r1   = RowFragment(buffer, true, MRow(rm, lane_col + 16), lane_k);
            return __builtin_shufflevector(
                r0, r1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        }
        else
        {
            return Pad16(RowFragment(buffer, true, MRow(rm, lane_col), LaneK(step)));
        }
    }

    /// A's or B's fragment of repeat r at K step `step` through its tile window: the wave,
    /// buffer, repeat and step are a word offset from the window's.
    template <bool IsA>
    CK_TILE_DEVICE auto LdsFragment(index_t buffer, index_t r, index_t step) const
    {
        constexpr index_t atom        = IsA ? kAtomM : kAtomN;
        constexpr index_t repeats     = IsA ? kRm : kRn;
        constexpr auto repeat_offsets = [] {
            array<index_t, repeats> offsets{};
            for(index_t i = 0; i < repeats; ++i)
                offsets[i] = Policy::template GetLdsRowOffset<Problem, IsA>(i * atom);
            return offsets;
        }();
        constexpr index_t base = Policy::template GetLdsOperand<Problem, IsA>().base / 4;
        constexpr index_t wave_words =
            Policy::template GetLdsRowOffset<Problem, IsA>(repeats * atom);
        const index_t wave_index = IsA ? wave_m : wave_n;
        const index_t offset     = buffer / 4 + base + wave_index * wave_words + repeat_offsets[r] +
                               step * (kStepBytes / 4);
        const auto& window = [&]() -> const auto& {
            if constexpr(IsA)
                return a_window;
            else
                return b_window;
        }();
        const auto tile = window.load_with_offset(offset, number<-1>{}, bool_constant<false>{});
        using Vec       = ext_vector_t<int32_t, decltype(tile)::get_thread_buffer_size()>;
        return tile.get_thread_buffer().template get_as<Vec>()[number<0>{}];
    }

    CK_TILE_DEVICE BFrag BFragment(index_t buffer, index_t rn, index_t step) const
    {
        const RowFrag f = [&] {
            if constexpr(kTileReadB)
                return LdsFragment<false>(buffer, rn, step);
            else
                return RowFragment(buffer, false, NRow(rn, lane_col), LaneK(step));
        }();
        if constexpr(kMma32)
            return f;
        else
            return Pad16(f);
    }

    /// Per-lane E8M0 words of one K step and repeat count: 16 B reads of four repeats, or one
    /// read of all of them.
    template <index_t Count>
    CK_TILE_DEVICE void ScaleWords(int32_t (&out)[Count], index_t base) const
    {
        if constexpr(Count >= 4)
        {
            static_for<0, Count / 4, 1>{}([&](auto chunk) {
                constexpr index_t i = decltype(chunk)::value;
                const int32x4_t w   = Load4(base + i * 16);
                out[i * 4 + 0]      = w[0];
                out[i * 4 + 1]      = w[1];
                out[i * 4 + 2]      = w[2];
                out[i * 4 + 3]      = w[3];
            });
        }
        else if constexpr(Count == 2)
        {
            const int32x2_t w = *reinterpret_cast<const CK_TILE_LDS_ADDR int32x2_t*>(smem + base);
            out[0]            = w[0];
            out[1]            = w[1];
        }
        else
        {
            out[0] = *reinterpret_cast<const CK_TILE_LDS_ADDR int32_t*>(smem + base);
        }
    }

    /// A's or B's per-lane E8M0 words of K step `step` through the scale window.
    template <bool IsA, index_t Count>
    CK_TILE_DEVICE void LdsScaleWords(int32_t (&out)[Count], index_t buffer, index_t step) const
    {
        constexpr auto op = Policy::template GetLdsOperand<Problem, IsA, true>();
        constexpr index_t step_words =
            static_cast<index_t>(Policy::template GetLdsScaleByteOffset<uint32_t>(op, 1u, 0u) -
                                 op.base) /
            4;
        const auto& window = [&]() -> const auto& {
            if constexpr(IsA)
                return as_window;
            else
                return bs_window;
        }();
        const auto tile = window.load_with_offset(
            buffer / 4 + op.base / 4 + step * step_words, number<-1>{}, bool_constant<false>{});
        static_for<0, Count, 1>{}([&](auto i) { out[i] = tile.get_thread_buffer()[i]; });
    }

    CK_TILE_LDS_ADDR uint8_t* smem;
    index_t lane;
    index_t lane_col;
    index_t lane_half;
    index_t wave_m;
    index_t wave_n;
    ALdsWindow a_window;
    BLdsWindow b_window;
    AScaleWindow as_window;
    BScaleWindow bs_window;
};

} // namespace ck_tile
