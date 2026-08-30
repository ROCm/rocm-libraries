// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct FmhaD192ScoreFragmentMapping
{
    using Pair = fp32x2_t;

    struct Coordinate
    {
        index_t msb;
        index_t su;
        index_t pair;
        index_t element;
    };

    static constexpr index_t kNumMIter         = 2;
    static constexpr index_t kNumNIter         = 8;
    static constexpr index_t kNumMsb           = 4;
    static constexpr index_t kNumSu            = 4;
    static constexpr index_t kPairsPerFragment = 4;
    static constexpr index_t kElementsPerPair  = 2;
    static constexpr index_t kElementsPerWmma  = kPairsPerFragment * kElementsPerPair;
    static constexpr index_t kPairsPerMsb      = kNumSu * kPairsPerFragment;
    static constexpr index_t kElementsPerMsb   = kPairsPerMsb * kElementsPerPair;
    static constexpr index_t kThreadBufferSize = kNumMsb * kElementsPerMsb;
    static constexpr index_t kNIterPerSu       = 2;

    static_assert(kNumMIter * kNumNIter * kElementsPerWmma == kThreadBufferSize);
    static_assert(kNumSu * kNIterPerSu == kNumNIter);

    CK_TILE_HOST_DEVICE static constexpr index_t GetMIter(index_t msb) { return msb / 2; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetNIterInSu(index_t msb) { return msb % 2; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetFullNIter(index_t msb, index_t su)
    {
        return su * kNIterPerSu + GetNIterInSu(msb);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t
    GetThreadBufferOffset(index_t msb, index_t pair, index_t element)
    {
        const index_t su               = pair / kPairsPerFragment;
        const index_t pair_in_fragment = pair % kPairsPerFragment;
        const index_t fragment         = GetMIter(msb) * kNumNIter + GetFullNIter(msb, su);
        return fragment * kElementsPerWmma + pair_in_fragment * kElementsPerPair + element;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetPairBufferOffset(index_t msb, index_t pair)
    {
        return GetThreadBufferOffset(msb, pair, 0) / kElementsPerPair;
    }

    template <index_t Msb, index_t PairIndex, typename ScoreTensor>
    CK_TILE_HOST_DEVICE static constexpr Pair LoadPair(const ScoreTensor& score)
    {
        static_assert(Msb >= 0 && Msb < kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < kPairsPerMsb);
        static_assert(std::is_same_v<remove_cvref_t<typename ScoreTensor::DataType>, float>);
        static_assert(ScoreTensor::get_thread_buffer_size() == kThreadBufferSize);

        constexpr index_t pair_offset = GetPairBufferOffset(Msb, PairIndex);
        return score.get_thread_buffer().template get_as<Pair>(number<pair_offset>{});
    }

    template <index_t Msb, index_t PairIndex, typename ScoreTensor>
    CK_TILE_HOST_DEVICE static constexpr void StorePair(ScoreTensor& score, const Pair& value)
    {
        static_assert(Msb >= 0 && Msb < kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < kPairsPerMsb);
        static_assert(std::is_same_v<remove_cvref_t<typename ScoreTensor::DataType>, float>);
        static_assert(ScoreTensor::get_thread_buffer_size() == kThreadBufferSize);

        constexpr index_t pair_offset = GetPairBufferOffset(Msb, PairIndex);
        score.get_thread_buffer().template set_as<Pair>(number<pair_offset>{}, value);
    }

    CK_TILE_HOST_DEVICE static constexpr Coordinate DecodeThreadBufferOffset(index_t offset)
    {
        const index_t fragment            = offset / kElementsPerWmma;
        const index_t element_in_fragment = offset % kElementsPerWmma;
        const index_t m_iter              = fragment / kNumNIter;
        const index_t full_n_iter         = fragment % kNumNIter;
        const index_t su                  = full_n_iter / kNIterPerSu;
        const index_t n_iter_in_su        = full_n_iter % kNIterPerSu;
        const index_t pair_in_fragment    = element_in_fragment / kElementsPerPair;

        return Coordinate{m_iter * 2 + n_iter_in_su,
                          su,
                          su * kPairsPerFragment + pair_in_fragment,
                          element_in_fragment % kElementsPerPair};
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsValidCoordinate(const Coordinate& coordinate)
    {
        return coordinate.msb >= 0 && coordinate.msb < kNumMsb && coordinate.su >= 0 &&
               coordinate.su < kNumSu && coordinate.pair >= 0 && coordinate.pair < kPairsPerMsb &&
               coordinate.element >= 0 && coordinate.element < kElementsPerPair &&
               coordinate.su == coordinate.pair / kPairsPerFragment;
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidateBijection()
    {
        bool seen[kThreadBufferSize] = {};

        for(index_t msb = 0; msb < kNumMsb; ++msb)
        {
            for(index_t pair = 0; pair < kPairsPerMsb; ++pair)
            {
                for(index_t element = 0; element < kElementsPerPair; ++element)
                {
                    const index_t offset      = GetThreadBufferOffset(msb, pair, element);
                    const index_t pair_offset = GetPairBufferOffset(msb, pair);
                    if(offset < 0 || offset >= kThreadBufferSize || seen[offset])
                    {
                        return false;
                    }

                    if(offset / kElementsPerPair != pair_offset)
                    {
                        return false;
                    }

                    seen[offset]                = true;
                    const Coordinate coordinate = DecodeThreadBufferOffset(offset);
                    if(!IsValidCoordinate(coordinate) || coordinate.msb != msb ||
                       coordinate.pair != pair || coordinate.element != element)
                    {
                        return false;
                    }
                }
            }
        }

        for(bool visited : seen)
        {
            if(!visited)
            {
                return false;
            }
        }

        return true;
    }
};

static_assert(FmhaD192ScoreFragmentMapping::ValidateBijection());

struct FmhaD192Scale
{
    float attention_scale;
    float scale_log2;

    CK_TILE_HOST_DEVICE static constexpr FmhaD192Scale FromKernelScale(float scale_s)
    {
#if CK_TILE_FMHA_FWD_FAST_EXP2
        return FmhaD192Scale{scale_s / log2e_v<float>, scale_s};
#else
        return FmhaD192Scale{scale_s, scale_s * log2e_v<float>};
#endif
    }
};

struct FmhaD192ScoreFragments
{
    using Mapping = FmhaD192ScoreFragmentMapping;
    using Pair    = typename Mapping::Pair;
    using Storage = array<array<Pair, Mapping::kPairsPerMsb>, Mapping::kNumMsb>;

    Storage pairs{};

    template <index_t Msb, index_t PairIndex>
    CK_TILE_HOST_DEVICE constexpr Pair& Get()
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < Mapping::kPairsPerMsb);
        return pairs[number<Msb>{}][number<PairIndex>{}];
    }

    template <index_t Msb, index_t PairIndex>
    CK_TILE_HOST_DEVICE constexpr Pair Get() const
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < Mapping::kPairsPerMsb);
        return pairs[number<Msb>{}][number<PairIndex>{}];
    }

    template <typename ScoreTensor>
    CK_TILE_HOST_DEVICE static constexpr FmhaD192ScoreFragments Load(const ScoreTensor& score)
    {
        FmhaD192ScoreFragments result{};
        static_ford<sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}([&](auto indices) {
            constexpr index_t msb            = indices[number<0>{}];
            constexpr index_t pair           = indices[number<1>{}];
            result.template Get<msb, pair>() = Mapping::template LoadPair<msb, pair>(score);
        });
        return result;
    }

    template <typename ScoreTensor>
    CK_TILE_HOST_DEVICE constexpr void Store(ScoreTensor& score) const
    {
        static_ford<sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}([&](auto indices) {
            constexpr index_t msb  = indices[number<0>{}];
            constexpr index_t pair = indices[number<1>{}];
            Mapping::template StorePair<msb, pair>(score, Get<msb, pair>());
        });
    }
};

struct FmhaD192ProbabilityFragments
{
    using Mapping   = FmhaD192ScoreFragmentMapping;
    using Pair      = bf16x2_t;
    using PvOperand = array<Pair, 2 * Mapping::kPairsPerFragment>;
    using Storage   = array<array<Pair, Mapping::kPairsPerMsb>, Mapping::kNumMsb>;

    struct SourceCoordinate
    {
        index_t msb;
        index_t pair;
        index_t element;
    };

    Storage pairs{};

    template <index_t Msb, index_t PairIndex>
    CK_TILE_HOST_DEVICE constexpr Pair& Get()
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < Mapping::kPairsPerMsb);
        return pairs[number<Msb>{}][number<PairIndex>{}];
    }

    template <index_t Msb, index_t PairIndex>
    CK_TILE_HOST_DEVICE constexpr Pair Get() const
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < Mapping::kPairsPerMsb);
        return pairs[number<Msb>{}][number<PairIndex>{}];
    }

    template <index_t Su, index_t MHalf>
    CK_TILE_HOST_DEVICE constexpr PvOperand MakePvOperand() const
    {
        static_assert(Su >= 0 && Su < Mapping::kNumSu);
        static_assert(MHalf >= 0 && MHalf < 2);
        PvOperand result{};

        static_for<0, 2 * Mapping::kPairsPerFragment, 1>{}([&](auto i) {
            constexpr index_t sibling = decltype(i)::value / Mapping::kPairsPerFragment;
            constexpr index_t pair =
                Su * Mapping::kPairsPerFragment + decltype(i)::value % Mapping::kPairsPerFragment;
            result[number<decltype(i)::value>{}] = Get<2 * MHalf + sibling, pair>();
        });
        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr SourceCoordinate
    GetPvSourceCoordinate(index_t su, index_t m_half, index_t element)
    {
        const index_t sibling = element / 8;
        const index_t in_msb  = element % 8;
        return SourceCoordinate{
            2 * m_half + sibling, su * Mapping::kPairsPerFragment + in_msb / 2, in_msb % 2};
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidatePvOperandMapping()
    {
        bool seen[Mapping::kThreadBufferSize] = {};
        for(index_t su = 0; su < Mapping::kNumSu; ++su)
        {
            for(index_t m_half = 0; m_half < 2; ++m_half)
            {
                for(index_t element = 0; element < 16; ++element)
                {
                    const auto source    = GetPvSourceCoordinate(su, m_half, element);
                    const index_t offset = source.msb * Mapping::kElementsPerMsb +
                                           source.pair * Mapping::kElementsPerPair + source.element;
                    if(source.msb < 0 || source.msb >= Mapping::kNumMsb || source.pair < 0 ||
                       source.pair >= Mapping::kPairsPerMsb || source.element < 0 ||
                       source.element >= Mapping::kElementsPerPair || seen[offset])
                    {
                        return false;
                    }
                    seen[offset] = true;
                }
            }
        }

        for(bool value : seen)
        {
            if(!value)
            {
                return false;
            }
        }
        return true;
    }
};

struct FmhaD192SoftmaxState
{
    using Mapping = FmhaD192ScoreFragmentMapping;

    FmhaD192ScoreFragments score{};
    array<float, Mapping::kNumMsb> old_max{};
    array<float, Mapping::kNumMsb> local_max{};
    array<float, Mapping::kNumMsb> logical_row_max{};
    array<float, Mapping::kNumMsb> delta{};
    array<float, Mapping::kNumMsb> exp_delta{};
    array<float, Mapping::kNumMsb> row_sum{};
};

using FmhaD192PreviousSoftmaxState = FmhaD192SoftmaxState;
using FmhaD192CurrentSoftmaxState  = FmhaD192SoftmaxState;

struct FmhaD192SoftmaxClosureState
{
    using Mapping = FmhaD192ScoreFragmentMapping;
    using Pair    = typename Mapping::Pair;

    struct PerMsb
    {
        array<float, Mapping::kNumSu> group_max{};
        float permute_input{};
        float permuted_max{};
        float pre_max_log2e_scl{};

        float exponent_max{};
        Pair scale_log2_pair{};
        float scaled_max_0{};
        float scaled_max_1{};
        float scaled_max_scalar{};
        Pair scaled_max_pair{};
        float exp_delta_copy{};

        array<Pair, Mapping::kPairsPerMsb / 2> sum_l0{};
        array<Pair, Mapping::kPairsPerMsb / 4> sum_l1{};
        array<Pair, Mapping::kPairsPerMsb / 8> sum_l2{};
        Pair final_pair{};
        float final_sum{};
    };

    array<PerMsb, Mapping::kNumMsb> msb{};
};

struct FmhaD192OneTileFlush
{
    static constexpr bool kHasNextTile = false;
};

struct FmhaD192MultiTileFirstSteady
{
    static constexpr bool kHasNextTile = true;
};

struct FmhaD192CrossTileBackEdge
{
    template <bool HasNextTile, typename OneTileConsumer, typename MultiTileConsumer>
    CK_TILE_DEVICE static void Transfer(FmhaD192SoftmaxState& state,
                                        FmhaD192SoftmaxClosureState& closure,
                                        OneTileConsumer& consume_one_tile,
                                        MultiTileConsumer& consume_multi_tile)
    {
        if constexpr(HasNextTile)
        {
            consume_multi_tile(FmhaD192MultiTileFirstSteady{}, state, closure);
        }
        else
        {
            consume_one_tile(FmhaD192OneTileFlush{}, state, closure);
        }
    }
};

static_assert(sizeof(FmhaD192ScoreFragments) == 128 * sizeof(float));
static_assert(sizeof(FmhaD192ProbabilityFragments) == 128 * sizeof(bf16_t));
static_assert(FmhaD192ProbabilityFragments::ValidatePvOperandMapping());

} // namespace ck_tile
