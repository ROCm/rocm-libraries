// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
#include <hip/hip_fp16.h>
#endif
#include <type_traits>

namespace hip_kernel_provider::batchnorm
{

template <typename IndexType>
struct StaticUnrollImpl
{
    struct Swallow
    {
        template <typename... Ts>
        __forceinline__ __host__ __device__ constexpr Swallow(Ts&&... /*unused*/)
        {
        }
    };

    template <IndexType... Is>
    struct Sequence
    {
    };

    template <typename Seq, typename... Seqs>
    struct SequenceMerge
    {
        using type = typename SequenceMerge<Seq, typename SequenceMerge<Seqs...>::type>::type;
    };

    template <IndexType... Xs, IndexType... Ys>
    struct SequenceMerge<Sequence<Xs...>, Sequence<Ys...>>
    {
        using type = Sequence<Xs..., Ys...>;
    };

    template <typename Seq>
    struct SequenceMerge<Seq>
    {
        using type = Seq;
    };

    template <IndexType NSize, typename F>
    struct SequenceGen
    {
        template <IndexType IBegin, IndexType NRemain, typename G>
        struct SequenceGenImpl
        {
            static constexpr IndexType N_REMAIN_LEFT = NRemain / 2;
            static constexpr IndexType N_REMAIN_RIGHT = NRemain - N_REMAIN_LEFT;
            static constexpr IndexType I_MIDDLE = IBegin + N_REMAIN_LEFT;

            using type = typename SequenceMerge<
                typename SequenceGenImpl<IBegin, N_REMAIN_LEFT, G>::type,
                typename SequenceGenImpl<I_MIDDLE, N_REMAIN_RIGHT, G>::type>::type;
        };

        template <IndexType I, typename G>
        struct SequenceGenImpl<I, 1, G>
        {
            using type = Sequence<G{}(I)>;
        };

        template <IndexType I, typename G>
        struct SequenceGenImpl<I, 0, G>
        {
            using type = Sequence<>;
        };

        using type = typename SequenceGenImpl<0, NSize, F>::type;
    };

    // arithmetic sequence
    template <IndexType IBegin, IndexType IEnd, IndexType Increment>
    struct ArithmeticSequenceGen
    {
        struct F
        {
            constexpr IndexType operator()(IndexType i) const
            {
                return i * Increment + IBegin;
            }
        };

        using type0 = typename SequenceGen<(IEnd - IBegin) / Increment, F>::type;
        using type1 = Sequence<>;

        static constexpr bool K_HAS_CONTENT
            = (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

        using type = typename std::conditional_t<K_HAS_CONTENT, type0, type1>;
    };

    template <class>
    struct StaticForImpl;

    template <IndexType... Is>
    struct StaticForImpl<Sequence<Is...>>
    {
        template <class F>
        __forceinline__ __host__ __device__ constexpr void operator()(F f) const
        {
            Swallow{(f(Is), 0)...};
        }
    };

    template <IndexType NBegin, IndexType NEnd, IndexType Increment>
    struct StaticFor
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "Wrong! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");

        template <class F>
        __forceinline__ __host__ __device__ constexpr void operator()(F f) const
        {
            StaticForImpl<typename ArithmeticSequenceGen<NBegin, NEnd, Increment>::type>{}(f);
        }
    };
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride>
struct StaticNounroll
{
    template <typename Func>
    __forceinline__ __host__ __device__ constexpr StaticNounroll(
        Func&& f) // NOLINT(bugprone-forwarding-reference-overload)
    {
        ItemType i = Start;
        while(i < End)
        {
            f(i);
            i += Stride;
        }
    }
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride>
struct StaticUnrollFull
{
    static constexpr ItemType ACTUAL_END
        = (End - Start) % Stride == 0 ? End : ((End - Start) / Stride + 1) * Stride;
    template <typename F>
    __forceinline__ __host__ __device__ constexpr StaticUnrollFull(F f)
    {
        typename StaticUnrollImpl<ItemType>::template StaticFor<Start, ACTUAL_END, Stride>{}(f);
    }
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride, ItemType Hint>
struct StaticUnrollCount
{
    static_assert(Hint > 0, "Hint must be a positive integer.");
    static constexpr ItemType UNROLL_END = Start + (Stride * Hint);

    template <typename F>
    __forceinline__ __host__ __device__ constexpr StaticUnrollCount(F f)
    {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if constexpr(Hint == 1 || (End - Start) <= Hint)
        {
            StaticNounroll<ItemType, Start, End, Stride>{f};
        }
        else
        {
            StaticUnrollFull<ItemType, Start, UNROLL_END, Stride>{f};
            StaticNounroll<ItemType, UNROLL_END, End, Stride>{f};
        }
    }
};

} // namespace hip_kernel_provider::batchnorm
