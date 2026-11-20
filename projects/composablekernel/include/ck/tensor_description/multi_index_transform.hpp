// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/multi_index.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {

template <typename LowLength>
struct PassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr PassThrough() = default;

    __host__ __device__ constexpr PassThrough(const LowLength& low_length)
        : up_lengths_{make_tuple(low_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("PassThrough, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLength,
          typename LeftPadLength,
          typename RightPadLength,
          bool SkipIsValidCheck = false>
struct Pad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr Pad() = default;

    __host__ __device__ constexpr Pad(const LowLength& low_length,
                                      const LeftPadLength& left_pad_length,
                                      const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length + right_pad_length)},
          left_pad_length_{left_pad_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              [[maybe_unused]] const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              [[maybe_unused]] const UpIdx& idx_up,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck ||
               ((idx_up[Number<0>{}] >= left_pad_length_) &&
                (idx_up[Number<0>{}] < up_lengths_[Number<0>{}] - right_pad_length_));
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Pad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_length %d", index_t{left_pad_length_});
        printf("right_pad_length %d", index_t{right_pad_length_});
        printf("}");
    }
};

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
struct LeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;

    __host__ __device__ constexpr LeftPad() = default;

    __host__ __device__ constexpr LeftPad(const LowLength& low_length,
                                          const LeftPadLength& left_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length)}, left_pad_length_{left_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              [[maybe_unused]] const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              [[maybe_unused]] const UpIdx& idx_up,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] >= left_pad_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("LeftPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_length_ %d", index_t{left_pad_length_});
        printf("}");
    }
};

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
struct RightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LowLength low_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr RightPad() = default;

    __host__ __device__ constexpr RightPad(const LowLength& low_length,
                                           const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + right_pad_length)},
          low_length_{low_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] < low_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LowLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("RightPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("low_length_ %d", index_t{low_length_});
        printf("left_pad_length_ %d", index_t{right_pad_length_});
        printf("}");
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
// UpLengths and Coefficients can be either of the followings:
//   1) Tuple of index_t, which is known at run-time, or
//   2) Tuple of Number, which is known at compile-time, or
//   3) Tuple of mixture of index_t and Number, which is known partially at run-time and partially
//   at compile-time
template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct Embed
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    __host__ __device__ constexpr Embed() = default;

    __host__ __device__ constexpr Embed(const UpLengths& up_lengths,
                                        const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Embed, ");
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("coefficients_ ");
        print_multi_index(coefficients_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses regular to do lowering of
// multi-index and use carry-and-borrow check to do lowering of multi-index delta
template <typename LowLengths>
struct Merge_v1_carry_check
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v1_carry_check() = default;

    __host__ __device__ constexpr Merge_v1_carry_check(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        auto tmp = idx_up[Number<0>{}];

        // normal division
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1a(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] + idx_diff_low_const[i]);
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto carry = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto borrow = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto idx_low_tmp = idx_low[i] - borrow;

                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto carry = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1b(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto carry = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto borrow = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto negative_idx_low_tmp = borrow - idx_low[i];

                bool do_borrow = negative_idx_low_tmp > idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            auto carry = decltype(idx_low[Number<0>{}]){0};

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                auto idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_2(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx& /* idx_up_new */,
                                                Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        //   computed at run-time each time this function is called, and can be
        //   very expensive.
        LowerIndex idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        auto tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] + do_carry;

                auto idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_carry = idx_low_tmp >= low_lengths_[i];

#if 0
                // TODO: use exec-mask inline asm, which use 1 VALU
                if(do_carry)
                {
                    idx_diff_low(i) -= low_lengths_[i];
                }
#elif 1
                // this use 2 VALU
                idx_diff_low(i) = do_carry ? idx_diff_low[i] - low_lengths_[i] : idx_diff_low[i];
#elif 1
                // this use 2 VALU
                auto idx_diff_low_tmp = idx_diff_low[i] - low_lengths_[i];
                idx_diff_low(i)       = do_carry ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] + do_carry;

            idx_low(I0) += idx_diff_low[I0];
        }
        else if constexpr(Hack == 2)
        {
            // do borrow check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] - do_borrow;

                auto idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_borrow = idx_low_tmp < 0;

#if 0
                // TODO: use exec-mask inline asm
                if(do_borrow)
                {
                    idx_diff_low(i) += low_lengths_[i];
                }
#elif 1
                idx_diff_low(i) = do_borrow ? idx_diff_low[i] + low_lengths_[i] : idx_diff_low[i];
#elif 1
                auto idx_diff_low_tmp = idx_diff_low[i] + low_lengths_[i];
                idx_diff_low(i)       = do_borrow ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] - do_borrow;

            idx_low(I0) += idx_diff_low[I0];
        }
        else
        {
            // not implemented
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
#if 1
        UpdateLowerIndex_1a(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#elif 0
        UpdateLowerIndex_1b(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#else
        UpdateLowerIndex_2(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#endif
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v1_carry_check, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_multiplier
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return MagicDivision::CalculateMagicMultiplier(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_shift
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return MagicDivision::CalculateMagicShift(LowLengths{}[i]);
    }
};

// Implementation of "Merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For Merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct Merge_v2_magic_division
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    using LowLengthsMagicDivisorMultipiler = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_multiplier<LowLengths>{},
        Number<NDimLow>{}));

    using LowLengthsMagicDivisorShift = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_shift<LowLengths>{},
        Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsMagicDivisorMultipiler low_lengths_magic_divisor_multiplier_;
    LowLengthsMagicDivisorShift low_lengths_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v2_magic_division() = default;

    __host__ __device__ constexpr Merge_v2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicMultiplier(low_lengths[i]); },
              Number<NDimLow>{})},
          low_lengths_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicShift(low_lengths[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_magic_divisor_multiplier_[i],
                                               this->low_lengths_magic_divisor_shift_[i]);
            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;
        });

        idx_low(Number<0>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up_new[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_magic_divisor_multiplier_[i],
                                               this->low_lengths_magic_divisor_shift_[i]);

            index_t idx_low_old = idx_low[i];

            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;

            idx_diff_low(i) = idx_low[i] - idx_low_old;
        });

        idx_diff_low(Number<0>{}) = tmp - idx_low(Number<0>{});

        idx_low(Number<0>{}) = tmp;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorMultipiler>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorShift>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v2_magic_division, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_magic_divisor_multiplier_ ");
        print_multi_index(low_lengths_magic_divisor_multiplier_);
        printf("low_lengths_magic_divisor_shift_ ");
        print_multi_index(low_lengths_magic_divisor_shift_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For Merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct Merge_v2r2_magic_division
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    using LowLengthsScanMagicDivisorMultipiler = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_multiplier<LowLengthsScan>{},
        Number<NDimLow>{}));

    using LowLengthsScanMagicDivisorShift = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_shift<LowLengthsScan>{},
        Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    LowLengthsScanMagicDivisorMultipiler low_lengths_scan_magic_divisor_multiplier_;
    LowLengthsScanMagicDivisorShift low_lengths_scan_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v2r2_magic_division() = default;

    __host__ __device__ constexpr Merge_v2r2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          low_lengths_scan_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicMultiplier(low_lengths_scan_[i]); },
              Number<NDimLow>{})},
          low_lengths_scan_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicShift(low_lengths_scan_[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        auto tmp = idx_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&, this](auto i) {
            idx_low(i) =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_scan_magic_divisor_multiplier_[i],
                                               this->low_lengths_scan_magic_divisor_shift_[i]);

            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        auto tmp = idx_up_new[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&, this](auto i) {
            auto idx_low_old = idx_low[i];

            idx_low(i) =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_scan_magic_divisor_multiplier_[i],
                                               this->low_lengths_scan_magic_divisor_shift_[i]);

            idx_diff_low(i) = idx_low[i] - idx_low_old;

            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_diff_low(Number<NDimLow - 1>{}) = tmp - idx_low[Number<NDimLow - 1>{}];

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScanMagicDivisorMultipiler>::value &&
               is_known_at_compile_time<LowLengthsScanMagicDivisorShift>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v2r2_magic_division, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan ");
        print_multi_index(low_lengths_scan_);
        printf("low_lengths_scan_magic_divisor_multiplier_ ");
        print_multi_index(low_lengths_scan_magic_divisor_multiplier_);
        printf("low_lengths_scan_magic_divisor_shift_ ");
        print_multi_index(low_lengths_scan_magic_divisor_shift_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses division and mod. It is supposed to
// be used for low_lengths that are known at compile time and are power of 2, otherwise performance
// will be very bad
template <typename LowLengths>
struct Merge_v3_division_mod
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v3_division_mod() = default;

    __host__ __device__ constexpr Merge_v3_division_mod(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        auto tmp = idx_up[Number<0>{}];

        // division and mod
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp %= this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0   = Number<0>{};
        constexpr auto INm1 = Number<NDimLow - 1>{};

        auto tmp = idx_up_new[I0];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            const auto tmp2 = idx_low[i];
            idx_low(i)      = tmp / this->low_lengths_scan_[i];
            idx_diff_low(i) = idx_low[i] - tmp2;
            tmp %= this->low_lengths_scan_[i];
        });

        const auto tmp2    = idx_low[INm1];
        idx_low(INm1)      = tmp;
        idx_diff_low(INm1) = idx_low[INm1] - tmp2;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v3_direct_division_mod, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename UpLengths, bool Use24BitIntegerCalculation>
struct UnMerge
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    using UpLengthsScan =
        decltype(container_reverse_exclusive_scan(UpLengths{}, math::multiplies{}, Number<1>{}));

    UpLengths up_lengths_;
    UpLengthsScan up_lengths_scan_;

    __host__ __device__ constexpr UnMerge() = default;

    __host__ __device__ constexpr UnMerge(const UpLengths& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              container_reverse_exclusive_scan(up_lengths, math::multiplies{}, Number<1>{})}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        if constexpr(!Use24BitIntegerCalculation)
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}(
                [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
        }
        else
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}([&](auto i) {
                idx_low(Number<0>{}) =
                    (0x00ffffff & idx_low[Number<0>{}]) +
                    (0x00ffffff & idx_up[i]) * (0x00ffffff & up_lengths_scan_[i]);
            });
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        CalculateLowerIndex(idx_diff_low, idx_diff_up);

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<UpLengthsScan>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("UnMerge, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("up_lengths_scan_");
        print_multi_index(up_lengths_scan_);
        printf("}");
    }
};

/**
 * @brief Transformation struct for convolution backward data output indices to GEMM indices.
 *
 * This struct is responsible for mapping the output tensor indices (N, Ho, Wo, K) from the
 * convolution backward data operation to the corresponding indices (K0, M, K1) used in the
 * implicit GEMM computation. It encapsulates the necessary parameters and transformation logic
 * required to efficiently perform the index conversion.
 *
 * MergePadding = false: the lower dimensions are the Ho / Wo padded output tensor (outer Pad
 * transforms report Ho / Wo out of range) and this transform only reports the GEMM rows / columns
 * beyond the real M and K as invalid.
 *
 * MergePadding = true: the lower dimensions are the output tensor itself and no separate Pad
 * transform is needed. This transform reports as invalid the output rows that fall outside
 * [0, Ho) x [0, Wo) and the GEMM rows / columns beyond the real M and K. The latter are made
 * invalid by adding a large value to the Ho coordinate (the M part to the term that only depends
 * on m, the K part to the term that only depends on k), so that the unsigned range check fails.
 * A poisoned coordinate must never wrap back into [0, Ho), which holds when Ho is at most 2^29
 * and the Ho coordinate of every valid row stays above -2^29.
 *
 * Indices must be non-negative and below 2^31 (magic division limitation).
 */
template <bool MergePadding = false>
struct ConvBwdDataImplicitGemmOutTransform
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using LowerIndex = MultiIndex<4>; // N, Ho, Wo, K
    using UpperIndex = MultiIndex<3>; // K0, M, K1

    index_t N_, Ho_, Wo_, K_;
    index_t XDot_;
    index_t HTilde_, WTilde_;
    index_t WTildeSlice_, TildeSlice_;
    index_t IHTildeSliceBegin_, IWTildeSliceBegin_;
    index_t HRatio_, WRatio_;
    index_t XDotSlice_K_;
    index_t MPad_, KPad_;
    Tuple<index_t, index_t, index_t> up_lengths_; // K0_, MPadded, K1_;

    static constexpr uint32_t kPoison = 1u << 30;

    Tuple<index_t, index_t, index_t, index_t>
        low_lengths_magic_divisor_multiplier_; // XDotSlice_K_, K_, TildeSlice_, WTildeSlice_
    Tuple<index_t, index_t, index_t, index_t>
        low_lengths_magic_divisor_shift_; // XDotSlice_K_, K_, TildeSlice_, WTildeSlice_

    __host__ __device__ ConvBwdDataImplicitGemmOutTransform() = default;

    __host__ __device__ constexpr ConvBwdDataImplicitGemmOutTransform(index_t N,
                                                                      index_t Ho,
                                                                      index_t Wo,
                                                                      index_t K,
                                                                      index_t XDot,
                                                                      index_t HTilde,
                                                                      index_t WTilde,
                                                                      index_t WTildeSlice,
                                                                      index_t HWTildeSlice,
                                                                      index_t IHTildeSliceBegin,
                                                                      index_t IWTildeSliceBegin,
                                                                      index_t HRatio,
                                                                      index_t WRatio,
                                                                      index_t XDotSlice_K,
                                                                      index_t K0,
                                                                      index_t MPadded,
                                                                      index_t K1,
                                                                      index_t MPad,
                                                                      index_t KPad)
        : N_{N},
          Ho_{Ho},
          Wo_{Wo},
          K_{K},
          XDot_{XDot},
          HTilde_{HTilde},
          WTilde_{WTilde},
          WTildeSlice_{WTildeSlice},
          TildeSlice_{HWTildeSlice},
          IHTildeSliceBegin_{IHTildeSliceBegin},
          IWTildeSliceBegin_{IWTildeSliceBegin},
          HRatio_{HRatio},
          WRatio_{WRatio},
          XDotSlice_K_{XDotSlice_K},
          MPad_{MPad},
          KPad_{KPad},
          up_lengths_{make_tuple(K0, MPadded, K1)},
          low_lengths_magic_divisor_multiplier_{
              MagicDivision::CalculateMagicMultiplier(XDotSlice_K_),
              MagicDivision::CalculateMagicMultiplier(K_),
              MagicDivision::CalculateMagicMultiplier(TildeSlice_),
              MagicDivision::CalculateMagicMultiplier(WTildeSlice_)},
          low_lengths_magic_divisor_shift_{MagicDivision::CalculateMagicShift(XDotSlice_K_),
                                           MagicDivision::CalculateMagicShift(K_),
                                           MagicDivision::CalculateMagicShift(TildeSlice_),
                                           MagicDivision::CalculateMagicShift(WTildeSlice_)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 4; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 3; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr auto CalculateLowerIndexN(const UpIdx& idx_up) const
    {
        index_t NStep{0}, HStep{0}, WStep{0};
        // Merge
        // NStep = M_id / TildeSlice_
        NStep = MagicDivision::DoMagicDivision(idx_up[I1],
                                               this->low_lengths_magic_divisor_multiplier_[I2],
                                               this->low_lengths_magic_divisor_shift_[I2]);
        HStep = idx_up[I1] - NStep * TildeSlice_;
        // HStep = HStep / WTildeSlice_
        HStep = MagicDivision::DoMagicDivision(HStep,
                                               this->low_lengths_magic_divisor_multiplier_[I3],
                                               this->low_lengths_magic_divisor_shift_[I3]);
        WStep = idx_up[I1] - NStep * TildeSlice_ - HStep * WTildeSlice_;
        // Slice
        HStep += IHTildeSliceBegin_;
        WStep += IWTildeSliceBegin_;

        return make_tuple(NStep, HStep, WStep, 0);
    }

    template <typename UpIdx>
    __host__ __device__ constexpr auto CalculateLowerIndexK(const UpIdx& idx_up) const
    {
        // UnMerge
        //  K_idx <- K0_idx * K1 + K1_idx
        index_t K_idx = idx_up[I0] * up_lengths_[I2] + idx_up[I2];
        // Merge
        // YStep = K_idx / XDotSlice_K_
        index_t YStep =
            MagicDivision::DoMagicDivision(K_idx,
                                           this->low_lengths_magic_divisor_multiplier_[I0],
                                           this->low_lengths_magic_divisor_shift_[I0]);
        index_t KStep = K_idx - YStep * XDotSlice_K_;
        // Xstep = KStep / K_
        index_t XStep =
            MagicDivision::DoMagicDivision(KStep,
                                           this->low_lengths_magic_divisor_multiplier_[I1],
                                           this->low_lengths_magic_divisor_shift_[I1]);
        KStep -= XStep * K_;
        // Embed
        YStep *= HRatio_;
        XStep *= WRatio_;

        return make_tuple(0, YStep, XStep, KStep);
    }

    // MergePadding: the lower index of a padding row / column is made invalid by adding a large
    // value to the Ho coordinate. The M and K parts are kept in separate terms (unsigned
    // arithmetic, no signed overflow) so that each can be hoisted independently.
    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateMergedLowerIndex(LowIdx& idx_low,
                                                                 const UpIdx& idx_up) const
    {
        const auto n_part = CalculateLowerIndexN(idx_up);
        const auto k_part = CalculateLowerIndexK(idx_up);

        const index_t K_idx = idx_up[I0] * up_lengths_[I2] + idx_up[I2];
        const uint32_t m_poison =
            static_cast<uint32_t>(idx_up[I1]) >= static_cast<uint32_t>(up_lengths_[I1] - MPad_)
                ? kPoison
                : 0u;
        const uint32_t k_poison =
            static_cast<uint32_t>(K_idx) >=
                    static_cast<uint32_t>(up_lengths_[I0] * up_lengths_[I2] - KPad_)
                ? kPoison
                : 0u;

        idx_low(I0) = n_part[I0];
        idx_low(I1) = static_cast<index_t>(static_cast<uint32_t>(n_part[I1]) + m_poison +
                                           static_cast<uint32_t>(k_part[I1]) + k_poison);
        idx_low(I2) = n_part[I2] + k_part[I2];
        idx_low(I3) = k_part[I3];
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        if constexpr(MergePadding)
        {
            CalculateMergedLowerIndex(idx_low, idx_up);
        }
        else
        {
            idx_low = CalculateLowerIndexN(idx_up) + CalculateLowerIndexK(idx_up);
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& /* idx_diff_up */,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up,
                                              Number<Hack>) const
    {
        LowIdx low_old = idx_low;
        CalculateLowerIndex(idx_low, idx_up);
        idx_diff_low = idx_low - low_old;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return false;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        if constexpr(MergePadding)
        {
            LowerIndex idx_low;
            CalculateLowerIndex(idx_low, idx_up);
            // Unsigned compares check both bounds: a negative or poisoned index becomes >= 2^30,
            // larger than any valid length.
            return static_cast<uint32_t>(idx_low[I1]) < static_cast<uint32_t>(Ho_) &&
                   static_cast<uint32_t>(idx_low[I2]) < static_cast<uint32_t>(Wo_);
        }
        else
        {
            // Padding
            index_t K_idx = idx_up[Number<0>{}] * up_lengths_[Number<2>{}] + idx_up[Number<2>{}];
            const auto& M_idx = idx_up[Number<1>{}];

            bool pad_valid = M_idx < up_lengths_[Number<1>{}] - MPad_ &&
                             K_idx < up_lengths_[Number<0>{}] * up_lengths_[Number<2>{}] - KPad_;
            return pad_valid;
        }
    }

    // The lane mask replaces the validity of the whole descriptor, so it is only available when
    // the Ho / Wo bounds are part of this transform.
    static constexpr bool HasValidLaneMask = MergePadding;

    // Lane mask version of the validity check (wave32, gfx125): one ballot per dimension so that
    // each ballot argument is a single compare, which is lowered to a v_cmp writing an SGPR lane
    // mask. Must be called by all the lanes of the wave.
    template <typename UpIdx>
    __device__ uint32_t IsValidLaneMask(const UpIdx& idx_up) const
    {
        static_assert(MergePadding, "the lane mask needs the Ho / Wo bounds in the transform");
#if defined(__gfx125__)
        LowerIndex idx_low;
        CalculateLowerIndex(idx_low, idx_up);

        return __builtin_amdgcn_ballot_w32(static_cast<uint32_t>(idx_low[I1]) <
                                           static_cast<uint32_t>(Ho_)) &
               __builtin_amdgcn_ballot_w32(static_cast<uint32_t>(idx_low[I2]) <
                                           static_cast<uint32_t>(Wo_));
#else
        ignore = idx_up;
        return 0;
#endif
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime() { return false; }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("ConvBwdDataImplicitGemmOutTransform, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

/**
 * @brief Map forward-convolution implicit-GEMM input indices to tensor indices.
 *
 * The upper dimensions are the padded GEMM dimensions M_padded >= M and K_padded >= K, with
 * M=(N, O_0..O_{D-1}) and K=(F_0..F_{D-1}, C) flattened. The lower dimensions are the input tensor
 * coordinates (N, I_0..I_{D-1}, C). D is NDimSpatial. Indices must be non-negative and below 2^31
 * (magic division limitation).
 *
 * The rows and columns beyond the real M and K (GEMM padding) are reported as invalid by this
 * transform itself, so a separate right-pad transform (and its compares) is not needed. This is
 * done by pushing the first spatial lower coordinate out of range: the M part of the poison is
 * added to the term that only depends on m and the K part to the term that only depends on k, so
 * each of them can be hoisted or shared independently of the other. The lower index of a padding
 * row / column is therefore meaningless. A poisoned coordinate must never wrap back into
 * [0, length), which holds when the first spatial input length is at most 2^29 and the first
 * spatial lower coordinate of every row stays above -2^29.
 */
template <index_t NDimSpatial>
struct ConvFwdDataImplicitGemmInTransform
{
    static_assert(NDimSpatial >= 1, "NDimSpatial must be positive");

    using LowerIndex  = MultiIndex<NDimSpatial + 2>;
    using UpperIndex  = MultiIndex<2>;
    using UpLengths   = Tuple<index_t, index_t>;
    using SpatialDims = MultiIndex<NDimSpatial>;

    SpatialDims in_lengths_;
    SpatialDims conv_strides_;
    SpatialDims conv_dilations_;
    SpatialDims left_pads_;
    // negated lengths so that remainder = dividend + quotient * negated_length
    SpatialDims neg_out_lengths_;
    SpatialDims neg_filter_lengths_;
    index_t nC_;
    UpLengths up_lengths_;
    // M side: divisors O_{D-1}, ..., O_0 (the last quotient is n)
    // K side: divisors C, F_{D-1}, ..., F_1 (the last quotient is f_0)
    SpatialDims m_magic_multiplier_;
    SpatialDims m_magic_shift_;
    SpatialDims k_magic_multiplier_;
    SpatialDims k_magic_shift_;
    // real (unpadded) GEMM lengths
    index_t m_valid_;
    index_t k_valid_;

    static constexpr uint32_t kPoison = 1u << 30;

    __host__ __device__ constexpr ConvFwdDataImplicitGemmInTransform() = default;

    __host__
        __device__ constexpr ConvFwdDataImplicitGemmInTransform(index_t N,
                                                                index_t C,
                                                                const SpatialDims& in_lengths,
                                                                const SpatialDims& filter_lengths,
                                                                const SpatialDims& out_lengths,
                                                                const SpatialDims& conv_strides,
                                                                const SpatialDims& conv_dilations,
                                                                const SpatialDims& left_pads,
                                                                index_t M_padded,
                                                                index_t K_padded)
        : in_lengths_{in_lengths},
          conv_strides_{conv_strides},
          conv_dilations_{conv_dilations},
          left_pads_{left_pads},
          neg_out_lengths_{},
          neg_filter_lengths_{},
          nC_{-C},
          up_lengths_{},
          m_magic_multiplier_{},
          m_magic_shift_{},
          k_magic_multiplier_{},
          k_magic_shift_{},
          m_valid_{0},
          k_valid_{0}
    {
        index_t m_length = N;
        index_t k_length = C;
        static_for<0, NDimSpatial, 1>{}([&](auto d) {
            neg_out_lengths_(d)    = -out_lengths[d];
            neg_filter_lengths_(d) = -filter_lengths[d];
            m_length *= out_lengths[d];
            k_length *= filter_lengths[d];

            constexpr index_t rev = NDimSpatial - 1 - d;
            m_magic_multiplier_(d) =
                MagicDivision::CalculateMagicMultiplier(out_lengths[Number<rev>{}]);
            m_magic_shift_(d) = MagicDivision::CalculateMagicShift(out_lengths[Number<rev>{}]);
        });
        k_magic_multiplier_(Number<0>{}) = MagicDivision::CalculateMagicMultiplier(C);
        k_magic_shift_(Number<0>{})      = MagicDivision::CalculateMagicShift(C);
        static_for<1, NDimSpatial, 1>{}([&](auto j) {
            k_magic_multiplier_(j) =
                MagicDivision::CalculateMagicMultiplier(filter_lengths[Number<NDimSpatial - j>{}]);
            k_magic_shift_(j) =
                MagicDivision::CalculateMagicShift(filter_lengths[Number<NDimSpatial - j>{}]);
        });
        m_valid_    = m_length;
        k_valid_    = k_length;
        up_lengths_ = make_tuple(M_padded, K_padded);
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension()
    {
        return NDimSpatial + 2;
    }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 2; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const [[clang::lifetimebound]]
    {
        return up_lengths_;
    }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimSpatial + 2 && UpIdx::Size() == 2,
                      "wrong! inconsistent # of dimensions");

        // M = (n, o_0, ..., o_{D-1}) -> innermost output dimension first
        index_t quotient = idx_up[Number<0>{}];
        SpatialDims out_idx;
        static_for<0, NDimSpatial, 1>{}([&](auto j) {
            constexpr index_t dim = NDimSpatial - 1 - j;
            const index_t next =
                MagicDivision::DoMagicDivision(quotient, m_magic_multiplier_[j], m_magic_shift_[j]);
            out_idx(Number<dim>{}) = quotient + next * neg_out_lengths_[Number<dim>{}];
            quotient               = next;
        });
        idx_low(Number<0>{}) = quotient;

        // K = (f_0, ..., f_{D-1}, c) -> c first, then innermost filter dimension
        quotient     = idx_up[Number<1>{}];
        index_t next = MagicDivision::DoMagicDivision(
            quotient, k_magic_multiplier_[Number<0>{}], k_magic_shift_[Number<0>{}]);
        idx_low(Number<NDimSpatial + 1>{}) = quotient + next * nC_;
        quotient                           = next;
        SpatialDims filter_idx;
        static_for<1, NDimSpatial, 1>{}([&](auto j) {
            constexpr index_t dim = NDimSpatial - j;
            next =
                MagicDivision::DoMagicDivision(quotient, k_magic_multiplier_[j], k_magic_shift_[j]);
            filter_idx(Number<dim>{}) = quotient + next * neg_filter_lengths_[Number<dim>{}];
            quotient                  = next;
        });
        filter_idx(Number<0>{}) = quotient;

        // Padding rows / columns (m >= M, k >= K) are made invalid by adding a large value to
        // the first spatial coordinate, so that the unsigned range check fails. The M and K
        // contributions are kept in separate terms (unsigned arithmetic, no signed overflow).
        const uint32_t m_poison =
            static_cast<uint32_t>(idx_up[Number<0>{}]) >= static_cast<uint32_t>(m_valid_) ? kPoison
                                                                                          : 0u;
        const uint32_t k_poison =
            static_cast<uint32_t>(idx_up[Number<1>{}]) >= static_cast<uint32_t>(k_valid_) ? kPoison
                                                                                          : 0u;
        static_for<0, NDimSpatial, 1>{}([&](auto d) {
            uint32_t m_part = static_cast<uint32_t>(out_idx[d] * conv_strides_[d] - left_pads_[d]);
            uint32_t k_part = static_cast<uint32_t>(filter_idx[d] * conv_dilations_[d]);
            if constexpr(d == 0)
            {
                m_part += m_poison;
                k_part += k_poison;
            }
            idx_low(Number<d + 1>{}) = static_cast<index_t>(m_part + k_part);
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        const LowIdx old_idx = idx_low;
        CalculateLowerIndex(idx_low, idx_up_new);
        idx_diff_low = idx_low - old_idx;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return false;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        LowerIndex idx_low;
        CalculateLowerIndex(idx_low, idx_up);
        // Indices are signed 32-bit values and lengths are nonnegative. Converting a negative
        // index to uint32_t produces a value >= 2^31, larger than any valid signed length;
        // nonnegative indices retain their value. Each unsigned comparison therefore checks
        // both bounds, eliminating separate sign checks and their instruction dependencies.
        bool valid = true;
        static_for<0, NDimSpatial, 1>{}([&](auto d) {
            valid &= static_cast<uint32_t>(idx_low[Number<d + 1>{}]) <
                     static_cast<uint32_t>(in_lengths_[d]);
        });
        return valid;
    }

    // PROTOTYPE: one ballot per spatial dimension keeps each term a single compare so it is
    // lowered to a v_cmp writing an SGPR lane mask (wave32 / gfx125 only).
    static constexpr bool HasValidLaneMask = true;

    template <typename UpIdx>
    __device__ uint32_t IsValidLaneMask(const UpIdx& idx_up) const
    {
#if defined(__gfx125__)
        LowerIndex idx_low;
        CalculateLowerIndex(idx_low, idx_up);
        uint32_t mask = 0xffffffffu;
        static_for<0, NDimSpatial, 1>{}([&](auto d) {
            mask &= __builtin_amdgcn_ballot_w32(static_cast<uint32_t>(idx_low[Number<d + 1>{}]) <
                                                static_cast<uint32_t>(in_lengths_[d]));
        });
        return mask;
#else
        ignore = idx_up;
        return 0;
#endif
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime() { return false; }

    __host__ __device__ void Print() const
    {
        printf("{ConvFwdDataImplicitGemmInTransform, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowerIndex>
struct Freeze
{
    LowerIndex low_idx_;

    __host__ __device__ constexpr Freeze() = default;

    __host__ __device__ constexpr Freeze(const LowerIndex& low_idx) : low_idx_{low_idx} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Tuple<>{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& /* idx_up */) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 0,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = low_idx_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& /* idx_diff_up */,
                                                     LowIdx& /* idx_low */,
                                                     const UpIdx& /* idx_up_new */,
                                                     Number<Hack>)
    {
        idx_diff_low(Number<0>{}) = 0;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowerIndex>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Freeze");
        printf("low_idx_ %d", index_t{low_idx_});
    }
};

// Insert a dangling upper dimension without lower dimension
template <typename UpperLength>
struct Insert
{
    using UpLengths = decltype(make_tuple(UpperLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr Insert() = default;

    __host__ __device__ constexpr Insert(const UpperLength& up_length)
        : up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 0; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::Size() == 0 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void
    UpdateLowerIndex(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&, Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 0 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 0 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpperLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Insert");
        print_multi_index(up_lengths_);
    }
};

template <typename VectorSize, typename UpLength>
struct Vectorize
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(UpLength{}));

    UpLengths up_lengths_;
    VectorSize vector_size_;

    __host__ __device__ constexpr Vectorize() = default;

    __host__ __device__ constexpr Vectorize(const VectorSize& vector_size,
                                            const UpLength& up_length)
        : vector_size_{vector_size}, up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = vector_size_ * idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = vector_size_ * idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Vectorize, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLength, typename SliceBegin, typename SliceEnd>
struct Slice
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(SliceEnd{} - SliceBegin{}));

    UpLengths up_lengths_;
    SliceBegin slice_begin_;
    SliceEnd slice_end_;

    __host__ __device__ constexpr Slice() = default;

    __host__ __device__ constexpr Slice(const LowLength&,
                                        const SliceBegin& slice_begin,
                                        const SliceEnd& slice_end)
        : up_lengths_{make_tuple(slice_end - slice_begin)},
          slice_begin_{slice_begin},
          slice_end_{slice_end}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] + slice_begin_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx&) const
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<SliceBegin>::value &&
               is_known_at_compile_time<SliceEnd>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Slice, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("slice_begin_ %d", index_t{slice_begin_});
        printf("slice_end %d", index_t{slice_end_});
        printf("}");
    }
};

/*
 * \brief lower_idx = upper_idx % modulus.
 * TODO: Need an improved implementation since the modulo operation is expensive.
 */
template <typename Modulus, typename UpLength>
struct Modulo
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;
    using UpLengths  = decltype(make_tuple(UpLength{}));

    Modulus modulus_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Modulo() = default;

    __host__ __device__ constexpr Modulo(const Modulus& modulus, const UpLength& up_length)
        : modulus_{modulus}, up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] % modulus_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& up_idx,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        const auto idx_low_old = idx_low;
        idx_low(I0)            = (up_idx(I0) + idx_diff_up(I0)) % modulus_;
        idx_diff_low(I0)       = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Modulus, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLengths, bool ApplyModulo>
struct Xor
{
    using LowerIndex = MultiIndex<2>;
    using UpperIndex = MultiIndex<2>;

    using UpLengths = LowLengths;

    UpLengths up_lengths_;

    __host__ __device__ constexpr Xor() : up_lengths_{} {}

    __host__ __device__ constexpr Xor(const LowLengths& low_lengths) : up_lengths_{low_lengths} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 2; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 2; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 2 && UpIdx::Size() == 2,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];

        if constexpr(ApplyModulo)
        {
            idx_low(Number<1>{}) =
                idx_up[Number<1>{}] ^ (idx_up[Number<0>{}] % up_lengths_[Number<1>{}]);
        }
        else
        {
            idx_low(Number<1>{}) = idx_up[Number<1>{}] ^ idx_up[Number<0>{}];
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 2 && UpIdxDiff::Size() == 2 && LowIdx::Size() == 2 &&
                          UpIdx::Size() == 2,
                      "wrong! inconsistent # of dimension");

        const auto idx_low_old = idx_low;

        CalculateLowerIndex(idx_low, idx_up);

        idx_diff_low = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Xor{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        printf("}");
    }
};
} // namespace ck
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
