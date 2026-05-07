// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
#include <vector>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck {

__device__ int static err = 0;

#if 0
template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 3, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 5, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));

    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 6, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d6_t __attribute__((ext_vector_type(6)));

    using type = d6_t;

    union
    {
        d6_t d6_;
        StaticallyIndexedArray<d1_t, 6> d1x6_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d3_t, 2> d3x2_;
        StaticallyIndexedArray<d6_t, 1> d6x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 7, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d7_t __attribute__((ext_vector_type(7)));

    using type = d7_t;

    union
    {
        d7_t d7_;
        StaticallyIndexedArray<d1_t, 7> d1x7_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d7_t, 1> d7x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 12, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    using type = d12_t;

    union
    {
        d12_t d12_;
        StaticallyIndexedArray<d1_t, 12> d1x12_;
        StaticallyIndexedArray<d2_t, 6> d2x6_;
        StaticallyIndexedArray<d3_t, 4> d3x4_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d6_t, 2> d6x2_;
        StaticallyIndexedArray<d12_t, 1> d12x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 13, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d13_t __attribute__((ext_vector_type(13)));

    using type = d13_t;

    union
    {
        d13_t d13_;
        StaticallyIndexedArray<d1_t, 13> d1x13_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
        StaticallyIndexedArray<d13_t, 1> d13x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 14, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d14_t __attribute__((ext_vector_type(14)));

    using type = d14_t;

    union
    {
        d14_t d14_;
        StaticallyIndexedArray<d1_t, 14> d1x14_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
        StaticallyIndexedArray<d14_t, 1> d14x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d14_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x14_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d14_t>::value)
        {
            return data_.d14x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d14_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x14_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d14_t>::value)
        {
            return data_.d14x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 18, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d18_t __attribute__((ext_vector_type(18)));

    using type = d18_t;

    union
    {
        d18_t d18_;
        StaticallyIndexedArray<d1_t, 18> d1x18_;
        StaticallyIndexedArray<d2_t, 9> d2x9_;
        StaticallyIndexedArray<d3_t, 6> d3x6_;
        StaticallyIndexedArray<d6_t, 3> d6x3_;
        StaticallyIndexedArray<d9_t, 2> d9x2_;
        StaticallyIndexedArray<d18_t, 1> d18x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 24, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(242)));
    using type = d24_t;

    union
    {
        d24_t d24_;
        StaticallyIndexedArray<d1_t, 24> d1x24_;
        StaticallyIndexedArray<d2_t, 12> d2x12_;
        StaticallyIndexedArray<d3_t, 8> d3x8_;
        StaticallyIndexedArray<d4_t, 6> d4x6_;
        StaticallyIndexedArray<d6_t, 4> d6x4_;
        StaticallyIndexedArray<d8_t, 3> d8x3_;
        StaticallyIndexedArray<d12_t, 2> d12x2_;
        StaticallyIndexedArray<d24_t, 1> d24x12_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_ = {d32_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    // __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    // __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 36, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    using type = d36_t;

    union
    {
        d36_t d36_;
        StaticallyIndexedArray<d1_t, 36> d1x36_;
        StaticallyIndexedArray<d2_t, 18> d2x18_;
        StaticallyIndexedArray<d3_t, 12> d3x12_;
        StaticallyIndexedArray<d4_t, 9> d4x9_;
        StaticallyIndexedArray<d6_t, 6> d6x6_;
        StaticallyIndexedArray<d9_t, 4> d9x4_;
        StaticallyIndexedArray<d12_t, 3> d12x3_;
        StaticallyIndexedArray<d18_t, 2> d18x2_;
        StaticallyIndexedArray<d36_t, 1> d36x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 72, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    using type = d72_t;

    union
    {
        d72_t d72_;
        StaticallyIndexedArray<d1_t, 72> d1x72_;
        StaticallyIndexedArray<d2_t, 36> d2x36_;
        StaticallyIndexedArray<d3_t, 24> d3x24_;
        StaticallyIndexedArray<d4_t, 18> d4x18_;
        StaticallyIndexedArray<d6_t, 12> d6x12_;
        StaticallyIndexedArray<d8_t, 9> d8x9_;
        StaticallyIndexedArray<d9_t, 8> d9x8_;
        StaticallyIndexedArray<d12_t, 6> d12x6_;
        StaticallyIndexedArray<d18_t, 4> d18x4_;
        StaticallyIndexedArray<d24_t, 3> d24x3_;
        StaticallyIndexedArray<d36_t, 2> d36x2_;
        StaticallyIndexedArray<d72_t, 1> d72x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 128, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_ = {d128_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 144, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    typedef T d144_t __attribute__((ext_vector_type(144)));
    using type = d144_t;

    union
    {
        d144_t d144_;
        StaticallyIndexedArray<d1_t, 144> d1x144_;
        StaticallyIndexedArray<d2_t, 72> d2x72_;
        StaticallyIndexedArray<d3_t, 48> d3x48_;
        StaticallyIndexedArray<d4_t, 36> d4x36_;
        StaticallyIndexedArray<d6_t, 24> d6x24_;
        StaticallyIndexedArray<d9_t, 16> d9x16_;
        StaticallyIndexedArray<d12_t, 12> d12x12_;
        StaticallyIndexedArray<d18_t, 8> d18x8_;
        StaticallyIndexedArray<d36_t, 4> d36x4_;
        StaticallyIndexedArray<d48_t, 4> d48x3_;
        StaticallyIndexedArray<d72_t, 2> d72x2_;
        StaticallyIndexedArray<d144_t, 1> d144x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_ = {d256_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 10, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d5_t __attribute__((ext_vector_type(5)));
    typedef T d10_t __attribute__((ext_vector_type(10)));
    using type = d10_t;

    union
    {
        d10_t d10_;
        StaticallyIndexedArray<d1_t, 10> d1x10_;
        StaticallyIndexedArray<d2_t, 5> d2x5_;
        StaticallyIndexedArray<d5_t, 2> d5x2_;
        StaticallyIndexedArray<d10_t, 1> d10x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 20, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));
    typedef T d10_t __attribute__((ext_vector_type(10)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    using type = d20_t;

    union
    {
        d20_t d20_;
        StaticallyIndexedArray<d1_t, 20> d1x20_;
        StaticallyIndexedArray<d2_t, 10> d2x10_;
        StaticallyIndexedArray<d4_t, 5> d4x5_;
        StaticallyIndexedArray<d5_t, 4> d5x4_;
        StaticallyIndexedArray<d10_t, 2> d10x2_;
        StaticallyIndexedArray<d20_t, 1> d20x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 9, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    using type = d9_t;

    union
    {
        d9_t d9_;
        StaticallyIndexedArray<d1_t, 9> d1x9_;
        StaticallyIndexedArray<d3_t, 3> d3x3_;
        StaticallyIndexedArray<d9_t, 1> d9x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }
};
#endif

template <typename T, index_t N, typename Enable = void>
struct non_native_vector_base;

template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>
{
    using data_t = typename scalar_type<T>::type; // select data_t based on the size of T
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    using data_v = NativeVectorT<data_t, N>;
    using type   = non_native_vector_base<T, N>;

    union alignas(math::next_power_of_two<N * sizeof(T)>())
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray_v2<data_t, N> dxN;
        StaticallyIndexedArray_v2<T, N> dTxN;
        StaticallyIndexedArray_v2<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{data_v(a)} {}
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }
};

// implementation for f6x16 and f6x32
template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>
{
    using data_t    = typename scalar_type<T>::type; // select data_t based on declared base type
    using element_t = typename T::element_type; // select element_t based on declared element type
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    static constexpr size_t size_factor = sizeof(data_t) / sizeof(element_t);
    using data_v                        = NativeVectorT<element_t, N * size_factor>;
    using type                          = non_native_vector_base<T, N>;

    union alignas(math::next_power_of_two<N * sizeof(T)>())
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray_v2<data_t, N> dxN;
        StaticallyIndexedArray_v2<T, N> dTxN;
        StaticallyIndexedArray_v2<data_v, 1> dNx1;
    } data_;

    // Broadcast single value to vector
    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{}
    {
        // TODO: consider removing initialization similar to vector_type<T, 256>

        ck::static_for<0, N, 1>{}([&](auto i) {
            data_.dxN(i) = a; // broadcast value to all elements
        });
    }

    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }

    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};

    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr non_native_vector_base(element_t v) : data_{data_v(v)} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }

    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }

    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return err; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, data_v> || is_same_v<X, T>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, data_v> || is_same_v<X, T>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>>
{
    using type                           = typename non_native_vector_base<T, N>::data_t;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>>
{
    using type                           = typename non_native_vector_base<T, N>::element_t;
    static constexpr index_t vector_size = N * non_native_vector_base<T, N>::size_factor;
};

/**
 * @brief Helper struct to determine the storage type for vector_type
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 * @tparam Enable SFINAE helper
 */
template <typename T, index_t Rank, typename Enable = void>
struct vector_type_storage;

/**
 * @brief Vector storage type for native scalar types.
 * @tparam T The element type of the vector
 * @note For Rank = 1 and native types, the storage type is simply T itself (scalar)
 */
template <typename T>
struct vector_type_storage<T, 1, enable_if_t<is_native_type<T>()>>
{
    using type = T;
};

/**
 * @brief Vector storage type for native vector types.
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 *
 * Assigns a native vector type based on the element type and rank.
 * For boolean types, uses a C-style array `T[Rank]`, otherwise uses
 * the `NativeVectorT<T, Rank>` template specialization.
 *
 * @note Special handling note:
 * Sub-byte sizes such as bool have different sizes in ext_vector_type (via NativeVectorT) vs array
 * types due to packing. Builtin vector types pack bool elements, while C++ arrays use 1 byte per
 * bool as a standard (minimum write size = 1 byte). e.g., ext_vector_type(bool, 4) is packed as
 * minimum 1 byte, while bool[4] is 4 bytes. vector_type::AsType, aliases with
 * StaticallyIndexedArray_v2 which is C-style array under the hood, so we need to avoid using
 * ext_vector_type with bool due to potential for data slicing errors.
 */
template <typename T, index_t Rank>
struct vector_type_storage<T, Rank, enable_if_t<is_native_type<T>() && (Rank > 1)>>
{
    using type = std::conditional_t<std::is_same_v<bool, T>, T[Rank], NativeVectorT<T, Rank>>;
};

/**
 * @brief Vector storage type for non-native vector types.
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 * @note For non-native types, the storage type is non_native_vector_base<T, Rank>
 */
template <typename T, index_t Rank>
struct vector_type_storage<T, Rank, enable_if_t<!is_native_type<T>()>>
{
    using type = non_native_vector_base<T, Rank>;
};

/**
 * @brief Convenience wrapper for vector_type_storage
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 */
template <typename T, index_t Rank>
using vector_type_storage_t = typename vector_type_storage<T, Rank>::type;

/**
 * @brief Trait to check whether one vector storage class is the same as another (e.g., same scalar,
 * or same vector class).
 * @tparam Lhs The source storage type
 * @tparam Rhs The comparator storage type
 *
 * Same storage classes are:
 * - Same type
 * - Same template vector types with matching base type (may have different ranks)
 * - C-style arrays of same base type (may have different ranks)
 */
template <typename Lhs, typename Rhs>
struct is_same_vector_storage_class : public false_type
{
};

/**
 * @brief Template native vector types of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 */
template <typename T, index_t LhsRank, index_t RhsRank>
struct is_same_vector_storage_class<NativeVectorT<T, LhsRank>, NativeVectorT<T, RhsRank>>
    : true_type
{
};

/**
 * @brief Template non-native vector types of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 * @tparam Enable SFINAE helper
 */
template <typename T, index_t LhsRank, index_t RhsRank, typename Enable>
struct is_same_vector_storage_class<non_native_vector_base<T, LhsRank, Enable>,
                                    non_native_vector_base<T, RhsRank, Enable>> : true_type
{
};

/**
 * @brief C-style arrays of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 */
template <typename T, index_t LhsRank, index_t RhsRank>
struct is_same_vector_storage_class<T[LhsRank], T[RhsRank]> : true_type
{
};

/**
 * @brief Convenience evaluator for is_same_vector_storage_class
 * @tparam Lhs The source storage type
 * @tparam Rhs The comparator storage type
 */
template <typename Lhs, typename Rhs>
static constexpr bool is_same_vector_storage_class_v =
    is_same_vector_storage_class<Lhs, Rhs>::value;

// Fwd declaration
template <typename T, index_t Rank>
struct vector_type;

/**
 * @brief Trait to extract element type and rank from vector_type and related types
 * @tparam T The vector type
 */
template <typename T>
struct vector_type_traits
{
    using element_type            = T;
    static constexpr index_t Rank = 1;
};

/**
 * @brief Specialization of vector_type_traits for vector_type
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<vector_type<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Specialization of vector_type_traits for non_native_vector_base
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<non_native_vector_base<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Specialization of vector_type_traits for NativeVectorT
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<NativeVectorT<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Vector type wrapper
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 */
template <typename T, index_t Rank>
struct vector_type
{
    /// @brief Internal storage type for vector_type.
    using StorageT = vector_type_storage_t<T, Rank>;
    using type     = StorageT;
    StorageT data_;

    /// @brief Default constructor for vector_type
    __host__ __device__ constexpr vector_type() : data_{} {}

    /// @brief Constructor for native vector initialization
    __host__ __device__ constexpr vector_type(StorageT v) : data_{v} {}

    /**
     * @brief Validates whether a type can be used in an AsType cast operation for vector_type
     * class.
     *
     * This function checks if a given type X can be legally used as an alias to either reinterpret
     * or slice (iterate) through the local storage type StorageT. The validation ensures type
     * safety and structural compatibility between the source and target vector types.
     *
     * @tparam X The target type to validate for AsType casting.
     *
     * @return constexpr bool True if the type is valid for AsType casting, false otherwise
     *
     * @note Requirements for a valid AsType<X> cast on vector_type<T, Rank>:
     *       1. The value type of X must match the storage value type (T)
     *       2. X must be either:
     *          a) A scalar type (T) where RankX == 1, OR
     *          b) A vector class that matches the storage vector class (e.g., both are
     * NativeVectorT or non_native_vector_base) where:
     *             - RankX is a power of 2, OR
     *               RankX == 3, OR
     *               RankX == Storage Rank
     *             - RankX must be <= Storage Rank
     * @example
     * auto srcVec = vector_type<float, 8>{};  // T = float, Rank = 8, native vector storage
     * auto result = srcVec.AsType<X>();       // Where datatype X could be:
     * X = NativeVectorT<float, 4>;            // OK: native vector T, RankX = 4 (power of 2)
     * X = float;                              // OK: scalar T, RankX = 1
     * X = NativeVectorT<float, 5>;            // ERROR: RankX not a power of 2, ==3, or ==Rank
     * X = int;                                // ERROR: Invalid scalar cast, T != int
     * X = float[4];                           // ERROR: Invalid type, storage vector class doesn't
     *                                         // match (native vector != C-array)
     */
    template <typename X>
    static constexpr bool is_as_type_cast_valid()
    {
        using TraitsX = vector_type_traits<X>;

        // Checks storage classes match, with same base type (may have different ranks)
        constexpr bool is_valid_cast =
            is_same_vector_storage_class_v<StorageT, X> || // Matching vector storage
            is_same_v<X, T>;                               // Matching scalar type

        // Validate vector ranks
        constexpr bool is_valid_rank = (math::is_power_of_two_integer(TraitsX::Rank) ||
                                        (TraitsX::Rank == 3) || (TraitsX::Rank == Rank)) &&
                                       (TraitsX::Rank <= Rank);

        return is_valid_cast && is_valid_rank;
    }

    /**
     * @brief Allows casting the vector_type to another type X via aliasing or slicing.
     * Use cases are to expose the internal storage type, or to slice the vector into smaller
     * vectors for iteration purposes.
     * @tparam X The target type to validate for AsType casting.
     * @returns a reference to the reinterpreted data as StaticallyIndexedArray_v2<X, Rank / RankX>.
     * Rigid control of allowable casts is enforced via static_assert to ensure type safety.
     * See is_as_type_cast_valid() for requirements.
     */
    template <typename X>
    __host__ __device__ constexpr auto const& AsType() const [[clang::lifetimebound]]
    {
        // Make this a hard error if the datatype X is not a valid cast.
        static_assert(is_as_type_cast_valid<X>(), "Datatype X is not a valid AsType cast");

        using TraitsX = vector_type_traits<X>;

        // Calculate the new rank after slicing.
        // Note: We might end up with incomplete quantization from slicing
        // when Rank % TraitsX::Rank != 0, so take the floor division.
        constexpr index_t newRank = Rank / TraitsX::Rank;

        // Determine the cast type:
        // - Scalar T if slicing to scalar or vector size of 1,
        // - X otherwise.
        using CastT   = conditional_t<TraitsX::Rank == 1, T, X>;
        using ResultT = StaticallyIndexedArray_v2<CastT, newRank>;

        // As a rule, the aliasing type should not be larger than the original type.
        static_assert(sizeof(ResultT) <= sizeof(vector_type),
                      "Resulting aliasing cannot be larger than original type");

        // Re-cast as vectorized type.
        return *(bit_cast<ResultT const*>(this));
    }

    /**
     * @brief Allows casting the vector_type to another type X via aliasing or slicing.
     * Use cases are to expose the internal storage type, or to slice the vector into smaller
     * vectors for iteration purposes.
     * @tparam X The target type to validate for AsType casting.
     * @returns a reference to the reinterpreted data as StaticallyIndexedArray_v2<X, Rank / RankX>.
     * Rigid control of allowable casts is enforced via static_assert to ensure type safety.
     * See is_as_type_cast_valid() for requirements.
     */
    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        // Make this a hard error if the datatype X is not a valid cast.
        static_assert(is_as_type_cast_valid<X>(), "Datatype X is not a valid AsType cast");

        using TraitsX = vector_type_traits<X>;

        // Calculate the new rank after slicing.
        // Note: We might end up with incomplete quantization from slicing
        // when Rank % TraitsX::Rank != 0, so take the floor division.
        constexpr index_t newRank = Rank / TraitsX::Rank;

        // Determine the cast type:
        // - Scalar T if slicing to scalar or vector size of 1,
        // - X otherwise.
        using CastT   = conditional_t<TraitsX::Rank == 1, T, X>;
        using ResultT = StaticallyIndexedArray_v2<CastT, newRank>;

        // As a rule, the aliasing type should not be larger than the original type.
        static_assert(sizeof(ResultT) <= sizeof(vector_type),
                      "Resulting aliasing cannot be larger than original type");

        // Re-cast as vectorized type.
        return *(bit_cast<ResultT*>(this));
    }
};

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<NativeVectorT<T, V>, N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

/**
 * @brief scalar_type trait override for vector_type
 * @tparam T The vector type
 * @tparam N The number of elements in the vector
 */
template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = typename scalar_type<T>::type;
    static constexpr index_t vector_size = N;
};

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<NativeVectorT<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

#if 0
template <typename T>
struct vector_type<T, 128, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d8_t   = non_native_vector_base<T, 8>;
    using d16_t  = non_native_vector_base<T, 16>;
    using d32_t  = non_native_vector_base<T, 32>;
    using d64_t  = non_native_vector_base<T, 64>;
    using d128_t = non_native_vector_base<T, 128>;

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d8_t   = non_native_vector_base<T, 8>;
    using d16_t  = non_native_vector_base<T, 16>;
    using d32_t  = non_native_vector_base<T, 32>;
    using d64_t  = non_native_vector_base<T, 64>;
    using d128_t = non_native_vector_base<T, 128>;
    using d256_t = non_native_vector_base<T, 256>;

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 3, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d3_t = non_native_vector_base<T, 3>;

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 6, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d3_t = non_native_vector_base<T, 3>;
    using d6_t = non_native_vector_base<T, 6>;

    using type = d6_t;

    union
    {
        d6_t d6_;
        StaticallyIndexedArray<d1_t, 6> d1x6_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d3_t, 2> d3x2_;
        StaticallyIndexedArray<d6_t, 1> d6x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 12, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d12_t = non_native_vector_base<T, 12>;
    using type  = d12_t;

    union
    {
        d12_t d12_;
        StaticallyIndexedArray<d1_t, 12> d1x12_;
        StaticallyIndexedArray<d2_t, 6> d2x6_;
        StaticallyIndexedArray<d3_t, 4> d3x4_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d6_t, 2> d6x2_;
        StaticallyIndexedArray<d12_t, 1> d12x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 18, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d18_t = non_native_vector_base<T, 18>;

    using type = d18_t;

    union
    {
        d18_t d18_;
        StaticallyIndexedArray<d1_t, 18> d1x18_;
        StaticallyIndexedArray<d2_t, 9> d2x9_;
        StaticallyIndexedArray<d3_t, 6> d3x6_;
        StaticallyIndexedArray<d6_t, 3> d6x3_;
        StaticallyIndexedArray<d9_t, 2> d9x2_;
        StaticallyIndexedArray<d18_t, 1> d18x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 24, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d12_t = non_native_vector_base<T, 12>;
    using d24_t = non_native_vector_base<T, 242>;
    using type  = d24_t;

    union
    {
        d24_t d24_;
        StaticallyIndexedArray<d1_t, 24> d1x24_;
        StaticallyIndexedArray<d2_t, 12> d2x12_;
        StaticallyIndexedArray<d3_t, 8> d3x8_;
        StaticallyIndexedArray<d4_t, 6> d4x6_;
        StaticallyIndexedArray<d6_t, 4> d6x4_;
        StaticallyIndexedArray<d8_t, 3> d8x3_;
        StaticallyIndexedArray<d12_t, 2> d12x2_;
        StaticallyIndexedArray<d24_t, 1> d24x12_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 36, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d12_t = non_native_vector_base<T, 12>;
    using d18_t = non_native_vector_base<T, 18>;
    using d36_t = non_native_vector_base<T, 36>;
    using type  = d36_t;

    union
    {
        d36_t d36_;
        StaticallyIndexedArray<d1_t, 36> d1x36_;
        StaticallyIndexedArray<d2_t, 18> d2x18_;
        StaticallyIndexedArray<d3_t, 12> d3x12_;
        StaticallyIndexedArray<d4_t, 9> d4x9_;
        StaticallyIndexedArray<d6_t, 6> d6x6_;
        StaticallyIndexedArray<d9_t, 4> d9x4_;
        StaticallyIndexedArray<d12_t, 3> d12x3_;
        StaticallyIndexedArray<d18_t, 2> d18x2_;
        StaticallyIndexedArray<d36_t, 1> d36x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 72, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d12_t = non_native_vector_base<T, 12>;
    using d18_t = non_native_vector_base<T, 18>;
    using d24_t = non_native_vector_base<T, 24>;
    using d36_t = non_native_vector_base<T, 36>;
    using d72_t = non_native_vector_base<T, 72>;
    using type  = d72_t;

    union
    {
        d72_t d72_;
        StaticallyIndexedArray<d1_t, 72> d1x72_;
        StaticallyIndexedArray<d2_t, 36> d2x36_;
        StaticallyIndexedArray<d3_t, 24> d3x24_;
        StaticallyIndexedArray<d4_t, 18> d4x18_;
        StaticallyIndexedArray<d6_t, 12> d6x12_;
        StaticallyIndexedArray<d8_t, 9> d8x9_;
        StaticallyIndexedArray<d9_t, 8> d9x8_;
        StaticallyIndexedArray<d12_t, 6> d12x6_;
        StaticallyIndexedArray<d18_t, 4> d18x4_;
        StaticallyIndexedArray<d24_t, 3> d24x3_;
        StaticallyIndexedArray<d36_t, 2> d36x2_;
        StaticallyIndexedArray<d72_t, 1> d72x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 144, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d3_t   = non_native_vector_base<T, 3>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d6_t   = non_native_vector_base<T, 6>;
    using d9_t   = non_native_vector_base<T, 9>;
    using d12_t  = non_native_vector_base<T, 12>;
    using d18_t  = non_native_vector_base<T, 18>;
    using d36_t  = non_native_vector_base<T, 36>;
    using d48_t  = non_native_vector_base<T, 48>;
    using d72_t  = non_native_vector_base<T, 72>;
    using d144_t = non_native_vector_base<T, 144>;
    using type   = d144_t;

    union
    {
        d144_t d144_;
        StaticallyIndexedArray<d1_t, 144> d1x144_;
        StaticallyIndexedArray<d2_t, 72> d2x72_;
        StaticallyIndexedArray<d3_t, 48> d3x48_;
        StaticallyIndexedArray<d4_t, 36> d4x36_;
        StaticallyIndexedArray<d6_t, 24> d6x24_;
        StaticallyIndexedArray<d9_t, 16> d9x16_;
        StaticallyIndexedArray<d12_t, 12> d12x12_;
        StaticallyIndexedArray<d18_t, 8> d18x8_;
        StaticallyIndexedArray<d36_t, 4> d36x4_;
        StaticallyIndexedArray<d48_t, 4> d48x3_;
        StaticallyIndexedArray<d72_t, 2> d72x2_;
        StaticallyIndexedArray<d144_t, 1> d144x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 5, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d5_t = non_native_vector_base<T, 5>;
    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d5_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d5_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 10, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d5_t  = non_native_vector_base<T, 5>;
    using d10_t = non_native_vector_base<T, 10>;
    using type  = d10_t;

    union
    {
        d10_t d10_;
        StaticallyIndexedArray<d1_t, 10> d1x10_;
        StaticallyIndexedArray<d2_t, 5> d2x5_;
        StaticallyIndexedArray<d5_t, 2> d5x2_;
        StaticallyIndexedArray<d10_t, 1> d10x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 20, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d5_t  = non_native_vector_base<T, 5>;
    using d10_t = non_native_vector_base<T, 10>;
    using d20_t = non_native_vector_base<T, 20>;
    using type  = d20_t;

    union
    {
        d20_t d20_;
        StaticallyIndexedArray<d1_t, 20> d1x20_;
        StaticallyIndexedArray<d2_t, 10> d2x10_;
        StaticallyIndexedArray<d4_t, 5> d4x5_;
        StaticallyIndexedArray<d5_t, 4> d5x4_;
        StaticallyIndexedArray<d10_t, 2> d10x2_;
        StaticallyIndexedArray<d20_t, 1> d20x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 9, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d3_t = non_native_vector_base<T, 3>;
    using d9_t = non_native_vector_base<T, 9>;
    using type = d9_t;

    union
    {
        d9_t d9_;
        StaticallyIndexedArray<d1_t, 9> d1x9_;
        StaticallyIndexedArray<d3_t, 3> d3x3_;
        StaticallyIndexedArray<d9_t, 1> d9x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }
};
#endif
// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

using half3_t  = typename vector_type<half_t, 3>::type;
using half6_t  = typename vector_type<half_t, 6>::type;
using half9_t  = typename vector_type<half_t, 9>::type;
using half18_t = typename vector_type<half_t, 18>::type;
using half36_t = typename vector_type<half_t, 36>::type;
using half5_t  = typename vector_type<half_t, 5>::type;
using half10_t = typename vector_type<half_t, 10>::type;
using half14_t = typename vector_type<half_t, 14>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

using bhalf3_t  = typename vector_type<bhalf_t, 3>::type;
using bhalf6_t  = typename vector_type<bhalf_t, 6>::type;
using bhalf9_t  = typename vector_type<bhalf_t, 9>::type;
using bhalf18_t = typename vector_type<bhalf_t, 18>::type;
using bhalf36_t = typename vector_type<bhalf_t, 36>::type;
using bhalf5_t  = typename vector_type<bhalf_t, 5>::type;
using bhalf10_t = typename vector_type<bhalf_t, 10>::type;

// i16
using int16x2_t = typename vector_type<int16_t, 2>::type;
using int16x4_t = typename vector_type<int16_t, 4>::type;
using int16x8_t = typename vector_type<int16_t, 8>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x6_t  = typename vector_type<int32_t, 6>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

using int32x3_t  = typename vector_type<int32_t, 3>::type;
using int32x5_t  = typename vector_type<int32_t, 5>::type;
using int32x7_t  = typename vector_type<int32_t, 7>::type;
using int32x9_t  = typename vector_type<int32_t, 9>::type;
using int32x18_t = typename vector_type<int32_t, 18>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

using int8x3_t  = typename vector_type<int8_t, 3>::type;
using int8x6_t  = typename vector_type<int8_t, 6>::type;
using int8x9_t  = typename vector_type<int8_t, 9>::type;
using int8x12_t = typename vector_type<int8_t, 12>::type;
using int8x18_t = typename vector_type<int8_t, 18>::type;
using int8x24_t = typename vector_type<int8_t, 24>::type;
using int8x36_t = typename vector_type<int8_t, 36>::type;
using int8x72_t = typename vector_type<int8_t, 72>::type;
using int8x5_t  = typename vector_type<int8_t, 5>::type;
using int8x10_t = typename vector_type<int8_t, 10>::type;
using int8x20_t = typename vector_type<int8_t, 20>::type;

// f8
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

using f8x3_fnuz_t  = typename vector_type<f8_fnuz_t, 3>::type;
using f8x5_fnuz_t  = typename vector_type<f8_fnuz_t, 5>::type;
using f8x6_fnuz_t  = typename vector_type<f8_fnuz_t, 6>::type;
using f8x9_fnuz_t  = typename vector_type<f8_fnuz_t, 9>::type;
using f8x10_fnuz_t = typename vector_type<f8_fnuz_t, 10>::type;
using f8x12_fnuz_t = typename vector_type<f8_fnuz_t, 12>::type;
using f8x18_fnuz_t = typename vector_type<f8_fnuz_t, 18>::type;
using f8x20_fnuz_t = typename vector_type<f8_fnuz_t, 20>::type;
using f8x24_fnuz_t = typename vector_type<f8_fnuz_t, 24>::type;
using f8x36_fnuz_t = typename vector_type<f8_fnuz_t, 36>::type;
using f8x72_fnuz_t = typename vector_type<f8_fnuz_t, 72>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

using bf8x3_fnuz_t  = typename vector_type<bf8_fnuz_t, 3>::type;
using bf8x5_fnuz_t  = typename vector_type<bf8_fnuz_t, 5>::type;
using bf8x6_fnuz_t  = typename vector_type<bf8_fnuz_t, 6>::type;
using bf8x9_fnuz_t  = typename vector_type<bf8_fnuz_t, 9>::type;
using bf8x10_fnuz_t = typename vector_type<bf8_fnuz_t, 10>::type;
using bf8x12_fnuz_t = typename vector_type<bf8_fnuz_t, 12>::type;
using bf8x18_fnuz_t = typename vector_type<bf8_fnuz_t, 18>::type;
using bf8x20_fnuz_t = typename vector_type<bf8_fnuz_t, 20>::type;
using bf8x24_fnuz_t = typename vector_type<bf8_fnuz_t, 24>::type;
using bf8x36_fnuz_t = typename vector_type<bf8_fnuz_t, 36>::type;
using bf8x72_fnuz_t = typename vector_type<bf8_fnuz_t, 72>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

using f8x3_ocp_t  = typename vector_type<f8_ocp_t, 3>::type;
using f8x5_ocp_t  = typename vector_type<f8_ocp_t, 5>::type;
using f8x6_ocp_t  = typename vector_type<f8_ocp_t, 6>::type;
using f8x9_ocp_t  = typename vector_type<f8_ocp_t, 9>::type;
using f8x10_ocp_t = typename vector_type<f8_ocp_t, 10>::type;
using f8x12_ocp_t = typename vector_type<f8_ocp_t, 12>::type;
using f8x18_ocp_t = typename vector_type<f8_ocp_t, 18>::type;
using f8x20_ocp_t = typename vector_type<f8_ocp_t, 20>::type;
using f8x24_ocp_t = typename vector_type<f8_ocp_t, 24>::type;
using f8x36_ocp_t = typename vector_type<f8_ocp_t, 36>::type;
using f8x72_ocp_t = typename vector_type<f8_ocp_t, 72>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

using bf8x3_ocp_t  = typename vector_type<bf8_ocp_t, 3>::type;
using bf8x5_ocp_t  = typename vector_type<bf8_ocp_t, 5>::type;
using bf8x6_ocp_t  = typename vector_type<bf8_ocp_t, 6>::type;
using bf8x9_ocp_t  = typename vector_type<bf8_ocp_t, 9>::type;
using bf8x10_ocp_t = typename vector_type<bf8_ocp_t, 10>::type;
using bf8x12_ocp_t = typename vector_type<bf8_ocp_t, 12>::type;
using bf8x18_ocp_t = typename vector_type<bf8_ocp_t, 18>::type;
using bf8x20_ocp_t = typename vector_type<bf8_ocp_t, 20>::type;
using bf8x24_ocp_t = typename vector_type<bf8_ocp_t, 24>::type;
using bf8x36_ocp_t = typename vector_type<bf8_ocp_t, 36>::type;
using bf8x72_ocp_t = typename vector_type<bf8_ocp_t, 72>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

using f8x3_t  = f8x3_ocp_t;
using f8x5_t  = f8x5_ocp_t;
using f8x6_t  = f8x6_ocp_t;
using f8x9_t  = f8x9_ocp_t;
using f8x10_t = f8x10_ocp_t;
using f8x12_t = f8x12_ocp_t;
using f8x18_t = f8x18_ocp_t;
using f8x20_t = f8x20_ocp_t;
using f8x24_t = f8x24_ocp_t;
using f8x36_t = f8x36_ocp_t;
using f8x72_t = f8x72_ocp_t;

// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;

using bf8x3_t  = bf8x3_ocp_t;
using bf8x5_t  = bf8x5_ocp_t;
using bf8x6_t  = bf8x6_ocp_t;
using bf8x9_t  = bf8x9_ocp_t;
using bf8x10_t = bf8x10_ocp_t;
using bf8x12_t = bf8x12_ocp_t;
using bf8x18_t = bf8x18_ocp_t;
using bf8x20_t = bf8x20_ocp_t;
using bf8x24_t = bf8x24_ocp_t;
using bf8x36_t = bf8x36_ocp_t;
using bf8x72_t = bf8x72_ocp_t;

#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t  = f8x2_fnuz_t;
using f8x4_t  = f8x4_fnuz_t;
using f8x8_t  = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

using f8x3_t  = f8x3_fnuz_t;
using f8x5_t  = f8x5_fnuz_t;
using f8x6_t  = f8x6_fnuz_t;
using f8x9_t  = f8x9_fnuz_t;
using f8x10_t = f8x10_fnuz_t;
using f8x12_t = f8x12_fnuz_t;
using f8x18_t = f8x18_fnuz_t;
using f8x20_t = f8x20_fnuz_t;
using f8x24_t = f8x24_fnuz_t;
using f8x36_t = f8x36_fnuz_t;
using f8x72_t = f8x72_fnuz_t;

// bf8
using bf8x2_t  = bf8x2_fnuz_t;
using bf8x4_t  = bf8x4_fnuz_t;
using bf8x8_t  = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;

using bf8x3_t  = bf8x3_fnuz_t;
using bf8x5_t  = bf8x5_fnuz_t;
using bf8x6_t  = bf8x6_fnuz_t;
using bf8x9_t  = bf8x9_fnuz_t;
using bf8x10_t = bf8x10_fnuz_t;
using bf8x12_t = bf8x12_fnuz_t;
using bf8x18_t = bf8x18_fnuz_t;
using bf8x20_t = bf8x20_fnuz_t;
using bf8x24_t = bf8x24_fnuz_t;
using bf8x36_t = bf8x36_fnuz_t;
using bf8x72_t = bf8x72_fnuz_t;
#endif

// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

using uint8x3_t  = typename vector_type<uint8_t, 3>::type;
using uint8x6_t  = typename vector_type<uint8_t, 6>::type;
using uint8x9_t  = typename vector_type<uint8_t, 9>::type;
using uint8x12_t = typename vector_type<uint8_t, 12>::type;
using uint8x18_t = typename vector_type<uint8_t, 18>::type;
using uint8x24_t = typename vector_type<uint8_t, 24>::type;
using uint8x36_t = typename vector_type<uint8_t, 36>::type;
using uint8x72_t = typename vector_type<uint8_t, 72>::type;
using uint8x5_t  = typename vector_type<uint8_t, 5>::type;
using uint8x10_t = typename vector_type<uint8_t, 10>::type;
using uint8x20_t = typename vector_type<uint8_t, 20>::type;

//__fp16
using fp16x2_t  = typename vector_type<__fp16, 2>::type;
using fp16x4_t  = typename vector_type<__fp16, 4>::type;
using fp16x8_t  = typename vector_type<__fp16, 8>::type;
using fp16x16_t = typename vector_type<__fp16, 16>::type;

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using int4x8_t  = typename vector_type<int32_t, 1>::type;
using int4x16_t = typename vector_type<int32_t, 2>::type;
using int4x32_t = typename vector_type<int32_t, 4>::type;
using int4x64_t = typename vector_type<int32_t, 8>::type;

using int4x24_t  = typename vector_type<int32_t, 3>::type;
using int4x48_t  = typename vector_type<int32_t, 6>::type;
using int4x72_t  = typename vector_type<int32_t, 9>::type;
using int4x144_t = typename vector_type<int32_t, 18>::type;
using int4x40_t  = typename vector_type<int32_t, 5>::type;

using uint4x8_t  = typename vector_type<uint32_t, 1>::type;
using uint4x16_t = typename vector_type<uint32_t, 2>::type;
using uint4x32_t = typename vector_type<uint32_t, 4>::type;
using uint4x64_t = typename vector_type<uint32_t, 8>::type;
#endif

enum class MTX_FMT
{
    MTX_FMT_FP8_E4M3, // FP8
    MTX_FMT_FP8_E5M2, // BF8
    MTX_FMT_FP6_E2M3, // FP6
    MTX_FMT_FP6_E3M2, // BF6
    MTX_FMT_FP4_E2M1, // FP4
    MTX_FMT_DEFAULT,
};

template <MTX_FMT MX_TYPE = MTX_FMT::MTX_FMT_DEFAULT>
struct MxType_t
{
    using type_t = unsigned _BitInt(8);
    using vec_t  = int32x8_t; // used for wmma
    MxType_t()   = default;
    MxType_t(type_t data) { m_data = data; }
    MxType_t(f8_ocp_t data) { m_data = data.data; }
    MxType_t(bf8_ocp_t data) { m_data = data.data; }
    MxType_t(const MxType_t& other) { m_data = other.m_data; }
    type_t m_data;
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP4_E2M1> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP4_E2M1;
    static constexpr index_t vec_size         = 4;
    static constexpr index_t dwords_per_wmmak = 8;
    static constexpr index_t BITS             = 4;
    // the below function is used for data preparation in host side; not used in device side
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        std::vector<type_t> in_packed;
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 2)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            in_packed.push_back((val0 & 0xf) | (val1 << 4));
        }

        return in_packed;
    }
#endif
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP6_E3M2> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType = MTX_FMT::MTX_FMT_FP6_E3M2;
    // this is used per K-dimension how many int32_t need to load; used in wmma_op shader
    static constexpr index_t vec_size         = 6;
    static constexpr index_t dwords_per_wmmak = 12;
    static constexpr index_t BITS             = 6;
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        std::vector<type_t> in_packed;
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 4)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            type_t val2 = in[i + 2];
            type_t val3 = in[i + 3];
            in_packed.push_back((val0 & 0x3f) | ((val1 & 0x3) << 6));
            in_packed.push_back((val1 & 0x3C) >> 2 | ((val2 & 0xf) << 4));
            in_packed.push_back((val2 & 0x3C) >> 4 | ((val3 & 0x3f) << 2));
        }
        return in_packed;
    }
#endif
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP6_E2M3> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP6_E2M3;
    static constexpr index_t vec_size         = 6;
    static constexpr index_t dwords_per_wmmak = 12;
    static constexpr index_t BITS             = 6;
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        std::vector<type_t> in_packed;
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 4)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            type_t val2 = in[i + 2];
            type_t val3 = in[i + 3];
            in_packed.push_back((val0 & 0x3f) | ((val1 & 0x3) << 6));
            in_packed.push_back((val1 & 0x3C) >> 2 | ((val2 & 0xf) << 4));
            in_packed.push_back((val2 & 0x3C) >> 4 | ((val3 & 0x3f) << 2));
        }
        return in_packed;
    }
#endif
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP8_E4M3> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    using type_t                              = f8_t;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP8_E4M3;
    static constexpr index_t vec_size         = 8;
    static constexpr index_t dwords_per_wmmak = 16;
    static constexpr index_t BITS             = 8;
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in) { return in; }
#endif
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP8_E5M2> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    using type_t                              = bf8_t;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP8_E5M2;
    static constexpr index_t vec_size         = 8;
    static constexpr index_t dwords_per_wmmak = 16;
    static constexpr index_t BITS             = 8;
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in) { return in; }
#endif
};

template <typename T>
struct is_mx_type_t : ck::false_type
{
};

// Specialize the template for ck::MxType_t
template <MTX_FMT MX_TYPE>
struct is_mx_type_t<MxType_t<MX_TYPE>> : ck::true_type
{
};

// Helper variable template
template <typename T>
inline constexpr bool is_mx_type_t_v = is_mx_type_t<T>::value;

template <typename T, typename = void>
struct view_type
{
    using srcType  = T;
    using viewType = T;
};

template <typename T>
struct view_type<T, enable_if_t<is_mx_type_t<T>::value>>
{
    using srcType  = typename T::type_t;
    using viewType = int32_t;
};

template <typename T>
using src_type_t = typename view_type<T>::srcType;

template <typename T>
using view_type_t = typename view_type<T>::viewType;

template <typename T, typename = void>
struct DataPerVGPR;

template <typename T>
struct DataPerVGPR<T, enable_if_t<is_same_v<T, half_t> || is_same_v<T, bhalf_t>>>
{
    static constexpr index_t value = 2;
};

template <typename T>
struct DataPerVGPR<T,
                   enable_if_t<is_same_v<T, f8_t> || is_same_v<T, bf8_t> || is_same_v<T, int8_t>>>
{
    static constexpr index_t value = 4;
};

// f4
using f4x2_t   = typename vector_type<f4x2_pk_t, 1>::type;
using f4x4_t   = typename vector_type<f4x2_pk_t, 2>::type;
using f4x8_t   = typename vector_type<f4x2_pk_t, 4>::type;
using f4x16_t  = typename vector_type<f4x2_pk_t, 8>::type;
using f4x32_t  = typename vector_type<f4x2_pk_t, 16>::type;
using f4x64_t  = typename vector_type<f4x2_pk_t, 32>::type;
using f4x128_t = typename vector_type<f4x2_pk_t, 64>::type;

// f6
using f6x16_t   = typename vector_type<f6x16_pk_t, 1>::type;
using f6x16x2_t = typename vector_type<f6x16_pk_t, 2>::type;
using f6x16x4_t = typename vector_type<f6x16_pk_t, 4>::type;
using f6x32_t   = typename vector_type<f6x32_pk_t, 1>::type;
using f6x32x2_t = typename vector_type<f6x32_pk_t, 2>::type;
using f6x64_t   = typename vector_type<f6x32_pk_t, 2>::type;

// bf6
using bf6x16_t   = typename vector_type<bf6x16_pk_t, 1>::type;
using bf6x16x2_t = typename vector_type<bf6x16_pk_t, 2>::type;
using bf6x16x4_t = typename vector_type<bf6x16_pk_t, 4>::type;
using bf6x32_t   = typename vector_type<bf6x32_pk_t, 1>::type;
using bf6x32x2_t = typename vector_type<bf6x32_pk_t, 2>::type;
using bf6x64_t   = typename vector_type<bf6x32_pk_t, 2>::type;

#ifndef CK_CODE_GEN_RTC
// e8m0
using e8m0x4_bexp_t = typename vector_type<e8m0_bexp_t, 4>::type;
using e8m0x8_bexp_t = typename vector_type<e8m0_bexp_t, 8>::type;
// e4m3
using e4m3x4_scale_t = typename vector_type<e4m3_scale_t, 4>::type;
using e4m3x8_scale_t = typename vector_type<e4m3_scale_t, 8>::type;
// e5m3
using e5m3x4_scale_t = typename vector_type<e5m3_scale_t, 4>::type;
using e5m3x8_scale_t = typename vector_type<e5m3_scale_t, 8>::type;
#endif

// pack int4
using pk_i4x2_t = typename vector_type<pk_i4_t, 2>::type;
using pk_i4x4_t = typename vector_type<pk_i4_t, 4>::type;
using pk_i4x8_t = typename vector_type<pk_i4_t, 8>::type;

} // namespace ck
#pragma clang diagnostic pop
