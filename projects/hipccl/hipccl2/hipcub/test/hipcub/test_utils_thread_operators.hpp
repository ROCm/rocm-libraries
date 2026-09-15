// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCUB_TEST_TEST_UTILS_THREAD_OPERATORS_HPP_
#define HIPCUB_TEST_TEST_UTILS_THREAD_OPERATORS_HPP_

#include "test_utils.hpp"

#include <hipcub/device/device_reduce.hpp>
#include <hipcub/thread/thread_operators.hpp>
#include <type_traits>

/**
 * \brief ExtendedFloatBoolOp general functor - Because hipcub::Equality() and Inequality()
 * don't work with input types <test_utils::half, float> and <test_utils::bfloat16, float>.
 */
template<typename BoolOpT>
struct ExtendedFloatBoolOp
{
    BoolOpT eq_op;

    HIPCUB_HOST_DEVICE inline ExtendedFloatBoolOp() {}

    template<class T>
    HIPCUB_HOST_DEVICE bool operator()(T a, T b) const
    {
        return eq_op(a.raw(), b.raw());
    }

    HIPCUB_HOST_DEVICE bool operator()(float a, float b) const
    {
        return eq_op(a, b);
    }

    HIPCUB_HOST_DEVICE bool operator()(test_utils::half a, test_utils::half b) const
    {
        return this->operator()(test_utils::native_half(a), test_utils::native_half(b));
    }

    HIPCUB_HOST_DEVICE bool operator()(test_utils::bfloat16 a, test_utils::bfloat16 b) const
    {
        return this->operator()(test_utils::native_bfloat16(a), test_utils::native_bfloat16(b));
    }

    HIPCUB_HOST_DEVICE bool operator()(float a, test_utils::half b) const
    {
        return this->operator()(a, float(b));
    }

    HIPCUB_HOST_DEVICE bool operator()(float a, test_utils::bfloat16 b) const
    {
        return this->operator()(a, float(b));
    }
};

/**
 * \brief ExtendedFloatBinOp general functor - Because hipcub::Sum(), Difference(), Division(),
 * Max() and Min() don't work with input types <test_utils::half, test_utils::half>,
 * <test_utils::bfloat16, test_utils::bfloat16> and
 * <test_utils::half, float> and <test_utils::bfloat16, float>.
 *
 * When using e.g. a constant input iterator of value 2 the CPU accumulator fails to keep adding
 * 2 to 4096 because of precision limitations, as 2 (in half binary representation
 * 0 10000 0000000000 = 1.0 x 2e1) needs to be converted to be able to sum it with 4096
 *  (in half binary representation 0 11011 000000000 = 1.0 x 2e12), that is, the mantisa of 2
 * needs to be shifted to the left 11 times, but that yields a 0 and thus 4096 + 2 = 4096.
 */
template<typename BinOpT>
struct ExtendedFloatBinOp
{
    BinOpT alg_op;

    HIPCUB_HOST_DEVICE inline ExtendedFloatBinOp() {}

    template<class T>
    HIPCUB_HOST_DEVICE T operator()(T a, T b) const
    {
        T result{};
        result.__x = alg_op(a.raw(), b.raw());
        return result;
    }

    HIPCUB_HOST_DEVICE float operator()(float a, float b) const
    {
        return alg_op(a, b);
    }

    HIPCUB_HOST_DEVICE test_utils::half operator()(test_utils::half a, test_utils::half b) const
    {
        return test_utils::native_to_half(
            this->operator()(test_utils::native_half(a), test_utils::native_half(b)));
    }

    HIPCUB_HOST_DEVICE test_utils::bfloat16 operator()(test_utils::bfloat16 a,
                                                       test_utils::bfloat16 b) const
    {
        return test_utils::native_to_bfloat16(
            this->operator()(test_utils::native_bfloat16(a), test_utils::native_bfloat16(b)));
    }

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::half b) const
    {
        return this->operator()(a, float(b));
    }

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::bfloat16 b) const
    {
        return this->operator()(a, float(b));
    }
};

/**
 * \brief Common type specialization - Because min and max don't work with
 * <float, test_utils::half>.
 */
template<>
struct std::common_type<float, test_utils::half>
{
    using type = float;
};

/**
 * \brief Common type specialization - Because min and max don't work with
 * <float, test_utils::bfloat16>.
 */
template<>
struct std::common_type<float, test_utils::bfloat16>
{
    using type = float;
};

/**
 * \brief ArgMax functor - Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMax
{
    template<typename OffsetT,
             class T,
             std::enable_if_t<std::is_same<T, test_utils::half>::value
                                  || std::is_same<T, test_utils::bfloat16>::value,
                              bool>
             = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair<OffsetT, T>
                                       operator()(const hipcub::KeyValuePair<OffsetT, T>& a,
                   const hipcub::KeyValuePair<OffsetT, T>& b) const
    {
        const hipcub::KeyValuePair<OffsetT, float> native_a(a.key, a.value);
        const hipcub::KeyValuePair<OffsetT, float> native_b(b.key, b.value);

        if((native_b.value > native_a.value)
           || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};
/**
 * \brief ArgMin functor - Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMin
{
    template<typename OffsetT,
             class T,
             std::enable_if_t<std::is_same<T, test_utils::half>::value
                                  || std::is_same<T, test_utils::bfloat16>::value,
                              bool>
             = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair<OffsetT, T>
                                       operator()(const hipcub::KeyValuePair<OffsetT, T>& a,
                   const hipcub::KeyValuePair<OffsetT, T>& b) const
    {
        const hipcub::KeyValuePair<OffsetT, float> native_a(a.key, a.value);
        const hipcub::KeyValuePair<OffsetT, float> native_b(b.key, b.value);

        if((native_b.value < native_a.value)
           || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};

/**
 * \brief Common type specialization - Because some thread operators do not work with
 * <custom_test_type<T>, custom_test_type<U>> for different types T and U.
 */
template<class T, class U>
struct std::common_type<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    using type = test_utils::custom_test_type<typename std::common_type<T, U>::type>;
};

/**
 * \brief CustomTestOp generic functor - Because some thread operators don't work with
 * <custom_test_type<T>, custom_test_type<U>> for different types T and U.
 */
template<typename BinaryOpT>
struct CustomTestOp
{
    BinaryOpT binary_op;

    HIPCUB_HOST_DEVICE inline CustomTestOp() {}

    template<typename T, typename U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(test_utils::custom_test_type<T> t,
                                                        test_utils::custom_test_type<U> u) const
        -> decltype(auto)
    {
        using common_type = typename std::common_type<test_utils::custom_test_type<T>,
                                                      test_utils::custom_test_type<U>>::type;
        const common_type common_t(t);
        const common_type common_u(u);
        return binary_op(common_t, common_u);
    }
};

// Equality functor selector.
template<typename OpT, typename T, typename U>
struct EqualitySelector
{
    using type = OpT;
};

template<typename OpT, typename U>
struct EqualitySelector<OpT, test_utils::half, U>
{
    using type = ExtendedFloatBoolOp<OpT>;
};

template<typename OpT, typename U>
struct EqualitySelector<OpT, test_utils::bfloat16, U>
{
    using type = ExtendedFloatBoolOp<OpT>;
};

// Algebraic functor selector.
template<typename OpT, typename T, typename U>
struct AlgebraicSelector
{
    using type = OpT;
};

template<typename OpT, typename T, typename U>
struct AlgebraicSelector<OpT, test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    using type = CustomTestOp<OpT>;
};

template<typename OpT, typename U>
struct AlgebraicSelector<OpT, test_utils::half, U>
{
    using type = ExtendedFloatBinOp<OpT>;
};

template<typename OpT, typename U>
struct AlgebraicSelector<OpT, test_utils::bfloat16, U>
{
    using type = ExtendedFloatBinOp<OpT>;
};

// Max functor selector.
template<typename T, typename U>
struct MaxSelector
{
    using type = hipcub::Max;
};

template<typename T, typename U>
struct MaxSelector<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    using type = CustomTestOp<hipcub::Max>;
};

template<typename U>
struct MaxSelector<test_utils::half, U>
{
    using type = ExtendedFloatBinOp<hipcub::Max>;
};

template<typename U>
struct MaxSelector<test_utils::bfloat16, U>
{
    using type = ExtendedFloatBinOp<hipcub::Max>;
};

// Min functor selector.
template<typename T, typename U>
struct MinSelector
{
    using type = hipcub::Min;
};

template<typename T, typename U>
struct MinSelector<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    using type = CustomTestOp<hipcub::Min>;
};

template<typename U>
struct MinSelector<test_utils::half, U>
{
    using type = ExtendedFloatBinOp<hipcub::Min>;
};

template<typename U>
struct MinSelector<test_utils::bfloat16, U>
{
    using type = ExtendedFloatBinOp<hipcub::Min>;
};

// ArgMax functor selector
template<typename T>
struct ArgMaxSelector
{
    using type = hipcub::ArgMax;
};

#ifdef __HIP_PLATFORM_NVIDIA__
template<>
struct ArgMaxSelector<test_utils::half>
{
    using type = ArgMax;
};

template<>
struct ArgMaxSelector<test_utils::bfloat16>
{
    using type = ArgMax;
};
#endif

// ArgMin functor selector
template<typename T>
struct ArgMinSelector
{
    using type = hipcub::ArgMin;
};

#ifdef __HIP_PLATFORM_NVIDIA__
template<>
struct ArgMinSelector<test_utils::half>
{
    using type = ArgMin;
};

template<>
struct ArgMinSelector<test_utils::bfloat16>
{
    using type = ArgMin;
};
#endif

#endif // HIPCUB_TEST_TEST_UTILS_THREAD_OPERATORS_HPP_
