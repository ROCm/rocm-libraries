// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"
#include "rocprim/types.hpp"

#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>

#include <cstdint>
#include <type_traits>

template<class T>
struct device_plus
{
    __device__ inline constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<class InputType, class Function, class ExpectedType = InputType>
struct RocprimTypeInvokeResultParams
{
    using input_type    = InputType;
    using function      = Function;
    using expected_type = ExpectedType;
};

template<class Params>
class RocprimInvokeResultBinOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

using RocprimInvokeResultBinOpTestsParams = ::testing::Types<
    RocprimTypeInvokeResultParams<uint16_t, rocprim::plus<uint16_t>>,
    RocprimTypeInvokeResultParams<int32_t, rocprim::plus<int32_t>>,
    RocprimTypeInvokeResultParams<float, rocprim::plus<float>>,
    RocprimTypeInvokeResultParams<int32_t, device_plus<int32_t>>,
    RocprimTypeInvokeResultParams<rocprim::bfloat16, device_plus<rocprim::bfloat16>>,
    RocprimTypeInvokeResultParams<rocprim::half, device_plus<rocprim::half>>,
    RocprimTypeInvokeResultParams<int32_t, rocprim::equal_to<int32_t>, bool>,
    RocprimTypeInvokeResultParams<rocprim::bfloat16, rocprim::equal_to<rocprim::bfloat16>, bool>,
    RocprimTypeInvokeResultParams<rocprim::half, rocprim::equal_to<rocprim::half>, bool>>;

TYPED_TEST_SUITE(RocprimInvokeResultBinOpTests, RocprimInvokeResultBinOpTestsParams);

TYPED_TEST(RocprimInvokeResultBinOpTests, HostInvokeResult)
{
    using input_type      = typename TestFixture::input_type;
    using binary_function = typename TestFixture::function;
    using expected_type   = typename TestFixture::expected_type;

    using resulting_type = ::rocprim::accumulator_t<binary_function, input_type>;

    // Compile and check on host
    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}

template<typename FromType, typename ToType>
struct static_cast_op
{
    __device__ inline constexpr ToType operator()(FromType a) const
    {
        return static_cast<ToType>(a);
    }
};

template<class Params>
class RocprimInvokeResultUnOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

using RocprimInvokeResultUnOpTestsParams = ::testing::Types<
    RocprimTypeInvokeResultParams<uint16_t, static_cast_op<uint16_t, float>, float>,
    RocprimTypeInvokeResultParams<double,
                                  static_cast_op<double, rocprim::bfloat16>,
                                  rocprim::bfloat16>,
    RocprimTypeInvokeResultParams<uint8_t, rocprim::identity<uint8_t>>>;

TYPED_TEST_SUITE(RocprimInvokeResultUnOpTests, RocprimInvokeResultUnOpTestsParams);

TYPED_TEST(RocprimInvokeResultUnOpTests, HostInvokeResult)
{
    using input_type     = typename TestFixture::input_type;
    using unary_function = typename TestFixture::function;
    using expected_type  = typename TestFixture::expected_type;

    using resulting_type = rocprim::invoke_result_t<unary_function, input_type>;

    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}
