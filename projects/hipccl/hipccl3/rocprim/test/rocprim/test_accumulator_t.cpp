// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../common/utils_custom_type.hpp"

#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits_functions.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>

template<typename T>
struct const_ref_op
{
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE
    constexpr const T&
        operator()([[maybe_unused]] T a, [[maybe_unused]] T b)
    {
        return value;
    }

private:
    static T value;
};

// Params for tests
template<typename InputType, typename ScanOp>
struct AccumulatorParams
{
    using input_type = InputType;
    using op_type    = ScanOp;
};

template<typename Params>
class RocprimAccumulatorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using op_type    = typename Params::op_type;
};

using input_types = ::testing::Types<int8_t,
                                     uint8_t,
                                     int16_t,
                                     uint16_t,
                                     int32_t,
                                     uint32_t,
                                     int64_t,
                                     uint64_t,
                                     ::rocprim::int128_t,
                                     ::rocprim::uint128_t,
                                     float,
                                     double,
                                     ::rocprim::half,
                                     ::rocprim::bfloat16,
                                     ::rocprim::native_half,
                                     ::rocprim::native_bfloat16,
                                     ::common::custom_type<short, short, true>,
                                     ::common::custom_type<int, int, true>,
                                     ::common::custom_type<size_t, size_t, true>,
                                     ::common::custom_type<float, float, true>,
                                     ::common::custom_type<double, double, true>>;

template<typename InputType>
using binary_ops_template = ::testing::Types<::rocprim::less<InputType>,
                                             ::rocprim::less_equal<InputType>,
                                             ::rocprim::greater<InputType>,
                                             ::rocprim::greater_equal<InputType>,
                                             ::rocprim::equal_to<InputType>,
                                             ::rocprim::not_equal_to<InputType>,
                                             ::rocprim::plus<InputType>,
                                             ::rocprim::minus<InputType>,
                                             ::rocprim::multiplies<InputType>,
                                             ::rocprim::maximum<InputType>,
                                             ::rocprim::minimum<InputType>,
                                             const_ref_op<InputType>>;

template<typename...>
struct FlattenHelper;

template<typename... L>
struct FlattenHelper<::testing::Types<L...>>
{
    using type = ::testing::Types<L...>;
};

template<typename... L1, typename... L2, typename... Rest>
struct FlattenHelper<::testing::Types<L1...>, ::testing::Types<L2...>, Rest...>
{
    using type = typename FlattenHelper<::testing::Types<L1..., L2...>, Rest...>::type;
};

template<typename InputType, typename OpType>
struct GenerateEachInputTypeParams;

template<typename InputType, typename... OpType>
struct GenerateEachInputTypeParams<InputType, ::testing::Types<OpType...>>
{
    using type = ::testing::Types<AccumulatorParams<InputType, OpType>...>;
};

template<typename InputType>
struct GenerateAllParams;

template<typename... InputType>
struct GenerateAllParams<::testing::Types<InputType...>>
{
    using type = typename FlattenHelper<
        typename GenerateEachInputTypeParams<InputType,
                                             binary_ops_template<InputType>>::type...>::type;
};

using RocprimAccumulatorTestsParams = GenerateAllParams<input_types>::type;

TYPED_TEST_SUITE(RocprimAccumulatorTests, RocprimAccumulatorTestsParams);

// Test `accumulator_t` with `const_ref_op` and all binary operators in rocPRIM
// This is tested in compile time, so it can be compiled if it's compatible with all binary operators.
TYPED_TEST(RocprimAccumulatorTests, PointerToAccType)
{
    using T  = typename TestFixture::input_type;
    using Op = typename TestFixture::op_type;

    using acc_type                    = ::rocprim::accumulator_t<Op, T>;
    [[maybe_unused]] acc_type* unused = nullptr;
}
