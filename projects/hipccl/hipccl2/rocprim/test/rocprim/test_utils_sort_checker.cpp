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

#include "../../common/utils_device_ptr.hpp"

#include "test_seed.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_checker.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

template<class InputType, class OpType = rocprim::less<InputType>>
struct RocprimSortCheckerParams
{
    using input_type = InputType;
    using op_type    = OpType;
};

template<class Params>
class RocprimSortCheckerTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using op_type    = typename Params::op_type;
};

using RocprimSortCheckerTestParams = ::testing::Types<
    RocprimSortCheckerParams<unsigned int, rocprim::less<unsigned int>>,
    RocprimSortCheckerParams<unsigned int, test_utils::key_comparator<unsigned int, true, 3u, 32u>>,
    RocprimSortCheckerParams<double>>;

TYPED_TEST_SUITE(RocprimSortCheckerTests, RocprimSortCheckerTestParams);

TYPED_TEST(RocprimSortCheckerTests, TrueTest)
{
    using input_type = typename TestFixture::input_type;
    using op_type    = typename TestFixture::op_type;

    int device_id = test_common_utils::obtain_device_from_ctest();
    HIP_CHECK(hipSetDevice(device_id));

    const auto op = op_type{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            if(!size)
            {
                continue;
            }
            std::vector<input_type> input = test_utils::get_random_data<input_type>(
                size,
                0,
                rocprim::numeric_limits<input_type>::max(),
                ++seed_value);

            std::sort(input.begin(), input.end(), op);

            common::device_ptr<input_type> d_input(input);
            ASSERT_TRUE(test_utils::device_sort_check(d_input.get(), size, op));
        }
    }
    SUCCEED();
}

TEST(RocprimSortCheckerTests, FalseTest)
{
    using input_type = unsigned int;

    int device_id = test_common_utils::obtain_device_from_ctest();
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            if(size > 20)
            {
                return;
            }
            if(!size)
            {
                continue;
            }
            std::vector<input_type> input = test_utils::get_random_data<input_type>(
                size,
                0,
                rocprim::numeric_limits<input_type>::max(),
                ++seed_value);
            bool all_equal = true;
            for(const auto i : input)
            {
                all_equal = all_equal && (i == input[0]);
            }
            if(all_equal)
            {
                continue;
            }
            std::sort(input.begin(), input.end(), rocprim::less<input_type>{});

            common::device_ptr<input_type> d_input(input);
            ASSERT_FALSE(
                test_utils::device_sort_check(d_input.get(), size, rocprim::greater<input_type>{}));
        }
    }
    SUCCEED();
}
