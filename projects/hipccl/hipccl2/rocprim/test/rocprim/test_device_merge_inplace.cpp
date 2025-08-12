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

#include "../../common/utils.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

#include "test_utils_sort_checker.hpp"

#include <rocprim/device/device_merge_inplace.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

TEST(RocprimDeviceMergeInplaceTests, Basic)
{
    using value_type    = int;
    size_t storage_size = 0;

    int left_size  = 128 * 1024;
    int right_size = 128 * 1024;

    int total_size = left_size + right_size;

    std::vector<value_type> h_data(total_size);

    // fill all even values
    for(int i = 0; i < left_size; ++i)
    {
        h_data[i] = i * 2;
    }

    // fill all odd
    for(int i = 0; i < right_size; ++i)
    {
        h_data[left_size + i] = i * 2 + 1;
    }

    common::device_ptr<value_type>     d_data(h_data);
    std::vector<value_type>            h_expected(h_data);

    // get temporary storage
    HIP_CHECK(rocprim::merge_inplace(nullptr, storage_size, d_data.get(), left_size, right_size));

    common::device_ptr<void> d_temp_storage(storage_size);

    HIP_CHECK(rocprim::merge_inplace(d_temp_storage.get(),
                                     storage_size,
                                     d_data.get(),
                                     left_size,
                                     right_size));

    std::inplace_merge(h_expected.begin(), h_expected.begin() + left_size, h_expected.end());

    h_data = d_data.load();
    d_data.free_manually();

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_data, h_expected));
}

struct small_sizes
{
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple(0, 0),
            std::make_tuple(2, 1),
            std::make_tuple(10, 10),
            std::make_tuple(111, 111),
            std::make_tuple(128, 1289),
            std::make_tuple(12, 1000),
            std::make_tuple(123, 3000),
            std::make_tuple(1024, 512),
            std::make_tuple(2345, 49),
            std::make_tuple(17867, 41),
            std::make_tuple(17867, 34567),
            std::make_tuple(924353, 1723454),
            std::make_tuple(123, 33554432),
            std::make_tuple(33554432, 123),
            std::make_tuple(33554432, 33554432),
            std::make_tuple(34567, (1 << 17) - 1220),
        };
    }
};

struct large_sizes
{
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple((1ULL << 14) - 1652, (1ULL << 27) - 5839),
            std::make_tuple((1ULL << 27) - 2459, (1ULL << 14) - 2134),
            std::make_tuple((1ULL << 27) - 9532, (1ULL << 27) - 8421),
            std::make_tuple((1ULL << 32) + 5327, (1ULL << 32) + 9682),
        };
    }
};

template<typename T, T start, T increment>
struct linear_data_generator
{
    static constexpr bool is_random = false;

    auto get_iterator(seed_type /* seed */)
    {
        return rocprim::make_transform_iterator(rocprim::make_counting_iterator(0),
                                                [](T value) { return value * increment + start; });
    }

    auto get_max_size()
    {
        return increment == 0
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>((std::numeric_limits<T>::max() - start) / abs(increment));
    }
};

template<typename T, int increment, int max_duplicates>
struct random_data_generator
{
    static constexpr bool is_random = true;

    struct random_monotonic
    {
        using difference_type = std::ptrdiff_t;
        using value_type      = T;

        // not all integral types are valid for int distribution
        using dist_value_type
            = std::conditional_t<std::is_integral<T>::value
                                     && !common::is_valid_for_int_distribution<value_type>::value,
                                 int,
                                 value_type>;

        using val_dist_type = std::conditional_t<std::is_integral<T>::value,
                                                 common::uniform_int_distribution<dist_value_type>,
                                                 std::uniform_real_distribution<dist_value_type>>;
        using dup_dist_type = common::uniform_int_distribution<int>;

        seed_type seed;
        int       duplicates;

        std::mt19937  engine{std::random_device{}()};
        val_dist_type val_dist{dist_value_type{1}, dist_value_type{increment}};
        dup_dist_type dup_dist{1, max_duplicates};

        dist_value_type value = dist_value_type{0};

        random_monotonic(seed_type seed) : seed(seed) {}

        int operator*() const
        {
            return test_utils::saturate_cast<value_type>(value);
        }

        void next()
        {
            // consume a duplicate
            duplicates--;

            // if we have duplicates left over, do nothing
            if(duplicates > 0 || value >= std::numeric_limits<value_type>::max() - increment)
            {
                return;
            }

            // get new duplicates
            duplicates = max_duplicates > 1 ? dup_dist(engine) : 1;
            value += val_dist(engine);
        }

        random_monotonic& operator++()
        {
            // prefix
            next();
            return *this;
        }

        random_monotonic operator++(int)
        {
            // postfix
            random_monotonic retval{*this};
            next();
            return retval;
        }
    };

    auto get_iterator(seed_type seed)
    {
        return random_monotonic{seed};
    }

    auto get_max_size()
    {
        return static_cast<size_t>(std::numeric_limits<size_t>::max());
    }
};

template<typename ValueType,
         typename GenAType,
         typename GenBType,
         typename Sizes     = small_sizes,
         typename CompareOp = ::rocprim::less<ValueType>,
         bool UseGraphs     = false>
struct DeviceMergeInplaceParams
{
    using value_type = ValueType;
    using gen_a_type = GenAType;
    using gen_b_type = GenBType;
    using sizes      = Sizes;
};

typedef ::testing::Types<
    // linear even-odd
    DeviceMergeInplaceParams<int64_t,
                             linear_data_generator<int64_t, 0, 2>,
                             linear_data_generator<int64_t, 1, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 2>,
                             linear_data_generator<int32_t, 1, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             linear_data_generator<int16_t, 0, 2>,
                             linear_data_generator<int16_t, 1, 2>>,
    // linear edge cases
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 1>,
                             linear_data_generator<int32_t, 0, 4>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 4>,
                             linear_data_generator<int32_t, 0, 1>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 128, 0>,
                             linear_data_generator<int32_t, 0, 1>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 1>,
                             linear_data_generator<int32_t, 128, 0>>,
    // random data
    DeviceMergeInplaceParams<int64_t,
                             random_data_generator<int64_t, 2, 2>,
                             random_data_generator<int64_t, 2, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2, 2>,
                             random_data_generator<int32_t, 2, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             random_data_generator<int16_t, 2, 2>,
                             random_data_generator<int16_t, 2, 2>>,
    DeviceMergeInplaceParams<float,
                             random_data_generator<float, 2, 2>,
                             random_data_generator<float, 2, 2>>,
    // large input sizes
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2, 4>,
                             random_data_generator<int32_t, 2, 4>,
                             large_sizes>,
    DeviceMergeInplaceParams<int64_t,
                             random_data_generator<int64_t, 4, 4>,
                             random_data_generator<int64_t, 4, 4>,
                             large_sizes>>
    DeviceMergeInplaceTestsParams;

template<typename Params>
struct DeviceMergeInplaceTests : public testing::Test
{
    using value_type = typename Params::value_type;
    using gen_a_type = typename Params::gen_a_type;
    using gen_b_type = typename Params::gen_b_type;
    using sizes      = typename Params::sizes;
};

TYPED_TEST_SUITE(DeviceMergeInplaceTests, DeviceMergeInplaceTestsParams);

TYPED_TEST(DeviceMergeInplaceTests, MergeInplace)
{
    using value_type = typename TestFixture::value_type;
    using gen_a_type = typename TestFixture::gen_a_type;
    using gen_b_type = typename TestFixture::gen_b_type;
    using binary_op  = rocprim::less<value_type>;

    auto sizes = typename TestFixture::sizes{}();

    binary_op compare_op{};

    hipStream_t stream = hipStreamDefault;

    for(auto size : sizes)
    {
        size_t num_seeds = gen_a_type::is_random || gen_b_type::is_random ? number_of_runs : 1;

        size_t size_a     = std::get<0>(size);
        size_t size_b     = std::get<1>(size);
        size_t size_total = size_a + size_b;

        // hipMallocManaged() currently doesnt support zero byte allocation
        if((size_a == 0 || size_b == 0) && common::use_hmm())
        {
            continue;
        }

        auto gen_a = gen_a_type{};
        auto gen_b = gen_b_type{};

        // don't test sizes more than we can actually generate
        if(size_a > gen_a.get_max_size() || size_b > gen_b.get_max_size())
        {
            continue;
        }

        std::vector<value_type> h_data(size_total);

        size_t total_bytes = sizeof(value_type) * size_total;

        // Limit the total size to slightly saner numbers.
        if(total_bytes >= 1LL << 35 /* 32 GiB */)
        {
            continue;
        }

        size_t storage_size = 0;
        HIP_CHECK(rocprim::merge_inplace(nullptr,
                                         storage_size,
                                         static_cast<value_type*>(nullptr),
                                         size_a,
                                         size_b,
                                         compare_op,
                                         stream));

        // ensure tests always fit on device
        HIP_CHECK(hipSetDevice(hipGetStreamDeviceId(stream)));
        size_t free_vram;
        size_t available_vram;
        HIP_CHECK(hipMemGetInfo(&free_vram, &available_vram));
        if(available_vram < total_bytes + storage_size)
        {
            continue;
        }

        // We only allocate on the device *after* ensuring we have enough available vram.
        common::device_ptr<value_type> d_data = common::device_ptr<value_type>(h_data);
        common::device_ptr<void>       d_temp_storage(storage_size);

        for(size_t seed_index = 0; seed_index < num_seeds; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
            auto gen_a_it = gen_a.get_iterator(seed_value);
            auto gen_b_it = gen_b.get_iterator(seed_value + 1);

            // generate left array
            for(size_t i = 0; i < size_a; ++i)
            {
                h_data[i] = static_cast<value_type>(*(gen_a_it++));
            }

            // generate right array
            for(size_t i = 0; i < size_b; ++i)
            {
                h_data[size_a + i] = static_cast<value_type>(*(gen_b_it++));
            }

            // move input to device
            d_data.store_async(h_data, stream);

            // run merge in place
            HIP_CHECK(rocprim::merge_inplace(d_temp_storage.get(),
                                             storage_size,
                                             d_data.get(),
                                             size_a,
                                             size_b,
                                             compare_op,
                                             stream));

            // check if is sorted on device
            bool is_sorted
                = test_utils::device_sort_check(d_data.get(), size_total, compare_op, stream);

            // skip host-side reference check with large inputs
            if(size_total > 16ULL * 1024 * 1024)
            {
                // input too big, only check device sort
                ASSERT_TRUE(is_sorted);
                continue;
            }

            // compare with reference
            auto h_output = d_data.load_async(stream);

            // compute reference
            std::vector<value_type> h_reference(size_a + size_b);
            std::merge(h_data.begin(),
                       h_data.begin() + size_a,
                       h_data.begin() + size_a,
                       h_data.end(),
                       h_reference.begin());

            // assert on host first, as this will print the offending value and index
            ASSERT_NO_FATAL_FAILURE((test_utils::assert_eq(h_output, h_reference)));

            // then check the result from device for good measure
            ASSERT_TRUE(is_sorted);
        }

        d_data.free_manually();
        d_temp_storage.free_manually();
    }
}
