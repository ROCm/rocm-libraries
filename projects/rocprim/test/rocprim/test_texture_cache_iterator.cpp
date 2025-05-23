// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_utils_data_generation.hpp"

// required rocprim headers
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/texture_cache_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// Params for tests
template<class InputType>
struct RocprimTextureCacheIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimTextureCacheIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

using RocprimTextureCacheIteratorTestsParams
    = ::testing::Types<RocprimTextureCacheIteratorParams<int>,
                       RocprimTextureCacheIteratorParams<unsigned int>,
                       RocprimTextureCacheIteratorParams<unsigned char>,
                       RocprimTextureCacheIteratorParams<float>,
                       RocprimTextureCacheIteratorParams<unsigned long long>,
                       RocprimTextureCacheIteratorParams<common::custom_type<int, int, true>>,
                       RocprimTextureCacheIteratorParams<common::custom_type<float, float, true>>>;

TYPED_TEST_SUITE(RocprimTextureCacheIteratorTests, RocprimTextureCacheIteratorTestsParams);

template<class T>
struct transform
{
    __device__ __host__
    constexpr T operator()(const T& a) const
    {
        return a + 5;
    }
};

TYPED_TEST(RocprimTextureCacheIteratorTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::string deviceName = std::string(props.gcnArchName);
    if (deviceName.rfind("gfx94", 0) == 0 || deviceName.rfind("gfx120") == 0 || deviceName.rfind("gfx95") == 0) {
        // This is a gfx94x or gfx120x device, so skip this test
        GTEST_SKIP() << "Test not run on gfx94x, gfx120x or gfx95x as texture cache API is not supported";
    }

    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::texture_cache_iterator<T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const size_t size = 1024;

    hipStream_t stream = 0; // default

    std::vector<T> input(size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::get_random_value<T>(1, 200, seed_value);
        }

        std::vector<T> output(size);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(output.size());

        Iterator x;
        HIP_CHECK(x.bind_texture(d_input.get(), sizeof(T) * input.size()));

        // Calculate expected results on host
        std::vector<T> expected(size);
        std::transform(
            input.begin(),
            input.end(),
            expected.begin(),
            transform<T>()
        );

        // Run
        HIP_CHECK(
            rocprim::transform(x, d_output.get(), size, transform<T>(), stream, debug_synchronous));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output = d_output.load();

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }

        HIP_CHECK(x.unbind_texture());
    }
}
