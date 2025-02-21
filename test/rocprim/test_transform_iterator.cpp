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

#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
// #include "test_utils_types.hpp"

// required rocprim headers
#include <rocprim/config.hpp>
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

template<class T>
struct times_two
{
    ROCPRIM_HOST_DEVICE
    T operator()(const T& value) const
    {
        return 2 * value;
    }
};

template<class T>
struct plus_ten
{
    ROCPRIM_HOST_DEVICE
    T operator()(const T& value) const
    {
        return value + 10;
    }
};

// Params for tests
template<
    class InputType,
    class UnaryFunction = times_two<InputType>,
    class ValueType = InputType
>
struct RocprimTransformIteratorParams
{
    using input_type = InputType;
    using value_type = ValueType;
    using unary_function = UnaryFunction;
};

template<class Params>
class RocprimTransformIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using value_type = typename Params::value_type;
    using unary_function = typename Params::unary_function;
    const bool debug_synchronous = false;
};

using RocprimTransformIteratorTestsParams
    = ::testing::Types<RocprimTransformIteratorParams<int, plus_ten<long>>,
                       RocprimTransformIteratorParams<unsigned int>,
                       RocprimTransformIteratorParams<unsigned long>,
                       RocprimTransformIteratorParams<float, plus_ten<double>, double>>;

TYPED_TEST_SUITE(RocprimTransformIteratorTests, RocprimTransformIteratorTestsParams);

TYPED_TEST(RocprimTransformIteratorTests, TransformReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type = typename TestFixture::input_type;
    using value_type = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type = typename rocprim::transform_iterator<
        input_type*, unary_function, value_type
    >;

    hipStream_t stream = 0; // default

    const size_t size = 1024;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<input_type> input
            = test_utils::get_random_data_wrapped<input_type>(size, 1, 200, seed_value);
        std::vector<value_type> output(1);

        common::device_ptr<input_type> d_input(input);
        common::device_ptr<value_type> d_output(output.size());

        auto reduce_op = rocprim::plus<value_type>();
        unary_function transform;

        // Calculate expected results on host
        iterator_type x(input.data(), transform);
        value_type expected = std::accumulate(x, x + size, value_type(0), reduce_op);

        auto d_iter = iterator_type(d_input.get(), transform);
        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::reduce(nullptr,
                                  temp_storage_size_bytes,
                                  d_iter,
                                  d_output.get(),
                                  value_type(0),
                                  input.size(),
                                  reduce_op,
                                  stream));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);
        // Run
        HIP_CHECK(rocprim::reduce(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  d_iter,
                                  d_output.get(),
                                  value_type(0),
                                  input.size(),
                                  reduce_op,
                                  stream,
                                  TestFixture::debug_synchronous));

        HIP_CHECK(hipDeviceSynchronize());

        output = d_output.load();

        // Check if output values are as expected
        test_utils::assert_near(output[0], expected, test_utils::precision<value_type> * size);
    }

}
