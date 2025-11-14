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

#include <iterator>
#include <stdio.h>
#include <typeinfo>

#include <hipcub/iterator/arg_index_input_iterator.hpp>
#include <hipcub/iterator/cache_modified_input_iterator.hpp>
#include <hipcub/iterator/cache_modified_output_iterator.hpp>
#include <hipcub/iterator/constant_input_iterator.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>
#include <hipcub/iterator/tex_obj_input_iterator.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>

#include <hipcub/util_allocator.hpp>

#include "common_test_header.hpp"

hipcub::CachingDeviceAllocator  g_allocator;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#define INTEGER_SEED (0)

// Params for tests
template<class InputType>
struct IteratorParams
{
    using input_type = InputType;
};

template<class Params>
class HipcubIteratorTests : public ::testing::Test
{
    public:
    using input_type = typename Params::input_type;
};

using HipcubIteratorTestsParams = ::testing::Types<IteratorParams<int8_t>,
                                                   IteratorParams<int16_t>,
                                                   IteratorParams<int32_t>,
                                                   IteratorParams<int64_t>
                                                   //IteratorParams<float>
                                                   >;

static std::vector<int32_t> base_values = {0, 99};

// TODO need to implement the seeding like CUB
template<typename T>
__host__ __device__ __forceinline__ void
InitValue(uint32_t seed, T& value, uint32_t index = 0)
{
    (void) seed;
    value = (index > 0);
}

template <typename T>
struct TransformOp
{
    // Increment transform
    __host__ __device__ __forceinline__ T operator()(T input) const
    {
        T addend;
        InitValue(INTEGER_SEED, addend, 1);
        return input + addend;
    }
};

struct SelectOp
{
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(T input) const
    {
        (void) input;
        return true;
    }
};

//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
* Test random access input iterator
*/
template<
    typename InputIteratorT,
    typename T,
    typename OutputIteratorT = InputIteratorT
>
__global__ void Kernel(
    InputIteratorT    d_in,
    T                 *d_out,
    OutputIteratorT    *d_itrs)
{
    d_out[0] = *d_in;               // Value at offset 0
    d_out[1] = d_in[100];           // Value at offset 100
    d_out[2] = *(d_in + 1000);      // Value at offset 1000
    d_out[3] = *(d_in + 10000);     // Value at offset 10000

    d_in++;
    d_out[4] = d_in[0];             // Value at offset 1

    d_in += 20;
    d_out[5] = d_in[0];             // Value at offset 21
    d_itrs[0] = d_in;               // Iterator at offset 21

    d_in -= 10;
    d_out[6] = d_in[0];             // Value at offset 11;

    d_in -= 11;
    d_out[7] = d_in[0];             // Value at offset 0
    d_itrs[1] = d_in;               // Iterator at offset 0
}

template<typename IteratorType, typename T>
void iterator_test_function(IteratorType d_itr, std::vector<T> &h_reference)
{
    std::vector<T> output(h_reference.size());

    IteratorType* d_itrs = nullptr;
    HIP_CHECK(hipMalloc(&d_itrs, sizeof(IteratorType) * 2));

    IteratorType *h_itrs = (IteratorType*)malloc(sizeof(IteratorType) * 2);

    T* device_output;
    HIP_CHECK(g_allocator.DeviceAllocate((void**)&device_output, output.size() * sizeof(T)));

    // Run unguarded kernel
    Kernel<<<1, 1>>>(d_itr, device_output, d_itrs);

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
                )
            );

    HIP_CHECK(
            hipMemcpy(
                h_itrs, d_itrs,
                sizeof(IteratorType) * 2,
                hipMemcpyDeviceToHost
                )
            );

    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], h_reference[i]) << i;
    }

    IteratorType h_itr = d_itr + 21;
    ASSERT_TRUE(h_itr == h_itrs[0]);
    ASSERT_TRUE(d_itr == h_itrs[1]);

    HIP_CHECK(g_allocator.DeviceFree(device_output));
    HIP_CHECK(hipFree(d_itrs));
    free(h_itrs);

}

TYPED_TEST_SUITE(HipcubIteratorTests, HipcubIteratorTestsParams);

TYPED_TEST(HipcubIteratorTests, TestCacheModifiedInput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using IteratorType = hipcub::CacheModifiedInputIterator<hipcub::LOAD_CG, T>;

    constexpr uint32_t array_size = 8;
    constexpr int TEST_VALUES = 11000;

    std::vector<T> h_data(TEST_VALUES);
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
    }

    std::vector<T> h_reference(array_size);
    h_reference[0] = h_data[0];          // Value at offset 0
    h_reference[1] = h_data[100];        // Value at offset 100
    h_reference[2] = h_data[1000];       // Value at offset 1000
    h_reference[3] = h_data[10000];      // Value at offset 10000
    h_reference[4] = h_data[1];          // Value at offset 1
    h_reference[5] = h_data[21];         // Value at offset 21
    h_reference[6] = h_data[11];         // Value at offset 11
    h_reference[7] = h_data[0];          // Value at offset 0;

    T* d_data = nullptr;
    HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));

    HIP_CHECK(hipMemcpy(d_data, h_data.data(), TEST_VALUES * sizeof(T), hipMemcpyHostToDevice));

    IteratorType d_itr((T*) d_data);
    iterator_test_function<IteratorType, T>(d_itr, h_reference);

    HIP_CHECK(g_allocator.DeviceFree(d_data));
}

TYPED_TEST(HipcubIteratorTests, TestConstant)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using IteratorType            = rocprim::constant_iterator<T>;
    constexpr uint32_t array_size = 8;

    std::vector<T> h_reference(array_size);

    for(uint32_t base_index = 0; base_index < base_values.size(); base_index++)
    {
        T base_value = (T)base_values[base_index];

        IteratorType d_itr(base_value);

        for(uint32_t i = 0; i < h_reference.size(); i++)
        {
            h_reference[i] = base_value;
        }

        iterator_test_function<IteratorType, T>(d_itr, h_reference);
    }
}

TYPED_TEST(HipcubIteratorTests, TestCounting)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using IteratorType            = rocprim::counting_iterator<T>;
    constexpr uint32_t array_size = 8;

    std::vector<T> h_reference(array_size);

    for(uint32_t base_index = 0; base_index < base_values.size(); base_index++)
    {
        T base_value = (T)base_values[base_index];

        IteratorType d_itr(base_value);

        h_reference[0] = base_value + 0;          // Value at offset 0
        h_reference[1] = base_value + 100;        // Value at offset 100
        h_reference[2] = base_value + 1000;       // Value at offset 1000
        h_reference[3] = base_value + 10000;      // Value at offset 10000
        h_reference[4] = base_value + 1;          // Value at offset 1
        h_reference[5] = base_value + 21;         // Value at offset 21
        h_reference[6] = base_value + 11;         // Value at offset 11
        h_reference[7] = base_value + 0;          // Value at offset 0;

        iterator_test_function<IteratorType, T>(d_itr, h_reference);
    }
}

TYPED_TEST(HipcubIteratorTests, TestTransform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using CastT = typename TestFixture::input_type;
    using IteratorType        = rocprim::transform_iterator<CastT*, TransformOp<T>, T>;
    constexpr int TEST_VALUES = 11000;

    std::vector<T> h_data(TEST_VALUES);
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
    }

    // Allocate device arrays
    T* d_data = nullptr;
    HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));

    HIP_CHECK(
        hipMemcpy(
            d_data, h_data.data(),
            TEST_VALUES * sizeof(T),
            hipMemcpyHostToDevice
            )
        );

    TransformOp<T> op;

    // Initialize reference data
    constexpr uint32_t array_size = 8;
    std::vector<T> h_reference(array_size);
    h_reference[0] = op(h_data[0]);          // Value at offset 0
    h_reference[1] = op(h_data[100]);        // Value at offset 100
    h_reference[2] = op(h_data[1000]);       // Value at offset 1000
    h_reference[3] = op(h_data[10000]);      // Value at offset 10000
    h_reference[4] = op(h_data[1]);          // Value at offset 1
    h_reference[5] = op(h_data[21]);         // Value at offset 21
    h_reference[6] = op(h_data[11]);         // Value at offset 11
    h_reference[7] = op(h_data[0]);          // Value at offset 0;

    IteratorType d_itr((CastT*) d_data, op);

    iterator_test_function<IteratorType, T>(d_itr, h_reference);

    HIP_CHECK(g_allocator.DeviceFree(d_data));
}

TYPED_TEST(HipcubIteratorTests, TestTexObj)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::string deviceName = std::string(props.gcnArchName);
    if (deviceName.rfind("gfx94", 0) == 0 || deviceName.rfind("gfx120") == 0 || deviceName.rfind("gfx95") == 0) {
        // This is a gfx94x or gfx120x device, so skip this test
        GTEST_SKIP() << "Test not run on gfx94x, gfx120x or gfx95x as texture cache API is not supported";
    }

    HIP_CHECK(hipSetDevice(device_id));

    using T            = typename TestFixture::input_type;
    using CastT        = typename TestFixture::input_type;
    using IteratorType = hipcub::TexObjInputIterator<T>;

    //
    // Test iterator manipulation in kernel
    //

    constexpr uint32_t TEST_VALUES       = 11000;
    constexpr uint32_t DUMMY_OFFSET      = 500;
    constexpr uint32_t DUMMY_TEST_VALUES = TEST_VALUES - DUMMY_OFFSET;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        std::vector<T> output
            = test_utils::get_random_data<T>(TEST_VALUES, T(2), T(200), seed_value);

        // Allocate device arrays
        T* d_data  = nullptr;
        T* d_dummy = nullptr;
        HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
        HIP_CHECK(hipMemcpy(d_data, output.data(), sizeof(T) * TEST_VALUES, hipMemcpyHostToDevice));

        HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_dummy, sizeof(T) * DUMMY_TEST_VALUES));
        HIP_CHECK(hipMemcpy(d_dummy,
                            output.data() + DUMMY_OFFSET,
                            sizeof(T) * DUMMY_TEST_VALUES,
                            hipMemcpyHostToDevice));

        // Initialize reference data
        constexpr uint32_t array_size = 8;
        std::vector<T>     h_reference(array_size);
        h_reference[0] = output[0]; // Value at offset 0
        h_reference[1] = output[100]; // Value at offset 100
        h_reference[2] = output[1000]; // Value at offset 1000
        h_reference[3] = output[10000]; // Value at offset 10000
        h_reference[4] = output[1]; // Value at offset 1
        h_reference[5] = output[21]; // Value at offset 21
        h_reference[6] = output[11]; // Value at offset 11
        h_reference[7] = output[0]; // Value at offset 0;

        // Create and bind obj-based test iterator
        IteratorType d_obj_itr;
        HIP_CHECK(d_obj_itr.BindTexture((CastT*)d_data, sizeof(T) * TEST_VALUES));

        // Create and bind dummy iterator of same type to check with interference
        IteratorType d_obj_itr2;
        HIP_CHECK(d_obj_itr2.BindTexture((CastT*)d_dummy, sizeof(T) * DUMMY_TEST_VALUES));

        iterator_test_function<IteratorType, T>(d_obj_itr, h_reference);

        HIP_CHECK(d_obj_itr.UnbindTexture());
        HIP_CHECK(d_obj_itr2.UnbindTexture());

        HIP_CHECK(g_allocator.DeviceFree(d_data));
        HIP_CHECK(g_allocator.DeviceFree(d_dummy));
    }
}

TYPED_TEST(HipcubIteratorTests, TestTexRef)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::string deviceName = std::string(props.gcnArchName);
    if (deviceName.rfind("gfx94", 0) == 0 || deviceName.rfind("gfx120") == 0 || deviceName.rfind("gfx95") == 0){
        // This is a gfx94x or gfx120x device, so skip this test
        GTEST_SKIP() << "Test not run on gfx94x, gfx120x or gfx95x as texture cache API is not supported";
    }

    HIP_CHECK(hipSetDevice(device_id));

    using T            = typename TestFixture::input_type;
    using CastT        = typename TestFixture::input_type;
    using IteratorType = hipcub::TexObjInputIterator<T, std::ptrdiff_t>;

    //
    // Test iterator manipulation in kernel
    //

    constexpr uint32_t TEST_VALUES       = 11000;
    constexpr uint32_t DUMMY_OFFSET      = 500;
    constexpr uint32_t DUMMY_TEST_VALUES = TEST_VALUES - DUMMY_OFFSET;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        std::vector<T> output
            = test_utils::get_random_data<T>(TEST_VALUES, T(2), T(200), seed_value);

        // Allocate device arrays
        T* d_data  = nullptr;
        T* d_dummy = nullptr;
        HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
        HIP_CHECK(hipMemcpy(d_data, output.data(), sizeof(T) * TEST_VALUES, hipMemcpyHostToDevice));

        HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_dummy, sizeof(T) * DUMMY_TEST_VALUES));
        HIP_CHECK(hipMemcpy(d_dummy,
                            output.data() + DUMMY_OFFSET,
                            sizeof(T) * DUMMY_TEST_VALUES,
                            hipMemcpyHostToDevice));

        // Initialize reference data
        constexpr uint32_t array_size = 8;
        std::vector<T>     h_reference(array_size);
        h_reference[0] = output[0]; // Value at offset 0
        h_reference[1] = output[100]; // Value at offset 100
        h_reference[2] = output[1000]; // Value at offset 1000
        h_reference[3] = output[10000]; // Value at offset 10000
        h_reference[4] = output[1]; // Value at offset 1
        h_reference[5] = output[21]; // Value at offset 21
        h_reference[6] = output[11]; // Value at offset 11
        h_reference[7] = output[0]; // Value at offset 0;

        // Create and bind ref-based test iterator
        IteratorType d_ref_itr;
        HIP_CHECK(d_ref_itr.BindTexture((CastT*)d_data, sizeof(T) * TEST_VALUES));

        // Create and bind dummy iterator of same type to check with interference
        IteratorType d_ref_itr2;
        HIP_CHECK(d_ref_itr2.BindTexture((CastT*)d_dummy, sizeof(T) * DUMMY_TEST_VALUES));

        iterator_test_function<IteratorType, T>(d_ref_itr, h_reference);

        HIP_CHECK(d_ref_itr.UnbindTexture());
        HIP_CHECK(d_ref_itr2.UnbindTexture());

        HIP_CHECK(g_allocator.DeviceFree(d_data));
        HIP_CHECK(g_allocator.DeviceFree(d_dummy));
    }
}

TYPED_TEST(HipcubIteratorTests, TestTexTransform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::string deviceName = std::string(props.gcnArchName);
    if (deviceName.rfind("gfx94", 0) == 0 || deviceName.rfind("gfx120") == 0 || deviceName.rfind("gfx95") == 0) {
        // This is a gfx94x or gfx120x device, so skip this test
        GTEST_SKIP() << "Test not run on gfx94x, gfx120x or gfx95x as texture cache API is not supported";
    }

    HIP_CHECK(hipSetDevice(device_id));

    using T                   = typename TestFixture::input_type;
    using TextureIteratorType = hipcub::TexObjInputIterator<T, std::ptrdiff_t>;

    constexpr uint32_t TEST_VALUES = 11000;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        std::vector<T> output
            = test_utils::get_random_data<T>(TEST_VALUES, T(2), T(200), seed_value);

        // Allocate device arrays
        T* d_data = nullptr;
        HIP_CHECK(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
        HIP_CHECK(hipMemcpy(d_data, output.data(), sizeof(T) * TEST_VALUES, hipMemcpyHostToDevice));

        TransformOp<T> op;

        // Initialize reference data
        constexpr uint32_t array_size = 8;
        std::vector<T>     h_reference(array_size);
        h_reference[0] = op(output[0]); // Value at offset 0
        h_reference[1] = op(output[100]); // Value at offset 100
        h_reference[2] = op(output[1000]); // Value at offset 1000
        h_reference[3] = op(output[10000]); // Value at offset 10000
        h_reference[4] = op(output[1]); // Value at offset 1
        h_reference[5] = op(output[21]); // Value at offset 21
        h_reference[6] = op(output[11]); // Value at offset 11
        h_reference[7] = op(output[0]); // Value at offset 0;

        // Create and bind ref-based test iterator
        TextureIteratorType d_tex_itr;
        HIP_CHECK(d_tex_itr.BindTexture(d_data, sizeof(T) * TEST_VALUES));

        // Create transform iterator
        rocprim::transform_iterator<TextureIteratorType, TransformOp<T>, T> xform_itr(d_tex_itr,
                                                                                      op);

        iterator_test_function<rocprim::transform_iterator<TextureIteratorType, TransformOp<T>, T>,
                               T>(xform_itr, h_reference);
        HIP_CHECK(g_allocator.DeviceFree(d_data));
    }
}
