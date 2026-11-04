// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// Utils
#include "test_utils.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"

// hipcub API
#include <hipcub/device/device_find.hpp>

// Predicate functors
template<typename T>
struct value_predicate_t
{
    T                 value_;
    test_utils::equal equal_op{};

    _CCCL_HOST_DEVICE
    inline bool       operator()(const T x) const
    {
        return equal_op(x, value_);
    }
};

template<typename T>
struct custom_value_predicate_t
{
    test_utils::custom_test_type<T> value_;
    test_utils::equal               equal_op{};

    template<class U>
    _CCCL_HOST_DEVICE
    inline bool operator()(test_utils::custom_test_type<U> x)
    {
        return equal_op(x.x, value_.x);
    }
};

// Params for tests
template<class Type,
         class OutputType,
         class PredicateType = value_predicate_t<Type>,
         bool UseGraphs      = false>
struct DeviceFindIfParams
{
    using type                       = Type;
    using output_type                = OutputType;
    using predicate_type             = PredicateType;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class HipcubDeviceFindIfTests : public ::testing::Test
{
public:
    using type                       = typename Params::type;
    using output_type                = typename Params::output_type;
    using predicate_type             = typename Params::predicate_type;
    static constexpr bool use_graphs = Params::use_graphs;
};

using HipcubDeviceFindIfTestsParams
    = ::testing::Types<DeviceFindIfParams<int8_t, int32_t>,
                       DeviceFindIfParams<int16_t, int32_t>,
                       DeviceFindIfParams<int32_t, int32_t>,
                       DeviceFindIfParams<int64_t, int32_t>,
                       DeviceFindIfParams<test_utils::custom_test_type<int32_t>,
                                          int32_t,
                                          custom_value_predicate_t<int32_t>>,
#if _CCCL_HAS_INT128()
                       DeviceFindIfParams<__int128_t, int32_t>,
#endif

                       DeviceFindIfParams<int8_t, int64_t>,
                       DeviceFindIfParams<int16_t, int64_t>,
                       DeviceFindIfParams<int32_t, int64_t>,
                       DeviceFindIfParams<int64_t, int64_t>,
                       DeviceFindIfParams<test_utils::custom_test_type<int32_t>,
                                          int64_t,
                                          custom_value_predicate_t<int32_t>>
#if _CCCL_HAS_INT128()
                       ,
                       DeviceFindIfParams<__int128_t, int64_t>
#endif
                       >;

TYPED_TEST_SUITE(HipcubDeviceFindIfTests, HipcubDeviceFindIfTestsParams);

template<typename TestFixture, size_t MinOffset = 0, size_t MaxOffset = 1>
void test_find_if()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type           = typename TestFixture::type;
    using output_type    = typename TestFixture::output_type;
    using predicate_type = typename TestFixture::predicate_type;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            for(size_t offset = MinOffset; offset < MaxOffset; ++offset)
            {
                if(offset > 0 && size <= offset)
                {
                    continue;
                }

                SCOPED_TRACE(testing::Message() << "with offset = " << offset);

                // Starting point is an appoximate position of the first match we want to test for
                for(double starting_point : {0.0, 0.234, 0.876, 1.0, 100.0})
                {
                    SCOPED_TRACE(testing::Message() << "with starting_point = " << starting_point);

                    // Generate data
                    std::vector<type> input(size);
                    const type        find_value{42};

                    // Generate the input data in such a way that it does not contain any values from
                    // keys before the starting point
                    const size_t size1
                        = starting_point >= 1.0 ? size : static_cast<size_t>(size * starting_point);
                    const size_t size2 = size - size1;
                    if(size1 > 0)
                    {
                        auto input1
                            = test_utils::get_random_data<type>(size1, 43, 100, seed_value + 2);
                        std::copy(input1.begin(), input1.end(), input.begin());
                    }
                    if(size2 > 0)
                    {
                        auto input2
                            = test_utils::get_random_data<type>(size2, 0, 42, seed_value + 3);
                        std::copy(input2.begin(), input2.end(), input.begin() + size1);
                    }

                    // Explicitly test for boundary cases
                    if(size > 0)
                    {
                        if(starting_point == 0.0)
                        {
                            input[0] = find_value;
                        }
                        else if(starting_point == 1.0)
                        {
                            input[size - 1] = find_value;
                        }
                    }

                    type* d_input;
                    HIP_CHECK(
                        test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(type)));
                    HIP_CHECK(hipMemcpy(d_input,
                                        input.data(),
                                        input.size() * sizeof(type),
                                        hipMemcpyHostToDevice));
                    output_type* d_output;
                    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(output_type)));

                    // compare function
                    predicate_type predicate{find_value};

                    // Get size of d_temp_storage
                    size_t temp_storage_size_bytes;
                    HIP_CHECK(hipcub::DeviceFind::FindIf(nullptr,
                                                         temp_storage_size_bytes,
                                                         d_input + offset,
                                                         d_output,
                                                         predicate,
                                                         size - offset,
                                                         stream));

                    // temp_storage_size_bytes must be >0
                    ASSERT_GT(temp_storage_size_bytes, 0);

                    // allocate temporary storage
                    void* d_temp_storage{};
                    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

                    test_utils::GraphHelper gHelper;
                    if(TestFixture::use_graphs)
                    {
                        gHelper.startStreamCapture(stream);
                    }

                    // Run
                    HIP_CHECK(hipcub::DeviceFind::FindIf(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_input + offset,
                                                         d_output,
                                                         predicate,
                                                         size - offset,
                                                         stream));

                    if(TestFixture::use_graphs)
                    {
                        gHelper.createAndLaunchGraph(stream);
                    }

                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());

                    // Copy output to host
                    output_type output{};
                    HIP_CHECK(
                        hipMemcpy(&output, d_output, sizeof(output_type), hipMemcpyDeviceToHost));

                    HIP_CHECK(hipFree(d_input));
                    HIP_CHECK(hipFree(d_output));
                    HIP_CHECK(hipFree(d_temp_storage));

                    // Check
                    auto expected = std::find_if(input.begin() + offset, input.end(), predicate)
                                    - input.begin() - offset;

                    ASSERT_EQ(output, expected);

                    if(TestFixture::use_graphs)
                    {
                        gHelper.cleanupGraphHelper();
                    }
                }
            }
        }
    }
    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(HipcubDeviceFindIfTests, FindIf)
{
    test_find_if<TestFixture>();
}

TYPED_TEST(HipcubDeviceFindIfTests, FindIfUnaligned)
{
    test_find_if<TestFixture, 1 /*MinOffset*/, 4 /*MaxOffset*/>();
}

TEST(HipcubDeviceFindIfTests, FindIfIterator)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type                  = int32_t;
    using predicate_type        = value_predicate_t<type>;
    using iota_iterator_t       = test_utils::counting_iterator<type>;
    using transform_op_t        = _HIPCUB_STD::identity;
    using transform_output_op_t = _HIPCUB_STD::negate<type>;

    constexpr size_t min_size = 1;
    constexpr size_t max_size = 10000000;

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Get size
        const type size = test_utils::get_random_value(min_size, max_size, seed_value);
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        const type find_value = test_utils::get_random_value(min_size, max_size, seed_value + 1);
        SCOPED_TRACE(testing::Message() << "with find_value = " << find_value);
        const auto expected = -std::min(find_value, size);

        auto d_input_it = test_utils::make_transform_iterator(iota_iterator_t{0}, transform_op_t{});
        type* d_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(type)));
        auto d_output_it
            = test_utils::make_transform_output_iterator(d_output, transform_output_op_t{});

        // compare function
        predicate_type predicate{find_value};

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK(hipcub::DeviceFind::FindIf(nullptr,
                                             temp_storage_size_bytes,
                                             d_input_it,
                                             d_output_it,
                                             predicate,
                                             size,
                                             stream));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(hipcub::DeviceFind::FindIf(d_temp_storage,
                                             temp_storage_size_bytes,
                                             d_input_it,
                                             d_output_it,
                                             predicate,
                                             size,
                                             stream));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        type output{};
        HIP_CHECK(hipMemcpy(&output, d_output, sizeof(type), hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));

        // Check
        ASSERT_EQ(output, expected);
    }
}

struct NotDefaultConstructible
{
    int value_;

    _CCCL_HOST_DEVICE
    constexpr explicit NotDefaultConstructible(int value)
        : value_(value)
    {}

    _CCCL_HOST_DEVICE
    friend constexpr bool operator==(const NotDefaultConstructible& lhs,
                                     const NotDefaultConstructible& rhs)
    {
        return lhs.value_ == rhs.value_;
    }
    _CCCL_HOST_DEVICE
    friend constexpr bool operator!=(const NotDefaultConstructible& lhs,
                                     const NotDefaultConstructible& rhs)
    {
        return lhs.value_ != rhs.value_;
    }

    _CCCL_HOST_DEVICE
    constexpr operator int() const noexcept
    {
        return value_;
    }
};

struct index_to_value
{
    _CCCL_HOST_DEVICE
    NotDefaultConstructible operator()(int i)
    {
        return NotDefaultConstructible{static_cast<int>(i)};
    }
};

static_assert(!_HIPCUB_STD::is_default_constructible_v<NotDefaultConstructible>,
              "NotDefaultConstructible should not be default constructible");

TEST(HipcubDeviceFindIfTests, FindIfNonDefaultConstructible)
{
    using type           = NotDefaultConstructible;
    using output_type    = int;
    using predicate_type = value_predicate_t<type>;

    constexpr size_t min_size = 1;
    constexpr size_t max_size = 10000000;

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Get size
        const size_t size = test_utils::get_random_value(min_size, max_size, seed_value);
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        const output_type find_value = static_cast<int>(size - 1);
        SCOPED_TRACE(testing::Message() << "with find_value = " << find_value);

        std::vector<type> input(size, NotDefaultConstructible(0));
        output_type       i = 0;
        std::generate(input.begin(), input.end(), [&i] { return NotDefaultConstructible(i++); });

        type* d_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(type)));
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(type), hipMemcpyHostToDevice));
        output_type* d_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(output_type)));

        // compare function
        predicate_type predicate{NotDefaultConstructible(find_value)};

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK(hipcub::DeviceFind::FindIf(nullptr,
                                             temp_storage_size_bytes,
                                             d_input,
                                             d_output,
                                             predicate,
                                             size,
                                             stream));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(hipcub::DeviceFind::FindIf(d_temp_storage,
                                             temp_storage_size_bytes,
                                             d_input,
                                             d_output,
                                             predicate,
                                             size,
                                             stream));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output_type output{};
        HIP_CHECK(hipMemcpy(&output, d_output, sizeof(type), hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));

        // Check
        ASSERT_EQ(output, find_value);
    }
}
