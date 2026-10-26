// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include <hipcub/device/device_memcpy.hpp>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

#include <stdint.h>

template<class ValueType,
         class SizeType,
         bool     Shuffled   = false,
         uint32_t NumBuffers = 1024,
         uint32_t MaxSize    = 4 * 1024>
struct DeviceBatchMemcpyParams
{
    using value_type                      = ValueType;
    using size_type                       = SizeType;
    static constexpr bool     shuffled    = Shuffled;
    static constexpr uint32_t num_buffers = NumBuffers;
    static constexpr uint32_t max_size    = MaxSize;
};

template<class Params>
struct DeviceBatchMemcpyTests : public ::testing::Test
{
    using value_type                      = typename Params::value_type;
    using size_type                       = typename Params::size_type;
    static constexpr bool     shuffled    = Params::shuffled;
    static constexpr uint32_t num_buffers = Params::num_buffers;
    static constexpr uint32_t max_size    = Params::max_size;
};

using DeviceBatchMemcpyTestsParams = ::testing::Types<
    // Ignore copy/move

    // Unshuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false>,
    DeviceBatchMemcpyParams<uint32_t, uint32_t, false>,
    DeviceBatchMemcpyParams<uint64_t, uint32_t, false>,
// size_type: uint16_t
#ifdef __HIP_PLATFORM_AMD__ // Nvcc doesn't support uint16 as SizeType
    DeviceBatchMemcpyParams<uint8_t, uint16_t, false, 1024, 1024>,
#endif // __HIP_PLATFORM_AMD__
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, false, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, false, 1024, 128 * 1024>,

    // weird amount of buffers
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 3 * 1023>,
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 3 * 1025>,
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 1024 * 1024, 256>,

    // Shuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, uint32_t, true>,
    DeviceBatchMemcpyParams<uint32_t, uint32_t, true>,
    DeviceBatchMemcpyParams<uint64_t, uint32_t, true>,
// size_type: uint16_t
#ifdef __HIP_PLATFORM_AMD__ // Nvcc doesn't support uint16 as SizeType
    DeviceBatchMemcpyParams<uint8_t, uint16_t, true, 1024, 1024>,
#endif // __HIP_PLATFORM_AMD__
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, 1024, 128 * 1024>>;

TYPED_TEST_SUITE(DeviceBatchMemcpyTests, DeviceBatchMemcpyTestsParams);

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_memcpy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_memcpy(
//    [&a0 , &b0 , &c0 , &d0 ], // from (note the order is still just a, b, c, d!)
//    [&a0', &b0', &c0', &d0'], // to   (order is the same as above too!)
//    [3   , 2   , 1   , 2   ]) // size
//
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │b0 │b1 │a0 │a1 │a2 │d0 │d1 │c0 │ buffer x contains buffers a, b, c, d
// └───┴───┴───┴───┴───┴───┴───┴───┘ note that the order of buffers is shuffled!
//  ───┬─── ─────┬───── ───┬─── ───
//     └─────────┼─────────┼───┐
//           ┌───┘     ┌───┘   │ what batch_memcpy does
//           ▼         ▼       ▼
//  ─── ─────────── ─────── ───────
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │c0'│a0'│a1'│a2'│d0'│d1'│b0'│b1'│ buffer y contains buffers a', b', c', d'
// └───┴───┴───┴───┴───┴───┴───┴───┘
template<class T, class S, class RandomGenerator>
std::vector<T> shuffled_exclusive_scan(const std::vector<S>& input, RandomGenerator& rng)
{
    const size_t n = input.size();
    assert(n > 0);

    std::vector<T> result(n);
    std::vector<T> permute(n);

    std::iota(permute.begin(), permute.end(), 0);
    std::shuffle(permute.begin(), permute.end(), rng);

    T sum = 0;
    for(size_t i = 0; i < n; ++i)
    {
        result[permute[i]] = sum;
        sum += input[permute[i]];
    }

    return result;
}

TYPED_TEST(DeviceBatchMemcpyTests, SizeAndTypeVariation)
{
    // While on rocPRIM these can be variable via the config. CUB does not allow this.
    // Therefore we assume fixed size. Otherwise we would use:
    // - rocprim::batch_memcpy_config<>::wlev_size_threshold
    // - rocprim::batch_memcpy_config<>::blev_size_threshold;
    constexpr int32_t wlev_min_size = 128;
    constexpr int32_t blev_min_size = 1024;

    constexpr int32_t num_buffers = TestFixture::num_buffers;
    constexpr int32_t max_size    = TestFixture::max_size;
    constexpr bool    shuffled    = TestFixture::shuffled;

    constexpr int32_t num_tlev_buffers = num_buffers / 3;
    constexpr int32_t num_wlev_buffers = num_buffers / 3;

    using value_type         = typename TestFixture::value_type;
    using buffer_size_type   = typename TestFixture::size_type;
    using buffer_offset_type = uint32_t;
    using byte_offset_type   = size_t;

    using value_alias =
        typename std::conditional<test_utils::is_custom_test_type<value_type>::value,
                                  typename test_utils::inner_type<value_type>::type,
                                  value_type>::type;

    // Get random buffer sizes

    // Number of elements in each buffer.
    std::vector<buffer_size_type> h_buffer_num_elements(num_buffers);

    // Total number of bytes.
    byte_offset_type total_num_bytes    = 0;
    byte_offset_type total_num_elements = 0;

    uint32_t seed = 0;
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);
    std::default_random_engine rng{seed};

    for(buffer_offset_type i = 0; i < num_buffers; ++i)
    {
        buffer_size_type size;
        if(i < num_tlev_buffers)
        {
            size = test_utils::get_random_value<buffer_size_type>(1, wlev_min_size - 1, rng());
        } else if(i < num_tlev_buffers + num_wlev_buffers)
        {
            size = test_utils::get_random_value<buffer_size_type>(wlev_min_size,
                                                                  blev_min_size - 1,
                                                                  rng());
        } else
        {
            size = test_utils::get_random_value<buffer_size_type>(blev_min_size, max_size, rng());
        }

        // convert from number of bytes to number of elements
        size = test_utils::max(1, size / sizeof(value_type));
        size = test_utils::min(size, max_size);

        h_buffer_num_elements[i] = size;
        total_num_elements += size;
    }

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // Get the byte size of each buffer
    std::vector<buffer_size_type> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(value_type);
    }

    // And the total byte size
    total_num_bytes = total_num_elements * sizeof(value_type);

    // Device pointers
    value_type*       d_input{};
    value_type*       d_output{};
    value_type**      d_buffer_srcs{};
    value_type**      d_buffer_dsts{};
    buffer_size_type* d_buffer_sizes{};

    // Calculate temporary storage

    size_t temp_storage_bytes = 0;

    HIP_CHECK(hipcub::DeviceMemcpy::Batched(nullptr,
                                            temp_storage_bytes,
                                            d_buffer_srcs,
                                            d_buffer_dsts,
                                            d_buffer_sizes,
                                            num_buffers));

    void* d_temp_storage{};

    // Allocate memory.
    HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
    HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

    HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(*d_buffer_srcs)));
    HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(*d_buffer_dsts)));
    HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(*d_buffer_sizes)));

    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Generate data.
    std::vector<value_alias> h_input
        = test_utils::get_random_data<value_alias>(total_num_elements,
                                                   test_utils::numeric_limits<value_alias>::min(),
                                                   test_utils::numeric_limits<value_alias>::max(),
                                                   rng());

    // Generate the source and shuffled destination offsets.
    std::vector<buffer_offset_type> src_offsets;
    std::vector<buffer_offset_type> dst_offsets;

    if(shuffled)
    {
        src_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
    } else
    {
        src_offsets = std::vector<buffer_offset_type>(num_buffers);
        dst_offsets = std::vector<buffer_offset_type>(num_buffers);

        test_utils::host_exclusive_scan(h_buffer_num_elements.begin(),
                                        h_buffer_num_elements.end(),
                                        0,
                                        src_offsets.begin(),
                                        test_utils::plus{});
        test_utils::host_exclusive_scan(h_buffer_num_elements.begin(),
                                        h_buffer_num_elements.end(),
                                        0,
                                        dst_offsets.begin(),
                                        test_utils::plus{});
    }

    // Generate the source and destination pointers.
    std::vector<value_type*> h_buffer_srcs(num_buffers);
    std::vector<value_type*> h_buffer_dsts(num_buffers);

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = d_input + src_offsets[i];
        h_buffer_dsts[i] = d_output + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), total_num_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(*d_buffer_srcs),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(*d_buffer_dsts),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(*d_buffer_sizes),
                        hipMemcpyHostToDevice));

    // Run batched memcpy.
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_temp_storage,
                                            temp_storage_bytes,
                                            d_buffer_srcs,
                                            d_buffer_dsts,
                                            d_buffer_sizes,
                                            num_buffers,
                                            hipStreamDefault));
    // Verify results.
    std::vector<value_alias> h_output(total_num_elements);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, total_num_bytes, hipMemcpyDeviceToHost));

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        for(buffer_size_type j = 0; j < h_buffer_num_elements[i]; ++j)
        {
            auto input_index  = src_offsets[i] + j;
            auto output_index = dst_offsets[i] + j;

            ASSERT_TRUE(test_utils::bit_equal(h_input[input_index], h_output[output_index]));
        }
    }

    // De-allocate memory.
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    HIP_CHECK(hipFree(d_buffer_srcs));
    HIP_CHECK(hipFree(d_buffer_dsts));
    HIP_CHECK(hipFree(d_buffer_sizes));

    HIP_CHECK(hipFree(d_temp_storage));
}

TEST(DeviceMemcpyBatched, ZeroBuffersNoOp)
{
    using T         = uint8_t;
    T**     d_srcs  = nullptr;
    T**     d_dsts  = nullptr;
    size_t* d_sizes = nullptr;

    size_t temp_bytes = 0;
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(nullptr, temp_bytes, d_srcs, d_dsts, d_sizes, 0));
    void* d_temp = nullptr;
    if(temp_bytes)
    {
        HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    }

    // Should be a no-op without crashing
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_temp,
                                            temp_bytes,
                                            d_srcs,
                                            d_dsts,
                                            d_sizes,
                                            0,
                                            hipStreamDefault));
    if(d_temp)
    {
        HIP_CHECK(hipFree(d_temp));
    }
}

TEST(DeviceMemcpyBatched, ZeroSizeEntries)
{
    using T                               = uint8_t;
    const int                 num_buffers = 5;
    const std::vector<size_t> h_sizes     = {0, 1, 0, 7, 0};

    size_t total = 0;
    for(auto s : h_sizes)
    {
        total += s;
    }

    T* d_input  = nullptr;
    T* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, total));
    HIP_CHECK(hipMalloc(&d_output, total));

    // Build src/dst arrays
    std::vector<T*> h_srcs(num_buffers), h_dsts(num_buffers);
    size_t          offset = 0;
    for(int i = 0; i < num_buffers; ++i)
    {
        h_srcs[i] = d_input + offset;
        h_dsts[i] = d_output + offset;
        offset += h_sizes[i];
    }

    // Fill input
    std::vector<T> h_in(total);
    std::iota(h_in.begin(), h_in.end(), static_cast<T>(3));
    HIP_CHECK(hipMemcpy(d_input, h_in.data(), total, hipMemcpyHostToDevice));

    // Device arrays
    T**     d_srcs  = nullptr;
    T**     d_dsts  = nullptr;
    size_t* d_sizes = nullptr;
    HIP_CHECK(hipMalloc(&d_srcs, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_dsts, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_sizes, num_buffers * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_srcs, h_srcs.data(), num_buffers * sizeof(T*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dsts, h_dsts.data(), num_buffers * sizeof(T*), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_sizes, h_sizes.data(), num_buffers * sizeof(size_t), hipMemcpyHostToDevice));

    // Temp storage
    size_t temp_bytes = 0;
    HIP_CHECK(
        hipcub::DeviceMemcpy::Batched(nullptr, temp_bytes, d_srcs, d_dsts, d_sizes, num_buffers));
    void* d_temp = nullptr;
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));

    HIP_CHECK(
        hipcub::DeviceMemcpy::Batched(d_temp, temp_bytes, d_srcs, d_dsts, d_sizes, num_buffers));

    // Verify
    std::vector<T> h_out(total);
    HIP_CHECK(hipMemcpy(h_out.data(), d_output, total, hipMemcpyDeviceToHost));
    EXPECT_EQ(h_in, h_out);

    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_sizes));
    HIP_CHECK(hipFree(d_dsts));
    HIP_CHECK(hipFree(d_srcs));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
}

TEST(DeviceMemcpyBatched, NonDefaultStreamAndTempReuse)
{
    using T                                           = uint32_t;
    const int                             num_buffers = 8;
    std::mt19937                          rng(123);
    std::uniform_int_distribution<size_t> dist(1, 4096);
    std::vector<size_t>                   sizes(num_buffers);
    size_t                                total = 0;
    for(int i = 0; i < num_buffers; ++i)
    {
        sizes[i] = dist(rng) * sizeof(T);
        total += sizes[i];
    }

    T* d_in  = nullptr;
    T* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, total));
    HIP_CHECK(hipMalloc(&d_out, total));

    // Fill input
    std::vector<T> h_in(total / sizeof(T));
    for(size_t i = 0; i < h_in.size(); ++i)
        h_in[i] = static_cast<T>(i ^ 0xDEADBEEF);
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), total, hipMemcpyHostToDevice));

    // Build offset/size pairs
    struct Chunk
    {
        size_t offset;
        size_t size;
    };
    std::vector<Chunk> chunks(num_buffers);
    size_t             acc = 0;
    for(int i = 0; i < num_buffers; ++i)
    {
        chunks[i].offset = acc;
        chunks[i].size   = sizes[i];
        acc += sizes[i];
    }
    std::shuffle(chunks.begin(), chunks.end(), rng);

    std::vector<T*>     h_srcs(num_buffers), h_dsts(num_buffers);
    std::vector<size_t> h_sizes(num_buffers);
    for(int i = 0; i < num_buffers; ++i)
    {
        h_srcs[i]  = d_in + chunks[i].offset / sizeof(T);
        h_dsts[i]  = d_out + chunks[i].offset / sizeof(T);
        h_sizes[i] = chunks[i].size;
    }

    T**     d_srcs  = nullptr;
    T**     d_dsts  = nullptr;
    size_t* d_sizes = nullptr;
    HIP_CHECK(hipMalloc(&d_srcs, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_dsts, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_sizes, num_buffers * sizeof(size_t)));

    // Setup stream and event
    hipStream_t setup;
    hipEvent_t  ready;
    HIP_CHECK(hipStreamCreate(&setup));
    HIP_CHECK(hipEventCreateWithFlags(&ready, hipEventDisableTiming));

    HIP_CHECK(hipMemcpyAsync(d_srcs,
                             h_srcs.data(),
                             num_buffers * sizeof(T*),
                             hipMemcpyHostToDevice,
                             setup));
    HIP_CHECK(hipMemcpyAsync(d_dsts,
                             h_dsts.data(),
                             num_buffers * sizeof(T*),
                             hipMemcpyHostToDevice,
                             setup));
    HIP_CHECK(hipMemcpyAsync(d_sizes,
                             h_sizes.data(),
                             num_buffers * sizeof(size_t),
                             hipMemcpyHostToDevice,
                             setup));
    HIP_CHECK(hipEventRecord(ready, setup));

    // Query temp storage
    size_t temp_bytes = 0;
    HIP_CHECK(
        hipcub::DeviceMemcpy::Batched(nullptr, temp_bytes, d_srcs, d_dsts, d_sizes, num_buffers));

    void* d_tempA = nullptr;
    void* d_tempB = nullptr;
    HIP_CHECK(hipMalloc(&d_tempA, temp_bytes));
    HIP_CHECK(hipMalloc(&d_tempB, temp_bytes));

    hipStream_t streamA, streamB;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));
    HIP_CHECK(hipStreamWaitEvent(streamA, ready, 0));
    HIP_CHECK(hipStreamWaitEvent(streamB, ready, 0));

    // Launch batched memcpy
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_tempA,
                                            temp_bytes,
                                            d_srcs,
                                            d_dsts,
                                            d_sizes,
                                            num_buffers,
                                            streamA));
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_tempB,
                                            temp_bytes,
                                            d_srcs,
                                            d_dsts,
                                            d_sizes,
                                            num_buffers,
                                            streamB));

    HIP_CHECK(hipStreamSynchronize(streamA));
    HIP_CHECK(hipStreamSynchronize(streamB));

    // Verify
    std::vector<T> h_out(h_in.size());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, total, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < h_in.size(); ++i)
    {
        EXPECT_EQ(h_in[i], h_out[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    HIP_CHECK(hipEventDestroy(ready));
    HIP_CHECK(hipStreamDestroy(setup));
    HIP_CHECK(hipStreamDestroy(streamA));
    HIP_CHECK(hipStreamDestroy(streamB));
    HIP_CHECK(hipFree(d_tempA));
    HIP_CHECK(hipFree(d_tempB));
    HIP_CHECK(hipFree(d_sizes));
    HIP_CHECK(hipFree(d_dsts));
    HIP_CHECK(hipFree(d_srcs));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_in));
}

struct PackedPair
{
    uint16_t a;
    uint16_t b;
}; // alignment-sensitive
TEST(DeviceMemcpyBatched, PackedStructAlignment)
{
    using T                       = PackedPair;
    const int    num_buffers      = 4;
    const size_t elems_per_buffer = 1024;
    const size_t bytes_per_buffer = elems_per_buffer * sizeof(T);
    const size_t total_bytes      = num_buffers * bytes_per_buffer;

    T* d_in  = nullptr;
    T* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, total_bytes));
    HIP_CHECK(hipMalloc(&d_out, total_bytes));

    std::vector<T> h_in(num_buffers * elems_per_buffer);
    for(size_t i = 0; i < h_in.size(); ++i)
    {
        h_in[i] = T{static_cast<uint16_t>(i), static_cast<uint16_t>(~i)};
    }
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), total_bytes, hipMemcpyHostToDevice));

    std::vector<size_t> h_sizes(num_buffers, bytes_per_buffer);
    std::vector<T*>     h_srcs(num_buffers), h_dsts(num_buffers);
    for(int i = 0; i < num_buffers; ++i)
    {
        h_srcs[i] = d_in + i * elems_per_buffer;
        h_dsts[i] = d_out + i * elems_per_buffer;
    }

    T**     d_srcs  = nullptr;
    T**     d_dsts  = nullptr;
    size_t* d_sizes = nullptr;
    HIP_CHECK(hipMalloc(&d_srcs, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_dsts, num_buffers * sizeof(T*)));
    HIP_CHECK(hipMalloc(&d_sizes, num_buffers * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_srcs, h_srcs.data(), num_buffers * sizeof(T*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dsts, h_dsts.data(), num_buffers * sizeof(T*), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_sizes, h_sizes.data(), num_buffers * sizeof(size_t), hipMemcpyHostToDevice));

    size_t temp_bytes = 0;
    HIP_CHECK(
        hipcub::DeviceMemcpy::Batched(nullptr, temp_bytes, d_srcs, d_dsts, d_sizes, num_buffers));
    void* d_temp = nullptr;
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));

    HIP_CHECK(
        hipcub::DeviceMemcpy::Batched(d_temp, temp_bytes, d_srcs, d_dsts, d_sizes, num_buffers));

    std::vector<T> h_out(h_in.size());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, total_bytes, hipMemcpyDeviceToHost));
    EXPECT_TRUE(std::equal(h_in.begin(),
                           h_in.end(),
                           h_out.begin(),
                           [](const PackedPair& x, const PackedPair& y)
                           { return x.a == y.a && x.b == y.b; }));

    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_sizes));
    HIP_CHECK(hipFree(d_dsts));
    HIP_CHECK(hipFree(d_srcs));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_in));
}
