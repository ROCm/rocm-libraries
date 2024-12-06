// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "rocprim/detail/various.hpp"
#include "rocprim/device/device_copy.hpp"
#include "rocprim/device/device_memcpy.hpp"
#include "rocprim/intrinsics/thread.hpp"
#include "rocprim/iterator.hpp"

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstring>

#include <stdint.h>

template<class ValueType,
         class SizeType,
         bool         IsMemCpy,
         bool         Shuffled            = false,
         unsigned int NumBuffers          = 1024,
         unsigned int MaxSize             = 4 * 1024,
         bool         UseIndirectIterator = false>
struct DeviceBatchMemcpyParams
{
    using value_type                                    = ValueType;
    using size_type                                     = SizeType;
    static constexpr bool         isMemCpy              = IsMemCpy;
    static constexpr bool         shuffled              = Shuffled;
    static constexpr unsigned int num_buffers           = NumBuffers;
    static constexpr unsigned int max_size              = MaxSize;
    static constexpr bool         use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
struct RocprimDeviceBatchMemcpyTests : public ::testing::Test
{
    using value_type                                    = typename Params::value_type;
    using size_type                                     = typename Params::size_type;
    static constexpr bool         isMemCpy              = Params::isMemCpy;
    static constexpr bool         shuffled              = Params::shuffled;
    static constexpr unsigned int num_buffers           = Params::num_buffers;
    static constexpr unsigned int max_size              = Params::max_size;
    static constexpr bool         use_indirect_iterator = Params::use_indirect_iterator;
};

typedef ::testing::Types<
    // Ignore copy/move
    DeviceBatchMemcpyParams<test_utils::custom_non_copyable_type<uint8_t>,
                            unsigned int,
                            true,
                            false>,
    DeviceBatchMemcpyParams<test_utils::custom_non_moveable_type<uint8_t>,
                            unsigned int,
                            true,
                            false>,
    DeviceBatchMemcpyParams<test_utils::custom_non_default_type<uint8_t>,
                            unsigned int,
                            true,
                            false>,

    // Unshuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, unsigned int, true, false>,
    DeviceBatchMemcpyParams<unsigned int, unsigned int, true, false>,
    DeviceBatchMemcpyParams<uint64_t, unsigned int, true, false>,
    DeviceBatchMemcpyParams<uint8_t, unsigned int, false, false>,
    DeviceBatchMemcpyParams<unsigned int, unsigned int, false, false>,
    DeviceBatchMemcpyParams<uint64_t, unsigned int, false, false>,
    // size_type: uint16_t
    DeviceBatchMemcpyParams<uint8_t, uint16_t, true, false, 1024, 1024>,
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, false, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, false, 1024, 128 * 1024>,

    // weird amount of buffers
    DeviceBatchMemcpyParams<uint8_t, unsigned int, true, false, 3 * 1023>,
    DeviceBatchMemcpyParams<uint8_t, unsigned int, true, false, 3 * 1025>,
    DeviceBatchMemcpyParams<uint8_t, unsigned int, true, false, 1024 * 1024, 256>,

    // Shuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, unsigned int, true, true>,
    DeviceBatchMemcpyParams<unsigned int, unsigned int, true, true>,
    DeviceBatchMemcpyParams<uint64_t, unsigned int, true, true>,
    DeviceBatchMemcpyParams<uint8_t, unsigned int, false, true>,
    DeviceBatchMemcpyParams<unsigned int, unsigned int, false, true>,
    DeviceBatchMemcpyParams<uint64_t, unsigned int, false, true>,
    // size_type: uint16_t
    DeviceBatchMemcpyParams<uint8_t, uint16_t, true, true, 1024, 1024>,
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, true, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, true, 1024, 128 * 1024>,

    // Test iterator input for BatchCopy
    DeviceBatchMemcpyParams<unsigned int, unsigned int, false, false, 1024, 1024 * 4, true>>
    RocprimDeviceBatchMemcpyTestsParams;

TYPED_TEST_SUITE(RocprimDeviceBatchMemcpyTests, RocprimDeviceBatchMemcpyTestsParams);

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

template<bool IsMemCpy,
         class ContainerMemCpy,
         class ContainerCopy,
         class byte_offset_type,
         typename std::enable_if<IsMemCpy, int>::type = 0>
void init_input(ContainerMemCpy& h_input_for_memcpy,
                ContainerCopy& /*h_input_for_copy*/,
                std::mt19937_64& rng,
                byte_offset_type total_num_bytes)
{
    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    h_input_for_memcpy    = std::vector<unsigned char>(num_ints * sizeof(uint64_t));

    // generate_n for uninitialized memory, pragmatically use placement-new, since there are no
    // uint64_t objects alive yet in the storage.
    std::for_each(
        reinterpret_cast<uint64_t*>(h_input_for_memcpy.data()),
        reinterpret_cast<uint64_t*>(h_input_for_memcpy.data() + num_ints * sizeof(uint64_t)),
        [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });
}

template<bool IsMemCpy,
         class ContainerMemCpy,
         class ContainerCopy,
         class byte_offset_type,
         typename std::enable_if<!IsMemCpy, int>::type = 0>
void init_input(ContainerMemCpy& /*h_input_for_memcpy*/,
                ContainerCopy&   h_input_for_copy,
                std::mt19937_64& rng,
                byte_offset_type total_num_bytes)
{
    using value_type = typename ContainerCopy::value_type;

    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    const size_t num_of_elements
        = rocprim::detail::ceiling_div(num_ints * sizeof(uint64_t), sizeof(value_type));
    h_input_for_copy = std::vector<value_type>(num_of_elements);

    // generate_n for uninitialized memory, pragmatically use placement-new, since there are no
    // uint64_t objects alive yet in the storage.
    std::for_each(reinterpret_cast<uint64_t*>(h_input_for_copy.data()),
                  reinterpret_cast<uint64_t*>(h_input_for_copy.data()) + num_ints,
                  [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });
}

template<bool IsMemCpy,
         class InputBufferItType,
         class OutputBufferItType,
         class BufferSizeItType,
         typename std::enable_if<IsMemCpy, int>::type = 0>
void batch_copy(void*              temporary_storage,
                size_t&            storage_size,
                InputBufferItType  sources,
                OutputBufferItType destinations,
                BufferSizeItType   sizes,
                unsigned int       num_copies,
                hipStream_t        stream)
{
    HIP_CHECK(rocprim::batch_memcpy(temporary_storage,
                                    storage_size,
                                    sources,
                                    destinations,
                                    sizes,
                                    num_copies,
                                    stream));
}

template<bool IsMemCpy,
         class InputBufferItType,
         class OutputBufferItType,
         class BufferSizeItType,
         typename std::enable_if<!IsMemCpy, int>::type = 0>
void batch_copy(void*              temporary_storage,
                size_t&            storage_size,
                InputBufferItType  sources,
                OutputBufferItType destinations,
                BufferSizeItType   sizes,
                unsigned int       num_copies,
                hipStream_t        stream)
{
    HIP_CHECK(rocprim::batch_copy(temporary_storage,
                                  storage_size,
                                  sources,
                                  destinations,
                                  sizes,
                                  num_copies,
                                  stream));
}

template<bool IsMemCpy,
         class ContainerMemCpy,
         class ContainerCopy,
         class ptr,
         class OffsetContainer,
         class SizesContainer,
         class byte_offset_type,
         typename std::enable_if<IsMemCpy, int>::type = 0>
void check_result(ContainerMemCpy& h_input_for_memcpy,
                  ContainerCopy& /*h_input_for_copy*/,
                  ptr              d_output,
                  byte_offset_type total_num_bytes,
                  byte_offset_type /*total_num_elements*/,
                  int              num_buffers,
                  OffsetContainer& src_offsets,
                  OffsetContainer& dst_offsets,
                  SizesContainer&  h_buffer_num_bytes)
{
    using value_type                    = typename ContainerCopy::value_type;
    std::vector<unsigned char> h_output = std::vector<unsigned char>(total_num_bytes);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, total_num_bytes, hipMemcpyDeviceToHost));
    for(int i = 0; i < num_buffers; ++i)
    {
        ASSERT_EQ(std::memcmp(h_input_for_memcpy.data() + src_offsets[i] * sizeof(value_type),
                              h_output.data() + dst_offsets[i] * sizeof(value_type),
                              h_buffer_num_bytes[i]),
                  0)
            << "with index = " << i;
    }
}

template<bool IsMemCpy,
         class ContainerMemCpy,
         class ContainerCopy,
         class ptr,
         class OffsetContainer,
         class SizesContainer,
         class byte_offset_type,
         typename std::enable_if<!IsMemCpy, int>::type = 0>
void check_result(ContainerMemCpy& /*h_input_for_memcpy*/,
                  ContainerCopy&   h_input_for_copy,
                  ptr              d_output,
                  byte_offset_type total_num_bytes,
                  byte_offset_type total_num_elements,
                  int              num_buffers,
                  OffsetContainer& src_offsets,
                  OffsetContainer& dst_offsets,
                  SizesContainer&  h_buffer_num_bytes)
{
    using value_type                 = typename ContainerCopy::value_type;
    std::vector<value_type> h_output = std::vector<value_type>(total_num_elements);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, total_num_bytes, hipMemcpyDeviceToHost));
    for(int i = 0; i < num_buffers; ++i)
    {
        ASSERT_EQ(std::memcmp(h_input_for_copy.data() + src_offsets[i],
                              h_output.data() + dst_offsets[i],
                              h_buffer_num_bytes[i]),
                  0)
            << "with index = " << i;
    }
}

TYPED_TEST(RocprimDeviceBatchMemcpyTests, SizeAndTypeVariation)
{
    using value_type         = typename TestFixture::value_type;
    using buffer_size_type   = typename TestFixture::size_type;
    using buffer_offset_type = unsigned int;
    using byte_offset_type   = size_t;

    constexpr int  num_buffers           = TestFixture::num_buffers;
    constexpr int  max_size              = TestFixture::max_size;
    constexpr bool shuffled              = TestFixture::shuffled;
    constexpr bool isMemCpy              = TestFixture::isMemCpy;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    constexpr int wlev_min_size = rocprim::batch_memcpy_config<>::wlev_size_threshold;
    constexpr int blev_min_size = rocprim::batch_memcpy_config<>::blev_size_threshold;

    constexpr int wlev_min_elems = rocprim::detail::ceiling_div(wlev_min_size, sizeof(value_type));
    constexpr int blev_min_elems = rocprim::detail::ceiling_div(blev_min_size, sizeof(value_type));
    constexpr int max_elems      = max_size / sizeof(value_type);

    constexpr int enabled_size_categories
        = (blev_min_elems <= max_elems) + (wlev_min_elems <= max_elems) + 1;

    constexpr int num_blev
        = blev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int num_wlev
        = wlev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int num_tlev = num_buffers - num_blev - num_wlev;

    // Get random buffer sizes
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        seed_type seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
        std::mt19937_64 rng{seed_value};

        std::vector<buffer_size_type> h_buffer_num_elements(num_buffers);

        auto iter = h_buffer_num_elements.begin();

        if(num_tlev > 0)
            iter = test_utils::generate_random_data_n(iter, num_tlev, 1, wlev_min_elems - 1, rng);
        if(num_wlev > 0)
            iter = test_utils::generate_random_data_n(iter,
                                                      num_wlev,
                                                      wlev_min_elems,
                                                      blev_min_elems - 1,
                                                      rng);
        if(num_blev > 0)
            iter = test_utils::generate_random_data_n(iter,
                                                      num_blev,
                                                      blev_min_elems,
                                                      max_elems,
                                                      rng);

        const byte_offset_type total_num_elements = std::accumulate(h_buffer_num_elements.begin(),
                                                                    h_buffer_num_elements.end(),
                                                                    byte_offset_type{0});

        // Shuffle the sizes so that size classes aren't clustered
        std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

        // And the total byte size
        const byte_offset_type total_num_bytes = total_num_elements * sizeof(value_type);

        // Device pointers
        value_type*       d_input        = nullptr;
        value_type*       d_output       = nullptr;
        value_type**      d_buffer_srcs  = nullptr;
        value_type**      d_buffer_dsts  = nullptr;
        buffer_size_type* d_buffer_sizes = nullptr;

        // Calculate temporary storage

        size_t temp_storage_bytes = 0;

        batch_copy<isMemCpy>(nullptr,
                             temp_storage_bytes,
                             d_buffer_srcs,
                             d_buffer_dsts,
                             d_buffer_sizes,
                             num_buffers,
                             hipStreamDefault);

        void* d_temp_storage = nullptr;

        // Allocate memory.
        HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
        HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

        HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(*d_buffer_srcs)));
        HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(*d_buffer_dsts)));
        HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(*d_buffer_sizes)));

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

        // Generate data.
        std::vector<unsigned char> h_input_for_memcpy;
        std::vector<value_type>    h_input_for_copy;
        init_input<isMemCpy>(h_input_for_memcpy, h_input_for_copy, rng, total_num_bytes);

        // Generate the source and shuffled destination offsets.
        std::vector<buffer_offset_type> src_offsets;
        std::vector<buffer_offset_type> dst_offsets;

        if(shuffled)
        {
            src_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
            dst_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
        }
        else
        {
            src_offsets = std::vector<buffer_offset_type>(num_buffers);
            dst_offsets = std::vector<buffer_offset_type>(num_buffers);

            // Consecutive offsets (no shuffling).
            // src/dst offsets first element is 0, so skip that!
            std::partial_sum(h_buffer_num_elements.begin(),
                             h_buffer_num_elements.end() - 1,
                             src_offsets.begin() + 1);
            std::partial_sum(h_buffer_num_elements.begin(),
                             h_buffer_num_elements.end() - 1,
                             dst_offsets.begin() + 1);
        }

        // Get the byte size of each buffer
        std::vector<buffer_size_type> h_buffer_num_bytes(num_buffers);
        for(size_t i = 0; i < num_buffers; ++i)
        {
            h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(value_type);
        }

        // Generate the source and destination pointers.
        std::vector<value_type*> h_buffer_srcs(num_buffers);
        std::vector<value_type*> h_buffer_dsts(num_buffers);

        for(int i = 0; i < num_buffers; ++i)
        {
            h_buffer_srcs[i] = d_input + src_offsets[i];
            h_buffer_dsts[i] = d_output + dst_offsets[i];
        }

        // Prepare the batch memcpy.
        if(isMemCpy)
        {
            HIP_CHECK(hipMemcpy(d_input,
                                h_input_for_memcpy.data(),
                                total_num_bytes,
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_buffer_sizes,
                                h_buffer_num_bytes.data(),
                                h_buffer_num_bytes.size() * sizeof(*d_buffer_sizes),
                                hipMemcpyHostToDevice));
        }
        else
        {
            HIP_CHECK(hipMemcpy(d_input,
                                h_input_for_copy.data(),
                                total_num_bytes,
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_buffer_sizes,
                                h_buffer_num_elements.data(),
                                h_buffer_num_elements.size() * sizeof(*d_buffer_sizes),
                                hipMemcpyHostToDevice));
        }

        HIP_CHECK(hipMemcpy(d_buffer_srcs,
                            h_buffer_srcs.data(),
                            h_buffer_srcs.size() * sizeof(*d_buffer_srcs),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_buffer_dsts,
                            h_buffer_dsts.data(),
                            h_buffer_dsts.size() * sizeof(*d_buffer_dsts),
                            hipMemcpyHostToDevice));

        const auto input_src_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_srcs);
        const auto output_src_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_dsts);

        // Run batched memcpy.
        batch_copy<isMemCpy>(d_temp_storage,
                             temp_storage_bytes,
                             input_src_it,
                             output_src_it,
                             d_buffer_sizes,
                             num_buffers,
                             hipStreamDefault);

        // Verify results.
        check_result<isMemCpy>(h_input_for_memcpy,
                               h_input_for_copy,
                               d_output,
                               total_num_bytes,
                               total_num_elements,
                               num_buffers,
                               src_offsets,
                               dst_offsets,
                               h_buffer_num_bytes);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_buffer_sizes));
        HIP_CHECK(hipFree(d_buffer_dsts));
        HIP_CHECK(hipFree(d_buffer_srcs));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_input));
    }
}

struct GetIteratorToRange
{
    __host__ __device__ __forceinline__
    auto operator()(unsigned int index) const
    {
        return rocprim::make_constant_iterator(d_data_in[index]);
    }
    unsigned int* d_data_in;
};

struct GetPtrToRange
{
    __host__ __device__ __forceinline__
    auto operator()(unsigned int index) const
    {
        return d_data_out + d_offsets[index];
    }
    unsigned int* d_data_out;
    unsigned int* d_offsets;
};

struct GetRunLength
{
    __host__ __device__ __forceinline__
    unsigned int
        operator()(unsigned int index) const
    {
        return d_offsets[index + 1] - d_offsets[index];
    }
    unsigned int* d_offsets;
};

TEST(RocprimDeviceBatchMemcpyTests, IteratorTest)
{
    // Create the data and copy it to the device.
    const unsigned int num_ranges  = 5;
    const unsigned int num_outputs = 14;

    std::vector<unsigned int> h_data_in = {4, 2, 7, 3, 1}; // size should be num_ranges
    std::vector<unsigned int> h_data_out(num_outputs, 0); // size should be num_outputs
    std::vector<unsigned int> h_offsets
        = {0, 2, 5, 6, 9, 14}; // max value should be num_outputs, size should be (num_ranges + 1)

    unsigned int* d_data_in; // [4, 2, 7, 3, 1]
    unsigned int* d_data_out; // [0,                ...               ]
    unsigned int* d_offsets; // [0, 2, 5, 6, 9, 14]

    HIP_CHECK(hipMalloc(&d_data_in, sizeof(unsigned int) * num_ranges));
    HIP_CHECK(hipMalloc(&d_data_out, sizeof(unsigned int) * num_outputs));
    HIP_CHECK(hipMalloc(&d_offsets, sizeof(unsigned int) * (num_ranges + 1)));

    HIP_CHECK(hipMemcpy(d_data_in,
                        h_data_in.data(),
                        sizeof(unsigned int) * num_ranges,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_data_out,
                        h_data_out.data(),
                        sizeof(unsigned int) * num_outputs,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_offsets,
                        h_offsets.data(),
                        sizeof(unsigned int) * (num_ranges + 1),
                        hipMemcpyHostToDevice));

    // Returns a constant iterator to the element of the i-th run
    rocprim::counting_iterator<unsigned int> iota(0);
    auto iterators_in = rocprim::make_transform_iterator(iota, GetIteratorToRange{d_data_in});

    // Returns the run length of the i-th run
    auto sizes = rocprim::make_transform_iterator(iota, GetRunLength{d_offsets});

    // Returns pointers to the output range for each run
    auto ptrs_out = rocprim::make_transform_iterator(iota, GetPtrToRange{d_data_out, d_offsets});

    // Determine temporary device storage requirements
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    batch_copy<false>(d_temp_storage,
                      temp_storage_bytes,
                      iterators_in,
                      ptrs_out,
                      sizes,
                      num_ranges,
                      0);

    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Run batched copy algorithm (used to perform runlength decoding)
    batch_copy<false>(d_temp_storage,
                      temp_storage_bytes,
                      iterators_in,
                      ptrs_out,
                      sizes,
                      num_ranges,
                      0);

    // Copy results back to host and print
    HIP_CHECK(
        hipMemcpy(h_data_out.data(), d_data_out, sizeof(int) * num_outputs, hipMemcpyDeviceToHost));

    std::vector<unsigned int> expected = {4, 4, 2, 2, 2, 7, 3, 3, 3, 1, 1, 1, 1, 1};
    test_utils::assert_eq(expected, h_data_out);

    // Clean up
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_data_in));
    HIP_CHECK(hipFree(d_data_out));
    HIP_CHECK(hipFree(d_offsets));
}
