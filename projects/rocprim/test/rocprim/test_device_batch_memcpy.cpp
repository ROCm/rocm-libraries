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

#include "../../common/device_batch_memcpy.hpp"

#include "../../common/utils_device_ptr.hpp"
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_copy.hpp>
#include <rocprim/device/device_memcpy.hpp>
#include <rocprim/device/device_memcpy_config.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

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
    static constexpr bool         debug_synchronous     = false;
    static constexpr bool         isMemCpy              = Params::isMemCpy;
    static constexpr bool         shuffled              = Params::shuffled;
    static constexpr unsigned int num_buffers           = Params::num_buffers;
    static constexpr unsigned int max_size              = Params::max_size;
    static constexpr bool         use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceBatchMemcpyTestsParams = ::testing::Types<
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
    DeviceBatchMemcpyParams<unsigned int, unsigned int, false, false, 1024, 1024 * 4, true>>;

TYPED_TEST_SUITE(RocprimDeviceBatchMemcpyTests, RocprimDeviceBatchMemcpyTestsParams);

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
                hipStream_t        stream,
                bool               debug_synchronous)
{
    HIP_CHECK(rocprim::batch_memcpy(temporary_storage,
                                    storage_size,
                                    sources,
                                    destinations,
                                    sizes,
                                    num_copies,
                                    stream,
                                    debug_synchronous));
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
                hipStream_t        stream,
                bool               debug_synchronous)
{
    HIP_CHECK(rocprim::batch_copy(temporary_storage,
                                  storage_size,
                                  sources,
                                  destinations,
                                  sizes,
                                  num_copies,
                                  stream,
                                  debug_synchronous));
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
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;

    using config = rocprim::detail::
        wrapped_batch_memcpy_config<rocprim::default_config, value_type, isMemCpy>;

    rocprim::detail::target_arch target_arch;
    hipError_t success = rocprim::detail::host_target_arch(hipStreamDefault, target_arch);
    ASSERT_EQ(success, hipSuccess);

    const rocprim::detail::batch_memcpy_config_params params
        = rocprim::detail::dispatch_target_arch<config, false>(target_arch);

    const int32_t wlev_min_size = params.wlev_size_threshold;
    const int32_t blev_min_size = params.blev_size_threshold;

    const int32_t wlev_min_elems = rocprim::detail::ceiling_div(wlev_min_size, sizeof(value_type));
    const int32_t blev_min_elems = rocprim::detail::ceiling_div(blev_min_size, sizeof(value_type));
    constexpr int32_t max_elems  = max_size / sizeof(value_type);

    const int32_t enabled_size_categories
        = (blev_min_elems <= max_elems) + (wlev_min_elems <= max_elems) + 1;

    const int32_t num_blev
        = blev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    const int32_t num_wlev
        = wlev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    const int32_t num_tlev = num_buffers - num_blev - num_wlev;

    // Get random buffer sizes
    for(size_t seed_index = 0; seed_index < number_of_runs; ++seed_index)
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

        const byte_offset_type total_num_bytes = total_num_elements * sizeof(value_type);

        // Allocate memory.
        common::device_ptr<value_type> d_input(total_num_elements);
        common::device_ptr<value_type> d_output(total_num_elements);

        common::device_ptr<value_type*>      d_buffer_srcs(num_buffers);
        common::device_ptr<value_type*>      d_buffer_dsts(num_buffers);
        common::device_ptr<buffer_size_type> d_buffer_sizes(num_buffers);

        // Test if batch_copy also supports ptr-to-const type output.
        const auto dsts = d_buffer_dsts.get();

        // Calculate temporary storage
        size_t temp_storage_bytes = 0;
        batch_copy<isMemCpy>(nullptr,
                             temp_storage_bytes,
                             d_buffer_srcs.get(),
                             dsts,
                             d_buffer_sizes.get(),
                             num_buffers,
                             hipStreamDefault,
                             debug_synchronous);

        common::device_ptr<void> d_temp_storage(temp_storage_bytes);

        // Generate data.
        std::vector<unsigned char> h_input_for_memcpy;
        std::vector<value_type>    h_input_for_copy;
        common::init_input<isMemCpy>(h_input_for_memcpy, h_input_for_copy, rng, total_num_bytes);

        // Generate the source and shuffled destination offsets.
        std::vector<buffer_offset_type> src_offsets;
        std::vector<buffer_offset_type> dst_offsets;

        if(shuffled)
        {
            src_offsets
                = common::shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
            dst_offsets
                = common::shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
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
            h_buffer_srcs[i] = d_input.get() + src_offsets[i];
            h_buffer_dsts[i] = d_output.get() + dst_offsets[i];
        }

        // Prepare the batch memcpy.
        if(isMemCpy)
        {
            HIP_CHECK(hipMemcpy(d_input.get(),
                                h_input_for_memcpy.data(),
                                total_num_bytes,
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_buffer_sizes.get(),
                                h_buffer_num_bytes.data(),
                                h_buffer_num_bytes.size() * sizeof(buffer_size_type),
                                hipMemcpyHostToDevice));
        }
        else
        {
            HIP_CHECK(hipMemcpy(d_input.get(),
                                h_input_for_copy.data(),
                                total_num_bytes,
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_buffer_sizes.get(),
                                h_buffer_num_elements.data(),
                                h_buffer_num_elements.size() * sizeof(buffer_size_type),
                                hipMemcpyHostToDevice));
        }

        d_buffer_srcs.store(h_buffer_srcs);
        d_buffer_dsts.store(h_buffer_dsts);

        const auto input_src_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_srcs.get());
        const auto output_src_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_dsts.get());

        // Run batched memcpy.
        batch_copy<isMemCpy>(d_temp_storage.get(),
                             temp_storage_bytes,
                             input_src_it,
                             output_src_it,
                             d_buffer_sizes.get(),
                             num_buffers,
                             hipStreamDefault,
                             debug_synchronous);

        // Verify results.
        check_result<isMemCpy>(h_input_for_memcpy,
                               h_input_for_copy,
                               d_output.get(),
                               total_num_bytes,
                               total_num_elements,
                               num_buffers,
                               src_offsets,
                               dst_offsets,
                               h_buffer_num_bytes);
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

TYPED_TEST(RocprimDeviceBatchMemcpyTests, IteratorTest)
{
    // Create the data and copy it to the device.
    const unsigned int num_ranges        = 5;
    const unsigned int num_outputs       = 14;
    constexpr bool     debug_synchronous = TestFixture::debug_synchronous;

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
                      0,
                      debug_synchronous);

    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Run batched copy algorithm (used to perform runlength decoding)
    batch_copy<false>(d_temp_storage,
                      temp_storage_bytes,
                      iterators_in,
                      ptrs_out,
                      sizes,
                      num_ranges,
                      0,
                      debug_synchronous);

    // Copy results back to host and print
    HIP_CHECK(
        hipMemcpy(h_data_out.data(), d_data_out, sizeof(int) * num_outputs, hipMemcpyDeviceToHost));

    std::vector<unsigned int> expected = {4, 4, 2, 2, 2, 7, 3, 3, 3, 1, 1, 1, 1, 1};
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected, h_data_out));

    // Clean up
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_data_in));
    HIP_CHECK(hipFree(d_data_out));
    HIP_CHECK(hipFree(d_offsets));
}
