// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/device_batch_memcpy.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_copy.hpp>
#include <rocprim/device/device_memcpy.hpp>
#include <rocprim/device/device_memcpy_config.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template<bool IsMemCpy,
         typename InputBufferItType,
         typename OutputBufferItType,
         typename BufferSizeItType>
void batch_copy(void*              temporary_storage,
                size_t&            storage_size,
                InputBufferItType  sources,
                OutputBufferItType destinations,
                BufferSizeItType   sizes,
                uint32_t           num_copies,
                hipStream_t        stream)
{
    if constexpr(IsMemCpy)
    {
        HIP_CHECK(rocprim::batch_memcpy(temporary_storage,
                                        storage_size,
                                        sources,
                                        destinations,
                                        sizes,
                                        num_copies,
                                        stream));
    }
    else
    {
        HIP_CHECK(rocprim::batch_copy(temporary_storage,
                                      storage_size,
                                      sources,
                                      destinations,
                                      sizes,
                                      num_copies,
                                      stream));
    }
}

template<typename ValueType, typename BufferSizeType>
struct BatchMemcpyData
{
    size_t                             total_num_elements = 0;
    common::device_ptr<ValueType>      d_input;
    common::device_ptr<ValueType>      d_output;
    common::device_ptr<ValueType*>     d_buffer_srcs;
    common::device_ptr<ValueType*>     d_buffer_dsts;
    common::device_ptr<BufferSizeType> d_buffer_sizes;

    BatchMemcpyData()                       = default;
    BatchMemcpyData(const BatchMemcpyData&) = delete;

    BatchMemcpyData(BatchMemcpyData&& other) = default;

    BatchMemcpyData& operator=(BatchMemcpyData&& other) = default;

    BatchMemcpyData& operator=(const BatchMemcpyData&) = delete;

    size_t total_num_bytes() const
    {
        return total_num_elements * sizeof(ValueType);
    }

    ~BatchMemcpyData() {}
};

template<typename ValueType, typename BufferSizeType, bool IsMemCpy>
BatchMemcpyData<ValueType, BufferSizeType> prepare_data(hipStream_t    stream,
                                                        const uint32_t seed,
                                                        const int32_t  num_tlev_buffers,
                                                        const int32_t  num_wlev_buffers,
                                                        const int32_t  num_blev_buffers)
{
    const bool shuffle_buffers = false;

    BatchMemcpyData<ValueType, BufferSizeType> result;

    using Selector = rocprim::detail::batch_memcpy_config_selector<ValueType, IsMemCpy>;

    rocprim::detail::target_arch target_arch;
    hipError_t                   success = host_target_arch(stream, target_arch);

    rocprim::detail::gpu target_gpu;
    success = host_target_gpu(stream, target_gpu);

    if(success != hipSuccess)
    {
        return result;
    }

    const rocprim::detail::target get_target(target_arch, target_gpu);

    const auto params
        = rocprim::detail::get_config<Selector>(rocprim::default_config{}, get_target);

    const int32_t wlev_min_size = params.wlev_size_threshold;
    const int32_t blev_min_size = params.blev_size_threshold;

    const size_t num_buffers = num_tlev_buffers + num_wlev_buffers + num_blev_buffers;

    const int32_t wlev_min_elems = rocprim::detail::ceiling_div(wlev_min_size, sizeof(ValueType));
    const int32_t blev_min_elems = rocprim::detail::ceiling_div(blev_min_size, sizeof(ValueType));
    constexpr int32_t max_size   = 1024 * 1024;
    constexpr int32_t max_elems  = max_size / sizeof(ValueType);

    // Generate data
    std::mt19937_64 rng(seed);

    // Number of elements in each buffer.
    std::vector<BufferSizeType> h_buffer_num_elements(num_buffers);

    auto iter = h_buffer_num_elements.begin();

    iter = generate_random_data_n(iter, num_tlev_buffers, 1, wlev_min_elems - 1, rng);
    iter = generate_random_data_n(iter, num_wlev_buffers, wlev_min_elems, blev_min_elems - 1, rng);
    iter = generate_random_data_n(iter, num_blev_buffers, blev_min_elems, max_elems, rng);

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // Get the byte size of each buffer
    std::vector<BufferSizeType> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(ValueType);
    }

    result.total_num_elements
        = std::accumulate(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), size_t{0});

    std::vector<unsigned char> h_input_for_memcpy;
    std::vector<ValueType>     h_input_for_copy;
    common::init_input<IsMemCpy>(h_input_for_memcpy,
                                 h_input_for_copy,
                                 rng,
                                 result.total_num_elements * sizeof(ValueType));

    result.d_input.resize(result.total_num_elements);
    result.d_output.resize(result.total_num_elements);

    result.d_buffer_srcs.resize(num_buffers);
    result.d_buffer_dsts.resize(num_buffers);
    result.d_buffer_sizes.resize(num_buffers);

    using offset_type = size_t;

    // Generate the source and shuffled destination offsets.
    std::vector<offset_type> src_offsets;
    std::vector<offset_type> dst_offsets;

    if(shuffle_buffers)
    {
        src_offsets = common::shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
        dst_offsets = common::shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
    }
    else
    {
        src_offsets = std::vector<offset_type>(num_buffers);
        dst_offsets = std::vector<offset_type>(num_buffers);

        // Consecutive offsets (no shuffling).
        // src/dst offsets first element is 0, so skip that!
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         src_offsets.begin() + 1);
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         dst_offsets.begin() + 1);
    }

    // Generate the source and destination pointers.
    std::vector<ValueType*> h_buffer_srcs(num_buffers);
    std::vector<ValueType*> h_buffer_dsts(num_buffers);

    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = result.d_input.get() + src_offsets[i];
        h_buffer_dsts[i] = result.d_output.get() + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    if(IsMemCpy)
    {
        using cast_value_type = typename decltype(result.d_input)::value_type;
        result.d_input.store(std::vector<cast_value_type>(
            reinterpret_cast<cast_value_type*>(h_input_for_memcpy.data()),
            reinterpret_cast<cast_value_type*>(h_input_for_memcpy.data())
                + result.total_num_elements));
        result.d_buffer_sizes.store(h_buffer_num_bytes);
    }
    else
    {
        result.d_input.store(
            decltype(h_input_for_copy)(h_input_for_copy.data(),
                                       h_input_for_copy.data() + result.total_num_elements));
        result.d_buffer_sizes.store(h_buffer_num_elements);
    }
    result.d_buffer_srcs.store(h_buffer_srcs);
    result.d_buffer_dsts.store(h_buffer_dsts);

    return result;
}

template<typename OffsetType, int32_t BlockSize>
__launch_bounds__(BlockSize) __global__
void naive_kernel(void** in_ptr, void** out_ptr, const OffsetType* sizes)
{
    using underlying_type              = unsigned char;
    constexpr int32_t items_per_thread = 4;
    constexpr int32_t tile_size        = items_per_thread * BlockSize;

    const int32_t buffer_id = rocprim::flat_block_id();
    auto          in        = reinterpret_cast<underlying_type*>(in_ptr[buffer_id]);
    auto          out       = reinterpret_cast<underlying_type*>(out_ptr[buffer_id]);

    const auto items            = sizes[buffer_id];
    const auto size_in_elements = items / sizeof(underlying_type);
    const auto tiles            = size_in_elements / tile_size;

    auto num_items_to_copy = items;

    for(size_t i = 0; i < tiles; ++i)
    {
        underlying_type data[items_per_thread];
        rocprim::block_load_direct_blocked(rocprim::flat_block_thread_id(),
                                           in,
                                           data,
                                           num_items_to_copy);
        rocprim::block_store_direct_blocked(rocprim::flat_block_thread_id(),
                                            out,
                                            data,
                                            num_items_to_copy);

        in += tile_size;
        out += tile_size;
        num_items_to_copy -= tile_size;
    }
}

template<unsigned ItemSize,
         unsigned ItemAlignment,
         typename BufferSizeType,
         bool    IsMemCpy,
         int32_t NumTlevBuffers,
         int32_t NumWlevBuffers,
         int32_t NumBlevBuffers,
         typename Config = rocprim::default_config>
struct device_batch_memcpy_benchmark : public primbench::benchmark_interface
{
    device_batch_memcpy_benchmark(std::string_view subalgo) : m_subalgo(subalgo) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_batch_memcpy")
            .add("item_size", ItemSize)
            .add("item_alignment", ItemAlignment)
            .add("size_type", primbench::name<BufferSizeType>())
            .add("subalgo", m_subalgo)
            .add("num_tlev", NumTlevBuffers)
            .add("num_wlev", NumWlevBuffers)
            .add("num_blev", NumBlevBuffers)
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        if(m_subalgo == "naive_memcpy")
        {
            run_naive_benchmark(state);
        }
        else
        {
            run_benchmark(state);
        }
    }

private:
    using ValueType = custom_aligned_type<ItemSize, ItemAlignment>;

    void run_naive_benchmark(primbench::state& state)
    {
        const auto& stream = state.stream;
        const auto& seed   = state.seed;

        const auto data = prepare_data<ValueType, BufferSizeType, IsMemCpy>(stream,
                                                                            seed,
                                                                            NumTlevBuffers,
                                                                            NumWlevBuffers,
                                                                            NumBlevBuffers);

        constexpr size_t num_buffers = NumTlevBuffers + NumWlevBuffers + NumBlevBuffers;

        size_t items = data.total_num_elements;

        state.set_items(items);
        state.add_reads<ValueType>(items);

        state.run(
            [&]
            {
                naive_kernel<BufferSizeType, 256>
                    <<<num_buffers, 256, 0, stream>>>((void**)data.d_buffer_srcs.get(),
                                                      (void**)data.d_buffer_dsts.get(),
                                                      data.d_buffer_sizes.get());
            });
    }

    void run_benchmark(primbench::state& state)
    {
        const auto& stream = state.stream;
        const auto& seed   = state.seed;

        constexpr size_t num_buffers = NumTlevBuffers + NumWlevBuffers + NumBlevBuffers;

        size_t                                     temp_storage_bytes = 0;
        BatchMemcpyData<ValueType, BufferSizeType> data;
        batch_copy<IsMemCpy>(nullptr,
                             temp_storage_bytes,
                             data.d_buffer_srcs.get(),
                             data.d_buffer_dsts.get(),
                             data.d_buffer_sizes.get(),
                             num_buffers,
                             stream);

        common::device_ptr<void> d_temp_storage(temp_storage_bytes);

        data = prepare_data<ValueType, BufferSizeType, IsMemCpy>(stream,
                                                                 seed,
                                                                 NumTlevBuffers,
                                                                 NumWlevBuffers,
                                                                 NumBlevBuffers);

        size_t items = data.total_num_elements;

        state.set_items(items);
        state.add_reads<ValueType>(items);

        state.run(
            [&]
            {
                batch_copy<IsMemCpy>(d_temp_storage.get(),
                                     temp_storage_bytes,
                                     data.d_buffer_srcs.get(),
                                     data.d_buffer_dsts.get(),
                                     data.d_buffer_sizes.get(),
                                     num_buffers,
                                     stream);
            });
    }

    std::string m_subalgo;
};
