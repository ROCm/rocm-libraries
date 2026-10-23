// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "benchmark_utils.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/device/device_copy.hpp>
#include <hipcub/hipcub.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

constexpr int32_t max_size = 1024 * 1024;

constexpr int32_t wlev_min_size = 128;
constexpr int32_t blev_min_size = 1024;

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_copy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_copy(
//    [&a0 , &b0 , &c0 , &d0 ], // from (note the order is still just a, b, c,
//    d!)
//    [&a0', &b0', &c0', &d0'], // to   (order is the same as above too!)
//    [3   , 2   , 1   , 2   ]) // size
//
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │b0 │b1 │a0 │a1 │a2 │d0 │d1 │c0 │ buffer x contains buffers a, b, c, d
// └───┴───┴───┴───┴───┴───┴───┴───┘ note that the order of buffers is shuffled!
//  ───┬─── ─────┬───── ───┬─── ───
//     └─────────┼─────────┼───┐
//           ┌───┘     ┌───┘   │ what batch_copy does
//           ▼         ▼       ▼
//  ─── ─────────── ─────── ───────
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │c0'│a0'│a1'│a2'│d0'│d1'│b0'│b1'│ buffer y contains buffers a', b', c', d'
// └───┴───┴───┴───┴───┴───┴───┴───┘
template<class T, class S, class RandomGenerator>
std::vector<T> shuffled_exclusive_scan(const std::vector<S>& input, RandomGenerator& rng)
{
    const auto n = input.size();
    assert(n > 0);

    std::vector<T> result(n);
    std::vector<T> permute(n);

    std::iota(permute.begin(), permute.end(), 0);
    std::shuffle(permute.begin(), permute.end(), rng);

    for(T i = 0, sum = 0; i < n; ++i)
    {
        result[permute[i]] = sum;
        sum += input[permute[i]];
    }

    return result;
}

template<typename ValueType, typename BufferSizeType>
struct BatchCopyData
{
    size_t          items              = 0;
    ValueType*      d_input            = nullptr;
    ValueType*      d_output           = nullptr;
    ValueType**     d_buffer_srcs      = nullptr;
    ValueType**     d_buffer_dsts      = nullptr;
    BufferSizeType* d_buffer_sizes     = nullptr;

    BatchCopyData()                     = default;
    BatchCopyData(const BatchCopyData&) = delete;

    BatchCopyData(BatchCopyData&& other)
        : items{std::exchange(other.items, 0)}
        , d_input{std::exchange(other.d_input, nullptr)}
        , d_output{std::exchange(other.d_output, nullptr)}
        , d_buffer_srcs{std::exchange(other.d_buffer_srcs, nullptr)}
        , d_buffer_dsts{std::exchange(other.d_buffer_dsts, nullptr)}
        , d_buffer_sizes{std::exchange(other.d_buffer_sizes, nullptr)}
    {}

    BatchCopyData& operator=(BatchCopyData&& other)
    {
        items              = std::exchange(other.items, 0);
        d_input            = std::exchange(other.d_input, nullptr);
        d_output           = std::exchange(other.d_output, nullptr);
        d_buffer_srcs      = std::exchange(other.d_buffer_srcs, nullptr);
        d_buffer_dsts      = std::exchange(other.d_buffer_dsts, nullptr);
        d_buffer_sizes     = std::exchange(other.d_buffer_sizes, nullptr);
        return *this;
    };

    BatchCopyData& operator=(const BatchCopyData&) = delete;

    size_t get_bytes() const
    {
        return items * sizeof(ValueType);
    }

    ~BatchCopyData()
    {
        HIP_CHECK(hipFree(d_buffer_sizes));
        HIP_CHECK(hipFree(d_buffer_srcs));
        HIP_CHECK(hipFree(d_buffer_dsts));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_input));
    }
};

template<class ValueType, class BufferSizeType>
BatchCopyData<ValueType, BufferSizeType> prepare_data(const int32_t num_tlev_buffers = 1024,
                                                      const int32_t num_wlev_buffers = 1024,
                                                      const int32_t num_blev_buffers = 1024)
{
    const bool shuffle_buffers = false;

    BatchCopyData<ValueType, BufferSizeType> result;
    const size_t num_buffers = num_tlev_buffers + num_wlev_buffers + num_blev_buffers;

    constexpr int32_t wlev_min_elems
        = benchmark_utils::ceiling_div(wlev_min_size, sizeof(ValueType));
    constexpr int32_t blev_min_elems
        = benchmark_utils::ceiling_div(blev_min_size, sizeof(ValueType));
    constexpr int32_t max_elems = max_size / sizeof(ValueType);

    // Generate data
    std::mt19937_64 rng(std::random_device{}());

    // Number of elements in each buffer.
    std::vector<BufferSizeType> h_buffer_num_elements(num_buffers);

    auto iter = h_buffer_num_elements.begin();

    iter = benchmark_utils::generate_random_data_n(iter,
                                                   num_tlev_buffers,
                                                   1,
                                                   wlev_min_elems - 1,
                                                   rng);
    iter = benchmark_utils::generate_random_data_n(iter,
                                                   num_wlev_buffers,
                                                   wlev_min_elems,
                                                   blev_min_elems - 1,
                                                   rng);
    iter = benchmark_utils::generate_random_data_n(iter,
                                                   num_blev_buffers,
                                                   blev_min_elems,
                                                   max_elems,
                                                   rng);

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    result.items
        = std::accumulate(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), size_t{0});

    // Generate data.
    std::independent_bits_engine<std::mt19937_64, 64, unsigned long long> bits_engine{rng};

    const size_t num_ints = benchmark_utils::ceiling_div(result.get_bytes(), sizeof(unsigned long long));
    auto h_input = std::make_unique<unsigned char[]>(num_ints * sizeof(unsigned long long));

    std::for_each(reinterpret_cast<unsigned long long*>(h_input.get()),
                  reinterpret_cast<unsigned long long*>(h_input.get() + num_ints * sizeof(unsigned long long)),
                  [&bits_engine](unsigned long long& elem) { ::new(&elem) unsigned long long{bits_engine()}; });

    HIP_CHECK(hipMalloc(&result.d_input, result.get_bytes()));
    HIP_CHECK(hipMalloc(&result.d_output, result.get_bytes()));

    HIP_CHECK(hipMalloc(&result.d_buffer_srcs, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&result.d_buffer_dsts, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&result.d_buffer_sizes, num_buffers * sizeof(BufferSizeType)));

    using offset_type = size_t;

    // Generate the source and shuffled destination offsets.
    std::vector<offset_type> src_offsets;
    std::vector<offset_type> dst_offsets;

    if(shuffle_buffers)
    {
        src_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
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
        h_buffer_srcs[i] = result.d_input + src_offsets[i];
        h_buffer_dsts[i] = result.d_output + dst_offsets[i];
    }

    // Prepare the batch copy.
    HIP_CHECK(hipMemcpy(result.d_input, h_input.get(), result.get_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_sizes,
                        h_buffer_num_elements.data(),
                        h_buffer_num_elements.size() * sizeof(BufferSizeType),
                        hipMemcpyHostToDevice));

    return result;
}

template<int32_t ItemSize,
         int32_t ItemAlignment,
         class BufferSizeType,
         int32_t NumTlevBuffers = 1024,
         int32_t NumWlevBuffers = 1024,
         int32_t NumBlevBuffers = 1024>
class batch_copy_benchmark : public primbench::benchmark_interface
{
    using ValueType = benchmark_utils::custom_aligned_type<ItemSize, ItemAlignment>;

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_batch_copy")
            .add("lvl", "device")
            .add("item_size", ItemSize)
            .add("item_alignment", ItemAlignment)
            .add("data_type", primbench::name<BufferSizeType>())
            .add("number_of_tlev", NumTlevBuffers)
            .add("number_of_wlev", NumWlevBuffers)
            .add("number_of_blev", NumBlevBuffers);
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;

        BatchCopyData<ValueType, BufferSizeType> data
            = prepare_data<ValueType, BufferSizeType>(NumTlevBuffers,
                                                      NumWlevBuffers,
                                                      NumBlevBuffers);

        const size_t num_buffers = NumTlevBuffers + NumWlevBuffers + NumBlevBuffers;

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceCopy::Batched(d_temp_storage,
                                                  temp_storage_bytes,
                                                  data.d_buffer_srcs,
                                                  data.d_buffer_dsts,
                                                  data.d_buffer_sizes,
                                                  num_buffers,
                                                  stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

        state.set_items(data.items);
        state.add_writes<ValueType>(data.items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
    }
};

#define CREATE_BENCHMARK(IS, IA, T, num_tlev, num_wlev, num_blev) \
    executor.queue<batch_copy_benchmark<IS, IA, T, num_tlev, num_wlev, num_blev>>()

#define BENCHMARK_TYPE(item_size, item_alignment)                            \
    CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0),     \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0), \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000),   \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)

int32_t main(int32_t argc, char* argv[])
{
    primbench::settings settings;

    // The size is set to 1, as prepare_data() calculates it later.
    settings.size                 = 1;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    BENCHMARK_TYPE(1, 1);
    BENCHMARK_TYPE(1, 2);
    BENCHMARK_TYPE(1, 4);
    BENCHMARK_TYPE(1, 8);
    BENCHMARK_TYPE(2, 2);
    BENCHMARK_TYPE(4, 4);
    BENCHMARK_TYPE(8, 8);

    executor.run();
}
