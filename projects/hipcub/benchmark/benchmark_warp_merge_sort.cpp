// MIT License
//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../test/hipcub/test_utils_sort_comparator.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/util_ptx.hpp>
#include <hipcub/warp/warp_merge_sort.hpp>

#include <type_traits>

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs,
};

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_keys_benchmark(const T* input, T* output, Compare compare_op)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_tid     = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;
    T                  keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(flat_tid, input + block_offset, keys);

    constexpr unsigned int warps_per_block = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id         = threadIdx.x / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<T, ItemsPerThread, LogicalWarpSize>;
    __shared__ typename warp_merge_sort::TempStorage storage[warps_per_block];

    warp_merge_sort wsort{storage[warp_id]};
    wsort.Sort(keys, compare_op);

    hipcub::StoreDirectBlocked(flat_tid, output + block_offset, keys);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_keys_benchmark(const T* /*input*/, T* /*output*/, Compare /*compare_op*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__ __launch_bounds__(BlockSize)
void sort_keys(const T* input, T* output, Compare compare_op)
{
    sort_keys_benchmark<BlockSize, LogicalWarpSize, ItemsPerThread>(input, output, compare_op);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_pairs_benchmark(const T* input, T* output, Compare compare_op)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_tid     = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;
    T                  keys[ItemsPerThread];
    T                  values[ItemsPerThread];
    hipcub::LoadDirectBlocked(flat_tid, input + block_offset, keys);

    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        values[i] = keys[i] + T(1);
    }

    constexpr unsigned int warps_per_block = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id         = threadIdx.x / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<T, ItemsPerThread, LogicalWarpSize, T>;
    __shared__ typename warp_merge_sort::TempStorage storage[warps_per_block];

    warp_merge_sort wsort{storage[warp_id]};
    wsort.Sort(keys, values, compare_op);

    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        keys[i] += values[i];
    }

    hipcub::StoreDirectBlocked(flat_tid, output + block_offset, keys);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_pairs_benchmark(const T* /*input*/, T* /*output*/, Compare /*compare_op*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__ __launch_bounds__(BlockSize)
void sort_pairs(const T* input, T* output, Compare compare_op)
{
    sort_pairs_benchmark<BlockSize, LogicalWarpSize, ItemsPerThread>(input, output, compare_op);
}

template<typename T>
struct max_value
{
    static constexpr T value = std::numeric_limits<T>::max();
};

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_keys_segmented_benchmark(const T*            input,
                                   T*                  output,
                                   const unsigned int* segment_sizes,
                                   Compare             compare)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int max_segment_size   = LogicalWarpSize * ItemsPerThread;
    constexpr unsigned int segments_per_block = BlockSize / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<T, ItemsPerThread, LogicalWarpSize>;
    __shared__ typename warp_merge_sort::TempStorage storage[segments_per_block];

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    warp_merge_sort    wsort{storage[warp_id]};

    const unsigned int segment_id = blockIdx.x * segments_per_block + warp_id;

    const unsigned int segment_size = segment_sizes[segment_id];
    const unsigned int warp_offset  = segment_id * max_segment_size;
    T                  keys[ItemsPerThread];

    const unsigned int flat_tid = wsort.get_linear_tid();
    hipcub::LoadDirectBlocked(flat_tid, input + warp_offset, keys, segment_size);

    const T oob_default = max_value<T>::value;
    wsort.Sort(keys, compare, segment_size, oob_default);

    hipcub::StoreDirectBlocked(flat_tid, output + warp_offset, keys, segment_size);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_keys_segmented_benchmark(const T* /*input*/,
                                   T* /*output*/,
                                   const unsigned int* /*segment_sizes*/,
                                   Compare /*compare*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__ __launch_bounds__(BlockSize)
void sort_keys_segmented(const T*            input,
                         T*                  output,
                         const unsigned int* segment_sizes,
                         Compare             compare)
{
    sort_keys_segmented_benchmark<BlockSize, LogicalWarpSize, ItemsPerThread>(input,
                                                                              output,
                                                                              segment_sizes,
                                                                              compare);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_pairs_segmented_benchmark(const T*            input,
                                    T*                  output,
                                    const unsigned int* segment_sizes,
                                    Compare             compare)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int max_segment_size   = LogicalWarpSize * ItemsPerThread;
    constexpr unsigned int segments_per_block = BlockSize / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<T, ItemsPerThread, LogicalWarpSize, T>;
    __shared__ typename warp_merge_sort::TempStorage storage[segments_per_block];

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    warp_merge_sort    wsort{storage[warp_id]};

    const unsigned int segment_id = blockIdx.x * segments_per_block + warp_id;

    const unsigned int segment_size = segment_sizes[segment_id];
    const unsigned int warp_offset  = segment_id * max_segment_size;
    T                  keys[ItemsPerThread];
    T                  values[ItemsPerThread];

    const unsigned int flat_tid = wsort.get_linear_tid();
    hipcub::LoadDirectBlocked(flat_tid, input + warp_offset, keys, segment_size);

    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        if(flat_tid * ItemsPerThread + i < segment_size)
        {
            values[i] = keys[i] + T(1);
        }
    }

    const T oob_default = max_value<T>::value;
    wsort.Sort(keys, values, compare, segment_size, oob_default);

    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        if(flat_tid * ItemsPerThread + i < segment_size)
        {
            keys[i] += values[i];
        }
    }

    hipcub::StoreDirectBlocked(flat_tid, output + warp_offset, keys, segment_size);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__
auto sort_pairs_segmented_benchmark(const T* /*input*/,
                                    T* /*output*/,
                                    const unsigned int* /*segment_sizes*/,
                                    Compare /*compare*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__ __launch_bounds__(BlockSize)
void sort_pairs_segmented(const T*            input,
                          T*                  output,
                          const unsigned int* segment_sizes,
                          Compare             compare)
{
    sort_pairs_segmented_benchmark<BlockSize, LogicalWarpSize, ItemsPerThread>(input,
                                                                               output,
                                                                               segment_sizes,
                                                                               compare);
}

template<class T,
         unsigned int    BlockSize,
         unsigned int    LogicalWarpSize,
         unsigned int    ItemsPerThread,
         benchmark_kinds BenchmarkKind,
         class CompareOp = test_utils::less>
struct sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "warp_merge_sort")
                        .add("segmented", false)
                        .add("pairs", BenchmarkKind == benchmark_kinds::sort_pairs)
                        .add("data_type", primbench::name<T>())
                        .add("block_size", BlockSize)
                        .add("items_per_thread", ItemsPerThread)
                        .add("warp_size", LogicalWarpSize);

        return json;
    }

    void run(primbench::state& state) override
    {
        const auto& input_items = state.size;
        const auto& stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        const std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        T* d_input  = nullptr;
        T* d_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(input[0])));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                if constexpr(BenchmarkKind == benchmark_kinds::sort_keys)
                {
                    sort_keys<BlockSize, LogicalWarpSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input,
                            d_output,
                            CompareOp{});
                }
                else
                {
                    static_assert(BenchmarkKind == benchmark_kinds::sort_pairs);
                    sort_pairs<BlockSize, LogicalWarpSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input,
                            d_output,
                            CompareOp{});
                }
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

template<class T,
         unsigned int    BlockSize,
         unsigned int    LogicalWarpSize,
         unsigned int    ItemsPerThread,
         benchmark_kinds BenchmarkKind,
         class CompareOp = test_utils::less>
struct segmented_sort_benchmark : public primbench::benchmark_interface
{

    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "warp_merge_sort")
                        .add("segmented", true)
                        .add("pairs", BenchmarkKind == benchmark_kinds::sort_pairs)
                        .add("data_type", primbench::name<T>())
                        .add("block_size", BlockSize)
                        .add("items_per_thread", ItemsPerThread)
                        .add("warp_size", LogicalWarpSize);

        return json;
    }

    void run(primbench::state& state) override
    {
        const auto& input_items = state.size;
        const auto& stream      = state.stream;

        constexpr auto max_segment_size   = LogicalWarpSize * ItemsPerThread;
        constexpr auto segments_per_block = BlockSize / LogicalWarpSize;
        constexpr auto items_per_block    = BlockSize * ItemsPerThread;

        const auto num_blocks   = (input_items + items_per_block - 1) / items_per_block;
        const auto num_segments = num_blocks * segments_per_block;
        const auto items        = num_blocks * items_per_block;

        const std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        const auto segment_sizes
            = benchmark_utils::get_random_data<unsigned int>(num_segments, 0, max_segment_size);

        T*            d_input         = nullptr;
        T*            d_output        = nullptr;
        unsigned int* d_segment_sizes = nullptr;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_segment_sizes, num_segments * sizeof(segment_sizes[0])));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_segment_sizes,
                            segment_sizes.data(),
                            num_segments * sizeof(segment_sizes[0]),
                            hipMemcpyHostToDevice));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                if constexpr(BenchmarkKind == benchmark_kinds::sort_keys)
                {
                    sort_keys_segmented<BlockSize, LogicalWarpSize, ItemsPerThread>
                        <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input,
                                                                           d_output,
                                                                           d_segment_sizes,
                                                                           CompareOp{});
                }
                else
                {
                    static_assert(BenchmarkKind == benchmark_kinds::sort_pairs);

                    sort_pairs_segmented<BlockSize, LogicalWarpSize, ItemsPerThread>
                        <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input,
                                                                           d_output,
                                                                           d_segment_sizes,
                                                                           CompareOp{});
                }
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_segment_sizes));
    }
};

#define CREATE_BENCHMARK(T, BS, WS, IPT, BK)                                \
    if(WS <= device_warp_size)                                              \
    {                                                                       \
        if(segmented)                                                       \
        {                                                                   \
            executor.queue<segmented_sort_benchmark<T, BS, WS, IPT, BK>>(); \
        }                                                                   \
        else                                                                \
        {                                                                   \
            executor.queue<sort_benchmark<T, BS, WS, IPT, BK>>();           \
        }                                                                   \
    }

#define BENCHMARK_TYPE_WS(type, block, warp, kind) \
    CREATE_BENCHMARK(type, block, warp, 1, kind);  \
    CREATE_BENCHMARK(type, block, warp, 4, kind);  \
    CREATE_BENCHMARK(type, block, warp, 8, kind)

#define BENCHMARK_TYPE(type, block)                    \
    BENCHMARK_TYPE_WS(type, block, 4, BenchmarkKind);  \
    BENCHMARK_TYPE_WS(type, block, 16, BenchmarkKind); \
    BENCHMARK_TYPE_WS(type, block, 32, BenchmarkKind); \
    BENCHMARK_TYPE_WS(type, block, 64, BenchmarkKind)

template<benchmark_kinds BenchmarkKind>
void add_benchmarks(primbench::executor& executor,
                    const bool           segmented,
                    const unsigned int   device_warp_size)
{
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(uint8_t, 256);
    BENCHMARK_TYPE(int64_t, 256);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch    = 1000;
    settings.batch_window_size       = 3;
    settings.noise_tolerance_percent = 3;

    primbench::executor executor(argc, argv, settings);

    const auto device_warp_size = []
    {
        const int result = HIPCUB_HOST_WARP_THREADS;
        if(result > 0)
        {
            std::cout << "[HIP] Device warp size: " << result << std::endl;
        }
        else
        {
            std::cerr << "Failed to get device warp size! Aborting.\n";
            std::exit(1);
        }
        return static_cast<unsigned int>(result);
    }();

    add_benchmarks<benchmark_kinds::sort_keys>(executor, false, device_warp_size);
    add_benchmarks<benchmark_kinds::sort_pairs>(executor, false, device_warp_size);
    add_benchmarks<benchmark_kinds::sort_keys>(executor, true, device_warp_size);
    add_benchmarks<benchmark_kinds::sort_pairs>(executor, true, device_warp_size);

    executor.run();
}
