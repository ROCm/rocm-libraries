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

#include "common_benchmark_header.hpp"

#include "../test/hipcub/test_utils_sort_comparator.hpp"
// HIP API
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/util_ptx.hpp>
#include <hipcub/warp/warp_merge_sort.hpp>

#include <type_traits>

#ifndef DEFAULT_N
constexpr size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

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
__device__ auto sort_keys_benchmark(const T* input, T* output, Compare compare_op)
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
__device__ auto sort_keys_benchmark(const T* /*input*/, T* /*output*/, Compare /*compare_op*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__
    __launch_bounds__(BlockSize) void sort_keys(const T* input, T* output, Compare compare_op)
{
    sort_keys_benchmark<BlockSize, LogicalWarpSize, ItemsPerThread>(input, output, compare_op);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__device__ auto sort_pairs_benchmark(const T* input, T* output, Compare compare_op)
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
__device__ auto sort_pairs_benchmark(const T* /*input*/, T* /*output*/, Compare /*compare_op*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         typename T,
         typename Compare>
__global__
    __launch_bounds__(BlockSize) void sort_pairs(const T* input, T* output, Compare compare_op)
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
__device__ auto sort_keys_segmented_benchmark(const T*            input,
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
__device__ auto sort_keys_segmented_benchmark(const T* /*input*/,
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
__global__ __launch_bounds__(BlockSize) void sort_keys_segmented(const T*            input,
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
__device__ auto sort_pairs_segmented_benchmark(const T*            input,
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
__device__ auto sort_pairs_segmented_benchmark(const T* /*input*/,
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
__global__ __launch_bounds__(BlockSize) void sort_pairs_segmented(const T*            input,
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
         unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         class CompareOp     = test_utils::less,
         unsigned int Trials = 10>
void run_benchmark(benchmark::State&     state,
                   const benchmark_kinds benchmark_kind,
                   const hipStream_t     stream,
                   const size_t          N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    const std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    T* d_input  = nullptr;
    T* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(input[0])));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            for(unsigned int i = 0; i < Trials; ++i)
            {
                sort_keys<BlockSize, LogicalWarpSize, ItemsPerThread>
                    <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                   d_output,
                                                                                   CompareOp{});
            }
        } else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            for(unsigned int i = 0; i < Trials; ++i)
            {
                sort_pairs<BlockSize, LogicalWarpSize, ItemsPerThread>
                    <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                   d_output,
                                                                                   CompareOp{});
            }
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<class T,
         unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         class CompareOp     = test_utils::less,
         unsigned int Trials = 10>
void run_segmented_benchmark(benchmark::State&     state,
                             const benchmark_kinds benchmark_kind,
                             const hipStream_t     stream,
                             const size_t          N)
{
    constexpr auto max_segment_size   = LogicalWarpSize * ItemsPerThread;
    constexpr auto segments_per_block = BlockSize / LogicalWarpSize;
    constexpr auto items_per_block    = BlockSize * ItemsPerThread;

    const auto num_blocks   = (N + items_per_block - 1) / items_per_block;
    const auto num_segments = num_blocks * segments_per_block;
    const auto size         = num_blocks * items_per_block;

    const std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    const auto segment_sizes
        = benchmark_utils::get_random_data<unsigned int>(num_segments, 0, max_segment_size);

    T*            d_input         = nullptr;
    T*            d_output        = nullptr;
    unsigned int* d_segment_sizes = nullptr;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_segment_sizes, num_segments * sizeof(segment_sizes[0])));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_segment_sizes,
                        segment_sizes.data(),
                        num_segments * sizeof(segment_sizes[0]),
                        hipMemcpyHostToDevice));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            for(unsigned int i = 0; i < Trials; ++i)
            {
                sort_keys_segmented<BlockSize, LogicalWarpSize, ItemsPerThread>
                    <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input,
                                                                       d_output,
                                                                       d_segment_sizes,
                                                                       CompareOp{});
            }
        } else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            for(unsigned int i = 0; i < Trials; ++i)
            {
                sort_pairs_segmented<BlockSize, LogicalWarpSize, ItemsPerThread>
                    <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input,
                                                                       d_output,
                                                                       d_segment_sizes,
                                                                       CompareOp{});
            }
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_segment_sizes));
}

#define CREATE_BENCHMARK(T, BS, WS, IPT)                                                           \
    if(WS <= device_warp_size)                                                                     \
    {                                                                                              \
        benchmarks.push_back(benchmark::RegisterBenchmark(                                         \
            std::string("warp_merge_sort<data_type:" #T ",block_size:" #BS ",warp_size:" #WS       \
                        ",items_per_thread:" #IPT ">.sub_algorithm_name:"                          \
                        + name)                                                                    \
                .c_str(),                                                                          \
            segmented ? &run_benchmark<T, BS, WS, IPT> : &run_segmented_benchmark<T, BS, WS, IPT>, \
            benchmark_kind,                                                                        \
            stream,                                                                                \
            size));                                                                                \
    }

#define BENCHMARK_TYPE_WS(type, block, warp) \
    CREATE_BENCHMARK(type, block, warp, 1);  \
    CREATE_BENCHMARK(type, block, warp, 4);  \
    CREATE_BENCHMARK(type, block, warp, 8)

#define BENCHMARK_TYPE(type, block)     \
    BENCHMARK_TYPE_WS(type, block, 4);  \
    BENCHMARK_TYPE_WS(type, block, 16); \
    BENCHMARK_TYPE_WS(type, block, 32); \
    BENCHMARK_TYPE_WS(type, block, 64)

void add_benchmarks(const benchmark_kinds                         benchmark_kind,
                    const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const hipStream_t                             stream,
                    const size_t                                  size,
                    const bool                                    segmented,
                    const unsigned int                            device_warp_size)
{
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(uint8_t, 256);
    BENCHMARK_TYPE(long long, 256);
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    std::cout << "benchmark_warp_merge_sort" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    const auto device_warp_size = []
    {
        const int result = HIPCUB_HOST_WARP_THREADS;
        if(result > 0)
        {
            std::cout << "[HIP] Device warp size: " << result << std::endl;
        } else
        {
            std::cerr << "Failed to get device warp size! Aborting.\n";
            std::exit(1);
        }
        return static_cast<unsigned int>(result);
    }();

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmark_kinds::sort_keys,
                   "sort(keys)",
                   benchmarks,
                   stream,
                   size,
                   false,
                   device_warp_size);
    add_benchmarks(benchmark_kinds::sort_pairs,
                   "sort(keys, values)",
                   benchmarks,
                   stream,
                   size,
                   false,
                   device_warp_size);
    add_benchmarks(benchmark_kinds::sort_keys,
                   "segmented_sort(keys)",
                   benchmarks,
                   stream,
                   size,
                   true,
                   device_warp_size);
    add_benchmarks(benchmark_kinds::sort_pairs,
                   "segmented_sort(keys, values)",
                   benchmarks,
                   stream,
                   size,
                   true,
                   device_warp_size);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
