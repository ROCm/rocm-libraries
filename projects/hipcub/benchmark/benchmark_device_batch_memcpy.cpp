// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark/benchmark.h"
#include "cmdparser.hpp"
#include "common_benchmark_header.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/device/device_memcpy.hpp>
#include <hipcub/hipcub.hpp>

#ifdef __HIP_PLATFORM_AMD__
    // Only include this on AMD as it contains specialized config information
    #include <rocprim/device/device_memcpy_config.hpp>
#endif

#include <hip/hip_runtime.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <stdint.h>

constexpr uint32_t warmup_size = 5;
constexpr int32_t  max_size    = 1024 * 1024;

constexpr int32_t wlev_min_size = 128;
constexpr int32_t blev_min_size = 1024;

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_memcpy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_memcpy(
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
//           ┌───┘     ┌───┘   │ what batch_memcpy does
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

using offset_type = size_t;

template<typename ValueType, typename BufferSizeType>
struct BatchMemcpyData
{
    size_t          total_num_elements = 0;
    ValueType*      d_input            = nullptr;
    ValueType*      d_output           = nullptr;
    ValueType**     d_buffer_srcs      = nullptr;
    ValueType**     d_buffer_dsts      = nullptr;
    BufferSizeType* d_buffer_sizes     = nullptr;

    BatchMemcpyData()                       = default;
    BatchMemcpyData(const BatchMemcpyData&) = delete;

    BatchMemcpyData(BatchMemcpyData&& other)
        : total_num_elements{std::exchange(other.total_num_elements, 0)}
        , d_input{std::exchange(other.d_input, nullptr)}
        , d_output{std::exchange(other.d_output, nullptr)}
        , d_buffer_srcs{std::exchange(other.d_buffer_srcs, nullptr)}
        , d_buffer_dsts{std::exchange(other.d_buffer_dsts, nullptr)}
        , d_buffer_sizes{std::exchange(other.d_buffer_sizes, nullptr)}
    {}

    BatchMemcpyData& operator=(BatchMemcpyData&& other)
    {
        total_num_elements = std::exchange(other.total_num_elements, 0);
        d_input            = std::exchange(other.d_input, nullptr);
        d_output           = std::exchange(other.d_output, nullptr);
        d_buffer_srcs      = std::exchange(other.d_buffer_srcs, nullptr);
        d_buffer_dsts      = std::exchange(other.d_buffer_dsts, nullptr);
        d_buffer_sizes     = std::exchange(other.d_buffer_sizes, nullptr);
        return *this;
    };

    BatchMemcpyData& operator=(const BatchMemcpyData&) = delete;

    size_t total_num_bytes() const
    {
        return total_num_elements * sizeof(ValueType);
    }

    ~BatchMemcpyData()
    {
        HIP_CHECK(hipFree(d_buffer_sizes));
        HIP_CHECK(hipFree(d_buffer_srcs));
        HIP_CHECK(hipFree(d_buffer_dsts));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_input));
    }
};

template<class ValueType, class BufferSizeType>
BatchMemcpyData<ValueType, BufferSizeType> prepare_data(const int32_t num_tlev_buffers = 1024,
                                                        const int32_t num_wlev_buffers = 1024,
                                                        const int32_t num_blev_buffers = 1024)
{
    const bool shuffle_buffers = false;

    BatchMemcpyData<ValueType, BufferSizeType> result;
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

    // Get the byte size of each buffer
    std::vector<BufferSizeType> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(ValueType);
    }

    result.total_num_elements
        = std::accumulate(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), size_t{0});

    // Generate data.
    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints
        = benchmark_utils::ceiling_div(result.total_num_bytes(), sizeof(uint64_t));
    auto h_input = std::make_unique<unsigned char[]>(num_ints * sizeof(uint64_t));

    std::for_each(reinterpret_cast<uint64_t*>(h_input.get()),
                  reinterpret_cast<uint64_t*>(h_input.get() + num_ints * sizeof(uint64_t)),
                  [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });

    HIP_CHECK(hipMalloc(&result.d_input, result.total_num_bytes()));
    HIP_CHECK(hipMalloc(&result.d_output, result.total_num_bytes()));

    HIP_CHECK(hipMalloc(&result.d_buffer_srcs, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&result.d_buffer_dsts, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&result.d_buffer_sizes, num_buffers * sizeof(BufferSizeType)));

    // Generate the source and shuffled destination offsets.
    std::vector<offset_type> src_offsets;
    std::vector<offset_type> dst_offsets;

    if(shuffle_buffers)
    {
        src_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
    } else
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

    // Prepare the batch memcpy.
    HIP_CHECK(
        hipMemcpy(result.d_input, h_input.get(), result.total_num_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result.d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(BufferSizeType),
                        hipMemcpyHostToDevice));

    return result;
}

template<class ValueType, class BufferSizeType>
void run_benchmark(benchmark::State& state,
                   hipStream_t       stream,
                   const int32_t     num_tlev_buffers = 1024,
                   const int32_t     num_wlev_buffers = 1024,
                   const int32_t     num_blev_buffers = 1024)
{
    const size_t num_buffers = num_tlev_buffers + num_wlev_buffers + num_blev_buffers;

    size_t                                     temp_storage_bytes = 0;
    BatchMemcpyData<ValueType, BufferSizeType> data;
    HIP_CHECK(hipcub::DeviceMemcpy::Batched(nullptr,
                                            temp_storage_bytes,
                                            data.d_buffer_srcs,
                                            data.d_buffer_dsts,
                                            data.d_buffer_sizes,
                                            num_buffers));

    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    data = prepare_data<ValueType, BufferSizeType>(num_tlev_buffers,
                                                   num_wlev_buffers,
                                                   num_blev_buffers);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_temp_storage,
                                                temp_storage_bytes,
                                                data.d_buffer_srcs,
                                                data.d_buffer_dsts,
                                                data.d_buffer_sizes,
                                                num_buffers,
                                                stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        HIP_CHECK(hipcub::DeviceMemcpy::Batched(d_temp_storage,
                                                temp_storage_bytes,
                                                data.d_buffer_srcs,
                                                data.d_buffer_dsts,
                                                data.d_buffer_sizes,
                                                num_buffers,
                                                stream));

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }
    state.SetBytesProcessed(state.iterations() * data.total_num_bytes());
    state.SetItemsProcessed(state.iterations() * data.total_num_elements);

    HIP_CHECK(hipFree(d_temp_storage));
}

#define CREATE_BENCHMARK(IS, IA, T, num_tlev, num_wlev, num_blev)                                \
    benchmark::RegisterBenchmark(                                                                \
        std::string("device_batch_memcpy<data_type:" #T ",item_size:" #IS ",item_alignment:" #IA \
                    ",number_of_tlev:" #num_tlev ",number_of_wlev:" #num_wlev                    \
                    ",number_of_blev:" #num_blev ">.")                                           \
            .c_str(),                                                                            \
        [=](benchmark::State& state)                                                             \
        {                                                                                        \
            run_benchmark<benchmark_utils::custom_aligned_type<IS, IA>, T>(state,                \
                                                                           stream,               \
                                                                           num_tlev,             \
                                                                           num_wlev,             \
                                                                           num_blev);            \
        })

#define BENCHMARK_TYPE(item_size, item_alignment)                            \
    CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0),     \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0), \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000),   \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)

int32_t main(int32_t argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", 1024, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");

    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t  size   = parser.get<size_t>("size");
    const int32_t trials = parser.get<int>("trials");

    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    std::cout << "benchmark_device_adjacent_difference" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // HIP
    hipStream_t stream = hipStreamDefault; // default

    // Benchmark info
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;

    benchmarks = {BENCHMARK_TYPE(1, 1),
                  BENCHMARK_TYPE(1, 2),
                  BENCHMARK_TYPE(1, 4),
                  BENCHMARK_TYPE(1, 8),
                  BENCHMARK_TYPE(2, 2),
                  BENCHMARK_TYPE(4, 4),
                  BENCHMARK_TYPE(8, 8)};

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
