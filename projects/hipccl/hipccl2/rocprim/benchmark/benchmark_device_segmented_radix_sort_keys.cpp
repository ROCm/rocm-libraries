// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"
// CmdParser
#include "cmdparser.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_segmented_radix_sort.hpp>

#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif

namespace rp = rocprim;

namespace
{

constexpr unsigned int          warmup_size = 2;
constexpr size_t                min_size    = 30000;
constexpr std::array<size_t, 8> segment_counts{10, 100, 1000, 2500, 5000, 7500, 10000, 100000};
constexpr std::array<size_t, 4> segment_lengths{30, 256, 3000, 300000};
} // namespace

// This benchmark only handles the rocprim::segmented_radix_sort_keys function. The benchmark was separated into two (keys and pairs),
// because the binary became too large to link. Runs into a "relocation R_X86_64_PC32 out of range" error.
// This happens partially, because of the algorithm has 4 kernels, and decides at runtime which one to call.

template<class Key>
void run_sort_keys_benchmark(benchmark::State&   state,
                             size_t              num_segments,
                             size_t              mean_segment_length,
                             size_t              target_bytes,
                             const managed_seed& seed,
                             hipStream_t         stream)
{
    using offset_type = int;
    using key_type    = Key;

    // Calculate the number of elements 
    size_t target_size = target_bytes / sizeof(key_type);

    std::vector<offset_type> offsets;
    offsets.push_back(0);

    static constexpr int iseed = 716;
    engine_type          gen(iseed);

    std::normal_distribution<double> segment_length_dis(static_cast<double>(mean_segment_length),
                                                        0.1 * mean_segment_length);

    size_t offset = 0;
    for(size_t segment_index = 0; segment_index < num_segments;)
    {
        const double segment_length_candidate = std::round(segment_length_dis(gen));
        if(segment_length_candidate < 0)
        {
            continue;
        }
        const offset_type segment_length = static_cast<offset_type>(segment_length_candidate);
        offset += segment_length;
        offsets.push_back(offset);
        ++segment_index;
    }
    const size_t size           = offset;
    const size_t segments_count = offsets.size() - 1;

    std::vector<key_type> keys_input = get_random_data<key_type>(size,
                                                                 generate_limits<key_type>::min(),
                                                                 generate_limits<key_type>::max(),
                                                                 seed.get_0());

    size_t batch_size = 1;
    if(size < target_size)
    {
        batch_size = (target_size + size - 1) / size;
    }

    offset_type* d_offsets;
    HIP_CHECK(hipMalloc(&d_offsets, offsets.size() * sizeof(offset_type)));
    HIP_CHECK(hipMemcpy(d_offsets,
                        offsets.data(),
                        offsets.size() * sizeof(offset_type),
                        hipMemcpyHostToDevice));

    key_type* d_keys_input;
    key_type* d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(d_keys_input, keys_input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(rp::segmented_radix_sort_keys(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input,
                                            d_keys_output,
                                            size,
                                            segments_count,
                                            d_offsets,
                                            d_offsets + 1,
                                            0,
                                            sizeof(key_type) * 8,
                                            stream,
                                            false));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(rp::segmented_radix_sort_keys(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                size,
                                                segments_count,
                                                d_offsets,
                                                d_offsets + 1,
                                                0,
                                                sizeof(key_type) * 8,
                                                stream,
                                                false));
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

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(rp::segmented_radix_sort_keys(d_temporary_storage,
                                                    temporary_storage_bytes,
                                                    d_keys_input,
                                                    d_keys_output,
                                                    size,
                                                    segments_count,
                                                    d_offsets,
                                                    d_offsets + 1,
                                                    0,
                                                    sizeof(key_type) * 8,
                                                    stream,
                                                    false));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
}

template<class KeyT>
void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              size_t                                        max_bytes,
                              size_t                                        min_size,
                              size_t                                        target_size,
                              const managed_seed&                           seed,
                              hipStream_t                                   stream)
{
    // Calculate the number of elements 
    size_t max_size = max_bytes / sizeof(KeyT);

    std::string key_name   = Traits<KeyT>::name();
    std::string value_name = Traits<rocprim::empty_type>::name();
    for(const auto segment_count : segment_counts)
    {
        for(const auto segment_length : segment_lengths)
        {
            const auto number_of_elements = segment_count * segment_length;
            if(number_of_elements > max_size || number_of_elements < min_size)
            {
                continue;
            }
            benchmarks.push_back(benchmark::RegisterBenchmark(
                bench_naming::format_name(
                    "{lvl:device,algo:radix_sort_segmented,key_type:" + key_name + ",value_type:"
                    + value_name + ",segment_count:" + std::to_string(segment_count)
                    + ",segment_length:" + std::to_string(segment_length) + ",cfg:default_config}")
                    .c_str(),
                [=](benchmark::State& state)
                {
                    run_sort_keys_benchmark<KeyT>(state,
                                                  segment_count,
                                                  segment_length,
                                                  target_size,
                                                  seed,
                                                  stream);
                }));
        }
    }
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());

#ifdef BENCHMARK_CONFIG_TUNING
    // optionally run an evenly split subset of benchmarks, when making multiple program invocations
    parser.set_optional<int>("parallel_instance",
                             "parallel_instance",
                             0,
                             "parallel instance index");
    parser.set_optional<int>("parallel_instances",
                             "parallel_instances",
                             1,
                             "total parallel instances");
#endif

    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t bytes   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
#ifdef BENCHMARK_CONFIG_TUNING
    (void)min_size;
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        min_size,
                                                        seed,
                                                        stream);
#else
    add_sort_keys_benchmarks<float>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
    add_sort_keys_benchmarks<double>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
    add_sort_keys_benchmarks<int8_t>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
    add_sort_keys_benchmarks<uint8_t>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
    add_sort_keys_benchmarks<rocprim::half>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
    add_sort_keys_benchmarks<int>(benchmarks, bytes, min_size, bytes / 2, seed, stream);
#endif

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
