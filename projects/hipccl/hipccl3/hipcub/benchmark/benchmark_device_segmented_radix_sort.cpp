// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_benchmark_header.hpp"

// HIP API
#include "hipcub/device/device_segmented_radix_sort.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size = 4;
const unsigned int warmup_size = 2;

template<class Key>
void run_sort_keys_benchmark(benchmark::State& state,
                             size_t desired_segments,
                             hipStream_t stream, size_t size)
{
    using offset_type = int;
    using key_type = Key;

    // Generate data
    std::vector<offset_type> offsets;

    const double avg_segment_length = static_cast<double>(size) / desired_segments;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

    unsigned int segments_count = 0;
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = std::round(segment_length_dis(gen));
        offsets.push_back(offset);
        segments_count++;
        offset += segment_length;
    }
    offsets.push_back(size);

    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = benchmark_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
    }

    offset_type * d_offsets;
    HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
    HIP_CHECK(
        hipMemcpy(
            d_offsets, offsets.data(),
            (segments_count + 1) * sizeof(offset_type),
            hipMemcpyHostToDevice
        )
    );

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        hipcub::DeviceSegmentedRadixSort::SortKeys(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, size,
            segments_count, d_offsets, d_offsets + 1,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            hipcub::DeviceSegmentedRadixSort::SortKeys(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                segments_count, d_offsets, d_offsets + 1,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                hipcub::DeviceSegmentedRadixSort::SortKeys(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    segments_count, d_offsets, d_offsets + 1,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
}

template<class Key, class Value>
void run_sort_pairs_benchmark(benchmark::State& state,
                              size_t desired_segments,
                              hipStream_t stream, size_t size)
{
    using offset_type = int;
    using key_type = Key;
    using value_type = Value;

    // Generate data
    std::vector<offset_type> offsets;

    const double avg_segment_length = static_cast<double>(size) / desired_segments;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

    unsigned int segments_count = 0;
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = std::round(segment_length_dis(gen));
        offsets.push_back(offset);
        segments_count++;
        offset += segment_length;
    }
    offsets.push_back(size);

    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = benchmark_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
    }

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    offset_type * d_offsets;
    HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
    HIP_CHECK(
        hipMemcpy(
            d_offsets, offsets.data(),
            (segments_count + 1) * sizeof(offset_type),
            hipMemcpyHostToDevice
        )
    );

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    value_type * d_values_output;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        hipcub::DeviceSegmentedRadixSort::SortPairs(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, d_values_input, d_values_output, size,
            segments_count, d_offsets, d_offsets + 1,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            hipcub::DeviceSegmentedRadixSort::SortPairs(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                segments_count, d_offsets, d_offsets + 1,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                hipcub::DeviceSegmentedRadixSort::SortPairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    segments_count, d_offsets, d_offsets + 1,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(
        state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type))
    );
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_values_output));
}

#define CREATE_SORT_KEYS_BENCHMARK(Key, SEGMENTS) \
benchmark::RegisterBenchmark( \
    (std::string("sort_keys") + "<" #Key ">" + \
        "(~" + std::to_string(SEGMENTS) + " segments)" \
    ).c_str(), \
    [=](benchmark::State& state) { run_sort_keys_benchmark<Key>(state, SEGMENTS, stream, size); } \
)

#define BENCHMARK_KEY_TYPE(type) \
    CREATE_SORT_KEYS_BENCHMARK(type, 1), \
    CREATE_SORT_KEYS_BENCHMARK(type, 10), \
    CREATE_SORT_KEYS_BENCHMARK(type, 100), \
    CREATE_SORT_KEYS_BENCHMARK(type, 1000), \
    CREATE_SORT_KEYS_BENCHMARK(type, 10000)

void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              hipStream_t stream,
                              size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_KEY_TYPE(float),
        BENCHMARK_KEY_TYPE(double),
        BENCHMARK_KEY_TYPE(int8_t),
        BENCHMARK_KEY_TYPE(uint8_t),
        BENCHMARK_KEY_TYPE(int),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_SORT_PAIRS_BENCHMARK(Key, Value, SEGMENTS) \
benchmark::RegisterBenchmark( \
    (std::string("sort_pairs") + "<" #Key ", " #Value ">" + \
        "(~" + std::to_string(SEGMENTS) + " segments)" \
    ).c_str(), \
    [=](benchmark::State& state) { run_sort_pairs_benchmark<Key, Value>(state, SEGMENTS, stream, size); } \
)

#define BENCHMARK_PAIR_TYPE(type, value) \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 1), \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 10), \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 100), \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 1000), \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 10000)

void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                               hipStream_t stream,
                               size_t size)
{
    using custom_float2 = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_PAIR_TYPE(int, float),
        BENCHMARK_PAIR_TYPE(long long, double),
        BENCHMARK_PAIR_TYPE(int8_t, int8_t),
        BENCHMARK_PAIR_TYPE(uint8_t, uint8_t),
        BENCHMARK_PAIR_TYPE(int, custom_float2),
        BENCHMARK_PAIR_TYPE(long long, custom_double2),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_sort_keys_benchmarks(benchmarks, stream, size);
    add_sort_pairs_benchmarks(benchmarks, stream, size);

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
