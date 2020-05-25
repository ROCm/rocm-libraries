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

// CUB's implementation of single_pass_scan_operators has maybe uninitialized parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVCC__
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "common_benchmark_header.hpp"

// HIP API
#include "hipcub/device/device_reduce.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key, class Value>
void run_benchmark(benchmark::State& state, size_t max_length, hipStream_t stream, size_t size)
{
    using key_type = Key;
    using value_type = Value;

    // Generate data
    std::vector<key_type> keys_input(size);

    unsigned int unique_count = 0;
    std::vector<size_t> key_counts = benchmark_utils::get_random_data<size_t>(100000, 1, max_length);
    size_t offset = 0;
    while(offset < size)
    {
        const size_t key_count = key_counts[unique_count % key_counts.size()];
        const size_t end = std::min(size, offset + key_count);
        for(size_t i = offset; i < end; i++)
        {
            keys_input[i] = unique_count;
        }

        unique_count++;
        offset += key_count;
    }

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    key_type * d_keys_input;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    key_type * d_unique_output;
    value_type * d_aggregates_output;
    unsigned int * d_unique_count_output;
    HIP_CHECK(hipMalloc(&d_unique_output, unique_count * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_aggregates_output, unique_count * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_unique_count_output, sizeof(unsigned int)));

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;

    hipcub::Sum reduce_op;

    HIP_CHECK(
        hipcub::DeviceReduce::ReduceByKey(
            nullptr, temporary_storage_bytes,
            d_keys_input, d_unique_output, d_values_input,
            d_aggregates_output,
            d_unique_count_output,
            reduce_op, size,
            stream
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            hipcub::DeviceReduce::ReduceByKey(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input,
                d_unique_output, d_values_input, d_aggregates_output,
                d_unique_count_output,
                reduce_op, size,
                stream
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
                hipcub::DeviceReduce::ReduceByKey(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input,
                    d_unique_output, d_values_input, d_aggregates_output,
                    d_unique_count_output,
                    reduce_op, size,
                    stream
                )
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_unique_output));
    HIP_CHECK(hipFree(d_aggregates_output));
    HIP_CHECK(hipFree(d_unique_count_output));
}

#define CREATE_BENCHMARK(Key, Value) \
benchmark::RegisterBenchmark( \
    (std::string("reduce_by_key") + "<" #Key ", " #Value ">" + \
        "([1, " + std::to_string(max_length) + "])" \
    ).c_str(), \
    run_benchmark<Key, Value>, \
    max_length, stream, size \
)

void add_benchmarks(size_t max_length,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    using custom_float2 = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(int, float),
        CREATE_BENCHMARK(int, double),
        CREATE_BENCHMARK(int, custom_float2),
        CREATE_BENCHMARK(int, custom_double2),

        CREATE_BENCHMARK(int8_t, int8_t),
        CREATE_BENCHMARK(uint8_t, uint8_t),

        CREATE_BENCHMARK(long long, float),
        CREATE_BENCHMARK(long long, double),
        CREATE_BENCHMARK(long long, custom_float2),
        CREATE_BENCHMARK(long long, custom_double2),
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
    add_benchmarks(1000, benchmarks, stream, size);
    add_benchmarks(10, benchmarks, stream, size);

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
