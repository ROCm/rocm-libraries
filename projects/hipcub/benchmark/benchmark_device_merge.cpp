// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP API
#include <hipcub/device/device_merge.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class key_type>
struct CompareFunction
{
    HIPCUB_HOST_DEVICE
    inline constexpr bool
        operator()(const key_type& a, const key_type& b)
    {
        return a < b;
    }
};

template<class Key>
void run_merge_keys_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type = Key;

    CompareFunction<key_type> compare_function;

    const size_t size1 = size / 2;
    const size_t size2 = size - size1;

    std::vector<key_type> keys_input1 = benchmark_utils::get_random_data<key_type>(
        size1,
        benchmark_utils::generate_limits<key_type>::min(),
        benchmark_utils::generate_limits<key_type>::max());

    std::vector<key_type> keys_input2 = benchmark_utils::get_random_data<key_type>(
        size2,
        benchmark_utils::generate_limits<key_type>::min(),
        benchmark_utils::generate_limits<key_type>::max());

    std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
    std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

    key_type* d_keys_input1;
    HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input1,
                        keys_input1.data(),
                        size1 * sizeof(key_type),
                        hipMemcpyHostToDevice));

    key_type* d_keys_input2;
    HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input2,
                        keys_input2.data(),
                        size2 * sizeof(key_type),
                        hipMemcpyHostToDevice));

    key_type* d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys_input1,
                                             size1,
                                             d_keys_input2,
                                             size2,
                                             d_keys_output,
                                             compare_function,
                                             stream));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input1,
                                                 size1,
                                                 d_keys_input2,
                                                 size2,
                                                 d_keys_output,
                                                 compare_function,
                                                 stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_keys_input1,
                                                     size1,
                                                     d_keys_input2,
                                                     size2,
                                                     d_keys_output,
                                                     compare_function,
                                                     stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input1));
    HIP_CHECK(hipFree(d_keys_input2));
    HIP_CHECK(hipFree(d_keys_output));
}

template<class Key, class Value>
void run_merge_pairs_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type   = Key;
    using value_type = Value;

    CompareFunction<key_type> compare_function;

    const size_t size1 = size / 2;
    const size_t size2 = size - size1;

    std::vector<key_type> keys_input1 = benchmark_utils::get_random_data<key_type>(
        size1,
        benchmark_utils::generate_limits<key_type>::min(),
        benchmark_utils::generate_limits<key_type>::max());
    std::vector<key_type> keys_input2 = benchmark_utils::get_random_data<key_type>(
        size2,
        benchmark_utils::generate_limits<key_type>::min(),
        benchmark_utils::generate_limits<key_type>::max());

    std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
    std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

    key_type* d_keys_input1;
    HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input1,
                        keys_input1.data(),
                        size1 * sizeof(key_type),
                        hipMemcpyHostToDevice));

    key_type* d_keys_input2;
    HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input2,
                        keys_input2.data(),
                        size2 * sizeof(key_type),
                        hipMemcpyHostToDevice));

    key_type* d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));

    std::vector<value_type> values_input1(size1);
    std::iota(values_input1.begin(), values_input1.end(), 0);
    value_type* d_values_input1;
    HIP_CHECK(hipMalloc(&d_values_input1, size1 * sizeof(value_type)));
    HIP_CHECK(hipMemcpy(d_values_input1,
                        values_input1.data(),
                        size1 * sizeof(value_type),
                        hipMemcpyHostToDevice));

    std::vector<value_type> values_input2(size2);
    std::iota(values_input2.begin(), values_input2.end(), size1);
    value_type* d_values_input2;
    HIP_CHECK(hipMalloc(&d_values_input2, size2 * sizeof(value_type)));
    HIP_CHECK(hipMemcpy(d_values_input2,
                        values_input2.data(),
                        size2 * sizeof(value_type),
                        hipMemcpyHostToDevice));

    value_type* d_values_output;
    HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys_input1,
                                              d_values_input1,
                                              size1,
                                              d_keys_input2,
                                              d_values_input2,
                                              size2,
                                              d_keys_output,
                                              d_values_output,
                                              compare_function,
                                              stream));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input1,
                                                  d_values_input1,
                                                  size1,
                                                  d_keys_input2,
                                                  d_values_input2,
                                                  size2,
                                                  d_keys_output,
                                                  d_values_output,
                                                  compare_function,
                                                  stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys_input1,
                                                      d_values_input1,
                                                      size1,
                                                      d_keys_input2,
                                                      d_values_input2,
                                                      size2,
                                                      d_keys_output,
                                                      d_values_output,
                                                      compare_function,
                                                      stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size
                            * (sizeof(key_type) + sizeof(value_type)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input1));
    HIP_CHECK(hipFree(d_keys_input2));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input1));
    HIP_CHECK(hipFree(d_values_input2));
    HIP_CHECK(hipFree(d_values_output));
}

#define CREATE_MERGE_KEYS_BENCHMARK(T)                 \
    benchmarks.push_back(benchmark::RegisterBenchmark( \
        std::string("device_merge_keys"                \
                    "<key_data_type:" #T ">.")         \
            .c_str(),                                  \
        [=](benchmark::State& state) { run_merge_keys_benchmark<T>(state, stream, size); }));

#define CREATE_MERGE_PAIRS_BENCHMARK(T, V)                            \
    benchmarks.push_back(benchmark::RegisterBenchmark(                \
        std::string("device_merge_pairs<"                             \
                    ",key_data_type:" #T ",value_data_type:" #V ">.") \
            .c_str(),                                                 \
        [=](benchmark::State& state) { run_merge_pairs_benchmark<T, V>(state, stream, size); }));

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

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    std::cout << "benchmark_device_merge" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;

    using custom_float2      = benchmark_utils::custom_type<float, float>;
    using custom_double2     = benchmark_utils::custom_type<double, double>;
    using custom_char_double = benchmark_utils::custom_type<char, double>;
    using custom_double_char = benchmark_utils::custom_type<double, char>;

    CREATE_MERGE_KEYS_BENCHMARK(int)
    CREATE_MERGE_KEYS_BENCHMARK(long long)
    CREATE_MERGE_KEYS_BENCHMARK(int8_t)
    CREATE_MERGE_KEYS_BENCHMARK(uint8_t)
    CREATE_MERGE_KEYS_BENCHMARK(short)
    CREATE_MERGE_KEYS_BENCHMARK(double)
    CREATE_MERGE_KEYS_BENCHMARK(float)
    CREATE_MERGE_KEYS_BENCHMARK(custom_float2)
    CREATE_MERGE_KEYS_BENCHMARK(custom_double2)

    CREATE_MERGE_PAIRS_BENCHMARK(int, int)
    CREATE_MERGE_PAIRS_BENCHMARK(long long, long long)
    CREATE_MERGE_PAIRS_BENCHMARK(int8_t, int8_t)
    CREATE_MERGE_PAIRS_BENCHMARK(uint8_t, uint8_t)
    CREATE_MERGE_PAIRS_BENCHMARK(short, short)
    CREATE_MERGE_PAIRS_BENCHMARK(custom_char_double, custom_char_double)
    CREATE_MERGE_PAIRS_BENCHMARK(int, custom_double_char)
    CREATE_MERGE_PAIRS_BENCHMARK(custom_double2, custom_double2)

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
