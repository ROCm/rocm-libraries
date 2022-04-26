// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    bool Exclusive,
    class Config,
    class T,
    class BinaryFunction
>
auto run_device_scan(void * temporary_storage,
                     size_t& storage_size,
                     T * input,
                     T * output,
                     const T initial_value,
                     const size_t input_size,
                     BinaryFunction scan_op,
                     const hipStream_t stream,
                     const bool debug = false)
    -> typename std::enable_if<Exclusive, hipError_t>::type
{
    return rocprim::exclusive_scan<Config>(
        temporary_storage, storage_size,
        input, output, initial_value, input_size,
        scan_op, stream, debug
    );
}

template<
    bool Exclusive,
    class Config,
    class T,
    class BinaryFunction
>
auto run_device_scan(void * temporary_storage,
                     size_t& storage_size,
                     T * input,
                     T * output,
                     const T initial_value,
                     const size_t input_size,
                     BinaryFunction scan_op,
                     const hipStream_t stream,
                     const bool debug = false)
    -> typename std::enable_if<!Exclusive, hipError_t>::type
{
    (void) initial_value;
    return rocprim::inclusive_scan<Config>(
        temporary_storage, storage_size,
        input, output, input_size,
        scan_op, stream, debug
    );
}

template<
    bool Exclusive,
    typename Config,
    typename K,
    typename T,
    typename BinaryFunction,
    typename CompareFunction = rocprim::equal_to<K>
>
auto run_device_scan_by_key(void* temporary_storage,
                            size_t& storage_size,
                            const K* keys,
                            const T* input,
                            T* output,
                            const T initial_value,
                            const size_t input_size,
                            const BinaryFunction scan_op,
                            const CompareFunction compare,
                            const hipStream_t stream,
                            const bool debug = false)
    -> typename std::enable_if<Exclusive, hipError_t>::type
{
    return rocprim::exclusive_scan_by_key<Config>(
        temporary_storage, storage_size,
        keys, input, output, initial_value, input_size,
        scan_op, compare, stream, debug
    );
}

template<
    bool Exclusive,
    typename Config,
    typename K,
    typename T,
    typename BinaryFunction,
    typename CompareFunction
>
auto run_device_scan_by_key(void * temporary_storage,
                            size_t& storage_size,
                            const K* keys,
                            const T* input,
                            T* output,
                            const T /*initial_value*/,
                            const size_t input_size,
                            const BinaryFunction scan_op,
                            const CompareFunction compare,
                            const hipStream_t stream,
                            const bool debug = false)
    -> typename std::enable_if<!Exclusive, hipError_t>::type
{
    return rocprim::inclusive_scan_by_key<Config>(
        temporary_storage, storage_size,
        keys, input, output, input_size,
        scan_op, compare, stream, debug
    );
}

template<
    bool Exclusive,
    class T,
    class BinaryFunction,
    class Config
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   BinaryFunction scan_op)
{
    std::vector<T> input = get_random_data<T>(size, T(0), T(1000));
    T initial_value = T(123);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK((
        run_device_scan<Exclusive, Config>(
            d_temp_storage, temp_storage_size_bytes,
            d_input, d_output, initial_value, size,
            scan_op, stream
        )
    ));
    HIP_CHECK(hipMalloc(&d_temp_storage,temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 5; i++)
    {
        HIP_CHECK((
            run_device_scan<Exclusive, Config>(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, initial_value, size,
                scan_op, stream
            )
        ));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((
                run_device_scan<Exclusive, Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, initial_value, size,
                    scan_op, stream
                )
            ));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

template<
    bool Exclusive,
    typename K,
    typename T,
    typename BinaryFunction,
    typename CompareFunction = rocprim::equal_to<K>,
    typename Config = rocprim::default_config
>
void run_benchmark_by_key(benchmark::State& state,
                          const size_t size,
                          const hipStream_t stream,
                          const BinaryFunction scan_op,
                          const CompareFunction compare = CompareFunction())
{
    constexpr bool debug = false;
    const std::vector<T> input = get_random_data<T>(size, T(0), T(1000));

    const auto max_segment_length = state.range();
    const std::vector<K> keys
        = get_random_segments<K>(size, max_segment_length, std::random_device {}());

    T initial_value = T(123);
    T* d_input;
    K* d_keys;
    T* d_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_keys, keys.size() * sizeof(keys[0])));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(input[0])));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(keys[0]), hipMemcpyHostToDevice));

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;
    void* d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK((
        run_device_scan_by_key<Exclusive, Config>(
            d_temp_storage, temp_storage_size_bytes,
            d_keys, d_input, d_output, initial_value, size,
            scan_op, compare, stream, debug
        )
    ));
    HIP_CHECK(hipMalloc(&d_temp_storage,temp_storage_size_bytes));

    // Warm-up
    for(size_t i = 0; i < 5; i++)
    {
        HIP_CHECK((
            run_device_scan_by_key<Exclusive, Config>(
                d_temp_storage, temp_storage_size_bytes,
                d_keys, d_input, d_output, initial_value, size,
                scan_op, compare, stream, debug
            )
        ));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((
                run_device_scan_by_key<Exclusive, Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_keys, d_input, d_output, initial_value, size,
                    scan_op, compare, stream, debug
                )
            ));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        using seconds_d = std::chrono::duration<double>;

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds = std::chrono::duration_cast<seconds_d>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

#ifdef BENCHMARK_CONFIG_TUNING

template <
    bool EXCL,
    typename T, typename SCAN_OP,
    rocprim::block_scan_algorithm BSA,
    unsigned int BS
>
struct scan_add_benchmark_bind {

    template <unsigned int IPT>
    static std::string get_config_name() {
        return std::string {typeid(T).name()} + ", " + typeid(SCAN_OP).name() + ", scan_config<"
               + std::to_string(BS) + ", " + std::to_string(IPT) + ", "
               + std::string(BSA == rocprim::block_scan_algorithm::using_warp_scan
                                 ? "using_warp_scan"
                                 : "using_reduce_scan")
               + ">";
    }

    template<unsigned int IPT>
    struct add_benchmark {
        void operator()(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                        hipStream_t stream,
                        size_t size)
        {
            benchmarks.push_back(
                benchmark::RegisterBenchmark(
                    (std::string(EXCL ? "exclusive_scan" : "inclusive_scan")
                        + "<" + get_config_name<IPT>() + ">").c_str(),
                    run_benchmark<
                        EXCL, T, SCAN_OP,
                        rocprim::scan_config<
                            BS, IPT, true,
                            rocprim::block_load_method::block_load_transpose,
                            rocprim::block_store_method::block_store_transpose, BSA
                        >
                    >, size, stream, SCAN_OP()
                )
            );
        }
    };

    template<unsigned int IPT>
    struct add_benchmark_by_key {
        void operator()(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                        hipStream_t stream,
                        size_t size)
        {
            benchmarks.push_back(
                benchmark::RegisterBenchmark(
                    (std::string(EXCL ? "exclusive_scan_by_key" : "inclusive_scan_by_key")
                        + "<int, " + get_config_name<IPT>() + ">").c_str(),
                    run_benchmark_by_key<
                        EXCL, int, T, SCAN_OP, rocprim::equal_to<int>,
                        rocprim::scan_config<
                            BS, IPT, true,
                            rocprim::block_load_method::block_load_transpose,
                            rocprim::block_store_method::block_store_transpose, BSA
                        >
                    >, size, stream, SCAN_OP(), rocprim::equal_to<int>{}
                )->Range(1024,1024)
            );
        }
    };
};

template<
    bool EXCL,
    typename T, typename SCAN_OP,
    rocprim::block_scan_algorithm BSA,
    unsigned int BS, unsigned int IPT,
    unsigned int MaxItemsPerThread
>
void scan_benchmark_generate_ipt_grid(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
{
    using range = make_index_range<unsigned int, IPT, MaxItemsPerThread>;
    static_for_each<range, scan_add_benchmark_bind<EXCL, T, SCAN_OP, BSA, BS>::template add_benchmark>(
        benchmarks, stream, size);
}

template<
    bool EXCL,
    typename T, typename SCAN_OP,
    rocprim::block_scan_algorithm BSA,
    unsigned int BS, unsigned int IPT,
    unsigned int MaxItemsPerThread
>
void scan_benchmark_by_key_generate_ipt_grid(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
{
    using range = make_index_range<unsigned int, IPT, MaxItemsPerThread>;
    static_for_each<range, scan_add_benchmark_bind<EXCL, T, SCAN_OP, BSA, BS>::template add_benchmark_by_key>(
        benchmarks, stream, size);
}

constexpr rocprim::block_scan_algorithm using_warp_scan = rocprim::block_scan_algorithm::using_warp_scan;
constexpr rocprim::block_scan_algorithm reduce_then_scan = rocprim::block_scan_algorithm::reduce_then_scan;

#define CREATE_BENCHMARK(EXCL, T, SCAN_OP, MIPT)                                                                         \
    scan_benchmark_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 64, 1, MIPT>(benchmarks, stream, size);          \
    scan_benchmark_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 128, 1, MIPT>(benchmarks, stream, size);         \
    scan_benchmark_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 256, 1, MIPT>(benchmarks, stream, size);         \
    scan_benchmark_generate_ipt_grid<EXCL, T, SCAN_OP, reduce_then_scan, 256, 1, MIPT>(benchmarks, stream, size);        \
    scan_benchmark_by_key_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 64, 1, MIPT>(benchmarks, stream, size);   \
    scan_benchmark_by_key_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 128, 1, MIPT>(benchmarks, stream, size);  \
    scan_benchmark_by_key_generate_ipt_grid<EXCL, T, SCAN_OP, using_warp_scan, 256, 1, MIPT>(benchmarks, stream, size);  \
    scan_benchmark_by_key_generate_ipt_grid<EXCL, T, SCAN_OP, reduce_then_scan, 256, 1, MIPT>(benchmarks, stream, size);

#else // BENCHMARK_CONFIG_TUNING

#define CREATE_BENCHMARK(EXCL, T, SCAN_OP) \
benchmark::RegisterBenchmark(                                                         \
    (std::string(EXCL ? "exclusive_scan" : "inclusive_scan") +                        \
    ("<" #T ", " #SCAN_OP ">")).c_str(),                                              \
    run_benchmark<EXCL, T, SCAN_OP, rocprim::default_config>, size, stream, SCAN_OP() \
),                                                                                    \
benchmark::RegisterBenchmark(                                                         \
    (std::string(EXCL ? "exclusive_scan_by_key" : "inclusive_scan_by_key") +          \
    ("<int, " #T ", " #SCAN_OP ">")).c_str(),                                         \
    run_benchmark_by_key<EXCL, int, T, SCAN_OP, rocprim::equal_to<int>>,              \
    size, stream, SCAN_OP(), rocprim::equal_to<int>{}                                 \
)->RangeMultiplier(16)->Range(1,1<<16),

#endif // BENCHMARK_CONFIG_TUNING

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

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    using custom_double2 = custom_type<double, double>;

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_float2 = custom_type<float, float>;
#endif

    // Add benchmarks
    #ifdef BENCHMARK_CONFIG_TUNING
        // Compilation may never finish, if the compiler needs to compile too many kernels,
        // it is recommended to compile benchmarks only for 1-2 types when BENCHMARK_CONFIG_TUNING is used
        // (all other CREATE_BENCHMARK should be commented/removed).

        std::vector<benchmark::internal::Benchmark*> benchmarks;

        CREATE_BENCHMARK(false, int, rocprim::plus<int>, 20)
        CREATE_BENCHMARK(true, int, rocprim::plus<int>, 20)

        CREATE_BENCHMARK(false, float, rocprim::plus<float>, 20)
        CREATE_BENCHMARK(true, float, rocprim::plus<float>, 20)

        CREATE_BENCHMARK(false, double, rocprim::plus<double>, 15)
        CREATE_BENCHMARK(true, double, rocprim::plus<double>, 15)

        CREATE_BENCHMARK(false, long long, rocprim::plus<long long>, 15)
        CREATE_BENCHMARK(true, long long, rocprim::plus<long long>, 15)

        CREATE_BENCHMARK(false, custom_double2, rocprim::plus<custom_double2>, 15)
        CREATE_BENCHMARK(true, custom_double2, rocprim::plus<custom_double2>, 15)

        CREATE_BENCHMARK(false, int8_t, rocprim::plus<int8_t>, 20)
        CREATE_BENCHMARK(true, int8_t, rocprim::plus<int8_t>, 20)

        CREATE_BENCHMARK(false, rocprim::half, rocprim::plus<rocprim::half>, 30)
        CREATE_BENCHMARK(true, rocprim::half, rocprim::plus<rocprim::half>, 30)
    #else
        std::vector<benchmark::internal::Benchmark*> benchmarks =
        {
            CREATE_BENCHMARK(false, int, rocprim::plus<int>)
            CREATE_BENCHMARK(true, int, rocprim::plus<int>)

            CREATE_BENCHMARK(false, float, rocprim::plus<float>)
            CREATE_BENCHMARK(true, float, rocprim::plus<float>)

            CREATE_BENCHMARK(false, double, rocprim::plus<double>)
            CREATE_BENCHMARK(true, double, rocprim::plus<double>)

            CREATE_BENCHMARK(false, long long, rocprim::plus<long long>)
            CREATE_BENCHMARK(true, long long, rocprim::plus<long long>)

            CREATE_BENCHMARK(false, float2, rocprim::plus<float2>)
            CREATE_BENCHMARK(true, float2, rocprim::plus<float2>)

            CREATE_BENCHMARK(false, custom_float2, rocprim::plus<custom_float2>)
            CREATE_BENCHMARK(true, custom_float2, rocprim::plus<custom_float2>)

            CREATE_BENCHMARK(false, double2, rocprim::plus<double2>)
            CREATE_BENCHMARK(true, double2, rocprim::plus<double2>)

            CREATE_BENCHMARK(false, custom_double2, rocprim::plus<custom_double2>)
            CREATE_BENCHMARK(true, custom_double2, rocprim::plus<custom_double2>)

            CREATE_BENCHMARK(false, int8_t, rocprim::plus<int8_t>)
            CREATE_BENCHMARK(true, int8_t, rocprim::plus<int8_t>)

            CREATE_BENCHMARK(false, uint8_t, rocprim::plus<uint8_t>)
            CREATE_BENCHMARK(true, uint8_t, rocprim::plus<uint8_t>)

            CREATE_BENCHMARK(false, rocprim::half, rocprim::plus<rocprim::half>)
            CREATE_BENCHMARK(true, rocprim::half, rocprim::plus<rocprim::half>)
        };
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
