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
// SOFTWARE

// CUB's implementation of single_pass_scan_operators has maybe uninitialized parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVCC__
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "common_benchmark_header.hpp"

// HIP API
#include "hipcub/device/device_scan.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    bool Exclusive,
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
    return hipcub::DeviceScan::ExclusiveScan(
        temporary_storage, storage_size,
        input, output, scan_op, initial_value, input_size,
        stream, debug
    );
}

template<
    bool Exclusive,
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
    return hipcub::DeviceScan::InclusiveScan(
        temporary_storage, storage_size,
        input, output, scan_op, input_size,
        stream, debug
    );
}

template<
    bool Exclusive,
    class T,
    class BinaryFunction
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   BinaryFunction scan_op)
{
    std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(1000));
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
    size_t temp_storage_size_bytes = 0;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK((
        run_device_scan<Exclusive>(
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
            run_device_scan<Exclusive>(
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
                run_device_scan<Exclusive>(
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


#define CREATE_BENCHMARK(EXCL, T, SCAN_OP) \
benchmark::RegisterBenchmark( \
    (std::string(EXCL ? "exclusive_scan" : "inclusive_scan") + \
    ("<" #T ", " #SCAN_OP ">")).c_str(), \
    run_benchmark<EXCL, T, SCAN_OP>, size, stream, SCAN_OP() \
),


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

    using custom_double2 = benchmark_utils::custom_type<double, double>;
    using custom_float2 = benchmark_utils::custom_type<float, float>;

    // Compilation may never finish, if the compiler needs to compile too many kernels,
    // it is recommended to compile benchmarks only for 1-2 types when BENCHMARK_CONFIG_TUNING is used
    // (all other CREATE_*_BENCHMARK should be commented/removed).

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_BENCHMARK(false, int, hipcub::Sum)
        CREATE_BENCHMARK(true, int, hipcub::Sum)

        CREATE_BENCHMARK(false, float, hipcub::Sum)
        CREATE_BENCHMARK(true, float, hipcub::Sum)

        CREATE_BENCHMARK(false, double, hipcub::Sum)
        CREATE_BENCHMARK(true, double, hipcub::Sum)

        CREATE_BENCHMARK(false, long long, hipcub::Sum)
        CREATE_BENCHMARK(true, long long, hipcub::Sum)

        CREATE_BENCHMARK(false, custom_float2, hipcub::Sum)
        CREATE_BENCHMARK(true, custom_float2, hipcub::Sum)

        CREATE_BENCHMARK(false, custom_double2, hipcub::Sum)
        CREATE_BENCHMARK(true, custom_double2, hipcub::Sum)

        CREATE_BENCHMARK(false, int8_t, hipcub::Sum)
        CREATE_BENCHMARK(true, int8_t, hipcub::Sum)

        CREATE_BENCHMARK(false, uint8_t, hipcub::Sum)
        CREATE_BENCHMARK(true, uint8_t, hipcub::Sum)
    };

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
