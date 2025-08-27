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

// HIP API
#include <hipcub/warp/warp_load.hpp>

#include <type_traits>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto warp_load_benchmark(T* d_input, T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using WarpLoadT = ::hipcub::WarpLoad<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int      tile_size      = ItemsPerThread * LogicalWarpSize;

    const unsigned warp_id        = threadIdx.x / LogicalWarpSize;
    const unsigned global_warp_id = blockIdx.x * warps_in_block + warp_id;
    __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
    T                                          thread_data[ItemsPerThread];

    WarpLoadT(temp_storage[warp_id]).Load(d_input + global_warp_id * tile_size, thread_data);

#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned striped_global_idx
            = BlockSize * ItemsPerThread * blockIdx.x + BlockSize * i + threadIdx.x;
        d_output[striped_global_idx] = thread_data[i];
    }
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto warp_load_benchmark(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_load_kernel(T* d_input, T* d_output)
{
    warp_load_benchmark<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input, d_output);
}

template<class T,
         unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         unsigned                    Trials = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
    const unsigned     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(10));
    T*             d_input;
    T*             d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < Trials; i++)
        {
            warp_load_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
        }
        HIP_CHECK(hipPeekAtLastError())
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

#define CREATE_BENCHMARK(T, BS, IT, WS, ALG)                                              \
    benchmark::RegisterBenchmark(std::string("warp_load<data_type:" #T ",block_size:" #BS \
                                             ",items_per_thread:" #IT ",warp_size:" #WS   \
                                             ",sub_algorithm_name:" #ALG ">.")            \
                                     .c_str(),                                            \
                                 &run_benchmark<T, BS, IT, WS, ALG>,                      \
                                 stream,                                                  \
                                 size)

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

    std::cout << "benchmark_warp_load" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_VECTORIZE),
        // WARP_LOAD_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_TRANSPOSE),
        CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_DIRECT),
        CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_STRIPED),
        CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_VECTORIZE)
        // WARP_LOAD_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_TRANSPOSE)
    };

    if(::benchmark_utils::is_warp_size_supported(64))
    {
        std::vector<benchmark::internal::Benchmark*> additional_benchmarks{
            CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            // WARP_LOAD_TRANSPOSE removed because of shared memory limit
            // CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_VECTORIZE),
            // WARP_LOAD_TRANSPOSE removed because of shared memory limit
            // CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_TRANSPOSE),
            CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_DIRECT),
            CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_STRIPED),
            CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_VECTORIZE)
            // WARP_LOAD_TRANSPOSE removed because of shared memory limit
            // CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_TRANSPOSE)
        };
        benchmarks.insert(benchmarks.end(),
                          additional_benchmarks.begin(),
                          additional_benchmarks.end());
    }

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
