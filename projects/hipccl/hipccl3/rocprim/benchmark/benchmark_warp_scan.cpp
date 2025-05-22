// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_scan.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

enum class scan_type
{
    inclusive_scan,
    exclusive_scan,
    broadcast
};

template<class Runner, class T, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void kernel(const T* input, T* output, const T init)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        Runner::template run<T, VirtualWaveSize, Trials>(input, output, init);
    }
}

struct inclusive_scan
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().inclusive_scan(value, value, storage);
        }

        output[i] = value;
    }
};

struct exclusive_scan
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().exclusive_scan(value, value, init, storage);
        }

        output[i] = value;
    }
};

struct broadcast
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i        = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int warp_id  = i / VirtualWaveSize;
        const unsigned int src_lane = warp_id % VirtualWaveSize;
        auto               value    = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            value = wscan_t().broadcast(value, src_lane, storage);
        }

        output[i] = value;
    }
};

template<typename T,
         unsigned int BlockSize,
         unsigned int VirtualWaveSize,
         class Type,
         unsigned int Trials = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1) / BlockSize);
    // Allocate and fill memory
    std::vector<T>        input(size, (T)1);
    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Type, T, VirtualWaveSize, Trials>),
                               dim3(size / BlockSize),
                               dim3(BlockSize),
                               0,
                               stream,
                               d_input.get(),
                               d_output.get(),
                               input[0]);
        });

    state.set_throughput(Trials * size, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, WS)                                                              \
    executor.queue_fn(bench_naming::format_name("{lvl:warp,algo:scan,key_type:" #T ",subalgo:"   \
                                                + method_name + ",ws:" #WS ",cfg:{bs:" #BS "}}") \
                          .c_str(),                                                              \
                      run_benchmark<T, BS, WS, Benchmark>);

// clang-format off
#define BENCHMARK_TYPE(type)        \
    CREATE_BENCHMARK(type, 64, 64)  \
    CREATE_BENCHMARK(type, 128, 64) \
    CREATE_BENCHMARK(type, 256, 64) \
    CREATE_BENCHMARK(type, 256, 32) \
    CREATE_BENCHMARK(type, 256, 16) \
    CREATE_BENCHMARK(type, 63, 63)  \
    CREATE_BENCHMARK(type, 62, 31)  \
    CREATE_BENCHMARK(type, 60, 15)
// clang-format on

// clang-format off
#define BENCHMARK_TYPE_P2(type)     \
    CREATE_BENCHMARK(type, 64, 64)  \
    CREATE_BENCHMARK(type, 128, 64) \
    CREATE_BENCHMARK(type, 256, 64) \
    CREATE_BENCHMARK(type, 256, 32) \
    CREATE_BENCHMARK(type, 256, 16)
// clang-format on

template<typename Benchmark>
auto add_benchmarks(benchmark_utils::executor& executor, const std::string& method_name)
    -> std::enable_if_t<std::is_same<Benchmark, inclusive_scan>::value
                        || std::is_same<Benchmark, exclusive_scan>::value>
{
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;

    BENCHMARK_TYPE(int)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(custom_double2)
    BENCHMARK_TYPE(custom_int_double)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)
}

template<typename Benchmark>
auto add_benchmarks(benchmark_utils::executor& executor, const std::string& method_name)
    -> std::enable_if_t<std::is_same<Benchmark, broadcast>::value>
{
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;

    BENCHMARK_TYPE_P2(int)
    BENCHMARK_TYPE_P2(float)
    BENCHMARK_TYPE_P2(double)
    BENCHMARK_TYPE_P2(int8_t)
    BENCHMARK_TYPE_P2(uint8_t)
    BENCHMARK_TYPE_P2(rocprim::half)
    BENCHMARK_TYPE_P2(custom_double2)
    BENCHMARK_TYPE_P2(custom_int_double)
    BENCHMARK_TYPE_P2(rocprim::int128_t)
    BENCHMARK_TYPE_P2(rocprim::uint128_t)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<inclusive_scan>(executor, "inclusive_scan");
    add_benchmarks<exclusive_scan>(executor, "exclusive_scan");
    add_benchmarks<broadcast>(executor, "broadcast");

    executor.run();
}
