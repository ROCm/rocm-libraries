// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"

#include <hipcub/warp/warp_scan.hpp>

constexpr unsigned int Trials = 100;

enum class scan_type
{
    inclusive_scan,
    exclusive_scan,
    broadcast
};

template<class Runner, class T, unsigned int BlockSize, unsigned int WarpSize>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output, const T init)
{
    Runner::template run<T, WarpSize>(input, output, init);
}

struct inclusive_scan
{
    static constexpr const char* name = "inclusive_scan";

    template<class T, unsigned int WarpSize>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {
        (void)init;

        const unsigned int i     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        auto               value = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
        auto                                     scan_op = hipcub::Sum();
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wscan_t(storage).InclusiveScan(value, value, scan_op);
        }

        output[i] = value;
    }

    template<class T, unsigned int WarpSize>
    __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {}
};

struct exclusive_scan
{
    static constexpr const char* name = "exclusive_scan";

    template<class T, unsigned int WarpSize>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {
        const unsigned int i     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        auto               value = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
        auto                                     scan_op = hipcub::Sum();
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wscan_t(storage).ExclusiveScan(value, value, init, scan_op);
        }

        output[i] = value;
    }
    template<class T, unsigned int WarpSize>
        __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {}
};

struct broadcast
{
    static constexpr const char* name = "broadcast";

    template<class T, unsigned int WarpSize>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<(benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>
                             && benchmark_utils::is_power_of_two(WarpSize))>
    {
        (void)init;

        const unsigned int i        = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int warp_id  = i / WarpSize;
        const unsigned int src_lane = warp_id % WarpSize;
        auto               value    = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            value = wscan_t(storage).Broadcast(value, src_lane);
        }

        output[i] = value;
    }

    template<class T, unsigned int WarpSize>
    __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!(benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>
                              && benchmark_utils::is_power_of_two(WarpSize))>
    {}
};

template<class Benchmark, class T, unsigned int BlockSize, unsigned int WarpSize>
class warp_scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "warp_scan")
            .add("subalgo", Benchmark::name)
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("warp_size", WarpSize)
            .add("lvl", "warp");
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        const size_t items = BlockSize * ((input_items + BlockSize - 1) / BlockSize);

        std::vector<T> input(items, 1.0f);

        T* d_input;
        T* d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items * Trials);
        state.add_writes<T>(items * Trials);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, WarpSize>),
                                   dim3(items / BlockSize),
                                   dim3(BlockSize),
                                   0,
                                   stream,
                                   d_input,
                                   d_output,
                                   input[0]);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK_IMPL(T, BS, WS, OP) executor.queue<warp_scan_benchmark<OP, T, BS, WS>>()

#define CREATE_BENCHMARK(T, BS, WS) CREATE_BENCHMARK_IMPL(T, BS, WS, Benchmark)

#if HIPCUB_WARP_THREADS_MACRO == 32
    #define BENCHMARK_TYPE(type)         \
        CREATE_BENCHMARK(type, 60, 15),  \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 62, 31),  \
        CREATE_BENCHMARK(type, 256, 32)
#else
    #define BENCHMARK_TYPE(type)         \
        CREATE_BENCHMARK(type, 60, 15),  \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 62, 31),  \
        CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 63, 63),  \
        CREATE_BENCHMARK(type, 64, 64),  \
        CREATE_BENCHMARK(type, 128, 64), \
        CREATE_BENCHMARK(type, 256, 64)
#endif

#if HIPCUB_WARP_THREADS_MACRO == 32
    #define BENCHMARK_TYPE_P2(type)      \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 256, 32)
#else
    #define BENCHMARK_TYPE_P2(type)      \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 64, 64),  \
        CREATE_BENCHMARK(type, 128, 64), \
        CREATE_BENCHMARK(type, 256, 64)
#endif

template<typename Benchmark>
auto add_benchmarks(primbench::executor& executor)
    -> std::enable_if_t<std::is_same_v<Benchmark, inclusive_scan>
                        || std::is_same_v<Benchmark, exclusive_scan>>
{
    BENCHMARK_TYPE(int);
    BENCHMARK_TYPE(float);
    BENCHMARK_TYPE(double);
    BENCHMARK_TYPE(int8_t);
    BENCHMARK_TYPE(custom_double2);
    BENCHMARK_TYPE(custom_int_double);
}

template<typename Benchmark>
auto add_benchmarks(primbench::executor& executor)
    -> std::enable_if_t<std::is_same_v<Benchmark, broadcast>>
{
    BENCHMARK_TYPE_P2(int);
    BENCHMARK_TYPE_P2(float);
    BENCHMARK_TYPE_P2(double);
    BENCHMARK_TYPE_P2(int8_t);
    BENCHMARK_TYPE_P2(custom_double2);
    BENCHMARK_TYPE_P2(custom_int_double);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<inclusive_scan>(executor);
    add_benchmarks<exclusive_scan>(executor);
    add_benchmarks<broadcast>(executor);

    executor.run();
}
