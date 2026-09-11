// MIT License
//
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/block/block_scan.hpp>

constexpr unsigned int Trials = 100;

template<class Runner, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output, const T init)
{
    Runner::template run<T, BlockSize, ItemsPerThread>(input, output, init);
}

template<hipcub::BlockScanAlgorithm Algorithm, class Name>
struct inclusive_scan
{
    static const char* get_algorithm_name()
    {
        return Name::name;
    }

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
        __shared__ typename bscan_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bscan_t(storage).InclusiveScan(values, values, hipcub::Sum());
        }

        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<hipcub::BlockScanAlgorithm Algorithm, class Name>
struct exclusive_scan
{
    static const char* get_algorithm_name()
    {
        return Name::name;
    }

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
        __shared__ typename bscan_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bscan_t(storage).ExclusiveScan(values, values, init, hipcub::Sum());
        }

        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<class Benchmark, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
class block_scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_scan")
            .add("subalgo", Benchmark::get_algorithm_name())
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input(items, T(1));
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items * Trials);
        state.add_writes<T>(items * Trials);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread>),
                                   dim3(items / items_per_block),
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

#define CREATE_BENCHMARK(T, BS, IPT) executor.queue<block_scan_benchmark<Benchmark, T, BS, IPT>>()

#define BENCHMARK_TYPE(type, block)    \
    CREATE_BENCHMARK(type, block, 1);  \
    CREATE_BENCHMARK(type, block, 3);  \
    CREATE_BENCHMARK(type, block, 4);  \
    CREATE_BENCHMARK(type, block, 8);  \
    CREATE_BENCHMARK(type, block, 11); \
    CREATE_BENCHMARK(type, block, 16)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    // When block size is less than or equal to warp size
    BENCHMARK_TYPE(int, 64);
    BENCHMARK_TYPE(float, 64);
    BENCHMARK_TYPE(double, 64);
    BENCHMARK_TYPE(uint8_t, 64);

    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(float, 256);
    BENCHMARK_TYPE(double, 256);
    BENCHMARK_TYPE(uint8_t, 256);

    CREATE_BENCHMARK(custom_float2, 256, 1);
    CREATE_BENCHMARK(custom_float2, 256, 4);
    CREATE_BENCHMARK(custom_float2, 256, 8);

    CREATE_BENCHMARK(custom_double2, 256, 1);
    CREATE_BENCHMARK(custom_double2, 256, 4);
    CREATE_BENCHMARK(custom_double2, 256, 8);
}

// At the time of writing, BLOCK_SCAN_RAKING and BLOCK_SCAN_RAKING_MEMOIZE have the same values, so we can't switch on them like a normal enum.
// So we have to pass the algorithm names here
struct inclusive_raking_tag
{
    static constexpr const char* name = "inclusive_scan(block_scan_raking)";
};

struct inclusive_raking_memoize_tag
{
    static constexpr const char* name = "inclusive_scan(block_scan_raking_memoize)";
};

struct inclusive_warp_scans_tag
{
    static constexpr const char* name = "inclusive_scan(block_scan_warp_scans)";
};

struct exclusive_raking_tag
{
    static constexpr const char* name = "exclusive_scan(block_scan_raking)";
};

struct exclusive_raking_memoize_tag
{
    static constexpr const char* name = "exclusive_scan(block_scan_raking_memoize)";
};

struct exclusive_warp_scans_tag
{
    static constexpr const char* name = "exclusive_scan(block_scan_warp_scans)";
};

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch    = 100;
    settings.noise_tolerance_percent = 2;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<
        inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING, inclusive_raking_tag>>(
        executor);
    add_benchmarks<inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE,
                                  inclusive_raking_memoize_tag>>(executor);
    add_benchmarks<inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
                                  inclusive_warp_scans_tag>>(executor);

    add_benchmarks<
        exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING, exclusive_raking_tag>>(
        executor);
    add_benchmarks<exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE,
                                  exclusive_raking_memoize_tag>>(executor);
    add_benchmarks<exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
                                  exclusive_warp_scans_tag>>(executor);

    executor.run();
}
