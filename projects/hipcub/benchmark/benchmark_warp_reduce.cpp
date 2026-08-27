// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/warp/warp_reduce.hpp>

constexpr unsigned int Trials = 100;

template<unsigned int WarpSize, class T>
__device__
auto warp_reduce_benchmark_fn(const T* d_input, T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = d_input[i];

    using wreduce_t = hipcub::WarpReduce<T, WarpSize>;
    __shared__ typename wreduce_t::TempStorage storage;
    auto                                       reduce_op = hipcub::Sum();
#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        value = wreduce_t(storage).Reduce(value, reduce_op);
    }

    d_output[i] = value;
}

template<unsigned int WarpSize, class T>
__device__
auto warp_reduce_benchmark_fn(const T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
{}

template<unsigned int WarpSize, class T>
__global__ __launch_bounds__(64)
void warp_reduce_kernel(const T* d_input, T* d_output)
{
    warp_reduce_benchmark_fn<WarpSize>(d_input, d_output);
}

template<unsigned int WarpSize, class T, class Flag>
__device__
auto segmented_warp_reduce_benchmark_fn(const T* d_input, Flag* d_flags, T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = d_input[i];
    auto flag  = d_flags[i];

    using wreduce_t = hipcub::WarpReduce<T, WarpSize>;
    __shared__ typename wreduce_t::TempStorage storage;
#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        value = wreduce_t(storage).HeadSegmentedSum(value, flag);
    }

    d_output[i] = value;
}

template<unsigned int WarpSize, class T, class Flag>
__device__
auto segmented_warp_reduce_benchmark_fn(const T* /*d_input*/, Flag* /*d_flags*/, T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
{}

template<unsigned int WarpSize, class T, class Flag>
__global__ __launch_bounds__(64)
void segmented_warp_reduce_kernel_fn(const T* d_input, Flag* d_flags, T* d_output)
{
    segmented_warp_reduce_benchmark_fn<WarpSize>(d_input, d_flags, d_output);
}

template<bool Segmented, unsigned int WarpSize, unsigned int BlockSize, class T, class Flag>
inline auto execute_warp_reduce_kernel(
    T* input, T* output, Flag* /* flags */, size_t items, hipStream_t stream) ->
    typename std::enable_if<!Segmented>::type
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(warp_reduce_kernel<WarpSize, T>),
                       dim3(items / BlockSize),
                       dim3(BlockSize),
                       0,
                       stream,
                       input,
                       output);
    HIP_CHECK(hipPeekAtLastError());
}

template<bool Segmented, unsigned int WarpSize, unsigned int BlockSize, class T, class Flag>
inline auto execute_warp_reduce_kernel(
    T* input, T* output, Flag* flags, size_t items, hipStream_t stream) ->
    typename std::enable_if<Segmented>::type
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(segmented_warp_reduce_kernel_fn<WarpSize, T, Flag>),
                       dim3(items / BlockSize),
                       dim3(BlockSize),
                       0,
                       stream,
                       input,
                       flags,
                       output);
    HIP_CHECK(hipPeekAtLastError());
}

template<bool Segmented, class T, unsigned int WarpSize, unsigned int BlockSize>
class warp_reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "warp_reduce")
                        .add("segmented", Segmented)
                        .add("warp_size", WarpSize)
                        .add("block_size", BlockSize)
                        .add("data_type", primbench::name<T>());

        return json;
    }

    void run(primbench::state& state) override
    {
        using flag_type = unsigned char;

        const auto& input_size = state.size;
        const auto& stream     = state.stream;

        const auto items = BlockSize * ((input_size + BlockSize - 1) / BlockSize);

        std::vector<T>         input = benchmark_utils::get_random_data<T>(items, T(0), T(10));
        std::vector<flag_type> flags = benchmark_utils::get_random_data<flag_type>(items, 0, 1);
        T*                     d_input;
        flag_type*             d_flags;
        T*                     d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_flags, items * sizeof(flag_type)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(d_flags, flags.data(), items * sizeof(flag_type), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                execute_warp_reduce_kernel<Segmented, WarpSize, BlockSize>(d_input,
                                                                           d_output,
                                                                           d_flags,
                                                                           items,
                                                                           stream);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_flags));
    }
};

#define CREATE_BENCHMARK(T, WS, BS) executor.queue<warp_reduce_benchmark<Segmented, T, WS, BS>>()

// If warp size limit is 16
#define BENCHMARK_TYPE_WS16(type)   \
    CREATE_BENCHMARK(type, 15, 32); \
    CREATE_BENCHMARK(type, 16, 32);

// If warp size limit is 32
#define BENCHMARK_TYPE_WS32(type)   \
    BENCHMARK_TYPE_WS16(type);      \
    CREATE_BENCHMARK(type, 31, 32); \
    CREATE_BENCHMARK(type, 32, 32); \
    CREATE_BENCHMARK(type, 32, 64);

// If warp size limit is 64
#define BENCHMARK_TYPE_WS64(type)   \
    BENCHMARK_TYPE_WS32(type);      \
    CREATE_BENCHMARK(type, 37, 64); \
    CREATE_BENCHMARK(type, 61, 64); \
    CREATE_BENCHMARK(type, 64, 64);

template<bool Segmented>
void add_benchmarks(primbench::executor& executor)
{
#if HIPCUB_WARP_THREADS_MACRO == 16
    BENCHMARK_TYPE_WS16(int);
    BENCHMARK_TYPE_WS16(float);
    BENCHMARK_TYPE_WS16(double);
    BENCHMARK_TYPE_WS16(int8_t);
    BENCHMARK_TYPE_WS16(uint8_t);
#elif HIPCUB_WARP_THREADS_MACRO == 32
    BENCHMARK_TYPE_WS32(int);
    BENCHMARK_TYPE_WS32(float);
    BENCHMARK_TYPE_WS32(double);
    BENCHMARK_TYPE_WS32(int8_t);
    BENCHMARK_TYPE_WS32(uint8_t);
#else
    BENCHMARK_TYPE_WS64(int);
    BENCHMARK_TYPE_WS64(float);
    BENCHMARK_TYPE_WS64(double);
    BENCHMARK_TYPE_WS64(int8_t);
    BENCHMARK_TYPE_WS64(uint8_t);
#endif
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Add benchmarks
    add_benchmarks<false>(executor);
    add_benchmarks<true>(executor);

    executor.run();
}
