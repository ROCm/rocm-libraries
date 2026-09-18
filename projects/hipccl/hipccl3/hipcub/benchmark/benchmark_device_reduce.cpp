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

#include "hipcub/config.hpp"

#include <hipcub/device/device_reduce.hpp>

template<class T, class OutputT, class ReduceKernel>
class reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_reduce")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("op", ReduceKernel::name);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(1000));

        T*       d_input;
        OutputT* d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(OutputT)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        auto reduce = ReduceKernel::kernel;

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&] {
            HIP_CHECK(reduce(d_temp_storage, temp_storage_bytes, d_input, d_output, items, stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

template<typename T, typename Op>
struct Benchmark;

template<class T>
struct sum_kernel
{
    static constexpr const char* name = "sum";

    static constexpr hipError_t (*kernel)(void*, size_t&, T*, T*, int, hipStream_t)
        = &hipcub::DeviceReduce::Sum;
};

template<typename T>
struct Benchmark<T, hipcub::Sum>
{
    using type = reduce_benchmark<T, T, sum_kernel<T>>;
};

template<class T>
struct min_kernel
{
    static constexpr const char* name = "min";

    static constexpr hipError_t (*kernel)(void*, size_t&, T*, T*, int, hipStream_t)
        = &hipcub::DeviceReduce::Min;
};

template<typename T>
struct Benchmark<T, hipcub::Min>
{
    using type = reduce_benchmark<T, T, min_kernel<T>>;
};

template<class T>
struct argmin_kernel
{
    static constexpr const char* name = "argmin";

    using Difference = int;
    using Iterator   = hipcub::ArgIndexInputIterator<T*, Difference>;
    using KeyValue   = typename Iterator::value_type;

    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    static constexpr hipError_t (*kernel)(void*, size_t&, T*, KeyValue*, int, hipStream_t)
        = &hipcub::DeviceReduce::ArgMin;
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
};
template<typename T>
struct Benchmark<T, hipcub::ArgMin>
{
    using type = reduce_benchmark<T, typename argmin_kernel<T>::KeyValue, argmin_kernel<T>>;
};

#define CREATE_BENCHMARK(T, REDUCE_OP) executor.queue<Benchmark<T, REDUCE_OP>::type>()

#define CREATE_BENCHMARKS(REDUCE_OP)        \
    CREATE_BENCHMARK(int, REDUCE_OP);       \
    CREATE_BENCHMARK(int64_t, REDUCE_OP); \
    CREATE_BENCHMARK(float, REDUCE_OP);     \
    CREATE_BENCHMARK(double, REDUCE_OP);    \
    CREATE_BENCHMARK(int8_t, REDUCE_OP)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARKS(hipcub::Sum);
    CREATE_BENCHMARK(custom_double2, hipcub::Sum);
    CREATE_BENCHMARKS(hipcub::Min);
#ifdef HIPCUB_ROCPRIM_API
    CREATE_BENCHMARK(custom_double2, hipcub::Min);
#endif
    CREATE_BENCHMARKS(hipcub::ArgMin);
#ifdef HIPCUB_ROCPRIM_API
    CREATE_BENCHMARK(custom_double2, hipcub::ArgMin);
#endif

    executor.run();
}
