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
// SOFTWARE

// CUB's implementation of single_pass_scan_operators has maybe uninitialized
// parameters, disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVIDIA__
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "benchmark_utils.hpp"

#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>

template<bool Exclusive, class T, class BinaryFunction>
auto run_device_scan(void*             temporary_storage,
                     size_t&           storage_size,
                     T*                input,
                     T*                output,
                     const T           initial_value,
                     const size_t      input_size,
                     BinaryFunction    scan_op,
                     const hipStream_t stream) ->
    typename std::enable_if<Exclusive, hipError_t>::type
{
    return hipcub::DeviceScan::ExclusiveScan(temporary_storage,
                                             storage_size,
                                             input,
                                             output,
                                             scan_op,
                                             initial_value,
                                             input_size,
                                             stream);
}

template<bool Exclusive, class T, class BinaryFunction>
auto run_device_scan(void*             temporary_storage,
                     size_t&           storage_size,
                     T*                input,
                     T*                output,
                     const T           initial_value,
                     const size_t      input_size,
                     BinaryFunction    scan_op,
                     const hipStream_t stream) ->
    typename std::enable_if<!Exclusive, hipError_t>::type
{
    (void)initial_value;
    return hipcub::DeviceScan::InclusiveScan(temporary_storage,
                                             storage_size,
                                             input,
                                             output,
                                             scan_op,
                                             input_size,
                                             stream);
}

template<bool Exclusive, class T, class K, class BinaryFunction>
auto run_device_scan_by_key(void*             temporary_storage,
                            size_t&           storage_size,
                            K*                keys,
                            T*                input,
                            T*                output,
                            const T           initial_value,
                            const size_t      input_size,
                            BinaryFunction    scan_op,
                            const hipStream_t stream) ->
    typename std::enable_if<Exclusive, hipError_t>::type
{
    return hipcub::DeviceScan::ExclusiveScanByKey(temporary_storage,
                                                  storage_size,
                                                  keys,
                                                  input,
                                                  output,
                                                  scan_op,
                                                  initial_value,
                                                  static_cast<int>(input_size),
                                                  hipcub::Equality(),
                                                  stream);
}

template<bool Exclusive, class T, class K, class BinaryFunction>
auto run_device_scan_by_key(void*   temporary_storage,
                            size_t& storage_size,
                            K*      keys,
                            T*      input,
                            T*      output,
                            const T /*initial_value*/,
                            const size_t      input_size,
                            BinaryFunction    scan_op,
                            const hipStream_t stream) ->
    typename std::enable_if<!Exclusive, hipError_t>::type
{
    return hipcub::DeviceScan::InclusiveScanByKey(temporary_storage,
                                                  storage_size,
                                                  keys,
                                                  input,
                                                  output,
                                                  scan_op,
                                                  static_cast<int>(input_size),
                                                  hipcub::Equality(),
                                                  stream);
}

template<bool Exclusive, class T, class BinaryFunction, class Tag>
class scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_scan")
            .add("subalgo", "scan")
            .add("lvl", "device")
            .add("exclusive", Exclusive)
            .add("data_type", primbench::name<T>())
            .add("op", Tag::name);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        BinaryFunction scan_op{};

        std::vector<T> input         = benchmark_utils::get_random_data<T>(items, T(0), T(1000));
        T              initial_value = T(123);
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(run_device_scan<Exclusive>(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_output,
                                                 initial_value,
                                                 items,
                                                 scan_op,
                                                 stream));
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

template<bool Exclusive, class T, class BinaryFunction, class Tag>
class scan_by_key_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_scan")
            .add("subalgo", "scan_by_key")
            .add("lvl", "device")
            .add("exclusive", Exclusive)
            .add("data_type", primbench::name<T>())
            .add("op", Tag::name);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        BinaryFunction scan_op{};

        using Key                           = int;
        constexpr size_t max_segment_length = 100;

        const std::vector<Key> keys
            = benchmark_utils::get_random_segments<Key>(items,
                                                        max_segment_length,
                                                        std::random_device{}());
        const std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(1000));
        const T              initial_value = T(123);
        Key*                 d_keys;
        T*                   d_input;
        T*                   d_output;
        HIP_CHECK(hipMalloc(&d_keys, items * sizeof(Key)));
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_keys, keys.data(), items * sizeof(Key), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(run_device_scan_by_key<Exclusive>(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        d_input,
                                                        d_output,
                                                        initial_value,
                                                        items,
                                                        scan_op,
                                                        stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_keys));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

struct sum_tag
{
    static constexpr const char* name = "sum";
};

struct min_tag
{
    static constexpr const char* name = "min";
};

#define CREATE_BENCHMARK(EXCL, T, SCAN_OP, TAG)              \
    executor.queue<scan_benchmark<EXCL, T, SCAN_OP, TAG>>(); \
    executor.queue<scan_by_key_benchmark<EXCL, T, SCAN_OP, TAG>>()

#define CREATE_BENCHMARKS(SCAN_OP, TAG)                    \
    CREATE_BENCHMARK(false, int, SCAN_OP, TAG);            \
    CREATE_BENCHMARK(true, int, SCAN_OP, TAG);             \
    CREATE_BENCHMARK(false, float, SCAN_OP, TAG);          \
    CREATE_BENCHMARK(true, float, SCAN_OP, TAG);           \
    CREATE_BENCHMARK(false, double, SCAN_OP, TAG);         \
    CREATE_BENCHMARK(true, double, SCAN_OP, TAG);          \
    CREATE_BENCHMARK(false, int64_t, SCAN_OP, TAG);      \
    CREATE_BENCHMARK(true, int64_t, SCAN_OP, TAG);       \
    CREATE_BENCHMARK(false, custom_float2, SCAN_OP, TAG);  \
    CREATE_BENCHMARK(true, custom_float2, SCAN_OP, TAG);   \
    CREATE_BENCHMARK(false, custom_double2, SCAN_OP, TAG); \
    CREATE_BENCHMARK(true, custom_double2, SCAN_OP, TAG);  \
    CREATE_BENCHMARK(false, int8_t, SCAN_OP, TAG);         \
    CREATE_BENCHMARK(true, int8_t, SCAN_OP, TAG);          \
    CREATE_BENCHMARK(false, uint8_t, SCAN_OP, TAG);        \
    CREATE_BENCHMARK(true, uint8_t, SCAN_OP, TAG)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Compilation may never finish, if the compiler needs to compile too many
    // kernels, it is recommended to compile benchmarks only for 1-2 types when
    // BENCHMARK_CONFIG_TUNING is used (all other CREATE_*_BENCHMARK should be
    // commented/removed).

    // Add benchmarks
    CREATE_BENCHMARKS(hipcub::Sum, sum_tag);
    CREATE_BENCHMARKS(hipcub::Min, min_tag);

    executor.run();
}
