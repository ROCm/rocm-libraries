// MIT License
//
// Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// CUB's implementation of single_pass_scan_operators has maybe uninitialized
// parameters, disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVIDIA__
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "benchmark_utils.hpp"

// HIP API
#include <hipcub/device/device_reduce.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class Key, class Value, size_t MaxLength, class BinaryFunction>
class reduce_by_key_benchmark : public primbench::benchmark_interface
{
    static constexpr bool is_sum = std::is_same_v<BinaryFunction, benchmark_utils::plus>;
    static constexpr bool is_min = std::is_same_v<BinaryFunction, benchmark_utils::minimum>;
    static_assert(is_sum || is_min, "unknown binary function");

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_reduce_by_key")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>())
            .add("random_number_range", "[1, " + std::to_string(MaxLength) + "]")
            .add("reduce_op", is_sum ? "sum" : (is_min ? "min" : "unknown"));
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type   = Key;
        using value_type = Value;

        // Generate data
        std::vector<key_type> keys_input(size);

        unsigned int        unique_count = 0;
        std::vector<size_t> key_counts
            = benchmark_utils::get_random_data<size_t>(100000, 1, MaxLength);
        size_t offset = 0;
        while(offset < size)
        {
            const size_t key_count = key_counts[unique_count % key_counts.size()];
            const size_t end       = _HIPCUB_STD::min(size, offset + key_count);
            for(size_t i = offset; i < end; i++)
            {
                keys_input[i] = unique_count;
            }

            unique_count++;
            offset += key_count;
        }

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        key_type* d_keys_input;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        key_type*     d_unique_output;
        value_type*   d_aggregates_output;
        unsigned int* d_unique_count_output;
        HIP_CHECK(hipMalloc(&d_unique_output, unique_count * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_aggregates_output, unique_count * sizeof(value_type)));
        HIP_CHECK(hipMalloc(&d_unique_count_output, sizeof(unsigned int)));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        BinaryFunction reduce_op{};

        HIP_CHECK(hipcub::DeviceReduce::ReduceByKey(nullptr,
                                                    temporary_storage_bytes,
                                                    d_keys_input,
                                                    d_unique_output,
                                                    d_values_input,
                                                    d_aggregates_output,
                                                    d_unique_count_output,
                                                    reduce_op,
                                                    size,
                                                    stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<std::byte>(size * (sizeof(key_type) + sizeof(value_type)));

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceReduce::ReduceByKey(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys_input,
                                                            d_unique_output,
                                                            d_values_input,
                                                            d_aggregates_output,
                                                            d_unique_count_output,
                                                            reduce_op,
                                                            size,
                                                            stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_aggregates_output));
        HIP_CHECK(hipFree(d_unique_count_output));
    }
};

#define CREATE_BENCHMARK(Key, Value, REDUCE_OP) \
    executor.queue<reduce_by_key_benchmark<Key, Value, MaxLength, REDUCE_OP>>()

#define CREATE_BENCHMARKS(REDUCE_OP)                  \
    CREATE_BENCHMARK(int, float, REDUCE_OP);          \
    CREATE_BENCHMARK(int, double, REDUCE_OP);         \
    CREATE_BENCHMARK(int, custom_double2, REDUCE_OP); \
    CREATE_BENCHMARK(int8_t, int8_t, REDUCE_OP);      \
    CREATE_BENCHMARK(long long, float, REDUCE_OP);    \
    CREATE_BENCHMARK(long long, double, REDUCE_OP)

template<size_t MaxLength>
void add_benchmarks(primbench::executor& executor)
{
    CREATE_BENCHMARKS(benchmark_utils::plus);
    CREATE_BENCHMARK(long long, custom_double2, benchmark_utils::plus);
    CREATE_BENCHMARKS(benchmark_utils::minimum);
#ifdef HIPCUB_ROCPRIM_API
    CREATE_BENCHMARK(long long, custom_double2, benchmark_utils::minimum);
#endif
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<1000>(executor);
    add_benchmarks<10>(executor);

    executor.run();
}
