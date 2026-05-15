// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/device/device_merge_sort.hpp>
#include <hipcub/hipcub.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class key_type>
struct CompareFunction
{
    HIPCUB_DEVICE
    inline constexpr bool
        operator()(const key_type& a, const key_type& b)
    {
        return a < b;
    }
};

template<class Key>
class sort_keys_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_merge_sort")
            .add("subalgo", "keys")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>());
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type = Key;

        CompareFunction<key_type> compare_function;

        std::vector<key_type> keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(hipcub::DeviceMergeSort::SortKeysCopy(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        d_keys_output,
                                                        size,
                                                        compare_function,
                                                        stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<key_type>(size);

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceMergeSort::SortKeysCopy(d_temporary_storage,
                                                                temporary_storage_bytes,
                                                                d_keys_input,
                                                                d_keys_output,
                                                                size,
                                                                compare_function,
                                                                stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }
};

template<class Key, class Value>
class sort_pairs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_merge_sort")
            .add("subalgo", "pairs")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>());
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type   = Key;
        using value_type = Value;

        CompareFunction<key_type> compare_function;

        std::vector<key_type> keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        std::vector<value_type> values_input(size);
        for(size_t i = 0; i < size; i++)
        {
            values_input[i] = value_type(i);
        }

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(hipcub::DeviceMergeSort::SortPairsCopy(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_keys_input,
                                                         d_values_input,
                                                         d_keys_output,
                                                         d_values_output,
                                                         size,
                                                         compare_function,
                                                         stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        state.set_items(size);
        state.add_writes<char>(size * (sizeof(key_type) + sizeof(value_type)));

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceMergeSort::SortPairsCopy(d_temporary_storage,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input,
                                                                 d_values_input,
                                                                 d_keys_output,
                                                                 d_values_output,
                                                                 size,
                                                                 compare_function,
                                                                 stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }
};

#define CREATE_SORT_KEYS_BENCHMARK(T) executor.queue<sort_keys_benchmark<T>>()

#define CREATE_SORT_PAIRS_BENCHMARK(T, V) executor.queue<sort_pairs_benchmark<T, V>>()

void add_sort_keys_benchmarks(primbench::executor& executor)
{
    CREATE_SORT_KEYS_BENCHMARK(int);
    CREATE_SORT_KEYS_BENCHMARK(long long);
    CREATE_SORT_KEYS_BENCHMARK(int8_t);
    CREATE_SORT_KEYS_BENCHMARK(uint8_t);
    CREATE_SORT_KEYS_BENCHMARK(short);
}

void add_sort_pairs_benchmarks(primbench::executor& executor)
{
    CREATE_SORT_PAIRS_BENCHMARK(int, float);
    CREATE_SORT_PAIRS_BENCHMARK(int, double);
    CREATE_SORT_PAIRS_BENCHMARK(int, custom_float2);
    CREATE_SORT_PAIRS_BENCHMARK(int, custom_double2);
    CREATE_SORT_PAIRS_BENCHMARK(int, custom_char_double);
    CREATE_SORT_PAIRS_BENCHMARK(int, custom_double_char);

    CREATE_SORT_PAIRS_BENCHMARK(long long, float);
    CREATE_SORT_PAIRS_BENCHMARK(long long, double);
    CREATE_SORT_PAIRS_BENCHMARK(long long, custom_float2);
    CREATE_SORT_PAIRS_BENCHMARK(long long, custom_char_double);
    CREATE_SORT_PAIRS_BENCHMARK(long long, custom_double_char);
    CREATE_SORT_PAIRS_BENCHMARK(long long, custom_double2);

    CREATE_SORT_PAIRS_BENCHMARK(int8_t, int8_t);
    CREATE_SORT_PAIRS_BENCHMARK(uint8_t, uint8_t);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.min_gpu_ms_per_batch = 100;
    settings.size                 = DEFAULT_N;

    primbench::executor executor(argc, argv, settings);

    add_sort_keys_benchmarks(executor);
    add_sort_pairs_benchmarks(executor);

    executor.run();
}
