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

#include "benchmark_utils.hpp"

#include <hipcub/device/device_merge_sort.hpp>
#include <hipcub/hipcub.hpp>

template<class Key>
struct CompareFunction
{
    HIPCUB_DEVICE
    inline constexpr bool operator()(const Key& a, const Key& b)
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
            .add("subalgo", "sort_keys")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>());
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        CompareFunction<Key> compare_function;

        std::vector<Key> keys_input
            = benchmark_utils::get_random_data<Key>(items,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        Key* d_keys_input;
        Key* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, items * sizeof(Key)));
        HIP_CHECK(hipMalloc(&d_keys_output, items * sizeof(Key)));
        HIP_CHECK(
            hipMemcpy(d_keys_input, keys_input.data(), items * sizeof(Key), hipMemcpyHostToDevice));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceMergeSort::SortKeysCopy(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys_input,
                                                            d_keys_output,
                                                            items,
                                                            compare_function,
                                                            stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<Key>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
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
            .add("subalgo", "sort_pairs")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>());
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        CompareFunction<Key> compare_function;

        std::vector<Key> keys_input
            = benchmark_utils::get_random_data<Key>(items,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        std::vector<Value> values_input(items);
        for(size_t i = 0; i < items; i++)
        {
            values_input[i] = Value(i);
        }

        Key* d_keys_input;
        Key* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, items * sizeof(Key)));
        HIP_CHECK(hipMalloc(&d_keys_output, items * sizeof(Key)));
        HIP_CHECK(
            hipMemcpy(d_keys_input, keys_input.data(), items * sizeof(Key), hipMemcpyHostToDevice));

        Value* d_values_input;
        Value* d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, items * sizeof(Value)));
        HIP_CHECK(hipMalloc(&d_values_output, items * sizeof(Value)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            items * sizeof(Value),
                            hipMemcpyHostToDevice));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceMergeSort::SortPairsCopy(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_input,
                                                             d_values_input,
                                                             d_keys_output,
                                                             d_values_output,
                                                             items,
                                                             compare_function,
                                                             stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

        state.set_items(items);
        state.add_writes<Key>(items);
        state.add_writes<Value>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
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
    CREATE_SORT_KEYS_BENCHMARK(int64_t);
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

    CREATE_SORT_PAIRS_BENCHMARK(int64_t, float);
    CREATE_SORT_PAIRS_BENCHMARK(int64_t, double);
    CREATE_SORT_PAIRS_BENCHMARK(int64_t, custom_float2);
    CREATE_SORT_PAIRS_BENCHMARK(int64_t, custom_char_double);
    CREATE_SORT_PAIRS_BENCHMARK(int64_t, custom_double_char);
    CREATE_SORT_PAIRS_BENCHMARK(int64_t, custom_double2);

    CREATE_SORT_PAIRS_BENCHMARK(int8_t, int8_t);
    CREATE_SORT_PAIRS_BENCHMARK(uint8_t, uint8_t);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_sort_keys_benchmarks(executor);
    add_sort_pairs_benchmarks(executor);

    executor.run();
}
