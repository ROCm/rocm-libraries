// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/device/device_merge.hpp>

template<class Key>
struct CompareFunction
{
    HIPCUB_HOST_DEVICE
    inline constexpr bool operator()(const Key& a, const Key& b)
    {
        return a < b;
    }
};

template<class Key>
class merge_keys_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_merge")
            .add("subalgo", "merge_keys")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>());
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        CompareFunction<Key> compare_function;

        const size_t items1 = items / 2;
        const size_t items2 = items - items1;

        std::vector<Key> keys_input1
            = benchmark_utils::get_random_data<Key>(items1,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        std::vector<Key> keys_input2
            = benchmark_utils::get_random_data<Key>(items2,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

        Key* d_keys_input1;
        HIP_CHECK(hipMalloc(&d_keys_input1, items1 * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            items1 * sizeof(Key),
                            hipMemcpyHostToDevice));

        Key* d_keys_input2;
        HIP_CHECK(hipMalloc(&d_keys_input2, items2 * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            items2 * sizeof(Key),
                            hipMemcpyHostToDevice));

        Key* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_output, items * sizeof(Key)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_keys_input1,
                                                     items1,
                                                     d_keys_input2,
                                                     items2,
                                                     d_keys_output,
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
        HIP_CHECK(hipFree(d_keys_input1));
        HIP_CHECK(hipFree(d_keys_input2));
        HIP_CHECK(hipFree(d_keys_output));
    }
};

template<class Key, class Value>
class merge_pairs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_merge")
            .add("subalgo", "merge_pairs")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>());
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        CompareFunction<Key> compare_function;

        const size_t items1 = items / 2;
        const size_t items2 = items - items1;

        std::vector<Key> keys_input1
            = benchmark_utils::get_random_data<Key>(items1,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());
        std::vector<Key> keys_input2
            = benchmark_utils::get_random_data<Key>(items2,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

        Key* d_keys_input1;
        HIP_CHECK(hipMalloc(&d_keys_input1, items1 * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            items1 * sizeof(Key),
                            hipMemcpyHostToDevice));

        Key* d_keys_input2;
        HIP_CHECK(hipMalloc(&d_keys_input2, items2 * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            items2 * sizeof(Key),
                            hipMemcpyHostToDevice));

        Key* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_output, items * sizeof(Key)));

        std::vector<Value> values_input1(items1);
        std::iota(values_input1.begin(), values_input1.end(), 0);
        Value* d_values_input1;
        HIP_CHECK(hipMalloc(&d_values_input1, items1 * sizeof(Value)));
        HIP_CHECK(hipMemcpy(d_values_input1,
                            values_input1.data(),
                            items1 * sizeof(Value),
                            hipMemcpyHostToDevice));

        std::vector<Value> values_input2(items2);
        std::iota(values_input2.begin(), values_input2.end(), items1);
        Value* d_values_input2;
        HIP_CHECK(hipMalloc(&d_values_input2, items2 * sizeof(Value)));
        HIP_CHECK(hipMemcpy(d_values_input2,
                            values_input2.data(),
                            items2 * sizeof(Value),
                            hipMemcpyHostToDevice));

        Value* d_values_output;
        HIP_CHECK(hipMalloc(&d_values_output, items * sizeof(Value)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_keys_input1,
                                                      d_values_input1,
                                                      items1,
                                                      d_keys_input2,
                                                      d_values_input2,
                                                      items2,
                                                      d_keys_output,
                                                      d_values_output,
                                                      compare_function,
                                                      stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<Key>(items);
        state.add_writes<Value>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_keys_input1));
        HIP_CHECK(hipFree(d_keys_input2));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input1));
        HIP_CHECK(hipFree(d_values_input2));
        HIP_CHECK(hipFree(d_values_output));
    }
};

#define CREATE_MERGE_KEYS_BENCHMARK(T) executor.queue<merge_keys_benchmark<T>>()

#define CREATE_MERGE_PAIRS_BENCHMARK(T, V) executor.queue<merge_pairs_benchmark<T, V>>()

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_MERGE_KEYS_BENCHMARK(int);
    CREATE_MERGE_KEYS_BENCHMARK(int64_t);
    CREATE_MERGE_KEYS_BENCHMARK(int8_t);
    CREATE_MERGE_KEYS_BENCHMARK(uint8_t);
    CREATE_MERGE_KEYS_BENCHMARK(short);
    CREATE_MERGE_KEYS_BENCHMARK(double);
    CREATE_MERGE_KEYS_BENCHMARK(float);
    CREATE_MERGE_KEYS_BENCHMARK(custom_float2);
    CREATE_MERGE_KEYS_BENCHMARK(custom_double2);

    CREATE_MERGE_PAIRS_BENCHMARK(int, int);
    CREATE_MERGE_PAIRS_BENCHMARK(int64_t, int64_t);
    CREATE_MERGE_PAIRS_BENCHMARK(int8_t, int8_t);
    CREATE_MERGE_PAIRS_BENCHMARK(uint8_t, uint8_t);
    CREATE_MERGE_PAIRS_BENCHMARK(short, short);
    CREATE_MERGE_PAIRS_BENCHMARK(custom_char_double, custom_char_double);
    CREATE_MERGE_PAIRS_BENCHMARK(int, custom_double_char);
    CREATE_MERGE_PAIRS_BENCHMARK(custom_double2, custom_double2);

    executor.run();
}
