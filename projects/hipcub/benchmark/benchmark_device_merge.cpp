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

// HIP API
#include <hipcub/device/device_merge.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class key_type>
struct CompareFunction
{
    HIPCUB_HOST_DEVICE
    inline constexpr bool
        operator()(const key_type& a, const key_type& b)
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
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type = Key;

        CompareFunction<key_type> compare_function;

        const size_t size1 = size / 2;
        const size_t size2 = size - size1;

        std::vector<key_type> keys_input1 = benchmark_utils::get_random_data<key_type>(
            size1,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        std::vector<key_type> keys_input2 = benchmark_utils::get_random_data<key_type>(
            size2,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

        key_type* d_keys_input1;
        HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            size1 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys_input2;
        HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            size2 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input1,
                                                 size1,
                                                 d_keys_input2,
                                                 size2,
                                                 d_keys_output,
                                                 compare_function,
                                                 stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<key_type>(size);

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_keys_input1,
                                                         size1,
                                                         d_keys_input2,
                                                         size2,
                                                         d_keys_output,
                                                         compare_function,
                                                         stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
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
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type   = Key;
        using value_type = Value;

        CompareFunction<key_type> compare_function;

        const size_t size1 = size / 2;
        const size_t size2 = size - size1;

        std::vector<key_type> keys_input1 = benchmark_utils::get_random_data<key_type>(
            size1,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());
        std::vector<key_type> keys_input2 = benchmark_utils::get_random_data<key_type>(
            size2,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        std::sort(keys_input1.begin(), keys_input1.end(), compare_function);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_function);

        key_type* d_keys_input1;
        HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            size1 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys_input2;
        HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            size2 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));

        std::vector<value_type> values_input1(size1);
        std::iota(values_input1.begin(), values_input1.end(), 0);
        value_type* d_values_input1;
        HIP_CHECK(hipMalloc(&d_values_input1, size1 * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input1,
                            values_input1.data(),
                            size1 * sizeof(value_type),
                            hipMemcpyHostToDevice));

        std::vector<value_type> values_input2(size2);
        std::iota(values_input2.begin(), values_input2.end(), size1);
        value_type* d_values_input2;
        HIP_CHECK(hipMalloc(&d_values_input2, size2 * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input2,
                            values_input2.data(),
                            size2 * sizeof(value_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_output;
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input1,
                                                  d_values_input1,
                                                  size1,
                                                  d_keys_input2,
                                                  d_values_input2,
                                                  size2,
                                                  d_keys_output,
                                                  d_values_output,
                                                  compare_function,
                                                  stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<char>(size * (sizeof(key_type) + sizeof(value_type)));

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                                          temporary_storage_bytes,
                                                          d_keys_input1,
                                                          d_values_input1,
                                                          size1,
                                                          d_keys_input2,
                                                          d_values_input2,
                                                          size2,
                                                          d_keys_output,
                                                          d_values_output,
                                                          compare_function,
                                                          stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
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
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_MERGE_KEYS_BENCHMARK(int);
    CREATE_MERGE_KEYS_BENCHMARK(long long);
    CREATE_MERGE_KEYS_BENCHMARK(int8_t);
    CREATE_MERGE_KEYS_BENCHMARK(uint8_t);
    CREATE_MERGE_KEYS_BENCHMARK(short);
    CREATE_MERGE_KEYS_BENCHMARK(double);
    CREATE_MERGE_KEYS_BENCHMARK(float);
    CREATE_MERGE_KEYS_BENCHMARK(custom_float2);
    CREATE_MERGE_KEYS_BENCHMARK(custom_double2);

    CREATE_MERGE_PAIRS_BENCHMARK(int, int);
    CREATE_MERGE_PAIRS_BENCHMARK(long long, long long);
    CREATE_MERGE_PAIRS_BENCHMARK(int8_t, int8_t);
    CREATE_MERGE_PAIRS_BENCHMARK(uint8_t, uint8_t);
    CREATE_MERGE_PAIRS_BENCHMARK(short, short);
    CREATE_MERGE_PAIRS_BENCHMARK(custom_char_double, custom_char_double);
    CREATE_MERGE_PAIRS_BENCHMARK(int, custom_double_char);
    CREATE_MERGE_PAIRS_BENCHMARK(custom_double2, custom_double2);

    executor.run();
}
