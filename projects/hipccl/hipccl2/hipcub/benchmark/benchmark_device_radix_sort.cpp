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

#include "benchmark_utils.hpp"

#include <memory>
#include <type_traits>

#include <hipcub/device/device_radix_sort.hpp>

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      items,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             items,
                                             0,
                                             sizeof(Key) * 8,
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      items,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_input,
                                                       d_keys_output,
                                                       items,
                                                       0,
                                                       sizeof(Key) * 8,
                                                       stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      items,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             items,
                                             benchmark_utils::custom_type_decomposer<Key>{},
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      items,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_input,
        d_keys_output,
        items,
        benchmark_utils::custom_type_decomposer<Key>{},
        stream);
}

template<class Key, bool Descending = false>
class sort_keys_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_radix_sort")
            .add("subalgo", "sort_keys")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("descending", Descending);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

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
            HIP_CHECK(invoke_sort_keys<Descending>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   items,
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

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temp_storage,
                       size_t&     temp_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      items,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              d_values_input,
                                              d_values_output,
                                              items,
                                              0,
                                              sizeof(Key) * 8,
                                              stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temp_storage,
                       size_t&     temp_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      items,
                       hipStream_t stream)
    -> std::enable_if_t<Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_input,
                                                        d_keys_output,
                                                        d_values_input,
                                                        d_values_output,
                                                        items,
                                                        0,
                                                        sizeof(Key) * 8,
                                                        stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temp_storage,
                       size_t&     temp_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      items,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              d_values_input,
                                              d_values_output,
                                              items,
                                              benchmark_utils::custom_type_decomposer<Key>{},
                                              stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temp_storage,
                       size_t&     temp_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      items,
                       hipStream_t stream)
    -> std::enable_if_t<Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_input,
        d_keys_output,
        d_values_input,
        d_values_output,
        items,
        benchmark_utils::custom_type_decomposer<Key>{},
        stream);
}

template<class Key, class Value, bool Descending = false>
class sort_pairs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_radix_sort")
            .add("subalgo", "sort_pairs")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>())
            .add("descending", Descending);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

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
            HIP_CHECK(invoke_sort_pairs<Descending>(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_keys_input,
                                                    d_keys_output,
                                                    d_values_input,
                                                    d_values_output,
                                                    items,
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
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }
};

#define QUEUE_KEY(Key)                                 \
    executor.queue<sort_keys_benchmark<Key, false>>(); \
    executor.queue<sort_keys_benchmark<Key, true>>();

#define QUEUE_PAIR(Key, Value)                                 \
    executor.queue<sort_pairs_benchmark<Key, Value, false>>(); \
    executor.queue<sort_pairs_benchmark<Key, Value, true>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv);

    QUEUE_KEY(int)
    QUEUE_KEY(int64_t)
    QUEUE_KEY(int8_t)
    QUEUE_KEY(uint8_t)
    QUEUE_KEY(short)
    QUEUE_KEY(custom_int_t)

    QUEUE_PAIR(int, float);
    QUEUE_PAIR(int, double);
    QUEUE_PAIR(int, custom_float2);
    QUEUE_PAIR(int, custom_double2);
    QUEUE_PAIR(int, custom_char_double);
    QUEUE_PAIR(int, custom_double_char);

    QUEUE_PAIR(int64_t, float);
    QUEUE_PAIR(int64_t, double);
    QUEUE_PAIR(int64_t, custom_float2);
    QUEUE_PAIR(int64_t, custom_char_double);
    QUEUE_PAIR(int64_t, custom_double_char);
    QUEUE_PAIR(int64_t, custom_double2);

    QUEUE_PAIR(int8_t, int8_t);
    QUEUE_PAIR(uint8_t, uint8_t);

    QUEUE_PAIR(custom_int_t, float);

    executor.run();
}
