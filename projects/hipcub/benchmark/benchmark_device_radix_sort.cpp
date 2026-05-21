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

#include "common_benchmark_header.hpp"

#include <memory>
#include <type_traits>

// HIP API
#include <hipcub/device/device_radix_sort.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class Key>
std::vector<Key> generate_keys(size_t size)
{
    using key_type = Key;

    return benchmark_utils::get_random_data<key_type>(
        size,
        benchmark_utils::generate_limits<key_type>::min(),
        benchmark_utils::generate_limits<key_type>::max());
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             size,
                                             0,
                                             sizeof(Key) * 8,
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_input,
                                                       d_keys_output,
                                                       size,
                                                       0,
                                                       sizeof(Key) * 8,
                                                       stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             size,
                                             benchmark_utils::custom_type_decomposer<Key>{},
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temp_storage,
                      size_t&     temp_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_input,
        d_keys_output,
        size,
        benchmark_utils::custom_type_decomposer<Key>{},
        stream);
}

template<class Key, bool Descending = false>
class sort_keys_benchmark : public primbench::benchmark_interface
{
public:
    sort_keys_benchmark(std::shared_ptr<std::vector<Key>> keys_input) : m_keys_input(keys_input) {}

private:
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
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type = Key;
        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            m_keys_input->data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_sort_keys<Descending>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_keys_input,
                                               d_keys_output,
                                               size,
                                               stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<key_type>(size);

        state.run(
            [&]
            {
                HIP_CHECK(invoke_sort_keys<Descending>(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_keys_input,
                                                       d_keys_output,
                                                       size,
                                                       stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

    std::shared_ptr<std::vector<Key>> m_keys_input;
};

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temp_storage,
                       size_t&     temp_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      size,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              d_values_input,
                                              d_values_output,
                                              size,
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
                       size_t      size,
                       hipStream_t stream)
    -> std::enable_if_t<Descending && !benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_input,
                                                        d_keys_output,
                                                        d_values_input,
                                                        d_values_output,
                                                        size,
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
                       size_t      size,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && benchmark_utils::is_custom_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              d_values_input,
                                              d_values_output,
                                              size,
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
                       size_t      size,
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
        size,
        benchmark_utils::custom_type_decomposer<Key>{},
        stream);
}

template<class Key, class Value, bool Descending = false>
class sort_pairs_benchmark : public primbench::benchmark_interface
{
public:
    sort_pairs_benchmark(std::shared_ptr<std::vector<Key>> keys_input) : m_keys_input(keys_input) {}

private:
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
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using key_type   = Key;
        using value_type = Value;
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
                            m_keys_input->data(),
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
        HIP_CHECK(invoke_sort_pairs<Descending>(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                d_values_input,
                                                d_values_output,
                                                size,
                                                stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<std::byte>(size * (sizeof(key_type) + sizeof(value_type)));

        state.run(
            [&]
            {
                HIP_CHECK(invoke_sort_pairs<Descending>(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        d_keys_output,
                                                        d_values_input,
                                                        d_values_output,
                                                        size,
                                                        stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    const std::shared_ptr<std::vector<Key>> m_keys_input;
};

#define CREATE_SORT_KEYS_BENCHMARK(Key)                                                 \
    {                                                                                   \
        auto keys_input = std::make_shared<std::vector<Key>>(generate_keys<Key>(size)); \
        executor.queue<sort_keys_benchmark<Key, false>>(keys_input);                    \
        executor.queue<sort_keys_benchmark<Key, true>>(keys_input);                     \
    }

#define CREATE_SORT_PAIRS_BENCHMARK(Key, Value)                                         \
    {                                                                                   \
        auto keys_input = std::make_shared<std::vector<Key>>(generate_keys<Key>(size)); \
        executor.queue<sort_pairs_benchmark<Key, Value, false>>(keys_input);            \
        executor.queue<sort_pairs_benchmark<Key, Value, true>>(keys_input);             \
    }

void add_sort_keys_benchmarks(primbench::executor& executor, size_t size)
{
    CREATE_SORT_KEYS_BENCHMARK(int);
    CREATE_SORT_KEYS_BENCHMARK(long long);
    CREATE_SORT_KEYS_BENCHMARK(int8_t);
    CREATE_SORT_KEYS_BENCHMARK(uint8_t);
    CREATE_SORT_KEYS_BENCHMARK(short);
    CREATE_SORT_KEYS_BENCHMARK(custom_int_t);
}

void add_sort_pairs_benchmarks(primbench::executor& executor, size_t size)
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

    CREATE_SORT_PAIRS_BENCHMARK(custom_int_t, float);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv);

    const size_t size = executor.get("size", DEFAULT_N, "the size of the input array");

    add_sort_keys_benchmarks(executor, size);
    add_sort_pairs_benchmarks(executor, size);

    executor.run();
}
