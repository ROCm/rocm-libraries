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

// HIP API
#include <hipcub/hipcub.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class Key, size_t DesiredSegments, bool Descending, bool Stable>
class sort_keys_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_segmented_sort")
            .add("subalgo", "sort_keys")
            .add("key_type", primbench::name<Key>())
            .add("desired_segments", DesiredSegments)
            .add("ascending", !Descending)
            .add("stable", Stable);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using offset_type = int;
        using key_type    = Key;
        using sort_func   = hipError_t (*)(void*,
                                         size_t&,
                                         const key_type*,
                                         key_type*,
                                         int,
                                         int,
                                         offset_type*,
                                         offset_type*,
                                         hipStream_t);

        sort_func func_ascending = &hipcub::DeviceSegmentedSort::SortKeys<key_type, offset_type*>;
        sort_func func_descending
            = &hipcub::DeviceSegmentedSort::SortKeysDescending<key_type, offset_type*>;
        sort_func func_ascending_stable
            = &hipcub::DeviceSegmentedSort::StableSortKeys<key_type, offset_type*>;
        sort_func func_descending_stable
            = &hipcub::DeviceSegmentedSort::StableSortKeysDescending<key_type, offset_type*>;

        sort_func sorting = Descending ? (Stable ? func_descending_stable : func_descending)
                                       : (Stable ? func_ascending_stable : func_ascending);

        std::vector<offset_type> offsets;

        const double avg_segment_length = static_cast<double>(size) / DesiredSegments;

        std::random_device         rd;
        std::default_random_engine gen(rd());

        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        unsigned int segments_count = 0;
        size_t       offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            ++segments_count;
            offset += segment_length;
        }
        offsets.push_back(size);

        std::vector<key_type> keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

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
        HIP_CHECK(sorting(d_temporary_storage,
                          temporary_storage_bytes,
                          d_keys_input,
                          d_keys_output,
                          size,
                          segments_count,
                          d_offsets,
                          d_offsets + 1,
                          stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<key_type>(size);

        state.run(
            [&]
            {
                HIP_CHECK(sorting(d_temporary_storage,
                                  temporary_storage_bytes,
                                  d_keys_input,
                                  d_keys_output,
                                  size,
                                  segments_count,
                                  d_offsets,
                                  d_offsets + 1,
                                  stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }
};

template<class Key, class Value, size_t DesiredSegments, bool Descending, bool Stable>
class sort_pairs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_segmented_sort")
            .add("subalgo", "sort_pairs")
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("desired_segments", DesiredSegments)
            .add("ascending", !Descending)
            .add("stable", Stable);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        using offset_type = int;
        using key_type    = Key;
        using value_type  = Value;
        using sort_func   = hipError_t (*)(void*,
                                         size_t&,
                                         const key_type*,
                                         key_type*,
                                         const value_type*,
                                         value_type*,
                                         int,
                                         int,
                                         offset_type*,
                                         offset_type*,
                                         hipStream_t);

        sort_func func_ascending
            = &hipcub::DeviceSegmentedSort::SortPairs<key_type, value_type, offset_type*>;
        sort_func func_descending
            = &hipcub::DeviceSegmentedSort::SortPairsDescending<key_type, value_type, offset_type*>;
        sort_func func_ascending_stable
            = &hipcub::DeviceSegmentedSort::StableSortPairs<key_type, value_type, offset_type*>;
        sort_func func_descending_stable
            = &hipcub::DeviceSegmentedSort::
                  StableSortPairsDescending<key_type, value_type, offset_type*>;

        sort_func sorting = Descending ? (Stable ? func_descending_stable : func_descending)
                                       : (Stable ? func_ascending_stable : func_ascending);

        std::vector<offset_type> offsets;

        const double avg_segment_length = static_cast<double>(size) / DesiredSegments;

        std::random_device         rd;
        std::default_random_engine gen(rd());

        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        unsigned int segments_count = 0;
        size_t       offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            ++segments_count;
            offset += segment_length;
        }
        offsets.push_back(size);

        std::vector<key_type> keys_input = benchmark_utils::get_random_data<key_type>(
            size,
            benchmark_utils::generate_limits<key_type>::min(),
            benchmark_utils::generate_limits<key_type>::max());

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

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
        HIP_CHECK(sorting(d_temporary_storage,
                          temporary_storage_bytes,
                          d_keys_input,
                          d_keys_output,
                          d_values_input,
                          d_values_output,
                          size,
                          segments_count,
                          d_offsets,
                          d_offsets + 1,
                          stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(size);
        state.add_writes<std::byte>(size * (sizeof(key_type) + sizeof(value_type)));

        state.run(
            [&]
            {
                HIP_CHECK(sorting(d_temporary_storage,
                                  temporary_storage_bytes,
                                  d_keys_input,
                                  d_keys_output,
                                  d_values_input,
                                  d_values_output,
                                  size,
                                  segments_count,
                                  d_offsets,
                                  d_offsets + 1,
                                  stream));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }
};

#define CREATE_SORT_KEYS_BENCHMARK(Key, SEGMENTS)                       \
    executor.queue<sort_keys_benchmark<Key, SEGMENTS, false, false>>(); \
    executor.queue<sort_keys_benchmark<Key, SEGMENTS, true, false>>();  \
    executor.queue<sort_keys_benchmark<Key, SEGMENTS, false, true>>();  \
    executor.queue<sort_keys_benchmark<Key, SEGMENTS, true, true>>();

#define BENCHMARK_KEY_TYPE(type)            \
    CREATE_SORT_KEYS_BENCHMARK(type, 10);   \
    CREATE_SORT_KEYS_BENCHMARK(type, 100);  \
    CREATE_SORT_KEYS_BENCHMARK(type, 1000); \
    CREATE_SORT_KEYS_BENCHMARK(type, 10000)

void add_sort_keys_benchmarks(primbench::executor& executor)
{
    BENCHMARK_KEY_TYPE(float);
    BENCHMARK_KEY_TYPE(double);
    BENCHMARK_KEY_TYPE(int8_t);
    BENCHMARK_KEY_TYPE(uint8_t);
    BENCHMARK_KEY_TYPE(int);
}

#define CREATE_SORT_PAIRS_BENCHMARK(Key, Value, SEGMENTS)                       \
    executor.queue<sort_pairs_benchmark<Key, Value, SEGMENTS, false, false>>(); \
    executor.queue<sort_pairs_benchmark<Key, Value, SEGMENTS, true, false>>();  \
    executor.queue<sort_pairs_benchmark<Key, Value, SEGMENTS, false, true>>();  \
    executor.queue<sort_pairs_benchmark<Key, Value, SEGMENTS, true, true>>()

#define BENCHMARK_PAIR_TYPE(type, value)           \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 10);  \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 100); \
    CREATE_SORT_PAIRS_BENCHMARK(type, value, 10000)

void add_sort_pairs_benchmarks(primbench::executor& executor)
{
    BENCHMARK_PAIR_TYPE(int, float);
    BENCHMARK_PAIR_TYPE(long long, double);
    BENCHMARK_PAIR_TYPE(int8_t, int8_t);
    BENCHMARK_PAIR_TYPE(uint8_t, uint8_t);
    BENCHMARK_PAIR_TYPE(int, custom_float2);
    BENCHMARK_PAIR_TYPE(long long, custom_double2);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_sort_keys_benchmarks(executor);
    add_sort_pairs_benchmarks(executor);

    executor.run();
}
