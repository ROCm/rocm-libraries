// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/hipcub.hpp>

template<class Key, size_t DesiredSegments, bool Descending, bool Stable>
class sort_keys_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_segmented_sort")
            .add("subalgo", "sort_keys")
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("desired_segments", DesiredSegments)
            .add("ascending", !Descending)
            .add("stable", Stable);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using offset_type = int;
        using sort_func   = hipError_t (*)(void*,
                                         size_t&,
                                         const Key*,
                                         Key*,
                                         int,
                                         int,
                                         offset_type*,
                                         offset_type*,
                                         hipStream_t);

        sort_func func_ascending = &hipcub::DeviceSegmentedSort::SortKeys<Key, offset_type*>;
        sort_func func_descending
            = &hipcub::DeviceSegmentedSort::SortKeysDescending<Key, offset_type*>;
        sort_func func_ascending_stable
            = &hipcub::DeviceSegmentedSort::StableSortKeys<Key, offset_type*>;
        sort_func func_descending_stable
            = &hipcub::DeviceSegmentedSort::StableSortKeysDescending<Key, offset_type*>;

        sort_func sorting = Descending ? (Stable ? func_descending_stable : func_descending)
                                       : (Stable ? func_ascending_stable : func_ascending);

        std::vector<offset_type> offsets;

        const double avg_segment_length = static_cast<double>(items) / DesiredSegments;

        std::random_device         rd;
        std::default_random_engine gen(rd());

        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        unsigned int segments_count = 0;
        size_t       offset         = 0;
        while(offset < items)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            ++segments_count;
            offset += segment_length;
        }
        offsets.push_back(items);

        std::vector<Key> keys_input
            = benchmark_utils::get_random_data<Key>(items,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

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
            HIP_CHECK(sorting(d_temp_storage,
                              temp_storage_bytes,
                              d_keys_input,
                              d_keys_output,
                              items,
                              segments_count,
                              d_offsets,
                              d_offsets + 1,
                              stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<Key>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
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
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>())
            .add("desired_segments", DesiredSegments)
            .add("ascending", !Descending)
            .add("stable", Stable);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using offset_type = int;
        using sort_func   = hipError_t (*)(void*,
                                         size_t&,
                                         const Key*,
                                         Key*,
                                         const Value*,
                                         Value*,
                                         int,
                                         int,
                                         offset_type*,
                                         offset_type*,
                                         hipStream_t);

        sort_func func_ascending
            = &hipcub::DeviceSegmentedSort::SortPairs<Key, Value, offset_type*>;
        sort_func func_descending
            = &hipcub::DeviceSegmentedSort::SortPairsDescending<Key, Value, offset_type*>;
        sort_func func_ascending_stable
            = &hipcub::DeviceSegmentedSort::StableSortPairs<Key, Value, offset_type*>;
        sort_func func_descending_stable
            = &hipcub::DeviceSegmentedSort::StableSortPairsDescending<Key, Value, offset_type*>;

        sort_func sorting = Descending ? (Stable ? func_descending_stable : func_descending)
                                       : (Stable ? func_ascending_stable : func_ascending);

        std::vector<offset_type> offsets;

        const double avg_segment_length = static_cast<double>(items) / DesiredSegments;

        std::random_device         rd;
        std::default_random_engine gen(rd());

        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        unsigned int segments_count = 0;
        size_t       offset         = 0;
        while(offset < items)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            ++segments_count;
            offset += segment_length;
        }
        offsets.push_back(items);

        std::vector<Key> keys_input
            = benchmark_utils::get_random_data<Key>(items,
                                                    benchmark_utils::generate_limits<Key>::min(),
                                                    benchmark_utils::generate_limits<Key>::max());

        std::vector<Value> values_input(items);
        std::iota(values_input.begin(), values_input.end(), 0);

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

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
            HIP_CHECK(sorting(d_temp_storage,
                              temp_storage_bytes,
                              d_keys_input,
                              d_keys_output,
                              d_values_input,
                              d_values_output,
                              items,
                              segments_count,
                              d_offsets,
                              d_offsets + 1,
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
    BENCHMARK_PAIR_TYPE(int64_t, double);
    BENCHMARK_PAIR_TYPE(int8_t, int8_t);
    BENCHMARK_PAIR_TYPE(uint8_t, uint8_t);
    BENCHMARK_PAIR_TYPE(int, custom_float2);
    BENCHMARK_PAIR_TYPE(int64_t, custom_double2);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch    = 1000;
    settings.batch_window_size       = 3;
    settings.noise_tolerance_percent = 3;

    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    add_sort_keys_benchmarks(executor);
    add_sort_pairs_benchmarks(executor);

    executor.run();
}
