// MIT License
//
// Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/device/device_partition.hpp>

#include <chrono>
#include <vector>

namespace
{
template<typename T>
struct LessOp
{
    HIPCUB_HOST_DEVICE LessOp(const T& pivot) : pivot_{pivot} {}

    HIPCUB_HOST_DEVICE
    bool operator()(const T& val) const
    {
        return val < pivot_;
    }

private:
    T pivot_;
};
} // namespace

template<typename T, typename F, int Threshold>
class flagged_benchmark : public primbench::benchmark_interface
{
private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_partition")
            .add("subalgo", "flagged")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("flag_type", primbench::name<F>())
            .add("split_threshold", std::to_string(Threshold) + "%");
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        const auto select_op = LessOp<T>{T(Threshold)};
        const auto input
            = benchmark_utils::get_random_data<T>(items, static_cast<T>(0), static_cast<T>(100));

        std::vector<F> flags(items);
        for(unsigned int i = 0; i < items; i++)
        {
            flags[i] = static_cast<F>(select_op(input[i]));
        }

        T*            d_input               = nullptr;
        F*            d_flags               = nullptr;
        T*            d_output              = nullptr;
        unsigned int* d_num_selected_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_flags, input.size() * sizeof(F)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_num_selected_output, sizeof(unsigned int)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DevicePartition::Flagged(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_input,
                                                       d_flags,
                                                       d_output,
                                                       d_num_selected_output,
                                                       static_cast<int>(input.size()),
                                                       stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(input.size());
        state.add_writes<T>(input.size());

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_num_selected_output));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_flags));
        HIP_CHECK(hipFree(d_input));
    }
};

template<typename T, int Threshold>
class predicate_benchmark : public primbench::benchmark_interface
{
private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_partition")
            .add("subalgo", "predicate")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("split_threshold", std::to_string(Threshold) + "%");
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        const auto input
            = benchmark_utils::get_random_data<T>(items, static_cast<T>(0), static_cast<T>(100));

        T*            d_input               = nullptr;
        T*            d_output              = nullptr;
        unsigned int* d_num_selected_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_num_selected_output, sizeof(unsigned int)));

        const auto select_op = LessOp<T>{T(Threshold)};

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DevicePartition::If(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_input,
                                                  d_output,
                                                  d_num_selected_output,
                                                  static_cast<int>(input.size()),
                                                  select_op,
                                                  stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(input.size());
        state.add_writes<T>(input.size());

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_num_selected_output));
    }
};

template<typename T, int SmallThreshold, int LargeThreshold>
class threeway_benchmark : public primbench::benchmark_interface
{
private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_partition")
            .add("subalgo", "three_way")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("small_threshold", std::to_string(SmallThreshold) + "%")
            .add("large_threshold", std::to_string(LargeThreshold) + "%");
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        const auto input
            = benchmark_utils::get_random_data<T>(items, static_cast<T>(0), static_cast<T>(100));

        T*            d_input               = nullptr;
        T*            d_first_output        = nullptr;
        T*            d_second_output       = nullptr;
        T*            d_unselected_output   = nullptr;
        unsigned int* d_num_selected_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_first_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_second_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_unselected_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_num_selected_output, 2 * sizeof(unsigned int)));

        const auto select_first_part_op  = LessOp<T>{T(SmallThreshold)};
        const auto select_second_part_op = LessOp<T>{T(LargeThreshold)};

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DevicePartition::If(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_input,
                                                  d_first_output,
                                                  d_second_output,
                                                  d_unselected_output,
                                                  d_num_selected_output,
                                                  static_cast<int>(input.size()),
                                                  select_first_part_op,
                                                  select_second_part_op,
                                                  stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(input.size());
        state.add_writes<T>(input.size());

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_first_output));
        HIP_CHECK(hipFree(d_second_output));
        HIP_CHECK(hipFree(d_unselected_output));
        HIP_CHECK(hipFree(d_num_selected_output));
    }
};

#define CREATE_BENCHMARK_FLAGGED(T, T_FLAG, SPLIT_T) \
    executor.queue<flagged_benchmark<T, T_FLAG, SPLIT_T>>()

#define CREATE_BENCHMARK_PREDICATE(T, SPLIT_T) executor.queue<predicate_benchmark<T, SPLIT_T>>()

#define CREATE_BENCHMARK_THREEWAY(T, SMALL_T, LARGE_T) \
    executor.queue<threeway_benchmark<T, SMALL_T, LARGE_T>>()

#define BENCHMARK_FLAGGED_TYPE(type, flag_type)    \
    CREATE_BENCHMARK_FLAGGED(type, flag_type, 33); \
    CREATE_BENCHMARK_FLAGGED(type, flag_type, 50); \
    CREATE_BENCHMARK_FLAGGED(type, flag_type, 60); \
    CREATE_BENCHMARK_FLAGGED(type, flag_type, 90)

#define BENCHMARK_PREDICATE_TYPE(type)    \
    CREATE_BENCHMARK_PREDICATE(type, 33); \
    CREATE_BENCHMARK_PREDICATE(type, 50); \
    CREATE_BENCHMARK_PREDICATE(type, 60); \
    CREATE_BENCHMARK_PREDICATE(type, 90)

#define BENCHMARK_THREEWAY_TYPE(type)        \
    CREATE_BENCHMARK_THREEWAY(type, 33, 66); \
    CREATE_BENCHMARK_THREEWAY(type, 10, 66); \
    CREATE_BENCHMARK_THREEWAY(type, 50, 60); \
    CREATE_BENCHMARK_THREEWAY(type, 50, 90)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    BENCHMARK_FLAGGED_TYPE(int8_t, unsigned char);
    BENCHMARK_FLAGGED_TYPE(int, unsigned char);
    BENCHMARK_FLAGGED_TYPE(float, unsigned char);
    BENCHMARK_FLAGGED_TYPE(int64_t, uint8_t);
    BENCHMARK_FLAGGED_TYPE(double, int8_t);
    BENCHMARK_FLAGGED_TYPE(custom_float2, int8_t);
    BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char);

    BENCHMARK_PREDICATE_TYPE(int8_t);
    BENCHMARK_PREDICATE_TYPE(int);
    BENCHMARK_PREDICATE_TYPE(float);
    BENCHMARK_PREDICATE_TYPE(int64_t);
    BENCHMARK_PREDICATE_TYPE(double);
    BENCHMARK_PREDICATE_TYPE(custom_float2);
    BENCHMARK_PREDICATE_TYPE(custom_double2);

    BENCHMARK_THREEWAY_TYPE(int8_t);
    BENCHMARK_THREEWAY_TYPE(int);
    BENCHMARK_THREEWAY_TYPE(float);
    BENCHMARK_THREEWAY_TYPE(int64_t);
    BENCHMARK_THREEWAY_TYPE(double);
    BENCHMARK_THREEWAY_TYPE(custom_float2);
    BENCHMARK_THREEWAY_TYPE(custom_double2);

    executor.run();
}
