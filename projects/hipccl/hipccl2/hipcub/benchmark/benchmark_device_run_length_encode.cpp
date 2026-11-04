// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

// CUB's implementation of DeviceRunLengthEncode has unused parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVIDIA__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "benchmark_utils.hpp"

#include <hipcub/device/device_run_length_encode.hpp>

template<class T, size_t MaxLength>
class encode_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_run_length_encode")
            .add("subalgo", "encode")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("random_number_range", "[1, " + std::to_string(MaxLength) + "]");
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using Key        = T;
        using count_type = unsigned int;

        // Generate data
        std::vector<Key> input(items);

        unsigned int        runs_count = 0;
        std::vector<size_t> key_counts
            = benchmark_utils::get_random_data<size_t>(100000, 1, MaxLength);
        size_t offset = 0;
        while(offset < items)
        {
            const size_t key_count = key_counts[runs_count % key_counts.size()];
            const size_t end       = std::min(items, offset + key_count);
            for(size_t i = offset; i < end; i++)
            {
                input[i] = runs_count;
            }

            runs_count++;
            offset += key_count;
        }

        Key* d_input;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(Key), hipMemcpyHostToDevice));

        Key*        d_unique_output;
        count_type* d_counts_output;
        count_type* d_runs_count_output;
        HIP_CHECK(hipMalloc(&d_unique_output, runs_count * sizeof(Key)));
        HIP_CHECK(hipMalloc(&d_counts_output, runs_count * sizeof(count_type)));
        HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_input,
                                                            d_unique_output,
                                                            d_counts_output,
                                                            d_runs_count_output,
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
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_counts_output));
        HIP_CHECK(hipFree(d_runs_count_output));
    }
};

template<class T, size_t MaxLength>
class non_trivial_runs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_run_length_encode")
            .add("subalgo", "non_trivial_runs")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("random_number_range", "[1, " + std::to_string(MaxLength) + "]");
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using Key         = T;
        using offset_type = unsigned int;
        using count_type  = unsigned int;

        // Generate data
        std::vector<Key> input(items);

        unsigned int        runs_count = 0;
        std::vector<size_t> key_counts
            = benchmark_utils::get_random_data<size_t>(100000, 1, MaxLength);

        size_t offset = 0;
        while(offset < items)
        {
            const size_t key_count = key_counts[runs_count % key_counts.size()];
            const size_t end       = std::min(items, offset + key_count);
            for(size_t i = offset; i < end; i++)
            {
                input[i] = runs_count;
            }

            runs_count++;
            offset += key_count;
        }

        Key* d_input;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(Key)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(Key), hipMemcpyHostToDevice));

        offset_type* d_offsets_output;
        count_type*  d_counts_output;
        count_type*  d_runs_count_output;
        HIP_CHECK(hipMalloc(&d_offsets_output, runs_count * sizeof(offset_type)));
        HIP_CHECK(hipMalloc(&d_counts_output, runs_count * sizeof(count_type)));
        HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    d_input,
                                                                    d_offsets_output,
                                                                    d_counts_output,
                                                                    d_runs_count_output,
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
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_offsets_output));
        HIP_CHECK(hipFree(d_counts_output));
        HIP_CHECK(hipFree(d_runs_count_output));
    }
};

#define CREATE_ENCODE_BENCHMARK(T) executor.queue<encode_benchmark<T, MaxLength>>()

template<size_t MaxLength>
void add_encode_benchmarks(primbench::executor& executor)
{
    CREATE_ENCODE_BENCHMARK(int);
    CREATE_ENCODE_BENCHMARK(int64_t);

    CREATE_ENCODE_BENCHMARK(int8_t);
    CREATE_ENCODE_BENCHMARK(uint8_t);

    CREATE_ENCODE_BENCHMARK(custom_float2);
    CREATE_ENCODE_BENCHMARK(custom_double2);
}

#define CREATE_NON_TRIVIAL_RUNS_BENCHMARK(T) \
    executor.queue<non_trivial_runs_benchmark<T, MaxLength>>()

template<size_t MaxLength>
void add_non_trivial_runs_benchmarks(primbench::executor& executor)
{
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int);
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int64_t);

    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int8_t);
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(uint8_t);

    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_float2);
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_double2);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_encode_benchmarks<1000>(executor);
    add_encode_benchmarks<10>(executor);
    add_non_trivial_runs_benchmarks<1000>(executor);
    add_non_trivial_runs_benchmarks<10>(executor);

    executor.run();
}
