// MIT License
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/device/device_partition.hpp>

#include <chrono>
#include <vector>

#ifndef DEFAULT_N
constexpr size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

constexpr unsigned int batch_size  = 10;
constexpr unsigned int warmup_size = 5;

namespace
{
template<typename T>
struct LessOp
{
    HIPCUB_HOST_DEVICE LessOp(const T& pivot) : pivot_{pivot} {}

    HIPCUB_HOST_DEVICE bool operator()(const T& val) const
    {
        return val < pivot_;
    }

private:
    T pivot_;
};
} // namespace

template<typename T, typename F>
void run_flagged(benchmark::State& state,
                 const hipStream_t stream,
                 const T           threshold,
                 const size_t      size)
{
    const auto select_op = LessOp<T>{threshold};
    const auto input
        = benchmark_utils::get_random_data<T>(size, static_cast<T>(0), static_cast<T>(100));

    std::vector<F> flags(size);
    for(unsigned int i = 0; i < size; i++)
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

    // Allocate temporary storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_CHECK(hipcub::DevicePartition::Flagged(nullptr,
                                               temp_storage_bytes,
                                               d_input,
                                               d_flags,
                                               d_output,
                                               d_num_selected_output,
                                               static_cast<int>(input.size()),
                                               stream));
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Warm-up
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(F), hipMemcpyHostToDevice));
    for(unsigned int i = 0; i < warmup_size; ++i)
    {
        HIP_CHECK(hipcub::DevicePartition::Flagged(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_input,
                                                   d_flags,
                                                   d_output,
                                                   d_num_selected_output,
                                                   static_cast<int>(input.size()),
                                                   stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Run benchmark
    for(auto _ : state)
    {
        namespace chrono = std::chrono;
        using clock      = chrono::high_resolution_clock;

        const auto start = clock::now();
        for(unsigned int i = 0; i < batch_size; ++i)
        {
            HIP_CHECK(hipcub::DevicePartition::Flagged(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_input,
                                                       d_flags,
                                                       d_output,
                                                       d_num_selected_output,
                                                       static_cast<int>(input.size()),
                                                       stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        const auto end             = clock::now();
        using seconds_d            = chrono::duration<double>;
        const auto elapsed_seconds = chrono::duration_cast<seconds_d>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }

    state.SetItemsProcessed(state.iterations() * batch_size * input.size());
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * input.size() * sizeof(input[0])));

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_num_selected_output));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_input));
}

template<typename T>
void run_predicate(benchmark::State& state,
                   const hipStream_t stream,
                   const T           threshold,
                   const size_t      size)
{
    const auto input
        = benchmark_utils::get_random_data<T>(size, static_cast<T>(0), static_cast<T>(100));

    T*            d_input               = nullptr;
    T*            d_output              = nullptr;
    unsigned int* d_num_selected_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_num_selected_output, sizeof(unsigned int)));

    const auto select_op = LessOp<T>{threshold};

    // Allocate temporary storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_CHECK(hipcub::DevicePartition::If(nullptr,
                                          temp_storage_bytes,
                                          d_input,
                                          d_output,
                                          d_num_selected_output,
                                          static_cast<int>(input.size()),
                                          select_op,
                                          stream));
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Warm-up
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    for(unsigned int i = 0; i < warmup_size; ++i)
    {
        HIP_CHECK(hipcub::DevicePartition::If(d_temp_storage,
                                              temp_storage_bytes,
                                              d_input,
                                              d_output,
                                              d_num_selected_output,
                                              static_cast<int>(input.size()),
                                              select_op,
                                              stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Run benchmark
    for(auto _ : state)
    {
        namespace chrono = std::chrono;
        using clock      = chrono::high_resolution_clock;

        const auto start = clock::now();
        for(unsigned int i = 0; i < batch_size; ++i)
        {
            HIP_CHECK(hipcub::DevicePartition::If(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_input,
                                                  d_output,
                                                  d_num_selected_output,
                                                  static_cast<int>(input.size()),
                                                  select_op,
                                                  stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        const auto end             = clock::now();
        using seconds_d            = chrono::duration<double>;
        const auto elapsed_seconds = chrono::duration_cast<seconds_d>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }

    state.SetItemsProcessed(state.iterations() * batch_size * input.size());
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * input.size() * sizeof(input[0])));

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_num_selected_output));
}

template<typename T>
void run_threeway(benchmark::State& state,
                  const hipStream_t stream,
                  const T           small_threshold,
                  const T           large_threshold,
                  const size_t      size)
{
    const auto input
        = benchmark_utils::get_random_data<T>(size, static_cast<T>(0), static_cast<T>(100));

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

    const auto select_first_part_op  = LessOp<T>{small_threshold};
    const auto select_second_part_op = LessOp<T>{large_threshold};

    // Allocate temporary storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_CHECK(hipcub::DevicePartition::If(nullptr,
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
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Warm-up
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    for(unsigned int i = 0; i < warmup_size; ++i)
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
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Run benchmark
    for(auto _ : state)
    {
        namespace chrono = std::chrono;
        using clock      = chrono::high_resolution_clock;

        const auto start = clock::now();
        for(unsigned int i = 0; i < batch_size; ++i)
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
        }
        HIP_CHECK(hipDeviceSynchronize());

        const auto end             = clock::now();
        using seconds_d            = chrono::duration<double>;
        const auto elapsed_seconds = chrono::duration_cast<seconds_d>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }

    state.SetItemsProcessed(state.iterations() * batch_size * input.size());
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * input.size() * sizeof(input[0])));

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_first_output));
    HIP_CHECK(hipFree(d_second_output));
    HIP_CHECK(hipFree(d_unselected_output));
    HIP_CHECK(hipFree(d_num_selected_output));
}

#define CREATE_BENCHMARK_FLAGGED(T, T_FLAG, SPLIT_T)                                              \
    benchmark::RegisterBenchmark(std::string("device_parition_flagged<data_type:" #T              \
                                             ",flag_type:" #T_FLAG ">.(split_threshold:" #SPLIT_T \
                                             "%)")                                                \
                                     .c_str(),                                                    \
                                 &run_flagged<T, T_FLAG>,                                         \
                                 stream,                                                          \
                                 static_cast<T>(SPLIT_T),                                         \
                                 size)

#define CREATE_BENCHMARK_PREDICATE(T, SPLIT_T)                                                     \
    benchmark::RegisterBenchmark(                                                                  \
        std::string("device_parition_predicate<data_type:" #T ">.(split_threshold:" #SPLIT_T "%)") \
            .c_str(),                                                                              \
        &run_predicate<T>,                                                                         \
        stream,                                                                                    \
        static_cast<T>(SPLIT_T),                                                                   \
        size)

#define CREATE_BENCHMARK_THREEWAY(T, SMALL_T, LARGE_T)                                       \
    benchmark::RegisterBenchmark(std::string("device_parition_three_way"                     \
                                             "<data_type:" #T ">.(small_threshold:" #SMALL_T \
                                             "%,large_threshold:" #LARGE_T "%)")             \
                                     .c_str(),                                               \
                                 &run_threeway<T>,                                           \
                                 stream,                                                     \
                                 static_cast<T>(SMALL_T),                                    \
                                 static_cast<T>(LARGE_T),                                    \
                                 size)

#define BENCHMARK_FLAGGED_TYPE(type, flag_type)                                                   \
    CREATE_BENCHMARK_FLAGGED(type, flag_type, 33), CREATE_BENCHMARK_FLAGGED(type, flag_type, 50), \
        CREATE_BENCHMARK_FLAGGED(type, flag_type, 60),                                            \
        CREATE_BENCHMARK_FLAGGED(type, flag_type, 90)

#define BENCHMARK_PREDICATE_TYPE(type)                                          \
    CREATE_BENCHMARK_PREDICATE(type, 33), CREATE_BENCHMARK_PREDICATE(type, 50), \
        CREATE_BENCHMARK_PREDICATE(type, 60), CREATE_BENCHMARK_PREDICATE(type, 90)

#define BENCHMARK_THREEWAY_TYPE(type)                                                 \
    CREATE_BENCHMARK_THREEWAY(type, 33, 66), CREATE_BENCHMARK_THREEWAY(type, 10, 66), \
        CREATE_BENCHMARK_THREEWAY(type, 50, 60), CREATE_BENCHMARK_THREEWAY(type, 50, 90)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    std::cout << "benchmark_device_partition" << std::endl;

    // HIP
    const hipStream_t stream = 0; // default
    {
        hipDeviceProp_t devProp;
        int             device_id = 0;
        HIP_CHECK(hipGetDevice(&device_id));
        HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
        std::cout << "[HIP] Device name: " << devProp.name << std::endl;
    }

    using custom_float2  = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {
        BENCHMARK_FLAGGED_TYPE(int8_t, unsigned char),
        BENCHMARK_FLAGGED_TYPE(int, unsigned char),
        BENCHMARK_FLAGGED_TYPE(float, unsigned char),
        BENCHMARK_FLAGGED_TYPE(long long, uint8_t),
        BENCHMARK_FLAGGED_TYPE(double, int8_t),
        BENCHMARK_FLAGGED_TYPE(custom_float2, int8_t),
        BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char),

        BENCHMARK_PREDICATE_TYPE(int8_t),
        BENCHMARK_PREDICATE_TYPE(int),
        BENCHMARK_PREDICATE_TYPE(float),
        BENCHMARK_PREDICATE_TYPE(long long),
        BENCHMARK_PREDICATE_TYPE(double),
        BENCHMARK_PREDICATE_TYPE(custom_float2),
        BENCHMARK_PREDICATE_TYPE(custom_double2),

        BENCHMARK_THREEWAY_TYPE(int8_t),
        BENCHMARK_THREEWAY_TYPE(int),
        BENCHMARK_THREEWAY_TYPE(float),
        BENCHMARK_THREEWAY_TYPE(long long),
        BENCHMARK_THREEWAY_TYPE(double),
        BENCHMARK_THREEWAY_TYPE(custom_float2),
        BENCHMARK_THREEWAY_TYPE(custom_double2),
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
