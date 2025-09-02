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

#include "common_benchmark_header.hpp"

// HIP API
#include <hipcub/device/device_select.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class T, class FlagType>
void run_flagged_benchmark(benchmark::State& state,
                           size_t            size,
                           const hipStream_t stream,
                           float             true_probability)
{
    std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    std::vector<FlagType> flags
        = benchmark_utils::get_random_data01<FlagType>(size, true_probability);

    T*            d_input;
    FlagType*     d_flags;
    T*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(FlagType)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(FlagType), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes = 0;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSelect::Flagged(nullptr,
                                            temp_storage_size_bytes,
                                            d_input,
                                            d_flags,
                                            d_output,
                                            d_selected_count_output,
                                            input.size(),
                                            stream));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_flags,
                                                d_output,
                                                d_selected_count_output,
                                                input.size(),
                                                stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_input,
                                                    d_flags,
                                                    d_output,
                                                    d_selected_count_output,
                                                    input.size(),
                                                    stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipDeviceSynchronize());
}

template<class T>
struct SelectOperator
{
    float true_probability;
    SelectOperator(float true_probability_) : true_probability(true_probability_) {}
    HIPCUB_DEVICE
    inline constexpr bool
        operator()(const T& value)
    {
        return value < T(1000 * true_probability);
    }
};

template<class T>
void run_selectop_benchmark(benchmark::State& state,
                            size_t            size,
                            const hipStream_t stream,
                            float             true_probability)
{
    std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(1000));

    SelectOperator<T> select_op(true_probability);

    T*            d_input;
    T*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSelect::If(nullptr,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_output,
                                       d_selected_count_output,
                                       input.size(),
                                       select_op,
                                       stream));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           d_output,
                                           d_selected_count_output,
                                           input.size(),
                                           select_op,
                                           stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                               temp_storage_size_bytes,
                                               d_input,
                                               d_output,
                                               d_selected_count_output,
                                               input.size(),
                                               select_op,
                                               stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipDeviceSynchronize());
}

template<class T, class FlagType>
void run_flagged_if_benchmark(benchmark::State& state,
                              size_t            size,
                              const hipStream_t stream,
                              float             true_probability)
{
    std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    std::vector<FlagType> flags
        = benchmark_utils::get_random_data01<FlagType>(size, true_probability);

    SelectOperator<T> select_flag_op(true_probability);

    T*            d_input;
    FlagType*     d_flags;
    T*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(FlagType)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(FlagType), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes = 0;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(nullptr,
                                              temp_storage_size_bytes,
                                              d_input,
                                              d_flags,
                                              d_output,
                                              d_selected_count_output,
                                              input.size(),
                                              select_flag_op,
                                              stream));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_flags,
                                                  d_output,
                                                  d_selected_count_output,
                                                  input.size(),
                                                  select_flag_op,
                                                  stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(d_temp_storage,
                                                      temp_storage_size_bytes,
                                                      d_input,
                                                      d_flags,
                                                      d_output,
                                                      d_selected_count_output,
                                                      input.size(),
                                                      select_flag_op,
                                                      stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

template<class T>
void run_unique_benchmark(benchmark::State& state,
                          size_t            size,
                          const hipStream_t stream,
                          float             discontinuity_probability)
{
    hipcub::Sum op;

    std::vector<T> input(size);
    {
        auto input01 = benchmark_utils::get_random_data01<T>(size, discontinuity_probability);
        auto acc     = input01[0];
        input[0]     = acc;
        for(size_t i = 1; i < input01.size(); i++)
        {
            input[i] = op(acc, input01[i]);
        }
    }

    T*            d_input;
    T*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSelect::Unique(nullptr,
                                           temp_storage_size_bytes,
                                           d_input,
                                           d_output,
                                           d_selected_count_output,
                                           input.size(),
                                           stream));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                               temp_storage_size_bytes,
                                               d_input,
                                               d_output,
                                               d_selected_count_output,
                                               input.size(),
                                               stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                                   temp_storage_size_bytes,
                                                   d_input,
                                                   d_output,
                                                   d_selected_count_output,
                                                   input.size(),
                                                   stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

template<class KeyT, class ValueT>
void run_unique_by_key_benchmark(benchmark::State& state,
                                 size_t            size,
                                 const hipStream_t stream,
                                 float             discontinuity_probability)
{
    hipcub::Sum op;

    std::vector<KeyT> input_keys(size);
    {
        auto input01 = benchmark_utils::get_random_data01<KeyT>(size, discontinuity_probability);
        auto acc     = input01[0];

        input_keys[0] = acc;

        for(size_t i = 1; i < input01.size(); i++)
        {
            input_keys[i] = op(acc, input01[i]);
        }
    }

    const auto input_values
        = benchmark_utils::get_random_data<ValueT>(size, ValueT(-1000), ValueT(1000));

    KeyT*         d_keys_input;
    ValueT*       d_values_input;
    KeyT*         d_keys_output;
    ValueT*       d_values_output;
    unsigned int* d_selected_count_output;

    HIP_CHECK(hipMalloc(&d_keys_input, input_keys.size() * sizeof(input_keys[0])));
    HIP_CHECK(hipMalloc(&d_values_input, input_values.size() * sizeof(input_values[0])));
    HIP_CHECK(hipMalloc(&d_keys_output, input_keys.size() * sizeof(input_keys[0])));
    HIP_CHECK(hipMalloc(&d_values_output, input_values.size() * sizeof(input_values[0])));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

    HIP_CHECK(hipMemcpy(d_keys_input,
                        input_keys.data(),
                        input_keys.size() * sizeof(input_keys[0]),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values_input,
                        input_values.data(),
                        input_values.size() * sizeof(input_values[0]),
                        hipMemcpyHostToDevice));

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(nullptr,
                                                temp_storage_size_bytes,
                                                d_keys_input,
                                                d_values_input,
                                                d_keys_output,
                                                d_values_output,
                                                d_selected_count_output,
                                                input_keys.size(),
                                                stream));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_keys_input,
                                                    d_values_input,
                                                    d_keys_output,
                                                    d_values_output,
                                                    d_selected_count_output,
                                                    input_keys.size(),
                                                    stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        d_keys_input,
                                                        d_values_input,
                                                        d_keys_output,
                                                        d_values_output,
                                                        d_selected_count_output,
                                                        input_keys.size(),
                                                        stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size
                            * (sizeof(KeyT) + sizeof(ValueT)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

#define CREATE_SELECT_FLAGGED_BENCHMARK(T, F, p)                                                   \
    benchmark::RegisterBenchmark(                                                                  \
        std::string("device_select_flagged<data_type:" #T ",flag_type:" #F ",output_data_type:" #T \
                    ",selected_output_data_type:unsigned int>.(probability:" #p ")")               \
            .c_str(),                                                                              \
        &run_flagged_benchmark<T, F>,                                                              \
        size,                                                                                      \
        stream,                                                                                    \
        p)

#define CREATE_SELECT_IF_BENCHMARK(T, p)                                             \
    benchmark::RegisterBenchmark(                                                    \
        std::string("device_select_if<data_type:" #T ",output_data_type:" #T         \
                    ",selected_output_data_type:unsigned int>.(probability:" #p ")") \
            .c_str(),                                                                \
        &run_selectop_benchmark<T>,                                                  \
        size,                                                                        \
        stream,                                                                      \
        p)

#define CREATE_SELECT_FLAGGED_IF_BENCHMARK(T, F, p)                                  \
    benchmark::RegisterBenchmark(                                                    \
        std::string("device_select_flagged_if<data_type:" #T ",flag_type:" #F        \
                    ",output_data_type:" #T                                          \
                    ",selected_output_data_type:unsigned int>.(probability:" #p ")") \
            .c_str(),                                                                \
        &run_flagged_if_benchmark<T, F>,                                             \
        size,                                                                        \
        stream,                                                                      \
        p)

#define CREATE_UNIQUE_BENCHMARK(T, p)                                                \
    benchmark::RegisterBenchmark(                                                    \
        std::string("device_select_unique<data_type:" #T ",output_data_type:" #T     \
                    ",selected_output_data_type:unsigned int>.(probability:" #p ")") \
            .c_str(),                                                                \
        &run_unique_benchmark<T>,                                                    \
        size,                                                                        \
        stream,                                                                      \
        p)

#define CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, p)                                            \
    benchmark::RegisterBenchmark(                                                          \
        std::string("device_select_unique_by_key<Key data_type:" #K ",value_data_type:" #V \
                    ",selected_output_data_type:unsigned int>.(probability:" #p ")")       \
            .c_str(),                                                                      \
        &run_unique_by_key_benchmark<K, V>,                                                \
        size,                                                                              \
        stream,                                                                            \
        p)

#define BENCHMARK_FLAGGED_TYPE(type, value)                  \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.05f),     \
        CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.25f), \
        CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.5f),  \
        CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_IF_TYPE(type)                                                       \
    CREATE_SELECT_IF_BENCHMARK(type, 0.05f), CREATE_SELECT_IF_BENCHMARK(type, 0.25f), \
        CREATE_SELECT_IF_BENCHMARK(type, 0.5f), CREATE_SELECT_IF_BENCHMARK(type, 0.75f)

#define BENCHMARK_FLAGGED_IF_TYPE(type, value)                  \
    CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.05f),     \
        CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.25f), \
        CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.5f),  \
        CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_UNIQUE_TYPE(type)                                             \
    CREATE_UNIQUE_BENCHMARK(type, 0.05f), CREATE_UNIQUE_BENCHMARK(type, 0.25f), \
        CREATE_UNIQUE_BENCHMARK(type, 0.5f), CREATE_UNIQUE_BENCHMARK(type, 0.75f)

#define BENCHMARK_UNIQUE_BY_KEY_TYPE(key_type, value_type)           \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.05f),     \
        CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.25f), \
        CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.5f),  \
        CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.75f)

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

    std::cout << "benchmark_device_select" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    using custom_double2    = benchmark_utils::custom_type<double, double>;
    using custom_int_double = benchmark_utils::custom_type<int, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks
        = {BENCHMARK_FLAGGED_TYPE(int, unsigned char),
           BENCHMARK_FLAGGED_TYPE(float, unsigned char),
           BENCHMARK_FLAGGED_TYPE(double, unsigned char),
           BENCHMARK_FLAGGED_TYPE(uint8_t, uint8_t),
           BENCHMARK_FLAGGED_TYPE(int8_t, int8_t),
           BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char),

           BENCHMARK_IF_TYPE(int),
           BENCHMARK_IF_TYPE(float),
           BENCHMARK_IF_TYPE(double),
           BENCHMARK_IF_TYPE(uint8_t),
           BENCHMARK_IF_TYPE(int8_t),
           BENCHMARK_IF_TYPE(custom_int_double),

           BENCHMARK_FLAGGED_IF_TYPE(int, unsigned char),
           BENCHMARK_FLAGGED_IF_TYPE(float, unsigned char),
           BENCHMARK_FLAGGED_IF_TYPE(double, unsigned char),
           BENCHMARK_FLAGGED_IF_TYPE(uint8_t, uint8_t),
           BENCHMARK_FLAGGED_IF_TYPE(int8_t, int8_t),
           BENCHMARK_FLAGGED_IF_TYPE(custom_double2, unsigned char),

           BENCHMARK_UNIQUE_TYPE(int),
           BENCHMARK_UNIQUE_TYPE(float),
           BENCHMARK_UNIQUE_TYPE(double),
           BENCHMARK_UNIQUE_TYPE(uint8_t),
           BENCHMARK_UNIQUE_TYPE(int8_t),
           BENCHMARK_UNIQUE_TYPE(custom_int_double),

           BENCHMARK_UNIQUE_BY_KEY_TYPE(int, int),
           BENCHMARK_UNIQUE_BY_KEY_TYPE(float, double),
           BENCHMARK_UNIQUE_BY_KEY_TYPE(double, custom_double2),
           BENCHMARK_UNIQUE_BY_KEY_TYPE(uint8_t, uint8_t),
           BENCHMARK_UNIQUE_BY_KEY_TYPE(int8_t, double),
           BENCHMARK_UNIQUE_BY_KEY_TYPE(custom_int_double, custom_int_double)};

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
