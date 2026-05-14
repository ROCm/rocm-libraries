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
#include <hipcub/device/device_select.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 32 * primbench::MiB;
#endif

template<class T, class FlagType>
class flagged_benchmark : public primbench::benchmark_interface
{
public:
    flagged_benchmark(float true_probability) : m_true_probability(true_probability) {}

private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_select")
            .add("subalgo", "flagged")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("flag_type", primbench::name<FlagType>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(size,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        std::vector<FlagType> flags
            = benchmark_utils::get_random_data01<FlagType>(size, m_true_probability);

        T*            d_input;
        FlagType*     d_flags;
        T*            d_output;
        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(FlagType)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_flags,
                            flags.data(),
                            flags.size() * sizeof(FlagType),
                            hipMemcpyHostToDevice));
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

        state.set_items(size);
        state.add_writes<T>(size);

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        d_input,
                                                        d_flags,
                                                        d_output,
                                                        d_selected_count_output,
                                                        input.size(),
                                                        stream));
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_flags));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipDeviceSynchronize());
    }

    float m_true_probability;
};

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
class selectop_benchmark : public primbench::benchmark_interface
{
public:
    selectop_benchmark(float true_probability) : m_true_probability(true_probability) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_select")
            .add("subalgo", "if")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(1000));

        SelectOperator<T> select_op(m_true_probability);

        T*            d_input;
        T*            d_output;
        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
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

        state.set_items(size);
        state.add_writes<T>(size);

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                                   temp_storage_size_bytes,
                                                   d_input,
                                                   d_output,
                                                   d_selected_count_output,
                                                   input.size(),
                                                   select_op,
                                                   stream));
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipDeviceSynchronize());
    }

private:
    float m_true_probability;
};

template<class T, class FlagType>
class flagged_if_benchmark : public primbench::benchmark_interface
{
public:
    flagged_if_benchmark(float true_probability) : m_true_probability(true_probability) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_select")
            .add("subalgo", "flagged_if")
            .add("data_type", primbench::name<T>())
            .add("flag_type", primbench::name<FlagType>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(size,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        std::vector<FlagType> flags
            = benchmark_utils::get_random_data01<FlagType>(size, m_true_probability);

        SelectOperator<T> select_flag_op(m_true_probability);

        T*            d_input;
        FlagType*     d_flags;
        T*            d_output;
        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(FlagType)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_flags,
                            flags.data(),
                            flags.size() * sizeof(FlagType),
                            hipMemcpyHostToDevice));
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

        state.set_items(size);
        state.add_writes<T>(size);

        state.run(
            [&]
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
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_flags));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

private:
    float m_true_probability;
};

template<class T>
class unique_benchmark : public primbench::benchmark_interface
{
public:
    unique_benchmark(float discontinuity_probability)
        : m_discontinuity_probability(discontinuity_probability)
    {}

private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_select")
            .add("subalgo", "unique")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_discontinuity_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        hipcub::Sum op;

        std::vector<T> input(size);
        {
            auto input01 = benchmark_utils::get_random_data01<T>(size, m_discontinuity_probability);
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
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
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

        state.set_items(size);
        state.add_writes<T>(size);

        state.run(
            [&]
            {
                HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       stream));
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    float m_discontinuity_probability;
};

template<class KeyT, class ValueT>
class unique_by_key_benchmark : public primbench::benchmark_interface
{
public:
    unique_by_key_benchmark(float discontinuity_probability)
        : m_discontinuity_probability(discontinuity_probability)
    {}

private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_select")
            .add("subalgo", "unique_by_key")
            .add("key_data_type", primbench::name<KeyT>())
            .add("value_data_type", primbench::name<ValueT>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_discontinuity_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t size   = state.size;
        const auto&  stream = state.stream;

        hipcub::Sum op;

        std::vector<KeyT> input_keys(size);
        {
            auto input01
                = benchmark_utils::get_random_data01<KeyT>(size, m_discontinuity_probability);
            auto acc = input01[0];

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

        state.set_items(size);
        state.add_writes<std::byte>(size * (sizeof(KeyT) + sizeof(ValueT)));

        state.run(
            [&]
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
            });

        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    float m_discontinuity_probability;
};

#define CREATE_SELECT_FLAGGED_BENCHMARK(T, F, p) executor.queue<flagged_benchmark<T, F>>(p)

#define CREATE_SELECT_IF_BENCHMARK(T, p) executor.queue<selectop_benchmark<T>>(p)

#define CREATE_SELECT_FLAGGED_IF_BENCHMARK(T, F, p) executor.queue<flagged_if_benchmark<T, F>>(p)

#define CREATE_UNIQUE_BENCHMARK(T, p) executor.queue<unique_benchmark<T>>(p)

#define CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, p) executor.queue<unique_by_key_benchmark<K, V>>(p)

#define BENCHMARK_FLAGGED_TYPE(type, value)              \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.05f); \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.25f); \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.5f);  \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_IF_TYPE(type)              \
    CREATE_SELECT_IF_BENCHMARK(type, 0.05f); \
    CREATE_SELECT_IF_BENCHMARK(type, 0.25f); \
    CREATE_SELECT_IF_BENCHMARK(type, 0.5f);  \
    CREATE_SELECT_IF_BENCHMARK(type, 0.75f)

#define BENCHMARK_FLAGGED_IF_TYPE(type, value)              \
    CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.05f); \
    CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.25f); \
    CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.5f);  \
    CREATE_SELECT_FLAGGED_IF_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_UNIQUE_TYPE(type)       \
    CREATE_UNIQUE_BENCHMARK(type, 0.05f); \
    CREATE_UNIQUE_BENCHMARK(type, 0.25f); \
    CREATE_UNIQUE_BENCHMARK(type, 0.5f);  \
    CREATE_UNIQUE_BENCHMARK(type, 0.75f)

#define BENCHMARK_UNIQUE_BY_KEY_TYPE(key_type, value_type)       \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.05f); \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.25f); \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.5f);  \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(key_type, value_type, 0.75f)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    BENCHMARK_FLAGGED_TYPE(int, unsigned char);
    BENCHMARK_FLAGGED_TYPE(float, unsigned char);
    BENCHMARK_FLAGGED_TYPE(double, unsigned char);
    BENCHMARK_FLAGGED_TYPE(uint8_t, uint8_t);
    BENCHMARK_FLAGGED_TYPE(int8_t, int8_t);
    BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char);

    BENCHMARK_IF_TYPE(int);
    BENCHMARK_IF_TYPE(float);
    BENCHMARK_IF_TYPE(double);
    BENCHMARK_IF_TYPE(uint8_t);
    BENCHMARK_IF_TYPE(int8_t);
    BENCHMARK_IF_TYPE(custom_int_double);

    BENCHMARK_FLAGGED_IF_TYPE(int, unsigned char);
    BENCHMARK_FLAGGED_IF_TYPE(float, unsigned char);
    BENCHMARK_FLAGGED_IF_TYPE(double, unsigned char);
    BENCHMARK_FLAGGED_IF_TYPE(uint8_t, uint8_t);
    BENCHMARK_FLAGGED_IF_TYPE(int8_t, int8_t);
    BENCHMARK_FLAGGED_IF_TYPE(custom_double2, unsigned char);

    BENCHMARK_UNIQUE_TYPE(int);
    BENCHMARK_UNIQUE_TYPE(float);
    BENCHMARK_UNIQUE_TYPE(double);
    BENCHMARK_UNIQUE_TYPE(uint8_t);
    BENCHMARK_UNIQUE_TYPE(int8_t);
    BENCHMARK_UNIQUE_TYPE(custom_int_double);

    BENCHMARK_UNIQUE_BY_KEY_TYPE(int, int);
    BENCHMARK_UNIQUE_BY_KEY_TYPE(float, double);
    BENCHMARK_UNIQUE_BY_KEY_TYPE(double, custom_double2);
    BENCHMARK_UNIQUE_BY_KEY_TYPE(uint8_t, uint8_t);
    BENCHMARK_UNIQUE_BY_KEY_TYPE(int8_t, double);
    BENCHMARK_UNIQUE_BY_KEY_TYPE(custom_int_double, custom_int_double);

    executor.run();
}
