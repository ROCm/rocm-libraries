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

#include <hipcub/device/device_select.hpp>

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
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("flag_type", primbench::name<FlagType>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        std::vector<FlagType> flags
            = benchmark_utils::get_random_data01<FlagType>(items, m_true_probability);

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

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_input,
                                                    d_flags,
                                                    d_output,
                                                    d_selected_count_output,
                                                    input.size(),
                                                    stream));
        };

        launch();
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

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
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(1000));

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

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                               temp_storage_bytes,
                                               d_input,
                                               d_output,
                                               d_selected_count_output,
                                               input.size(),
                                               select_op,
                                               stream));
        };

        launch();
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

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
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("flag_type", primbench::name<FlagType>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_true_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        std::vector<FlagType> flags
            = benchmark_utils::get_random_data01<FlagType>(items, m_true_probability);

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

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_input,
                                                      d_flags,
                                                      d_output,
                                                      d_selected_count_output,
                                                      input.size(),
                                                      select_flag_op,
                                                      stream));
        };

        launch();
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

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
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("output_data_type", primbench::name<T>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_discontinuity_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        hipcub::Sum op{};

        std::vector<T> input(items);
        {
            auto input01
                = benchmark_utils::get_random_data01<T>(items, m_discontinuity_probability);
            auto acc = input01[0];
            input[0] = acc;
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

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_input,
                                                   d_output,
                                                   d_selected_count_output,
                                                   input.size(),
                                                   stream));
        };

        launch();
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    float m_discontinuity_probability;
};

template<class Key, class Value>
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
            .add("lvl", "device")
            .add("key_data_type", primbench::name<Key>())
            .add("value_data_type", primbench::name<Value>())
            .add("selected_output_data_type", "u32")
            .add("probability", m_discontinuity_probability);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        hipcub::Sum op{};

        std::vector<Key> input_keys(items);
        {
            auto input01
                = benchmark_utils::get_random_data01<Key>(items, m_discontinuity_probability);
            auto acc = input01[0];

            input_keys[0] = acc;

            for(size_t i = 1; i < input01.size(); i++)
            {
                input_keys[i] = op(acc, input01[i]);
            }
        }

        const auto input_values
            = benchmark_utils::get_random_data<Value>(items, Value(-1000), Value(1000));

        Key*          d_keys_input;
        Value*        d_values_input;
        Key*          d_keys_output;
        Value*        d_values_output;
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

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_input,
                                                        d_values_input,
                                                        d_keys_output,
                                                        d_values_output,
                                                        d_selected_count_output,
                                                        input_keys.size(),
                                                        stream));
        };

        launch();
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<Key>(items);
        state.add_writes<Value>(items);

        state.run(launch);

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

#define BENCHMARK_UNIQUE_BY_KEY_TYPE(Key, Value)       \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(Key, Value, 0.05f); \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(Key, Value, 0.25f); \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(Key, Value, 0.5f);  \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(Key, Value, 0.75f)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
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
