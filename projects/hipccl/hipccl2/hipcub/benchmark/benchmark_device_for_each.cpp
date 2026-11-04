// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "benchmark_utils.hpp"

#include <hipcub/device/device_for.hpp>

template<class T>
struct op_t
{
    unsigned int* d_count;

    HIPCUB_DEVICE
    void          operator()(T i)
    {
        // The data is non zero so atomic will never be activated.
        if(i == 0)
        {
            atomicAdd(d_count, 1);
        }
    }
};

template<class T>
class for_each_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_for_each")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>());
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        // Generate data
        std::vector<T> values_input(items, 4);

        T* d_input;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(
            hipMemcpy(d_input, values_input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        unsigned int* d_count;
        HIP_CHECK(hipMalloc(&d_count, sizeof(*d_count)));
        HIP_CHECK(hipMemset(d_count, 0, sizeof(*d_count)));
        op_t<T> device_op{d_count};

        state.set_items(items);
        state.add_reads<T>(items);

        state.run(
            [&] {
                HIP_CHECK(hipcub::DeviceFor::ForEach(d_input, d_input + items, device_op, stream));
            });

        HIP_CHECK(hipFree(d_count));
        HIP_CHECK(hipFree(d_input));
    }
};

#define CREATE_BENCHMARK(Value) executor.queue<for_each_benchmark<Value>>()

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Add benchmarks
    CREATE_BENCHMARK(float);
    CREATE_BENCHMARK(double);
    CREATE_BENCHMARK(custom_double2);
    CREATE_BENCHMARK(int8_t);
    CREATE_BENCHMARK(int64_t);

    executor.run();
}
