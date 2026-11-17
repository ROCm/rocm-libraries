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

#include "benchmark_utils.hpp"

#include <hipcub/device/device_segmented_reduce.hpp>

using OffsetType = int;

template<class T, class OutputT, class SegmentedReduceKernel, size_t DesiredSegments>
class reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_segmented_reduce")
            .add("lvl", "device")
            .add("reduce_op", SegmentedReduceKernel::name)
            .add("data_type", primbench::name<T>())
            .add("desired_segments", DesiredSegments);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        const auto segmented_reduce = SegmentedReduceKernel::kernel;

        using Value = T;

        // Generate data
        const unsigned int         seed = 123;
        std::default_random_engine gen(seed);

        const double avg_segment_length
            = std::max(1.0, static_cast<double>(items) / DesiredSegments);
        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        std::vector<OffsetType> offsets;
        unsigned int            segments_count = 0;
        size_t                  offset         = 0;
        while(offset < items)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(items);

        std::vector<Value> values_input(items);
        std::iota(values_input.begin(), values_input.end(), 0);

        OffsetType* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(OffsetType)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(OffsetType),
                            hipMemcpyHostToDevice));

        Value* d_values_input;
        HIP_CHECK(hipMalloc(&d_values_input, items * sizeof(Value)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            items * sizeof(Value),
                            hipMemcpyHostToDevice));

        OutputT* d_aggregates_output;
        HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(OutputT)));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(segmented_reduce(d_temp_storage,
                                       temp_storage_bytes,
                                       d_values_input,
                                       d_aggregates_output,
                                       segments_count,
                                       d_offsets,
                                       d_offsets + 1,
                                       stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<Value>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_aggregates_output));
    }
};

template<typename T, typename Op, size_t DesiredSegments>
struct Benchmark;

template<typename T>
struct min_kernel
{
    static constexpr const char* name = "min";

    static constexpr hipError_t (*kernel)(
        void*, size_t&, T*, T*, int, OffsetType*, OffsetType*, hipStream_t)
        = &hipcub::DeviceSegmentedReduce::Min;
};

template<typename T>
struct sum_kernel
{
    static constexpr const char* name = "sum";

    static constexpr hipError_t (*kernel)(
        void*, size_t&, T*, T*, int, OffsetType*, OffsetType*, hipStream_t)
        = &hipcub::DeviceSegmentedReduce::Sum;
};

template<typename T>
struct argmin_kernel
{
    using Difference = OffsetType;
    using Iterator   = typename hipcub::ArgIndexInputIterator<T*, Difference>;
    using KeyValue   = typename Iterator::value_type;

    static constexpr const char* name = "argmin";

    static constexpr hipError_t (*kernel)(
        void*, size_t&, T*, KeyValue*, int, OffsetType*, OffsetType*, hipStream_t)
        = &hipcub::DeviceSegmentedReduce::ArgMin;
};

template<typename T, size_t DesiredSegments>
struct Benchmark<T, hipcub::Min, DesiredSegments>
{
    using type = reduce_benchmark<T, T, min_kernel<T>, DesiredSegments>;
};

template<typename T, size_t DesiredSegments>
struct Benchmark<T, hipcub::Sum, DesiredSegments>
{
    using type = reduce_benchmark<T, T, sum_kernel<T>, DesiredSegments>;
};

template<typename T, size_t DesiredSegments>
struct Benchmark<T, hipcub::ArgMin, DesiredSegments>
{
    using type = reduce_benchmark<T,
                                  typename argmin_kernel<T>::KeyValue,
                                  argmin_kernel<T>,
                                  DesiredSegments>;
};

#define CREATE_BENCHMARK(T, SEGMENTS, REDUCE_OP) \
    executor.queue<Benchmark<T, REDUCE_OP, SEGMENTS>::type>()

#define BENCHMARK_TYPE(type, REDUCE_OP)     \
    CREATE_BENCHMARK(type, 1, REDUCE_OP);   \
    CREATE_BENCHMARK(type, 100, REDUCE_OP); \
    CREATE_BENCHMARK(type, 10000, REDUCE_OP)

#define CREATE_BENCHMARKS(REDUCE_OP)   \
    BENCHMARK_TYPE(float, REDUCE_OP);  \
    BENCHMARK_TYPE(double, REDUCE_OP); \
    BENCHMARK_TYPE(int8_t, REDUCE_OP); \
    BENCHMARK_TYPE(int, REDUCE_OP)

void add_benchmarks(primbench::executor& executor)
{
    CREATE_BENCHMARKS(hipcub::Sum);
    BENCHMARK_TYPE(custom_double2, hipcub::Sum);
    CREATE_BENCHMARKS(hipcub::Min);
#ifdef HIPCUB_ROCPRIM_API
    BENCHMARK_TYPE(custom_double2, hipcub::Min);
#endif
    CREATE_BENCHMARKS(hipcub::ArgMin);
#ifdef HIPCUB_ROCPRIM_API
    BENCHMARK_TYPE(custom_double2, hipcub::ArgMin);
#endif
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks(executor);

    executor.run();
}
