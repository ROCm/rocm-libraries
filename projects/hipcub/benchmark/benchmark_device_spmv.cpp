// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/device/device_spmv.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 32;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T>
void run_benchmark(benchmark::State& state,
                   size_t            size,
                   const hipStream_t stream,
                   float             probability)
{
    const T rand_min = T(1);
    const T rand_max = T(10);

    // generate a lexicograhically sorted list of (row, column) index tuples
    // number of nonzeroes cannot be guaranteed as duplicates may exist
    const int num_nonzeroes_attempt = static_cast<int>(
        std::min(static_cast<size_t>(INT_MAX),
                 static_cast<size_t>(probability * static_cast<float>(size * size))));
    std::vector<std::pair<int, int>> indices(num_nonzeroes_attempt);
    {
        std::vector<int> flat_indices
            = benchmark_utils::get_random_data<int>(2 * num_nonzeroes_attempt,
                                                    0,
                                                    size - 1,
                                                    2 * num_nonzeroes_attempt);
        for(int i = 0; i < num_nonzeroes_attempt; i++)
        {
            indices[i] = std::make_pair(flat_indices[2 * i], flat_indices[2 * i + 1]);
        }
        std::sort(indices.begin(), indices.end());
    }

    // generate the compressed sparse rows matrix
    std::pair<int, int> prev_cell     = std::make_pair(-1, -1);
    int                 num_nonzeroes = 0;
    std::vector<int>    row_offsets(size + 1);
    // this vector might be too large, but doing the allocation now eliminates a
    // scan
    std::vector<int> column_indices(num_nonzeroes_attempt);
    row_offsets[0]       = 0;
    int last_row_written = 0;
    for(int i = 0; i < num_nonzeroes_attempt; i++)
    {
        if(indices[i] != prev_cell)
        {
            // update the row offets if we go to the next row (or skip some)
            if(indices[i].first != last_row_written)
            {
                for(int j = last_row_written + 1; j <= indices[i].first; j++)
                {
                    row_offsets[j] = num_nonzeroes;
                }
                last_row_written = indices[i].first;
            }

            column_indices[num_nonzeroes++] = indices[i].second;

            prev_cell = indices[i];
        }
    }
    // fill in the entries for any missing rows
    for(int j = last_row_written + 1; j < static_cast<int>(size) + 1; j++)
    {
        row_offsets[j] = num_nonzeroes;
    }

    // generate the random data once the actual number of nonzeroes are known
    std::vector<T> values = benchmark_utils::get_random_data<T>(num_nonzeroes, rand_min, rand_max);

    std::vector<T> vector_x = benchmark_utils::get_random_data<T>(size, rand_min, rand_max);

    T*   d_values;
    int* d_row_offsets;
    int* d_column_indices;
    T*   d_vector_x;
    T*   d_vector_y;
    HIP_CHECK(hipMalloc(&d_values, values.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_row_offsets, row_offsets.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_column_indices, num_nonzeroes * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_vector_x, vector_x.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_vector_y, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_values, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_offsets,
                        row_offsets.data(),
                        row_offsets.size() * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_column_indices,
                        column_indices.data(),
                        num_nonzeroes * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_vector_x, vector_x.data(), vector_x.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    HIP_CHECK(hipcub::DeviceSpmv::CsrMV(nullptr,
                                        temp_storage_size_bytes,
                                        d_values,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_vector_x,
                                        d_vector_y,
                                        size,
                                        size,
                                        num_nonzeroes,
                                        stream));
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        HIP_CHECK(hipcub::DeviceSpmv::CsrMV(d_temp_storage,
                                            temp_storage_size_bytes,
                                            d_values,
                                            d_row_offsets,
                                            d_column_indices,
                                            d_vector_x,
                                            d_vector_y,
                                            size,
                                            size,
                                            num_nonzeroes,
                                            stream));
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
            HIP_CHECK(hipcub::DeviceSpmv::CsrMV(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_values,
                                                d_row_offsets,
                                                d_column_indices,
                                                d_vector_x,
                                                d_vector_y,
                                                size,
                                                size,
                                                num_nonzeroes,
                                                stream));
            HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * (num_nonzeroes + size) * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * (num_nonzeroes + size));

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_vector_y));
    HIP_CHECK(hipFree(d_vector_x));
    HIP_CHECK(hipFree(d_column_indices));
    HIP_CHECK(hipFree(d_row_offsets));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipDeviceSynchronize());
}

#define CREATE_BENCHMARK(T, p)                                                          \
    benchmark::RegisterBenchmark(                                                       \
        std::string("device_spmv_CsrMV<data_type:" #T ",probability:" #p ">.").c_str(), \
        &run_benchmark<T>,                                                              \
        size,                                                                           \
        stream,                                                                         \
        p)

#define BENCHMARK_TYPE(type)                                              \
    CREATE_BENCHMARK(type, 1.0e-6f), CREATE_BENCHMARK(type, 1.0e-5f),     \
        CREATE_BENCHMARK(type, 1.0e-4f), CREATE_BENCHMARK(type, 1.0e-3f), \
        CREATE_BENCHMARK(type, 1.0e-2f)

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

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    std::cout << "benchmark_device_spmv" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(unsigned int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
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
