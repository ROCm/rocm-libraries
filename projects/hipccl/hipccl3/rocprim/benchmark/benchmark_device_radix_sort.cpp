// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_radix_sort.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

#ifdef BENCHMARK_CONFIG_TUNING

template<
    typename Key, typename Value,
    unsigned int LRB, unsigned int SRB,
    unsigned int BlockSize1, unsigned int ItemsPerThread1,
    unsigned int BlockSize2, unsigned int ItemsPerThread2
>
auto sort_keys_add_benchmark(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
    -> typename std::enable_if<
        std::is_same<Value, ::rocprim::empty_type>::value, void
    >::type
{
    benchmarks.push_back(
        benchmark::RegisterBenchmark(
            (std::string("sort_keys") + "<" + Traits<Key>::TYPE_NAME + ",radix_sort_config<" +
             std::to_string(LRB) + ", " + std::to_string(SRB) + ", kernel_config<" +
             std::to_string(BlockSize1) + ", " + std::to_string(ItemsPerThread1) + ">, kernel_config<" +
             std::to_string(BlockSize2) + ", " + std::to_string(ItemsPerThread2) + ">>>").c_str(),
            [=](benchmark::State& state) {
                run_sort_keys_benchmark<
                    Key,
                    rocprim::radix_sort_config<
                        LRB,
                        SRB,
                        rocprim::kernel_config<BlockSize1, ItemsPerThread1>,
                        rocprim::kernel_config<BlockSize2, ItemsPerThread2>
                    >
                >(state, stream, size);
            }
        )
    );
}

template<
    typename Key, typename Value,
    unsigned int LRB, unsigned int SRB,
    unsigned int BlockSize1, unsigned int ItemsPerThread1,
    unsigned int BlockSize2, unsigned int ItemsPerThread2
>
auto sort_keys_add_benchmark(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
    -> typename std::enable_if<
        !std::is_same<Value, ::rocprim::empty_type>::value, void
    >::type
{
    benchmarks.push_back(
        benchmark::RegisterBenchmark(
            (std::string("sort_pairs") + "<" + Traits<Key>::TYPE_NAME + "," + Traits<Value>::TYPE_NAME +
             ",radix_sort_config<" + std::to_string(LRB) + ", " + std::to_string(SRB) + ", kernel_config<" +
             std::to_string(BlockSize1) + ", " + std::to_string(ItemsPerThread1) + ">, kernel_config<" +
             std::to_string(BlockSize2) + ", " + std::to_string(ItemsPerThread2) + ">>>").c_str(),
            [=](benchmark::State& state) {
                run_sort_pairs_benchmark<
                    Key, Value,
                    rocprim::radix_sort_config<
                        LRB,
                        SRB,
                        rocprim::kernel_config<BlockSize1, ItemsPerThread1>,
                        rocprim::kernel_config<BlockSize2, ItemsPerThread2>
                    >
                >(state, stream, size);
            }
        )
    );
}

template<
    typename Key, typename Value,
    unsigned int LRB, unsigned int SRB,
    unsigned int BlockSize1, unsigned int ItemsPerThread1,
    unsigned int BlockSize2, unsigned int ItemsPerThread2,
    unsigned int MaxItemsPerThread1
>
auto sort_keys_benchmark_generate_ipt1_grid(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
    -> typename std::enable_if< ItemsPerThread1 == MaxItemsPerThread1, void>::type
{
    sort_keys_add_benchmark<
        Key, Value, LRB, SRB,
        BlockSize1, ItemsPerThread1,
        BlockSize2, ItemsPerThread2
    >(benchmarks, stream, size);
}

template<
    typename Key, typename Value,
    unsigned int LRB, unsigned int SRB,
    unsigned int BlockSize1, unsigned int ItemsPerThread1,
    unsigned int BlockSize2, unsigned int ItemsPerThread2,
    unsigned int MaxItemsPerThread1
>
auto sort_keys_benchmark_generate_ipt1_grid(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
    -> typename std::enable_if< ItemsPerThread1 < MaxItemsPerThread1, void>::type
{
    sort_keys_add_benchmark<
        Key, Value, LRB, SRB,
        BlockSize1, ItemsPerThread1,
        BlockSize2, ItemsPerThread2
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, LRB, SRB,
        BlockSize1, ItemsPerThread1 + 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);
}

template<
    typename Key, typename Value,
    unsigned int BlockSize1,
    unsigned int BlockSize2, unsigned int ItemsPerThread2,
    unsigned int MaxItemsPerThread1
>
void sort_keys_benchmark_generate_radix_grid(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
{
    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, 4, 3,
        BlockSize1, 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, 5, 4,
        BlockSize1, 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, 6, 4,
        BlockSize1, 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, 7, 6,
        BlockSize1, 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_ipt1_grid<
        Key, Value, 8, 7,
        BlockSize1, 1,
        BlockSize2, ItemsPerThread2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);
}

template<
    typename Key, typename Value = ::rocprim::empty_type,
    unsigned int BlockSize1 = 256U,
    unsigned int BlockSize2 = 256U,
    unsigned int MaxItemsPerThread1 = 20U
>
void sort_keys_benchmark_generate(
    std::vector<benchmark::internal::Benchmark*>& benchmarks,
    hipStream_t stream,
    size_t size)
{
    sort_keys_benchmark_generate_radix_grid<
        Key, Value,
        BlockSize1,
        BlockSize2, 1,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_radix_grid<
        Key, Value,
        BlockSize1,
        BlockSize2, 2,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_radix_grid<
        Key, Value,
        BlockSize1,
        BlockSize2, 4,
        MaxItemsPerThread1
    >(benchmarks, stream, size);

    sort_keys_benchmark_generate_radix_grid<
        Key, Value,
        BlockSize1,
        BlockSize2, 8,
        MaxItemsPerThread1
    >(benchmarks, stream, size);
}

// Compilation may never finish, if the compiler needs to compile too many kernels,
// it is recommended to compile benchmarks only for 1-2 types when BENCHMARK_CONFIG_TUNING is used
// (all other sort_keys_benchmark_generate should be commented/removed).
void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              hipStream_t stream,
                              size_t size)
{
    sort_keys_benchmark_generate<int>(benchmarks, stream, size);
    sort_keys_benchmark_generate<long long>(benchmarks, stream, size);
    sort_keys_benchmark_generate<int8_t>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<uint8_t>(benchmarks, stream, size);
    sort_keys_benchmark_generate<rocprim::half, ::rocprim::empty_type, 256U, 256U, 30>(benchmarks, stream, size);
    sort_keys_benchmark_generate<short>(benchmarks, stream, size);
}

void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                               hipStream_t stream,
                               size_t size)
{
    //using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    sort_keys_benchmark_generate<int, float>(benchmarks, stream, size);
    sort_keys_benchmark_generate<int, double>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<int, float2, 256U, 256U, 15>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<int, custom_float2, 256U, 256U, 15>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<int, double2, 256U, 256U, 15>(benchmarks, stream, size);
    sort_keys_benchmark_generate<int, custom_double2, 256U, 256U, 15>(benchmarks, stream, size);

    sort_keys_benchmark_generate<long long, float>(benchmarks, stream, size);
    sort_keys_benchmark_generate<long long, double>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<long long, float2, 256U, 256U, 15>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<long long, custom_float2, 256U, 256U, 15>(benchmarks, stream, size);
    //sort_keys_benchmark_generate<long long, double2, 256U, 256U, 15>(benchmarks, stream, size);
    sort_keys_benchmark_generate<long long, custom_double2, 256U, 256U, 15>(benchmarks, stream, size);
    sort_keys_benchmark_generate<int8_t, int8_t>(benchmarks, stream, size);
    sort_keys_benchmark_generate<uint8_t, uint8_t>(benchmarks, stream, size);
    sort_keys_benchmark_generate<rocprim::half, rocprim::half, 256U, 256U, 30>(benchmarks, stream, size);
}

#endif // BENCHMARK_CONFIG_TUNING


int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_sort_keys_benchmarks(benchmarks, stream, size);
    add_sort_pairs_benchmarks(benchmarks, stream, size);

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
