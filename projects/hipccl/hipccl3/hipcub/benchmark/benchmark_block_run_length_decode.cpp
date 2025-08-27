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

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_run_length_decode.hpp>
#include <hipcub/block/block_store.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class ItemT,
         class OffsetT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread,
         unsigned Trials>
__global__
    __launch_bounds__(BlockSize) void block_run_length_decode_kernel(const ItemT*   d_run_items,
                                                                     const OffsetT* d_run_offsets,
                                                                     ItemT*         d_decoded_items,
                                                                     bool enable_store = false)
{
    using BlockRunLengthDecodeT
        = hipcub::BlockRunLengthDecode<ItemT, BlockSize, RunsPerThread, DecodedItemsPerThread>;

    ItemT   run_items[RunsPerThread];
    OffsetT run_offsets[RunsPerThread];

    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_items, run_items);
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_offsets, run_offsets);

    BlockRunLengthDecodeT block_run_length_decode(run_items, run_offsets);

    const OffsetT total_decoded_size
        = d_run_offsets[(hipBlockIdx_x + 1) * BlockSize * RunsPerThread]
          - d_run_offsets[hipBlockIdx_x * BlockSize * RunsPerThread];

#pragma nounroll
    for(unsigned i = 0; i < Trials; ++i)
    {
        OffsetT decoded_window_offset = 0;
        while(decoded_window_offset < total_decoded_size)
        {
            ItemT decoded_items[DecodedItemsPerThread];
            block_run_length_decode.RunLengthDecode(decoded_items, decoded_window_offset);

            if(enable_store)
            {
                hipcub::StoreDirectBlocked(global_thread_idx,
                                           d_decoded_items + decoded_window_offset,
                                           decoded_items);
            }

            decoded_window_offset += BlockSize * DecodedItemsPerThread;
        }
    }
}

template<class ItemT,
         class OffsetT,
         unsigned MinRunLength,
         unsigned MaxRunLength,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread,
         unsigned Trials = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr auto runs_per_block  = BlockSize * RunsPerThread;
    const auto     target_num_runs = 2 * N / (MinRunLength + MaxRunLength);
    const auto     num_runs
        = runs_per_block * ((target_num_runs + runs_per_block - 1) / runs_per_block);

    std::vector<ItemT>   run_items(num_runs);
    std::vector<OffsetT> run_offsets(num_runs + 1);

    std::default_random_engine prng(std::random_device{}());
    using ItemDistribution = std::conditional_t<std::is_integral<ItemT>::value,
                                                std::uniform_int_distribution<ItemT>,
                                                std::uniform_real_distribution<ItemT>>;
    ItemDistribution                       run_item_dist(0, 100);
    std::uniform_int_distribution<OffsetT> run_length_dist(MinRunLength, MaxRunLength);

    for(size_t i = 0; i < num_runs; ++i)
    {
        run_items[i] = run_item_dist(prng);
    }
    for(size_t i = 1; i < num_runs + 1; ++i)
    {
        const OffsetT next_run_length = run_length_dist(prng);
        run_offsets[i]                = run_offsets[i - 1] + next_run_length;
    }
    const OffsetT output_length = run_offsets.back();

    ItemT* d_run_items{};
    HIP_CHECK(hipMalloc(&d_run_items, run_items.size() * sizeof(ItemT)));
    HIP_CHECK(hipMemcpy(d_run_items,
                        run_items.data(),
                        run_items.size() * sizeof(ItemT),
                        hipMemcpyHostToDevice));

    OffsetT* d_run_offsets{};
    HIP_CHECK(hipMalloc(&d_run_offsets, run_offsets.size() * sizeof(OffsetT)));
    HIP_CHECK(hipMemcpy(d_run_offsets,
                        run_offsets.data(),
                        run_offsets.size() * sizeof(OffsetT),
                        hipMemcpyHostToDevice));

    ItemT* d_output{};
    HIP_CHECK(hipMalloc(&d_output, output_length * sizeof(ItemT)));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_run_length_decode_kernel<ItemT,
                                                                          OffsetT,
                                                                          BlockSize,
                                                                          RunsPerThread,
                                                                          DecodedItemsPerThread,
                                                                          Trials>),
                           dim3(num_runs / runs_per_block),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_run_items,
                           d_run_offsets,
                           d_output);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * output_length * sizeof(ItemT) * Trials);
    state.SetItemsProcessed(state.iterations() * output_length * Trials);

    HIP_CHECK(hipFree(d_run_items));
    HIP_CHECK(hipFree(d_run_offsets));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(IT, OT, MINRL, MAXRL, BS, RPT, DIPT)                               \
    benchmark::RegisterBenchmark(                                                           \
        std::string("block_run_length_decode<item_type:" #IT ",offset_type:" #OT            \
                    ",min_run_length:" #MINRL ",max_run_length:" #MAXRL ",block_size: " #BS \
                    ",runs_per_thread:" #RPT ",decoded_items_per_thread:" #DIPT ">.")       \
            .c_str(),                                                                       \
        &run_benchmark<IT, OT, MINRL, MAXRL, BS, RPT, DIPT>,                                \
        stream,                                                                             \
        size)

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

    std::cout << "benchmark_block_run_length_decode" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK(int, int, 1, 5, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 10, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 50, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 100, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 500, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 1000, 128, 2, 4),
        CREATE_BENCHMARK(int, int, 1, 5000, 128, 2, 4),

        CREATE_BENCHMARK(double, long long, 1, 5, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 10, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 50, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 100, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 500, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 1000, 128, 2, 4),
        CREATE_BENCHMARK(double, long long, 1, 5000, 128, 2, 4)};

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
