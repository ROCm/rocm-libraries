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

#include "benchmark_utils.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_run_length_decode.hpp>
#include <hipcub/block/block_store.hpp>

constexpr unsigned int Trials = 100;

template<class ItemT,
         class OffsetT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void block_run_length_decode_kernel(const ItemT*   d_run_items,
                                    const OffsetT* d_run_offsets,
                                    ItemT*         d_decoded_items,
                                    bool           enable_store = false)
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
         unsigned DecodedItemsPerThread>
class block_run_length_decode_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_run_length_decode")
            .add("lvl", "block")
            .add("item_type", primbench::name<ItemT>())
            .add("offset_type", primbench::name<OffsetT>())
            .add("min_run_length", MinRunLength)
            .add("max_run_length", MaxRunLength)
            .add("block_size", BlockSize)
            .add("runs_per_thread", RunsPerThread)
            .add("decoded_items_per_thread", DecodedItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr auto runs_per_block  = BlockSize * RunsPerThread;
        const auto     target_num_runs = 2 * input_items / (MinRunLength + MaxRunLength);
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

        state.set_items(Trials * output_length);
        state.add_writes<ItemT>(Trials * output_length);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(block_run_length_decode_kernel<ItemT,
                                                                   OffsetT,
                                                                   BlockSize,
                                                                   RunsPerThread,
                                                                   DecodedItemsPerThread>),
                    dim3(num_runs / runs_per_block),
                    dim3(BlockSize),
                    0,
                    stream,
                    d_run_items,
                    d_run_offsets,
                    d_output);
            });
    }
};

#define CREATE_BENCHMARK(IT, OT, MINRL, MAXRL, BS, RPT, DIPT) \
    executor.queue<block_run_length_decode_benchmark<IT, OT, MINRL, MAXRL, BS, RPT, DIPT>>()

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARK(int, int, 1, 5, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 10, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 50, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 100, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 500, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 1000, 128, 2, 4);
    CREATE_BENCHMARK(int, int, 1, 5000, 128, 2, 4);

    CREATE_BENCHMARK(double, int64_t, 1, 5, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 10, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 50, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 100, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 500, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 1000, 128, 2, 4);
    CREATE_BENCHMARK(double, int64_t, 1, 5000, 128, 2, 4);

    executor.run();
}
