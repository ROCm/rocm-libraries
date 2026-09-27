// MIT License
//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/warp/warp_store.hpp>

#include <type_traits>

constexpr const char* get_algorithm_name(hipcub::WarpStoreAlgorithm algorithm)
{
    switch(algorithm)
    {
        case hipcub::WarpStoreAlgorithm::WARP_STORE_DIRECT: return "warp_store_direct";
        case hipcub::WarpStoreAlgorithm::WARP_STORE_STRIPED: return "warp_store_striped";
        case hipcub::WarpStoreAlgorithm::WARP_STORE_VECTORIZE: return "warp_store_vectorize";
        case hipcub::WarpStoreAlgorithm::WARP_STORE_TRANSPOSE: return "warp_store_transpose";
    }

    return "unknown_algorithm";
}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__
auto warp_store_benchmark_device_fn(T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T thread_data[ItemsPerThread];
#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = static_cast<T>(i);
    }

    using WarpStoreT = ::hipcub::WarpStore<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned                          warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int                               tile_size      = ItemsPerThread * LogicalWarpSize;
    __shared__
    typename WarpStoreT::TempStorage            temp_storage[warps_in_block];
    const unsigned                              warp_id = threadIdx.x / LogicalWarpSize;
    const unsigned global_warp_id                       = blockIdx.x * warps_in_block + warp_id;

    WarpStoreT(temp_storage[warp_id]).Store(d_output + global_warp_id * tile_size, thread_data);
}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__
auto warp_store_benchmark_device_fn(T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize)
void warp_store_kernel(T* d_output)
{
    warp_store_benchmark_device_fn<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_output);
}

template<class T,
         unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm>
class warp_store_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "warp_store")
            .add("subalgo", get_algorithm_name(Algorithm))
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread)
            .add("warp_size", LogicalWarpSize)
            .add("lvl", "warp");
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
        const unsigned     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        T* d_output;
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                warp_store_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_output);
            });

        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, BS, IT, WS, ALG) \
    executor.queue<warp_store_benchmark<T, BS, IT, WS, ALG>>()

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_STORE_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_STORE_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_STORE_VECTORIZE);

    // WARP_STORE_TRANSPOSE removed because of shared memory limit
    // CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_STORE_TRANSPOSE);

    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_STORE_DIRECT);
    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_STORE_STRIPED);
    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_STORE_VECTORIZE);

    // WARP_STORE_TRANSPOSE removed because of shared memory limit
    // CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_STORE_TRANSPOSE);

    if(::benchmark_utils::is_warp_size_supported(64))
    {
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_STORE_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_STORE_TRANSPOSE);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_STORE_VECTORIZE);

        // WARP_STORE_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_STORE_TRANSPOSE);

        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_STORE_VECTORIZE);

        // WARP_STORE_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_STORE_TRANSPOSE);

        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_STORE_DIRECT);
        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_STORE_STRIPED);
        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_STORE_VECTORIZE);

        // WARP_STORE_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_STORE_TRANSPOSE);
    }

    executor.run();
}
