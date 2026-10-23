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

#include <hipcub/warp/warp_load.hpp>

#include <type_traits>

constexpr const char* getAlgorithmName(hipcub::WarpLoadAlgorithm algorithm)
{
    switch(algorithm)
    {
        case hipcub::WarpLoadAlgorithm::WARP_LOAD_DIRECT: return "warp_load_direct";
        case hipcub::WarpLoadAlgorithm::WARP_LOAD_STRIPED: return "warp_load_striped";
        case hipcub::WarpLoadAlgorithm::WARP_LOAD_VECTORIZE: return "warp_load_vectorize";
        case hipcub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE: return "warp_load_transpose";
    }

    return nullptr;
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__
auto warp_load_device_fn(T* d_input, T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using WarpLoadT = ::hipcub::WarpLoad<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int      tile_size      = ItemsPerThread * LogicalWarpSize;

    const unsigned warp_id = threadIdx.x / LogicalWarpSize;
    const unsigned global_warp_id = blockIdx.x * warps_in_block + warp_id;
    __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
    T                                          thread_data[ItemsPerThread];

    WarpLoadT(temp_storage[warp_id]).Load(d_input + global_warp_id * tile_size, thread_data);

#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned striped_global_idx
            = BlockSize * ItemsPerThread * blockIdx.x + BlockSize * i + threadIdx.x;
        d_output[striped_global_idx] = thread_data[i];
    }
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__
auto warp_load_device_fn(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize)
void warp_load_kernel(T* d_input, T* d_output)
{
    warp_load_device_fn<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input, d_output);
}

template<class T,
         unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm>
struct warp_load_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("data_type", primbench::name<T>())
                        .add("block_size", BlockSize)
                        .add("items_per_thread", ItemsPerThread)
                        .add("warp_size", LogicalWarpSize)
                        .add("algo", "warp_load")
                        .add("lvl", "warp")
                        .add("subalgo", getAlgorithmName(Algorithm));

        return json;
    }

    void run(primbench::state& state) override
    {
        const auto& input_items = state.size;
        const auto& stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(10));
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                warp_load_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                    d_output);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, BS, IT, WS, ALG) \
    executor.queue<warp_load_benchmark<T, BS, IT, WS, ALG>>()

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Add benchmarks
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 4, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 8, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 16, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(int, 256, 32, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(int, 256, 64, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 4, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 8, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_VECTORIZE);
    CREATE_BENCHMARK(double, 256, 16, 32, ::hipcub::WARP_LOAD_TRANSPOSE);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_VECTORIZE);

    // WARP_LOAD_TRANSPOSE removed because of shared memory limit
    // CREATE_BENCHMARK(double, 256, 32, 32, ::hipcub::WARP_LOAD_TRANSPOSE);

    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_DIRECT);
    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_STRIPED);
    CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_VECTORIZE);

    // WARP_LOAD_TRANSPOSE removed because of shared memory limit
    // CREATE_BENCHMARK(double, 256, 64, 32, ::hipcub::WARP_LOAD_TRANSPOSE)

    if(::benchmark_utils::is_warp_size_supported(64))
    {
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 4, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 8, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 16, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(int, 256, 32, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(int, 256, 64, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 4, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_VECTORIZE);
        CREATE_BENCHMARK(double, 256, 8, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_VECTORIZE);

        // WARP_LOAD_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 16, 64, ::hipcub::WARP_LOAD_TRANSPOSE);

        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_VECTORIZE);

        // WARP_LOAD_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 32, 64, ::hipcub::WARP_LOAD_TRANSPOSE);

        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_DIRECT);
        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_STRIPED);
        CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_VECTORIZE);

        // WARP_LOAD_TRANSPOSE removed because of shared memory limit
        // CREATE_BENCHMARK(double, 256, 64, 64, ::hipcub::WARP_LOAD_TRANSPOSE);
    }

    // Run benchmarks
    executor.run();
}
