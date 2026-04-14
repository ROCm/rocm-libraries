// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/block/block_scan.hpp>
#include <hipcub/block/block_store.hpp>

enum memory_operation_method
{
    direct,
    striped,
    vectorize,
    transpose,
    warp_transpose
};

enum kernel_operation
{
    no_operation,
    block_scan,
    custom_operation,
    atomics_no_collision,
    atomics_inter_block_collision,
    atomics_inter_warp_collision,
};

struct empty_storage_type
{};

template<kernel_operation Operation,
         typename T,
         unsigned int ItemsPerThread,
         unsigned int BlockSize = 0>
struct operation;

// no operation
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<no_operation, T, ItemsPerThread, BlockSize>
{
    using storage_type = empty_storage_type;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& /*storage*/, T (&)[ItemsPerThread], T* = nullptr) const
    {}
};

// custom operation
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<custom_operation, T, ItemsPerThread, BlockSize>
{
    using storage_type = empty_storage_type;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& storage,
                   T (&input)[ItemsPerThread],
                   T* global_mem_output = nullptr) const
    {
        (void)storage;
        (void)global_mem_output;

#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            input[i]                       = input[i] + 666;
            constexpr unsigned int repeats = 30;
#pragma unroll
            for(unsigned int j = 0; j < repeats; j++)
            {
                input[i] = input[i] * (input[j % ItemsPerThread]);
            }
        }
    }
};

// block scan
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<block_scan, T, ItemsPerThread, BlockSize>
{
    using block_scan_type =
        typename hipcub::BlockScan<T, BlockSize, hipcub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;
    using storage_type = typename block_scan_type::TempStorage;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& storage,
                   T (&input)[ItemsPerThread],
                   T* global_mem_output = nullptr)
    {
        (void)global_mem_output;

        // sync before re-using shared memory from load
        __syncthreads();
        block_scan_type(storage).InclusiveScan(input, input, hipcub::Sum());
    }
};

// atomics_no_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_no_collision, T, ItemsPerThread, BlockSize>
{
    using storage_type = empty_storage_type;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& storage,
                   T (&input)[ItemsPerThread],
                   T* global_mem_output = nullptr)
    {
        (void)storage;
        (void)input;

        const unsigned int index
            = threadIdx.x * ItemsPerThread + blockIdx.x * blockDim.x * ItemsPerThread;
#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_warp_collision, T, ItemsPerThread, BlockSize>
{
    using storage_type = empty_storage_type;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& storage,
                   T (&input)[ItemsPerThread],
                   T* global_mem_output = nullptr)
    {
        (void)storage;
        (void)input;

        const unsigned int index
            = (threadIdx.x % warpSize) * ItemsPerThread + blockIdx.x * blockDim.x * ItemsPerThread;
#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_block_collision, T, ItemsPerThread, BlockSize>
{
    using storage_type = empty_storage_type;

    HIPCUB_DEVICE
    inline void
        operator()(storage_type& storage,
                   T (&input)[ItemsPerThread],
                   T* global_mem_output = nullptr)
    {
        (void)storage;
        (void)input;

        const unsigned int index = threadIdx.x * ItemsPerThread;
#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

template<memory_operation_method MemOp>
struct memory_operation
{};

template<>
struct memory_operation<direct>
{
    static constexpr hipcub::BlockLoadAlgorithm load_type
        = hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT;
    static constexpr hipcub::BlockStoreAlgorithm store_type
        = hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT;
};

template<>
struct memory_operation<striped>
{
    static constexpr hipcub::BlockLoadAlgorithm load_type
        = hipcub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED;
    static constexpr hipcub::BlockStoreAlgorithm store_type
        = hipcub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED;
};

template<>
struct memory_operation<vectorize>
{
    static constexpr hipcub::BlockLoadAlgorithm load_type
        = hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE;
    static constexpr hipcub::BlockStoreAlgorithm store_type
        = hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE;
};

template<>
struct memory_operation<transpose>
{
    static constexpr hipcub::BlockLoadAlgorithm load_type
        = hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE;
    static constexpr hipcub::BlockStoreAlgorithm store_type
        = hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE;
};

template<>
struct memory_operation<warp_transpose>
{
    static constexpr hipcub::BlockLoadAlgorithm load_type
        = hipcub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE;
    static constexpr hipcub::BlockStoreAlgorithm store_type
        = hipcub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE;
};

template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         typename CustomOp>
__global__ __launch_bounds__(BlockSize) void operation_kernel(T* input, T* output, CustomOp op)
{
    using mem_op     = memory_operation<MemOp>;
    using load_type  = hipcub::BlockLoad<T, BlockSize, ItemsPerThread, mem_op::load_type>;
    using store_type = hipcub::BlockStore<T, BlockSize, ItemsPerThread, mem_op::store_type>;

    __shared__ union
    {
        typename load_type::TempStorage  load;
        typename store_type::TempStorage store;
        typename CustomOp::storage_type  operand;
    } storage;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     offset          = blockIdx.x * items_per_block;

    T items[ItemsPerThread];
    load_type(storage.load).Load(input + offset, items);

    op(storage.operand, items, output);
    // sync before re-using shared memory from load or from operand
    __syncthreads();
    store_type(storage.store).Store(output + offset, items);
}

template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         kernel_operation        KernelOp = no_operation>
void run_benchmark(benchmark::State& state, size_t size, const hipStream_t stream)
{
    const size_t   grid_size = size / (BlockSize * ItemsPerThread);
    std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    operation<KernelOp, T, ItemsPerThread, BlockSize> selected_operation;

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(operation_kernel<T, BlockSize, ItemsPerThread, MemOp>),
                           dim3(grid_size),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output,
                           selected_operation);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(operation_kernel<T, BlockSize, ItemsPerThread, MemOp>),
                dim3(grid_size),
                dim3(BlockSize),
                0,
                stream,
                d_input,
                d_output,
                selected_operation);
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<typename T>
void run_benchmark_memcpy(benchmark::State& state, size_t size, const hipStream_t stream)
{
    // Allocate device buffers
    // Note: since this benchmark only tests memcpy performance between device
    // buffers, we don't really need to copy data into these from the host -
    // whatever happens to be in memory will suffice.
    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK_IPT(METHOD, OPERATION, T, SIZE, BS, IPT)                             \
    benchmarks.push_back(benchmark::RegisterBenchmark(                                        \
        std::string("device_memory<method:" #METHOD ",operation:" #OPERATION ",data_type:" #T \
                    ",size:" #SIZE ",block_size:" #BS ",items_per_thread:" #IPT ">.")         \
            .c_str(),                                                                         \
        [=](benchmark::State& state)                                                          \
        { run_benchmark<T, BS, IPT, METHOD, OPERATION>(state, SIZE, stream); }));

#define CREATE_BENCHMARK_MEMCPY(T, SIZE)                                               \
    benchmarks.push_back(benchmark::RegisterBenchmark(                                 \
        std::string("device_memory_memcpy<data_type:" #T ",size:" #SIZE ">.").c_str(), \
        [=](benchmark::State& state) { run_benchmark_memcpy<T>(state, SIZE, stream); }));

// clang-format off
#define CREATE_BENCHMARK_BLOCK_SIZE(MEM_OP, OP, TYPE, SIZE, BLOCK_SIZE) \
    CREATE_BENCHMARK_IPT(MEM_OP, OP, TYPE, SIZE, BLOCK_SIZE, 1)         \
    CREATE_BENCHMARK_IPT(MEM_OP, OP, TYPE, SIZE, BLOCK_SIZE, 2)         \
    CREATE_BENCHMARK_IPT(MEM_OP, OP, TYPE, SIZE, BLOCK_SIZE, 4)         \
    CREATE_BENCHMARK_IPT(MEM_OP, OP, TYPE, SIZE, BLOCK_SIZE, 8)

#define CREATE_BENCHMARK_MEM_OP(MEM_OP, OP, TYPE, SIZE) \
    CREATE_BENCHMARK_BLOCK_SIZE(MEM_OP, OP, TYPE, SIZE, 256)

#define CREATE_BENCHMARK(OP, TYPE, SIZE)               \
    CREATE_BENCHMARK_MEM_OP(direct, OP, TYPE, SIZE)    \
    CREATE_BENCHMARK_MEM_OP(striped, OP, TYPE, SIZE)   \
    CREATE_BENCHMARK_MEM_OP(vectorize, OP, TYPE, SIZE) \
    CREATE_BENCHMARK_MEM_OP(transpose, OP, TYPE, SIZE) \
    CREATE_BENCHMARK_MEM_OP(warp_transpose, OP, TYPE, SIZE)
// clang-format on

template<typename T>
constexpr unsigned int megabytes(unsigned int size)
{
    return (size * (1024 * 1024 / sizeof(T)));
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const int trials = parser.get<int>("trials");

    std::cout << "benchmark_device_memory" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;

    // Simple memory copy from device to device, not running a kernel
    CREATE_BENCHMARK_MEMCPY(int, megabytes<int>(128))

    // clang-format off
    CREATE_BENCHMARK(no_operation,                  int, megabytes<int>(128))
    CREATE_BENCHMARK(block_scan,                    int, megabytes<int>(128))
    CREATE_BENCHMARK(custom_operation,              int, megabytes<int>(128))
    CREATE_BENCHMARK(atomics_no_collision,          int, megabytes<int>(128))
    CREATE_BENCHMARK(atomics_inter_block_collision, int, megabytes<int>(128))
    CREATE_BENCHMARK(atomics_inter_warp_collision,  int, megabytes<int>(128))
    // clang-format on

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
