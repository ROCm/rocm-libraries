#include <rocprim/device/config_types.hpp>

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

#include <iostream>

#include "benchmark_utils.hpp"

enum class stream_kind
{
    default_stream,
    per_thread_stream,
    explicit_stream,
    async_stream
};

static void BM_host_target_arch(benchmark::State& state, const stream_kind stream_kind)
{
    const hipStream_t stream = [stream_kind]() -> hipStream_t
    {
        hipStream_t stream = 0;
        switch(stream_kind)
        {
            case stream_kind::default_stream: return stream;
            case stream_kind::per_thread_stream: return hipStreamPerThread;
            case stream_kind::explicit_stream: HIP_CHECK(hipStreamCreate(&stream)); return stream;
            case stream_kind::async_stream:
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                return stream;
        }
    }();

    for(auto _ : state)
    {
        rocprim::detail::target_arch target_arch;
        HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
        benchmark::DoNotOptimize(target_arch);
    }

    if(stream_kind != stream_kind::default_stream && stream_kind != stream_kind::per_thread_stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

__global__ void empty_kernel() {}

// An empty kernel launch for baseline
static void BM_kernel_launch(benchmark::State& state)
{
    static constexpr hipStream_t stream = 0;

    for(auto _ : state)
    {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipGetLastError());
    }
    hipStreamSynchronize(stream);
}

BENCHMARK_CAPTURE(BM_host_target_arch, default_stream, stream_kind::default_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, per_thread_stream, stream_kind::per_thread_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, explicit_stream, stream_kind::explicit_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, async_stream, stream_kind::async_stream);
BENCHMARK(BM_kernel_launch);

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    add_common_benchmark_info();
    benchmark::RunSpecifiedBenchmarks();
}