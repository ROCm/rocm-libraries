
#include "benchmark_utils.hpp"
#include "cmdparser.hpp"
#include <rocprim/device/config_types.hpp>

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

#include <iostream>

#ifndef DEFAULT_N
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif


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
    HIP_CHECK(hipStreamSynchronize(stream));
}

#define CREATE_BENCHMARK(ST, SK)                \
    benchmark::RegisterBenchmark(               \
        bench_naming::format_name(              \
            "{lvl:na"                           \
            ",algo:" #ST                        \
            ",cfg:default_config}"              \
        ).c_str(),                              \
        &BM_host_target_arch,                   \
        SK                                      \
    )                                           \


int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", 100, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                    "name_format",
                                    "human",
                                    "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP

    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK(default_stream, stream_kind::default_stream),
        CREATE_BENCHMARK(per_thread_stream, stream_kind::per_thread_stream),
        CREATE_BENCHMARK(explicit_stream, stream_kind::explicit_stream),
        CREATE_BENCHMARK(async_stream, stream_kind::async_stream),
        benchmark::RegisterBenchmark(
            bench_naming::format_name("{lvl:na,algo:empty_kernel,cfg:default_config}").c_str(),
            BM_kernel_launch)};

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
