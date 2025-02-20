
#include "benchmark_utils.hpp"
#include <rocprim/device/config_types.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

enum class stream_kind
{
    default_stream,
    per_thread_stream,
    explicit_stream,
    async_stream
};

template<stream_kind StreamKind>
static void BM_host_target_arch(benchmark::State& gbench_state, benchmark_utils::state& state)
{
    const hipStream_t stream = []() -> hipStream_t
    {
        hipStream_t stream = 0;
        switch(StreamKind)
        {
            case stream_kind::default_stream: return stream;
            case stream_kind::per_thread_stream: return hipStreamPerThread;
            case stream_kind::explicit_stream: HIP_CHECK(hipStreamCreate(&stream)); return stream;
            case stream_kind::async_stream:
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                return stream;
        }
    }();

    state.run(gbench_state,
              [&]
              {
                  rocprim::detail::target_arch target_arch;
                  HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
                  benchmark::DoNotOptimize(target_arch);
              });

    state.set_items_processed_per_iteration<char>(gbench_state, 1);

    if(StreamKind != stream_kind::default_stream && StreamKind != stream_kind::per_thread_stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

__global__
void empty_kernel()
{}

// An empty kernel launch for baseline
static void BM_kernel_launch(benchmark::State& gbench_state, benchmark_utils::state& state)
{
    const auto& stream = state.stream;

    state.run(gbench_state,
              [&]
              {
                  empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
                  HIP_CHECK(hipGetLastError());
              });

    state.set_items_processed_per_iteration<char>(gbench_state, 1);
}

#define CREATE_BENCHMARK(ST, SK)                                                       \
    executor.queue_fn(                                                                 \
        bench_naming::format_name("{lvl:na,algo:" #ST ",cfg:default_config}").c_str(), \
        BM_host_target_arch<SK>);

int main(int argc, char** argv)
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0, true, 100);

    CREATE_BENCHMARK(default_stream, stream_kind::default_stream)
    CREATE_BENCHMARK(per_thread_stream, stream_kind::per_thread_stream)
    CREATE_BENCHMARK(explicit_stream, stream_kind::explicit_stream)
    CREATE_BENCHMARK(async_stream, stream_kind::async_stream)

    executor.queue_fn(
        bench_naming::format_name("{lvl:na,algo:empty_kernel,cfg:default_config}").c_str(),
        BM_kernel_launch);

    executor.run();
}
