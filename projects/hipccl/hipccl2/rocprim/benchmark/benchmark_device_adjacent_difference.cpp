#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

#include "cmdparser.hpp"

#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_adjacent_difference.hpp>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{

#ifndef DEFAULT_N
constexpr std::size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

constexpr unsigned int batch_size  = 10;
constexpr unsigned int warmup_size = 5;

template <typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template <typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template <typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_inplace(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
}

template <typename Config = rocprim::default_config,
          typename InputIt,
          typename OutputIt,
          typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right_inplace<Config>(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
}

template <typename T, bool left, bool in_place>
void run_benchmark(benchmark::State& state, const std::size_t size, const hipStream_t stream)
{
    using output_type = T;

    static constexpr bool debug_synchronous = false;

    // Generate data
    const std::vector<T> input = get_random_data<T>(size, 1, 100);

    T*           d_input;
    output_type* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(
        hipMemcpy(d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

    if(!in_place)
    {
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(output_type)));
    }

    static constexpr auto left_tag     = rocprim::detail::bool_constant<left> {};
    static constexpr auto in_place_tag = rocprim::detail::bool_constant<in_place> {};

    // Allocate temporary storage
    std::size_t temp_storage_size;
    void*       d_temp_storage = nullptr;

    const auto launch = [&] {
        return dispatch_adjacent_difference(left_tag,
                                            in_place_tag,
                                            d_temp_storage,
                                            temp_storage_size,
                                            d_input,
                                            d_output,
                                            size,
                                            rocprim::plus<> {},
                                            stream,
                                            debug_synchronous);
    };
    HIP_CHECK(launch());
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(launch());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Run
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(launch());
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    if(!in_place)
    {
        hipFree(d_output);
    }
    hipFree(d_temp_storage);
}

} // namespace

using namespace std::string_literals;

#define CREATE_BENCHMARK(T, left, in_place)                                        \
    benchmark::RegisterBenchmark(("adjacent_difference" + (left ? ""s : "_right"s) \
                                  + (in_place ? "_inplace"s : ""s) + "<" #T ">")   \
                                     .c_str(),                                     \
                                 run_benchmark<T, left, in_place>,                 \
                                 size,                                             \
                                 stream)

// clang-format off
#define CREATE_BENCHMARKS(T)           \
    CREATE_BENCHMARK(T, true,  false), \
    CREATE_BENCHMARK(T, true,  true),  \
    CREATE_BENCHMARK(T, false, false), \
    CREATE_BENCHMARK(T, false, true)
// clang-format on

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
    const hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    const std::vector<benchmark::internal::Benchmark*> benchmarks = {
        CREATE_BENCHMARKS(int),
        CREATE_BENCHMARKS(std::int64_t),

        CREATE_BENCHMARKS(uint8_t),
        CREATE_BENCHMARKS(rocprim::half),

        CREATE_BENCHMARKS(float),
        CREATE_BENCHMARKS(double),

        CREATE_BENCHMARKS(custom_float2),
        CREATE_BENCHMARKS(custom_double2),
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
