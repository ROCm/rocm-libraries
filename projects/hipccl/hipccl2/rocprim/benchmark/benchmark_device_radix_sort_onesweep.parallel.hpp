// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

namespace rp = rocprim;

constexpr const char* radix_rank_algorithm_name(rp::block_radix_rank_algorithm algorithm)
{
    switch(algorithm)
    {
        case rp::block_radix_rank_algorithm::basic: return "block_radix_rank_algorithm::basic";
        case rp::block_radix_rank_algorithm::basic_memoize:
            return "block_radix_rank_algorithm::basic_memoize";
        case rp::block_radix_rank_algorithm::match: return "block_radix_rank_algorithm::match";
    }
}

template<typename Config>
std::string config_name()
{
    constexpr rocprim::detail::radix_sort_onesweep_config_params params = Config();
    return "{histogram:{bs:" + std::to_string(params.histogram.block_size)
           + ",ipt:" + std::to_string(params.histogram.items_per_thread) + "},sort:{"
           + "bs:" + std::to_string(params.sort.block_size)
           + ",ipt:" + std::to_string(params.sort.items_per_thread)
           + "},bits_per_place:" + std::to_string(params.radix_bits_per_place)
           + ",algorithm:" + radix_rank_algorithm_name(params.radix_rank_algorithm) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_onesweep_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:radix_sort_onesweep,key_type:"
                                         + std::string(Traits<Key>::name())
                                         + ",value_type:" + std::string(Traits<Value>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    static std::vector<Key> generate_keys(size_t size)
    {
        using key_type = Key;

        if(std::is_floating_point<key_type>::value)
        {
            return get_random_data<key_type>(size,
                                             static_cast<key_type>(-1000),
                                             static_cast<key_type>(1000),
                                             size);
        }
        else
        {
            return get_random_data<key_type>(size,
                                             std::numeric_limits<key_type>::min(),
                                             std::numeric_limits<key_type>::max(),
                                             size);
        }
    }

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        auto keys_input = generate_keys(size);

        using key_type = Key;

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        bool                 is_result_in_output = true;
        rocprim::empty_type* d_values_ptr        = nullptr;
        HIP_CHECK((rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                       temporary_storage_bytes,
                                                                       d_keys_input,
                                                                       nullptr,
                                                                       d_keys_output,
                                                                       d_values_ptr,
                                                                       nullptr,
                                                                       d_values_ptr,
                                                                       size,
                                                                       is_result_in_output,
                                                                       0,
                                                                       sizeof(key_type) * 8,
                                                                       stream,
                                                                       false)));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK((rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                           temporary_storage_bytes,
                                                                           d_keys_input,
                                                                           nullptr,
                                                                           d_keys_output,
                                                                           d_values_ptr,
                                                                           nullptr,
                                                                           d_values_ptr,
                                                                           size,
                                                                           is_result_in_output,
                                                                           0,
                                                                           sizeof(key_type) * 8,
                                                                           stream,
                                                                           false)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(
                    (rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys_input,
                                                                         nullptr,
                                                                         d_keys_output,
                                                                         d_values_ptr,
                                                                         nullptr,
                                                                         d_values_ptr,
                                                                         size,
                                                                         is_result_in_output,
                                                                         0,
                                                                         sizeof(key_type) * 8,
                                                                         stream,
                                                                         false)));
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        auto keys_input = generate_keys(size);

        using key_type   = Key;
        using value_type = Value;

        std::vector<value_type> values_input(size);
        for(size_t i = 0; i < size; i++)
        {
            values_input[i] = value_type(i);
        }

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_output), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        bool is_result_in_output = true;
        HIP_CHECK((rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                       temporary_storage_bytes,
                                                                       d_keys_input,
                                                                       nullptr,
                                                                       d_keys_output,
                                                                       d_values_input,
                                                                       nullptr,
                                                                       d_values_output,
                                                                       size,
                                                                       is_result_in_output,
                                                                       0,
                                                                       sizeof(key_type) * 8,
                                                                       stream,
                                                                       false)));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK((rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                           temporary_storage_bytes,
                                                                           d_keys_input,
                                                                           nullptr,
                                                                           d_keys_output,
                                                                           d_values_input,
                                                                           nullptr,
                                                                           d_values_output,
                                                                           size,
                                                                           is_result_in_output,
                                                                           0,
                                                                           sizeof(key_type) * 8,
                                                                           stream,
                                                                           false)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(
                    (rp::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys_input,
                                                                         nullptr,
                                                                         d_keys_output,
                                                                         d_values_input,
                                                                         nullptr,
                                                                         d_values_output,
                                                                         size,
                                                                         is_result_in_output,
                                                                         0,
                                                                         sizeof(key_type) * 8,
                                                                         stream,
                                                                         false)));
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

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(key_type) + sizeof(value_type)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        do_run(state, size, stream);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<unsigned int BlockSize,
         unsigned int RadixBits,
         typename Key,
         typename Value = rocprim::empty_type>
struct device_radix_sort_onesweep_benchmark_generator
{
    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    static constexpr bool is_buildable()
    {
        using sharedmem_storage =
            typename rp::detail::onesweep_iteration_helper<Key,
                                                           Value,
                                                           size_t,
                                                           BlockSize,
                                                           ItemsPerThread,
                                                           RadixBits,
                                                           false,
                                                           RadixRankAlgorithm>::storage_type;
        return sizeof(sharedmem_storage) < TUNING_SHARED_MEMORY_MAX;
    }

    template<unsigned int                        ItemsPerThread,
             rocprim::block_radix_rank_algorithm RadixRankAlgorithm,
             typename Enable = void>
    struct create_ipt;

    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    struct create_ipt<ItemsPerThread,
                      RadixRankAlgorithm,
                      std::enable_if_t<(is_buildable<ItemsPerThread, RadixRankAlgorithm>())>>
    {
        using generated_config = rocprim::detail::radix_sort_onesweep_config<
            rocprim::kernel_config<BlockSize, ItemsPerThread>,
            rocprim::kernel_config<BlockSize, ItemsPerThread>,
            RadixBits,
            RadixRankAlgorithm>;
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<
                    device_radix_sort_onesweep_benchmark<Key, Value, generated_config>>());
        }
    };

    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    struct create_ipt<ItemsPerThread,
                      RadixRankAlgorithm,
                      std::enable_if_t<(!is_buildable<ItemsPerThread, RadixRankAlgorithm>())>>
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>&) {}
    };

    template<rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    static void create_algo(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        create_ipt<1u, RadixRankAlgorithm>()(storage);
        create_ipt<4u, RadixRankAlgorithm>()(storage);
        create_ipt<6u, RadixRankAlgorithm>()(storage);
        create_ipt<8u, RadixRankAlgorithm>()(storage);
        create_ipt<12u, RadixRankAlgorithm>()(storage);
        create_ipt<16u, RadixRankAlgorithm>()(storage);
        create_ipt<18u, RadixRankAlgorithm>()(storage);
        create_ipt<22u, RadixRankAlgorithm>()(storage);
    }

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        create_algo<rocprim::block_radix_rank_algorithm::basic>(storage);
        create_algo<rocprim::block_radix_rank_algorithm::match>(storage);
    }
};

#else // BENCHMARK_CONFIG_TUNING

    #define CREATE_RADIX_SORT_BENCHMARK(...)                                  \
        {                                                                     \
            const device_radix_sort_onesweep_benchmark<__VA_ARGS__> instance; \
            REGISTER_BENCHMARK(benchmarks, size, stream, instance);           \
        }

inline void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                     hipStream_t                                   stream,
                                     size_t                                        size)
{
    CREATE_RADIX_SORT_BENCHMARK(int)
    CREATE_RADIX_SORT_BENCHMARK(float)
    CREATE_RADIX_SORT_BENCHMARK(long long)
    CREATE_RADIX_SORT_BENCHMARK(int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(short)
}

inline void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                      hipStream_t                                   stream,
                                      size_t                                        size)
{
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    CREATE_RADIX_SORT_BENCHMARK(int, float)
    CREATE_RADIX_SORT_BENCHMARK(int, double)
    CREATE_RADIX_SORT_BENCHMARK(int, float2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(int, double2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_double2)

    CREATE_RADIX_SORT_BENCHMARK(long long, float)
    CREATE_RADIX_SORT_BENCHMARK(long long, double)
    CREATE_RADIX_SORT_BENCHMARK(long long, float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, double2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_double2)
    CREATE_RADIX_SORT_BENCHMARK(int8_t, int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t, uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half, rocprim::half)
}

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_
