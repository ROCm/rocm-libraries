// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_

#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

#include "../common/utils_custom_type.hpp"

// gbench
#include <benchmark/benchmark.h>

// HIP
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_search_n.hpp>
#include <rocprim/functional.hpp>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/types.hpp>
#endif

using custom_int2            = common::custom_type<int>;
using custom_double2         = common::custom_type<double>;
using custom_longlong_double = common::custom_type<long long, double>;

namespace
{
template<typename First, typename... Types>
struct type_arr
{
    using type = First;
    using next = type_arr<Types...>;
};
template<typename First>
struct type_arr<First>
{
    using type = First;
};
template<typename...>
using void_type = void;
template<typename T, typename = void>
constexpr bool is_type_arr_end = true;
template<typename T>
constexpr bool is_type_arr_end<T, void_type<typename T::next>> = false;

template<typename Config>
std::string search_n_config_name()
{
    const rocprim::detail::search_n_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",threshold:" + std::to_string(config.threshold) + "}";
}

#ifndef BENCHMARK_CONFIG_TUNING
template<>
std::string search_n_config_name<rocprim::default_config>()
{
    return "default_config";
}
#endif

template<size_t Value>
struct count_equal_to
{
    std::string name() const
    {
        return "count_equal_to<" + std::to_string(Value) + ">";
    }
    constexpr size_t resolve(size_t) const
    {
        return Value;
    }
};

template<size_t Value>
struct count_is_percent_of_size
{
    std::string name() const
    {
        return "count_is_percent_of_size<" + std::to_string(Value) + ">";
    }
    constexpr size_t resolve(size_t size) const
    {
        return size * Value / 100;
    }
};

} // namespace

template<class InputType,
         class OutputType,
         class CountCalculator,
         class Config = rocprim::default_config>
class benchmark_search_n : public config_autotune_interface
{

private:
    static constexpr size_t warmup_size = 10;
    static constexpr size_t batch_size  = 10;

public:
    void run(benchmark::State& state,
             size_t            size_byte,
             const managed_seed&, // not used
             hipStream_t stream) const override
    {
        InputType              h_noise{0};
        InputType              h_value{1};
        void*                  d_temp_storage    = nullptr;
        size_t                 temp_storage_size = 0;
        size_t                 size;
        size_t                 count;
        std::vector<InputType> input{};
        InputType*             d_input;
        OutputType*            d_output;
        InputType*             d_value;
        hipEvent_t             start;
        hipEvent_t             stop;

        size = size_byte / sizeof(InputType);

        count            = CountCalculator{}.resolve(size);
        size_t cur_tile  = 0;
        size_t last_tile = size / count - 1;
        input            = std::vector<InputType>(size, h_value);
        while(cur_tile != last_tile)
        {
            input[cur_tile * count + count - 1] = h_noise;
            ++cur_tile;
        }

        HIP_CHECK(hipMallocAsync(&d_value, sizeof(InputType), stream));
        HIP_CHECK(hipMallocAsync(&d_input, sizeof(InputType) * input.size(), stream));
        HIP_CHECK(hipMallocAsync(&d_output, sizeof(OutputType), stream));
        HIP_CHECK(
            hipMemcpyAsync(d_value, &h_value, sizeof(InputType), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_input,
                                 input.data(),
                                 sizeof(InputType) * input.size(),
                                 hipMemcpyHostToDevice,
                                 stream));

        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        auto launch_search_n = [&]()
        {
            HIP_CHECK(::rocprim::search_n<Config>(d_temp_storage,
                                                  temp_storage_size,
                                                  d_input,
                                                  d_output,
                                                  size,
                                                  count,
                                                  d_value,
                                                  rocprim::equal_to<InputType>{},
                                                  stream,
                                                  false));
        };

        // allocate temp memory
        launch_search_n();
        HIP_CHECK(hipMallocAsync(&d_temp_storage, temp_storage_size, stream));
        // Warm-up
        for(size_t i = 0; i < warmup_size; ++i)
        {
            launch_search_n();
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        // Run
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; ++i)
            {
                launch_search_n();
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Clean-up
        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(*(d_input)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_value));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        d_temp_storage    = nullptr;
        temp_storage_size = 0;
        decltype(input) tmp;
        input.swap(tmp); // clear input memspace
    }

    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:search_n,data_type:"
                                         + std::string(Traits<InputType>::name())
                                         + ",count_calculator:" + CountCalculator{}.name()
                                         + ",cfg:" + search_n_config_name<Config>() + "}")
            .c_str();
    }

    benchmark::internal::Benchmark* bench_register(const size_t       size_byte,
                                                   const managed_seed seed,
                                                   const hipStream_t  stream) const noexcept
    {
        return benchmark::RegisterBenchmark(name(),
                                            [this, size_byte, seed, stream](benchmark::State& state)
                                            { this->run(state, size_byte, seed, stream); });
    }
};

#ifndef BENCHMARK_CONFIG_TUNING
using destructor_t = std::function<void(void)>;
static std::vector<destructor_t> destructors;

static void clean_up_benchmarks_search_n()
{
    for(auto& i : destructors)
    {
        i();
    }
    destructors = {};
}

template<typename T>
inline void add_one_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                       const managed_seed                            seed,
                                       const hipStream_t                             stream,
                                       const size_t                                  size_byte)
{
    // normal
    auto equal_1
        = new benchmark_search_n<T, size_t, count_equal_to<1>>(); // benchmark for search_n case 1
    auto equal_6 = new benchmark_search_n<
        T,
        size_t,
        count_equal_to<6>>; // this benchmark is a selector for threshold 4 -> 8
    auto equal_10 = new benchmark_search_n<
        T,
        size_t,
        count_equal_to<10>>; // this benchmark is a selector for threshold 8 -> 12
    auto equal_14 = new benchmark_search_n<
        T,
        size_t,
        count_equal_to<14>>; // this benchmark is a selector for threshold 12 -> 16
    auto equal_25
        = new benchmark_search_n<T, size_t, count_equal_to<25>>; // benchmark for search_n case 2
    auto percent_50
        = new benchmark_search_n<T,
                                 size_t,
                                 count_is_percent_of_size<50>>(); // benchmark for search_n case 2
    auto percent_100
        = new benchmark_search_n<T,
                                 size_t,
                                 count_is_percent_of_size<100>>(); // benchmark for search_n case 2

    std::vector<benchmark::internal::Benchmark*> bs
        = {equal_1->bench_register(size_byte, seed, stream),
           equal_6->bench_register(size_byte, seed, stream),
           equal_10->bench_register(size_byte, seed, stream),
           equal_14->bench_register(size_byte, seed, stream),
           equal_25->bench_register(size_byte, seed, stream),
           percent_50->bench_register(size_byte, seed, stream),
           percent_100->bench_register(size_byte, seed, stream)};

    destructors.emplace_back(
        [=]()
        {
            delete equal_1;
            delete equal_6;
            delete equal_10;
            delete equal_14;
            delete equal_25;
            delete percent_50;
            delete percent_100;
        });

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

template<typename T, std::enable_if_t<!is_type_arr_end<T>, bool> = true>
inline void add_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                   const managed_seed                            seed,
                                   const hipStream_t                             stream,
                                   const size_t                                  size_byte)
{
    add_one_benchmark_search_n<typename T::type>(benchmarks, seed, stream, size_byte);
    add_benchmark_search_n<typename T::next>(benchmarks, seed, stream, size_byte);
}
template<typename T, std::enable_if_t<is_type_arr_end<T>, bool> = true>
inline void add_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                   const managed_seed                            seed,
                                   const hipStream_t                             stream,
                                   const size_t                                  size_byte)
{
    add_one_benchmark_search_n<typename T::type>(benchmarks, seed, stream, size_byte);
}

using benchmark_search_n_types = type_arr<custom_int2,
                                          custom_longlong_double,
                                          int8_t,
                                          int16_t,
                                          int32_t,
                                          int64_t,
                                          rocprim::int128_t,
                                          rocprim::uint128_t,
                                          rocprim::half,
                                          float,
                                          double>;

#else // BENCHMARK_CONFIG_TUNING

template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, size_t Threshold>
struct device_search_n_benchmark_generator
{
    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        using config = rocprim::search_n_config<BlockSize, ItemsPerThread, Threshold>;
        storage.emplace_back(
            std::make_unique<benchmark_search_n<T, size_t, count_equal_to<1>, config>>());
        storage.emplace_back(
            std::make_unique<benchmark_search_n<T, size_t, count_equal_to<6>, config>>());
        storage.emplace_back(
            std::make_unique<benchmark_search_n<T, size_t, count_equal_to<10>, config>>());
        storage.emplace_back(
            std::make_unique<benchmark_search_n<T, size_t, count_equal_to<14>, config>>());
        storage.emplace_back(
            std::make_unique<benchmark_search_n<T, size_t, count_equal_to<25>, config>>());
        storage.emplace_back(
            std::make_unique<
                benchmark_search_n<T, size_t, count_is_percent_of_size<50>, config>>());
        storage.emplace_back(
            std::make_unique<
                benchmark_search_n<T, size_t, count_is_percent_of_size<100>, config>>());
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_
