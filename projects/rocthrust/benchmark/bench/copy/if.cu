/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <string>

#include "bench_utils.hpp"

PRIMBENCH_REGISTER_TYPE(bench_utils::large_data, "large_data")

template <class T>
struct less_then_t
{
  T m_val;

  __host__ __device__ bool operator()(const T& val) const
  {
    return val < m_val;
  }
};

struct make_large_data
{
  THRUST_HOST_DEVICE bench_utils::large_data operator()(int i) const
  {
    return bench_utils::large_data(i % 31 - 15);
  }
};

struct large_data_value
{
  THRUST_HOST_DEVICE int operator()(const bench_utils::large_data& value) const
  {
    return static_cast<int>(value);
  }
};

template <typename T>
struct copy_benchmark : public primbench::benchmark_interface
{
  copy_benchmark(size_t items, size_t entropy)
      : m_items(items)
      , entropy(entropy)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "copy")
      .add("subalgo", "if")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("entropy", std::to_string(bench_utils::get_entropy_percentage(entropy)));
  }

  void run(primbench::state& state) override
  {
    T val = bench_utils::value_from_entropy<T>(bench_utils::get_entropy_percentage(entropy));
    less_then_t<T> select_op{val};

    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed);

    const auto selected_elements = thrust::count_if(in.cbegin(), in.cend(), select_op);

    thrust::device_vector<T> out(selected_elements);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(selected_elements);

    state.test([&] {
      const auto count = std::min(m_items, bench_utils::validation_items);
      thrust::device_vector<T> replay_output(count);
      if constexpr (std::is_same_v<T, bench_utils::large_data>)
      {
        thrust::device_vector<T> replay_input(count);
        const thrust::counting_iterator<int> first(0);
        thrust::transform(policy(alloc).on(state.stream), first, first + count, replay_input.begin(), make_large_data{});

        const auto end = thrust::copy_if(
          policy(alloc).on(state.stream),
          replay_input.cbegin(),
          replay_input.cend(),
          replay_output.begin(),
          less_then_t<T>{T(0)});
        thrust::host_vector<int> expected;
        for (size_t i = 0; i < count; ++i)
        {
          const int value = static_cast<int>(i % 31) - 15;
          if (value < 0)
          {
            expected.push_back(value);
          }
        }

        PRIMBENCH_ASSERT(static_cast<size_t>(end - replay_output.begin()), expected.size());
        thrust::device_vector<int> values(expected.size());
        thrust::transform(
          policy(alloc).on(state.stream),
          replay_output.cbegin(),
          replay_output.cbegin() + expected.size(),
          values.begin(),
          large_data_value{});
        HIP_CHECK(hipStreamSynchronize(state.stream));
        bench_utils::check_output(values, expected);
      }
      else
      {
        const auto input                            = bench_utils::read_range(in, 0, count);
        const thrust::device_vector<T> replay_input = input;
        const auto end                              = thrust::copy_if(
          policy(alloc).on(state.stream), replay_input.cbegin(), replay_input.cend(), replay_output.begin(), select_op);
        HIP_CHECK(hipStreamSynchronize(state.stream));
        thrust::host_vector<T> expected;
        std::copy_if(input.begin(), input.end(), std::back_inserter(expected), select_op);
        PRIMBENCH_ASSERT(static_cast<size_t>(end - replay_output.begin()), expected.size());
        bench_utils::check_output(replay_output, expected);
      }
    });

    state.run([&] {
      thrust::copy_if(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin(), select_op);
    });
  }

private:
  size_t m_items;
  size_t entropy;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
  {                                                     \
    executor.queue<copy_benchmark<T>>(size, 0);         \
    executor.queue<copy_benchmark<T>>(size, 2);         \
    executor.queue<copy_benchmark<T>>(size, 4200);      \
  }

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(int8_t)
  QUEUE(int16_t)
  QUEUE(int32_t)
  QUEUE(int64_t)

  QUEUE(float)
  QUEUE(double)

  QUEUE(bench_utils::large_data)

  executor.run();
}
