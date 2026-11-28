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

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>

template <typename T, typename K>
struct reduce_benchmark : public primbench::benchmark_interface
{
  reduce_benchmark(size_t items, size_t max_segment_size)
      : m_items(items)
      , max_segment_size(max_segment_size)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "reduce")
      .add("subalgo", "by_key")
      .add("value_type", primbench::name<T>())
      .add("key_type", primbench::name<K>())
      .add("elements", m_items)
      .add("max_segment_size", max_segment_size);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    constexpr size_t min_segment_size = 1;

    // Generate input and output
    thrust::device_vector<K> in_keys;
    thrust::device_vector<K> out_keys;
    thrust::device_vector<T> in_vals;

    in_keys  = bench_utils::generate.uniform.key_segments(m_items, state.seed, min_segment_size, max_segment_size);
    out_keys = in_keys;
    in_vals  = thrust::device_vector<T>(m_items);

    const size_t unique_keys = thrust::distance(out_keys.begin(), thrust::unique(out_keys.begin(), out_keys.end()));
    thrust::device_vector<T> out_vals;
    out_vals = thrust::device_vector<T>(unique_keys);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_reads<K>(m_items);
    state.add_writes<T>(unique_keys);
    state.add_writes<K>(unique_keys);

    state.test([&] {
      const auto input = bench_utils::arithmetic_input<T>(m_items);
      thrust::host_vector<K> keys(input.size());
      for (size_t i = 0; i < keys.size(); ++i)
      {
        keys[i] = K(i / max_segment_size);
      }

      const thrust::device_vector<K> d_keys  = keys;
      const thrust::device_vector<T> d_input = input;
      thrust::device_vector<K> result_keys(input.size());
      thrust::device_vector<T> result_values(input.size());
      const auto ends = thrust::reduce_by_key(
        policy(alloc).on(state.stream),
        d_keys.cbegin(),
        d_keys.cend(),
        d_input.cbegin(),
        result_keys.begin(),
        result_values.begin());
      HIP_CHECK(hipStreamSynchronize(state.stream));

      thrust::host_vector<K> expected_keys;
      thrust::host_vector<T> expected_values;
      for (size_t i = 0; i < input.size(); ++i)
      {
        if (i == 0 || keys[i] != keys[i - 1])
        {
          expected_keys.push_back(keys[i]);
          expected_values.push_back(input[i]);
        }
        else
        {
          expected_values.back() = T(expected_values.back() + input[i]);
        }
      }

      PRIMBENCH_ASSERT(static_cast<size_t>(ends.first - result_keys.begin()), expected_keys.size());
      PRIMBENCH_ASSERT(static_cast<size_t>(ends.second - result_values.begin()), expected_values.size());
      bench_utils::check_output(result_keys, expected_keys);
      bench_utils::check_output(result_values, expected_values);
    });

    state.run([&] {
      thrust::reduce_by_key(
        policy(alloc).on(state.stream),
        in_keys.begin(),
        in_keys.end(),
        in_vals.begin(),
        out_keys.begin(),
        out_vals.begin());
    });
  }

private:
  size_t m_items;
  size_t max_segment_size;
};

#define QUEUE(K, T)                                                     \
  for (size_t max_seg : max_segment_sizes)                              \
  {                                                                     \
    for (size_t size : bench_utils::sizes(2 * (sizeof(T) + sizeof(K)))) \
    {                                                                   \
      executor.queue<reduce_benchmark<T, K>>(size, max_seg);            \
    }                                                                   \
  }

#ifndef _MSC_VER
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)  \
    QUEUE(K, int128_t) \
    QUEUE(K, float)    \
    QUEUE(K, double)
#else
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)  \
    QUEUE(K, float)    \
    QUEUE(K, double)
#endif

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 200;
  settings.batch_window_size    = 5;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  constexpr size_t max_segment_sizes[] = {1, 4, 8};

  QUEUE_KEY(int8_t)
  QUEUE_KEY(int16_t)
  QUEUE_KEY(int32_t)
  QUEUE_KEY(int64_t)

#ifndef _MSC_VER
  QUEUE_KEY(int128_t)
#endif

  QUEUE_KEY(float)
  QUEUE_KEY(double)

  executor.run();
}
