// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/detail/config/namespace.h>
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

    const size_t unique_keys =
      _THRUST_STD::distance(out_keys.begin(), thrust::unique(out_keys.begin(), out_keys.end()));
    thrust::device_vector<T> out_vals;
    out_vals = thrust::device_vector<T>(unique_keys);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_reads<K>(m_items);
    state.add_writes<T>(unique_keys);
    state.add_writes<K>(unique_keys);

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
