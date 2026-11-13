// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

template <typename T, typename K>
struct inclusive_scan_benchmark : public primbench::benchmark_interface
{
  inclusive_scan_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "inclusive_scan")
      .add("subalgo", "by_key")
      .add("value_type", primbench::name<T>())
      .add("key_type", primbench::name<K>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    // Generate input
    thrust::device_vector<K> keys =
      bench_utils::generate.uniform.key_segments(m_items, state.seed, 0, 5200 /*magic numbers in thrust*/);
    thrust::device_vector<T> in_vals(m_items);

    thrust::device_vector<T> out_vals(m_items);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_reads<K>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::inclusive_scan_by_key(
        policy(alloc).on(state.stream), keys.cbegin(), keys.cend(), in_vals.cbegin(), out_vals.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE(K, T)                                                 \
  for (size_t size : bench_utils::sizes(2 * sizeof(T) + sizeof(K))) \
    executor.queue<inclusive_scan_benchmark<T, K>>(size);

#ifndef _MSC_VER
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)  \
    QUEUE(K, int128_t)
#else
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)
#endif

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 10;
  settings.min_gpu_ms_per_batch = 100;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

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
