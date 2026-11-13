// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

template <typename T>
struct unique_benchmark : public primbench::benchmark_interface
{
  unique_benchmark(size_t items, const size_t max_segment_size)
      : m_items(items)
      , max_segment_size(max_segment_size)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "unique")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("max_segment_size", max_segment_size);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    constexpr std::size_t min_segment_size = 1;
    thrust::device_vector<T> in =
      bench_utils::generate.uniform.key_segments(m_items, state.seed, min_segment_size, max_segment_size);

    thrust::device_vector<T> out(m_items);

    // not a warm-up run, we need to run once to determine the size of the output
    const auto new_end = thrust::unique_copy(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin());
    const size_t unique_items = _THRUST_STD::distance(out.begin(), new_end);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(unique_items);

    state.run([&] {
      thrust::unique_copy(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin());
    });
  }

private:
  size_t m_items;
  const size_t max_segment_size;
};

#define QUEUE(T, M)                                     \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
    executor.queue<unique_benchmark<T>>(size, M);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 100;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  constexpr size_t max_segment_sizes[] = {1, 4, 8};

  for (size_t m : max_segment_sizes)
  {
    QUEUE(int8_t, m)
    QUEUE(int16_t, m)
    QUEUE(int32_t, m)
    QUEUE(int64_t, m)

#ifndef _MSC_VER
    QUEUE(int128_t, m)
#endif

    QUEUE(float, m)
    QUEUE(double, m)
  }

  executor.run();
}
