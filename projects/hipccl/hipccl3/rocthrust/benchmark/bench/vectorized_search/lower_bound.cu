// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

template <typename T>
struct vectorized_search_benchmark : public primbench::benchmark_interface
{
  vectorized_search_benchmark(size_t items, const size_t needles_ratio)
      : m_items(items)
      , needles_ratio(needles_ratio)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "vectorized_search")
      .add("subalgo", "lower_bound")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("needles_ratio", needles_ratio);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    const auto needles = needles_ratio * static_cast<std::size_t>(static_cast<double>(m_items) / 100.0);

    thrust::device_vector<T> data = bench_utils::generate(m_items + needles, state.seed);
    thrust::device_vector<bool> result(needles);
    thrust::sort(data.begin(), data.begin() + m_items);

    state.set_items(needles);
    state.add_reads<T>(needles);

    state.run([&] {
      thrust::lower_bound(
        policy(alloc).on(state.stream),
        data.begin(),
        data.begin() + m_items,
        data.begin() + m_items,
        data.end(),
        result.begin());
    });
  }

private:
  size_t m_items;
  const size_t needles_ratio;
};

#define QUEUE(T, N)                                     \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
    executor.queue<vectorized_search_benchmark<T>>(size, N);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 200;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  const size_t needles[] = {1, 25, 50};

  for (size_t n : needles)
  {
    QUEUE(int8_t, n)
    QUEUE(int16_t, n)
    QUEUE(int32_t, n)
    QUEUE(int64_t, n)
  }

  executor.run();
}
