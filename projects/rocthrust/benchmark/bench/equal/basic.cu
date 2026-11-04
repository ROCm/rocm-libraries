// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

template <typename T>
struct equal_benchmark : public primbench::benchmark_interface
{
  equal_benchmark(size_t items, double common_prefix_ratio)
      : m_items(items)
      , common_prefix_ratio(common_prefix_ratio)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "equal")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("common_prefix_ratio", common_prefix_ratio);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in(m_items, T{1});
    thrust::device_vector<T> out(m_items, T{1});

    const auto same_elements = std::min(static_cast<size_t>(m_items * common_prefix_ratio), m_items);

    thrust::fill(policy(alloc).on(state.stream), out.begin() + same_elements, out.end(), T{2});

    state.set_items(std::max(same_elements, size_t(1)));
    state.add_reads<T>(std::max(same_elements, size_t(1)));
    state.add_writes<T>(std::max(same_elements, size_t(1)));

    state.run([&] {
      thrust::equal(policy(alloc).on(state.stream), in.begin(), in.end(), out.begin());
    });
  }

private:
  size_t m_items;
  double common_prefix_ratio;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
  {                                                     \
    executor.queue<equal_benchmark<T>>(size, 1.0);      \
    executor.queue<equal_benchmark<T>>(size, 0.5);      \
    executor.queue<equal_benchmark<T>>(size, 0.0);      \
  }

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 100;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(int8_t)
  QUEUE(int16_t)
  QUEUE(int32_t)
  QUEUE(int64_t)

  QUEUE(uint32_t)
  QUEUE(uint64_t)

  executor.run();
}
