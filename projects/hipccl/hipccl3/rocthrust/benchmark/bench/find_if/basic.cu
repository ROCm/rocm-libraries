// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/detail/internal_functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>

template <typename T>
struct find_if_benchmark : public primbench::benchmark_interface
{
  find_if_benchmark(size_t items, double common_prefix)
      : m_items(items)
      , common_prefix(common_prefix)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "find_if")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("mismatch_at", common_prefix);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    T val = 1;
    // set up input
    const auto mismatch_point = static_cast<std::size_t>(m_items * common_prefix);

    thrust::device_vector<T> in(m_items, thrust::no_init);
    thrust::fill(in.begin(), in.begin() + mismatch_point, T{0});
    thrust::fill(in.begin() + mismatch_point, in.end(), val);

    state.set_items(mismatch_point + 1);
    state.add_reads<T>(mismatch_point + 1);
    state.add_writes<size_t>(1);

    state.run([&] {
      thrust::find_if(
        policy(alloc).on(state.stream), in.begin(), in.end(), thrust::detail::equal_to_value<T>(val));
    });
  }

private:
  size_t m_items;
  double common_prefix;
};

#define QUEUE(T)                                     \
  for (size_t size : bench_utils::sizes(sizeof(T)))  \
  {                                                  \
    executor.queue<find_if_benchmark<T>>(size, 1.0); \
    executor.queue<find_if_benchmark<T>>(size, 0.5); \
    executor.queue<find_if_benchmark<T>>(size, 0.0); \
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

#ifndef _MSC_VER
  QUEUE(int128_t)
#endif

  QUEUE(float)
  QUEUE(double)

  executor.run();
}
