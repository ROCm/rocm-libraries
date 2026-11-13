// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>

template <typename T>
struct generator
{
  THRUST_DEVICE auto operator()() const -> T
  {
    return T{42};
  }
};

template <typename T>
struct generate_benchmark : public primbench::benchmark_interface
{
  generate_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "generate")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> output(m_items);

    state.set_items(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::generate(policy(alloc).on(state.stream), output.begin(), output.end(), generator<T>{});
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                    \
  for (size_t size : bench_utils::sizes(sizeof(T))) \
    executor.queue<generate_benchmark<T>>(size);

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
