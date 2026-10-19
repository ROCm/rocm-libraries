// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

template <class T>
struct square_t
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

template <typename T>
struct transform_reduce_benchmark : public primbench::benchmark_interface
{
  transform_reduce_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform_reduce")
      .add("subalgo", "sum")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(1);

    state.run([&] {
      thrust::transform_reduce(
        policy(alloc).on(state.stream), in.begin(), in.end(), square_t<T>{}, T{}, _THRUST_STD::plus<T>{});
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                    \
  for (size_t size : bench_utils::sizes(sizeof(T))) \
    executor.queue<transform_reduce_benchmark<T>>(size);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 5;
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
