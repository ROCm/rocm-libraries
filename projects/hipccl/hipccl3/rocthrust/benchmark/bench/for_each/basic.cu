// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include "bench_utils.hpp"

template <typename T>
struct for_each_benchmark : public primbench::benchmark_interface
{
  for_each_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "for_each")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    thrust::device_vector<T> in(m_items, T{1});

    auto op = [] __device__(T & x) {
      x = x * x;
    };

    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::for_each(policy(alloc).on(state.stream), in.begin(), in.end(), op);
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                    \
  for (size_t size : bench_utils::sizes(sizeof(T))) \
    executor.queue<for_each_benchmark<T>>(size);

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
