// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

template <typename T>
struct inclusive_scan_benchmark : public primbench::benchmark_interface
{
  inclusive_scan_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "inclusive_scan")
      .add("subalgo", "sum")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed);
    thrust::device_vector<T> out(m_items);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::inclusive_scan(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
    executor.queue<inclusive_scan_benchmark<T>>(size);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 200;
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
