// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/config/namespace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <string>

#include "bench_utils.hpp"

PRIMBENCH_REGISTER_TYPE(bench_utils::large_data, "large_data")

template <typename T>
struct copy_benchmark : public primbench::benchmark_interface
{
  copy_benchmark(size_t items, size_t entropy)
      : m_items(items)
      , entropy(entropy)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "copy")
      .add("subalgo", "if")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("entropy", std::to_string(bench_utils::get_entropy_percentage(entropy)));
  }

  void run(primbench::state& state) override
  {
    const T val = bench_utils::lerp_min_max<T>(bench_utils::get_entropy_percentage(entropy));
    bench_utils::less_then_t<T> select_op{val};

    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed);

    const auto selected_elements = thrust::count_if(in.cbegin(), in.cend(), select_op);

    thrust::device_vector<T> out(selected_elements);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(selected_elements);

    state.run([&] {
      thrust::copy_if(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin(), select_op);
    });
  }

private:
  size_t m_items;
  size_t entropy;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
  {                                                     \
    executor.queue<copy_benchmark<T>>(size, 0);         \
    executor.queue<copy_benchmark<T>>(size, 2);         \
    executor.queue<copy_benchmark<T>>(size, 4200);      \
  }

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(int8_t)
  QUEUE(int16_t)
  QUEUE(int32_t)
  QUEUE(int64_t)

  QUEUE(float)
  QUEUE(double)

  QUEUE(bench_utils::large_data)

  executor.run();
}
