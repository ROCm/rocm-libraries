// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/count.h>
#include <thrust/detail/config/namespace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <string>

#include "bench_utils.hpp"

template <typename T>
struct negate_if_stencil_benchmark : public primbench::benchmark_interface
{
  negate_if_stencil_benchmark(size_t items, size_t entropy)
      : m_items(items)
      , entropy(entropy)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform")
      .add("subalgo", "if_stencil")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("entropy", std::to_string(bench_utils::get_entropy_percentage(entropy)));
  }

  void run(primbench::state& state) override
  {
    const T val       = bench_utils::lerp_min_max<T>(bench_utils::get_entropy_percentage(entropy));
    auto transform_op = thrust::negate<T>{};
    bench_utils::less_then_t<T> select_op{val};

    bench_utils::caching_allocator_t alloc{}; // transform_if shouldn't allocate, but let's be consistent
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in      = bench_utils::generate(m_items, state.seed);
    thrust::device_vector<T> stencil = bench_utils::generate(m_items, state.seed + 1);
    thrust::device_vector<T> out(m_items);

    const auto selected_elements = thrust::count_if(in.cbegin(), in.cend(), select_op);

    state.set_items(m_items);
    state.add_reads<T>(m_items + selected_elements);
    state.add_writes<T>(selected_elements);

    state.run([&] {
      thrust::transform_if(
        policy(alloc).on(state.stream), in.begin(), in.end(), stencil.begin(), out.begin(), transform_op, select_op);
    });
  }

private:
  size_t m_items;
  size_t entropy;
};

#define QUEUE(T)                                                \
  for (size_t size : bench_utils::sizes(3 * sizeof(T)))         \
  {                                                             \
    executor.queue<negate_if_stencil_benchmark<T>>(size, 0);    \
    executor.queue<negate_if_stencil_benchmark<T>>(size, 2);    \
    executor.queue<negate_if_stencil_benchmark<T>>(size, 4200); \
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

  executor.run();
}
