// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include "thrust/detail/raw_pointer_cast.h"

template <typename T>
struct sequence_benchmark : public primbench::benchmark_interface
{
  sequence_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "tabulate")
      .add("subalgo", "sequence")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> out(m_items, thrust::no_init);

    state.set_items(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      // sequence is implemented via thrust::tabulate
      thrust::sequence(policy(alloc).on(state.stream), out.begin(), out.end());
    });
  }

private:
  size_t m_items;
};

template <class T>
struct seg_size_t
{
  T* d_offsets{};

  template <class OffsetT>
  __device__ T operator()(OffsetT i)
  {
    return static_cast<T>(d_offsets[i + 1] - d_offsets[i]);
  }
};

template <typename T>
struct tabulate_benchmark : public primbench::benchmark_interface
{
  tabulate_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "tabulate")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in(m_items + 1);
    thrust::device_vector<T> out(m_items, thrust::no_init);

    seg_size_t<T> op{thrust::raw_pointer_cast(in.data())};

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items + 1);

    state.run([&] {
      thrust::tabulate(policy(alloc).on(state.stream), out.begin(), out.end(), op);
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
  {                                                     \
    executor.queue<sequence_benchmark<T>>(size);        \
    executor.queue<tabulate_benchmark<T>>(size);        \
  }

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

  executor.run();
}
