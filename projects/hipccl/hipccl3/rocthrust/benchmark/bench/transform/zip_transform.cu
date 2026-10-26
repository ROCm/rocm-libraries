// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include _THRUST_LIBCXX_INCLUDE(__functional/address_stability.h)
#include _THRUST_LIBCXX_INCLUDE(iterator)

#include <cstdint>

// This benchmark is intended to be compared to the babelstream.nstream benchmark from basic.cu, so we can:
// * detect regressions in the unpacking of a zip_transform_iterator

// same variables as in basic.cu so we can compare results
constexpr auto startA      = 1; // BabelStream: 0.1
constexpr auto startB      = 2; // BabelStream: 0.2
constexpr auto startC      = 3; // BabelStream: 0.1
constexpr auto startScalar = 4; // BabelStream: 0.4

template <typename T>
inline bool is_valid(const size_t size)
{
  try
  {
    thrust::device_vector<T> a(size);
    thrust::device_vector<T> b(size);
    thrust::device_vector<T> c(size);
    return true;
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    return false;
  }
}

template <typename T>
struct nstream_zip_transform_benchmark : public primbench::benchmark_interface
{
  nstream_zip_transform_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform")
      .add("subalgo", "zip_transform.nstream")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{}; // transform shouldn't allocate, but let's be consistent
    thrust::detail::device_t policy{};

    thrust::device_vector<T> a(m_items, T{startA});
    thrust::device_vector<T> b(m_items, T{startB});
    thrust::device_vector<T> c(m_items, T{startC});

    state.set_items(m_items);
    state.add_reads<T>(3 * m_items);
    state.add_writes<T>(m_items);

    const T scalar = startScalar;
    auto lambda    = _THRUST_LIBCXX::proclaim_copyable_arguments(
      [=] THRUST_DEVICE(const T& ai, const T& bi, const T& ci) -> T {
        return ai + bi + scalar * ci;
      });
    _THRUST_LIBCXX::zip_transform_iterator begin{lambda, a.begin(), b.begin(), c.begin()};
    _THRUST_LIBCXX::zip_transform_iterator end{lambda, a.end(), b.end(), c.end()};

    state.run([&] {
      thrust::copy(policy(alloc).on(state.stream), begin, end, a.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE_SIZE(T, S)                                     \
  {                                                          \
    if (is_valid<T>(S))                                      \
      executor.queue<nstream_zip_transform_benchmark<T>>(S); \
  }

#define QUEUE(T)          \
  QUEUE_SIZE(T, 1u << 25) \
  QUEUE_SIZE(T, 1u << 31)

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // sizes are set explicitly by QUEUE.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 5;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(int8_t)
  QUEUE(int16_t)
  QUEUE(float)
  QUEUE(double)
#ifndef _MSC_VER
  QUEUE(int128_t)
#endif

  executor.run();
}
