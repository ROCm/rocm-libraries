// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <string>

template <typename T>
struct shuffle_benchmark : public primbench::benchmark_interface
{
  shuffle_benchmark(size_t items, std::string rng_engine)
      : m_items(items)
      , rng_engine(rng_engine)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "shuffle")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("rng_engine", rng_engine)
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    thrust::device_vector<T> data(m_items);
    auto do_engine = [&](auto&& engine_constructor) {
      state.run([&] {
        thrust::shuffle(policy(alloc).on(state.stream), data.begin(), data.end(), engine_constructor());
      });
    };

    if (rng_engine == "minstd")
    {
      do_engine([] {
        return thrust::random::minstd_rand{};
      });
    }
    else if (rng_engine == "ranlux24")
    {
      do_engine([] {
        return thrust::random::ranlux24{};
      });
    }
    else if (rng_engine == "ranlux48")
    {
      do_engine([] {
        return thrust::random::ranlux48{};
      });
    }
    else
    {
      do_engine([] {
        return thrust::random::taus88{};
      });
    }
  }

private:
  size_t m_items;
  std::string rng_engine;
};

#define QUEUE(T)                                            \
  for (size_t size : bench_utils::sizes(sizeof(T)))         \
  {                                                         \
    executor.queue<shuffle_benchmark<T>>(size, "minstd");   \
    executor.queue<shuffle_benchmark<T>>(size, "ranlux24"); \
    executor.queue<shuffle_benchmark<T>>(size, "ranlux48"); \
    executor.queue<shuffle_benchmark<T>>(size, "taus88");   \
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

  executor.run();
}
