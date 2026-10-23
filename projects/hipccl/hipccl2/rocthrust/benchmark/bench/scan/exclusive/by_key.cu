/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

template <typename T, typename K>
struct exclusive_scan_benchmark : public primbench::benchmark_interface
{
  exclusive_scan_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "exclusive_scan")
      .add("subalgo", "by_key")
      .add("value_type", primbench::name<T>())
      .add("key_type", primbench::name<K>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    // Generate input
    thrust::device_vector<K> keys =
      bench_utils::generate.uniform.key_segments(m_items, state.seed, 0, 5200 /*magic numbers in thrust*/);
    thrust::device_vector<T> in_vals(m_items);

    thrust::device_vector<T> out_vals(m_items);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_reads<K>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::exclusive_scan_by_key(
        policy(alloc).on(state.stream), keys.cbegin(), keys.cend(), in_vals.cbegin(), out_vals.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE(K, T)                                                 \
  for (size_t size : bench_utils::sizes(2 * sizeof(T) + sizeof(K))) \
    executor.queue<exclusive_scan_benchmark<T, K>>(size);

#ifndef _MSC_VER
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)  \
    QUEUE(K, int128_t) \
    QUEUE(K, float)    \
    QUEUE(K, double)
#else
#  define QUEUE_KEY(K) \
    QUEUE(K, int8_t)   \
    QUEUE(K, int16_t)  \
    QUEUE(K, int32_t)  \
    QUEUE(K, int64_t)  \
    QUEUE(K, float)    \
    QUEUE(K, double)
#endif

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 100;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE_KEY(int8_t)
  QUEUE_KEY(int16_t)
  QUEUE_KEY(int32_t)
  QUEUE_KEY(int64_t)

#ifndef _MSC_VER
  QUEUE_KEY(int128_t)
#endif

  QUEUE_KEY(float)
  QUEUE_KEY(double)

  executor.run();
}
