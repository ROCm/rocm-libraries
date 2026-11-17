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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <string>

#include "bench_utils.hpp"

template <typename T>
struct sort_benchmark : public primbench::benchmark_interface
{
  sort_benchmark(size_t items, const int entropy_reduction)
      : m_items(items)
      , entropy_reduction(entropy_reduction)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "sort")
      .add("subalgo", "keys")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("entropy", std::to_string(bench_utils::get_entropy_percentage(entropy_reduction)));
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    const auto entropy           = bench_utils::get_entropy_percentage(entropy_reduction) / 100.0f;
    thrust::device_vector<T> in  = bench_utils::generate(m_items, state.seed, entropy);
    thrust::device_vector<T> ref = in;

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    state.run_before_every_iteration([&] {
      thrust::copy(policy.on(state.stream), ref.begin(), ref.end(), in.begin());
    });

    state.run([&] {
      thrust::sort(policy(alloc).on(state.stream), in.begin(), in.end());
    });
  }

private:
  size_t m_items;
  const int entropy_reduction;
};

#define QUEUE(T, E)                                 \
  for (size_t size : bench_utils::sizes(sizeof(T))) \
    executor.queue<sort_benchmark<T>>(size, E);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  constexpr int entropy_reductions[] = {0, 2, 6}; // 1.000, 0.544, 0.000;

  for (int e : entropy_reductions)
  {
    QUEUE(int8_t, e)
    QUEUE(int16_t, e)
    QUEUE(int32_t, e)
    QUEUE(int64_t, e)

#ifndef _MSC_VER
    QUEUE(int128_t, e)
#endif

    QUEUE(float, e)
    QUEUE(double, e)
  }

  executor.run();
}
