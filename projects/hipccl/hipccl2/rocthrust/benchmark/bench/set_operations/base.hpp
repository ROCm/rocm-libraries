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

#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <string>

#include "bench_utils.hpp"

template <typename T, class OpT>
struct base_set_benchmark : public primbench::benchmark_interface
{
  base_set_benchmark(size_t items, const char* algo_name, const int entropy_reduction, const size_t input_size_ratio)
      : m_items(items)
      , algo_name(algo_name)
      , entropy_reduction(entropy_reduction)
      , input_size_ratio(input_size_ratio)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", algo_name)
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items)
      .add("entropy", std::to_string(bench_utils::get_entropy_percentage(entropy_reduction)))
      .add("input_size_ratio", input_size_ratio);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    const auto entropy    = bench_utils::get_entropy_percentage(entropy_reduction) / 100.0f;
    const auto items_in_A = static_cast<std::size_t>(static_cast<double>(input_size_ratio * m_items) / 100.0f);

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed, entropy);
    thrust::device_vector<T> out(m_items);

    thrust::sort(in.begin(), in.begin() + items_in_A);
    thrust::sort(in.begin() + items_in_A, in.end());

    OpT op{};

    // not a warm-up run, we need to run once to determine the size of the output
    const auto result_ends =
      op(policy(alloc).on(state.stream),
         in.cbegin(),
         in.cbegin() + items_in_A,
         in.cbegin() + items_in_A,
         in.cend(),
         out.begin());

    const size_t items_in_AB = thrust::distance(out.begin(), result_ends);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(items_in_AB);

    state.run([&] {
      op(policy(alloc).on(state.stream),
         in.cbegin(),
         in.cbegin() + items_in_A,
         in.cbegin() + items_in_A,
         in.cend(),
         out.begin());
    });
  }

private:
  size_t m_items;
  const char* algo_name;
  const int entropy_reduction;
  const size_t input_size_ratio;
};

#define QUEUE(T, OpT, algo_name, entropy_reduction, input_size_ratio) \
  for (size_t size : bench_utils::sizes(2 * sizeof(T)))               \
    executor.queue<base_set_benchmark<T, OpT>>(size, algo_name, entropy_reduction, input_size_ratio);

template <class OpT>
void queue_benchmarks(const char* algo_name, const primbench::executor& executor)
{
  constexpr int entropy_reductions[] = {0, 4}; // 1.000, 0.201;
  constexpr int input_size_ratios[]  = {25, 50, 75};

  for (int e : entropy_reductions)
  {
    for (int i : input_size_ratios)
    {
      QUEUE(int8_t, OpT, algo_name, e, i);
      QUEUE(int16_t, OpT, algo_name, e, i);
      QUEUE(int32_t, OpT, algo_name, e, i);
      QUEUE(int64_t, OpT, algo_name, e, i);
    }
  }
}
