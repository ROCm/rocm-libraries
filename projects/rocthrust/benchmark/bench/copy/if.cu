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
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <string>

#include "bench_utils.hpp"

PRIMBENCH_REGISTER_TYPE(bench_utils::large_data, "large_data")

template <class T>
struct less_then_t
{
  T m_val;

  __host__ __device__ bool operator()(const T& val) const
  {
    return val < m_val;
  }
};

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
    T val = bench_utils::value_from_entropy<T>(bench_utils::get_entropy_percentage(entropy));
    less_then_t<T> select_op{val};

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
