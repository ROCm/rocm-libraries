/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/transform.h>

#include <cstdint>

/*magic numbers used in Thrust*/
#define M0 0
#define M1 42

template <typename T>
inline bool is_valid(const size_t size)
{
  try
  {
    thrust::device_vector<T> in = bench_utils::generate(size, 123, M0, T{M0}, T{M1});
    thrust::device_vector<uint32_t> out(size);
    return true;
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    return false;
  }
}

template <class InT, class OutT>
struct fib_t
{
  __device__ OutT operator()(InT n)
  {
    OutT t1 = 0;
    OutT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    else if (n == 1)
    {
      return t1;
    }
    else if (n == 2)
    {
      return t2;
    }
    for (InT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }

    return t2;
  }
};

template <typename T>
struct fib_benchmark : public primbench::benchmark_interface
{
  fib_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform")
      .add("subalgo", "fib")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in = bench_utils::generate(m_items, state.seed, M0, T{M0}, T{M1});
    thrust::device_vector<uint32_t> out(m_items);

    fib_t<T, uint32_t> op{};

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<uint32_t>(m_items);

    state.run([&] {
      thrust::transform(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin(), op);
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
  {                                                     \
    if (!is_valid<T>(size))                             \
      continue;                                         \
    executor.queue<fib_benchmark<T>>(size);             \
  }

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 5;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(uint32_t)
  QUEUE(uint64_t)

  executor.run();
}
