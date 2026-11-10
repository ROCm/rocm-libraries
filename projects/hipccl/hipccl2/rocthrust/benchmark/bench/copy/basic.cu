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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// Non-trivially-copyable/relocatable type which is not allowed to be copied using std::memcpy or cudaMemcpy
struct non_trivial
{
  int a;
  int b;

  non_trivial() = default;

  THRUST_HOST_DEVICE explicit non_trivial(int i)
      : a(i)
      , b(i)
  {}

  // the user-defined copy constructor prevents the type from being trivially copyable
  THRUST_HOST_DEVICE non_trivial(const non_trivial& nt)
      : a(nt.a)
      , b(nt.b)
  {}

  non_trivial& operator=(const non_trivial&) = default;
};

static_assert(!_THRUST_STD::is_trivially_copyable<non_trivial>::value, ""); // as required by the C++ standard
static_assert(!thrust::is_trivially_relocatable<non_trivial>::value, ""); // thrust uses this check internally

PRIMBENCH_REGISTER_TYPE(non_trivial, "non_trivial")

template <typename T>
struct copy_benchmark : public primbench::benchmark_interface
{
  copy_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "copy")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> in(m_items, T{1});
    thrust::device_vector<T> out(m_items);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::copy(policy(alloc).on(state.stream), in.cbegin(), in.cend(), out.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
    executor.queue<copy_benchmark<T>>(size);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 75;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(int8_t)
  QUEUE(int16_t)
  QUEUE(int32_t)
  QUEUE(int64_t)

  QUEUE(uint8_t)
  QUEUE(uint16_t)
  QUEUE(uint32_t)
  QUEUE(uint64_t)

  QUEUE(float)
  QUEUE(double)

  QUEUE(non_trivial)

  executor.run();
}
