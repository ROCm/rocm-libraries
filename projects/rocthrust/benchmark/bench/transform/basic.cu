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
#include <thrust/detail/functional/address_stability.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <cstdint>
#include <string>

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

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// CCCL 3.0 has moved 'proclaim_copyable_arguments' to libcudacxx.
#  include _THRUST_LIBCXX_INCLUDE(functional)
// TODO: realize instances of this macros once the CCCL 2.x branch is removed.
#  define _THRUST_PROCLAIM_COPYABLE_ARGUMENTS _THRUST_PROCLAIM_COPYABLE_ARGUMENTS
#else
// TODO: remove this branch once we have remove CCCL 2.x support as this include will be removed!
#  include <thrust/functional.h>
#  define _THRUST_PROCLAIM_COPYABLE_ARGUMENTS thrust::detail::proclaim_copyable_arguments
#endif

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
struct transform_benchmark : public primbench::benchmark_interface
{
  transform_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform")
      .add("subalgo", "basic")
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
    executor.queue<transform_benchmark<T>>(size);       \
  }

// babelstream: BabelStream-inspired transform benchmarks
// https://github.com/UoB-HPC/BabelStream/blob/main/src/thrust/ThrustStream.cu

namespace babelstream
{
// Modified from BabelStream to also work for integers
constexpr auto startA      = 1;
constexpr auto startB      = 2;
constexpr auto startC      = 3;
constexpr auto startScalar = 4;

struct mul
{
  static constexpr const char* name       = "mul";
  static constexpr size_t reads_per_item  = 1;
  static constexpr size_t writes_per_item = 1;

  template <typename Policy, typename T>
  static void run(Policy policy, thrust::device_vector<T>&, thrust::device_vector<T>& b, thrust::device_vector<T>& c)
  {
    const T scalar = startScalar;
    thrust::transform(
      policy, c.begin(), c.end(), b.begin(), _THRUST_PROCLAIM_COPYABLE_ARGUMENTS([=] __device__(const T& ci) {
        return ci * scalar;
      }));
  }
};

struct add
{
  static constexpr const char* name       = "add";
  static constexpr size_t reads_per_item  = 2;
  static constexpr size_t writes_per_item = 1;

  template <typename Policy, typename T>
  static void run(Policy policy, thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& c)
  {
    thrust::transform(
      policy,
      a.begin(),
      a.end(),
      b.begin(),
      c.begin(),
      _THRUST_PROCLAIM_COPYABLE_ARGUMENTS([] __device__(const T& ai, const T& bi) -> T {
        return ai + bi;
      }));
  }
};

struct triad
{
  static constexpr const char* name       = "triad";
  static constexpr size_t reads_per_item  = 2;
  static constexpr size_t writes_per_item = 1;

  template <typename Policy, typename T>
  static void run(Policy policy, thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& c)
  {
    const T scalar = startScalar;
    thrust::transform(
      policy,
      b.begin(),
      b.end(),
      c.begin(),
      a.begin(),
      _THRUST_PROCLAIM_COPYABLE_ARGUMENTS([=] __device__(const T& bi, const T& ci) {
        return bi + scalar * ci;
      }));
  }
};

struct nstream
{
  static constexpr const char* name       = "nstream";
  static constexpr size_t reads_per_item  = 3;
  static constexpr size_t writes_per_item = 1;

  template <typename Policy, typename T>
  static void run(Policy policy, thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& c)
  {
    const T scalar = startScalar;
    thrust::transform(
      policy,
      thrust::make_zip_iterator(a.begin(), b.begin(), c.begin()),
      thrust::make_zip_iterator(a.end(), b.end(), c.end()),
      a.begin(),
      thrust::make_zip_function(
        _THRUST_PROCLAIM_COPYABLE_ARGUMENTS([=] __device__(const T& ai, const T& bi, const T& ci) {
          return ai + bi + scalar * ci;
        })));
  }
};

struct nstream_stable
{
  static constexpr const char* name       = "nstream_stable";
  static constexpr size_t reads_per_item  = 3;
  static constexpr size_t writes_per_item = 1;

  template <typename Policy, typename T>
  static void run(Policy policy, thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& c)
  {
    const T* a_start = thrust::raw_pointer_cast(a.data());
    const T* b_start = thrust::raw_pointer_cast(b.data());
    const T* c_start = thrust::raw_pointer_cast(c.data());
    const T scalar   = startScalar;
    thrust::transform(policy, a.begin(), a.end(), a.begin(), [=] __device__(const T& ai) {
      const auto i = &ai - a_start;
      return ai + b_start[i] + scalar * c_start[i];
    });
  }
};
} // namespace babelstream

template <typename T, class OpT>
struct transform_babel_benchmark : public primbench::benchmark_interface
{
  transform_babel_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "transform")
      .add("subalgo", std::string("babelstream.") + OpT::name)
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> a(m_items);
    thrust::device_vector<T> b(m_items);
    thrust::device_vector<T> c(m_items);

    thrust::fill(a.begin(), a.end(), T{babelstream::startA});
    thrust::fill(b.begin(), b.end(), T{babelstream::startB});
    thrust::fill(c.begin(), c.end(), T{babelstream::startC});

    state.set_items(m_items);
    state.add_reads<T>(m_items * OpT::reads_per_item);
    state.add_writes<T>(m_items * OpT::writes_per_item);

    state.run([&] {
      OpT::run(policy(alloc).on(state.stream), a, b, c);
    });
  }

private:
  size_t m_items;
};

#define QUEUE_BABEL_OP(T, S, OpT)                                        \
  {                                                                      \
    if (is_valid<T>(S))                                                  \
      executor.queue<transform_babel_benchmark<T, babelstream::OpT>>(S); \
  }

#define QUEUE_BABEL_SIZE(T, S)  \
  QUEUE_BABEL_OP(T, S, mul)     \
  QUEUE_BABEL_OP(T, S, add)     \
  QUEUE_BABEL_OP(T, S, triad)   \
  QUEUE_BABEL_OP(T, S, nstream) \
  QUEUE_BABEL_OP(T, S, nstream_stable)

#define QUEUE_BABEL(T)          \
  QUEUE_BABEL_SIZE(T, 1u << 25) \
  QUEUE_BABEL_SIZE(T, 1u << 31)

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 300;
  settings.batch_window_size    = 5;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(uint32_t)
  QUEUE(uint64_t)

  QUEUE_BABEL(int8_t)
  QUEUE_BABEL(int16_t)
  QUEUE_BABEL(float)
  QUEUE_BABEL(double)
#ifndef _MSC_VER
  QUEUE_BABEL(int128_t)
#endif

  executor.run();
}
