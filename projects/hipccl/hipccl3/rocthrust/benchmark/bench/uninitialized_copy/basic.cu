// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark utils
#include "bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/uninitialized_copy.h>

#include <cstdint>
#include <type_traits>

// Not allowed to be copied using std::memcpy or hipMemcpy.
struct no_copy
{
  uint32_t a;

  no_copy() = default;

  THRUST_HOST_DEVICE no_copy(uint32_t i)
      : a(i)
  {}

  // the user-defined copy constructor prevents the type from being trivially copyable
  THRUST_HOST_DEVICE no_copy(const no_copy& nt)
      : a(nt.a)
  {}
};

static_assert(std::is_trivially_default_constructible_v<no_copy>);
static_assert(!std::is_trivially_copyable_v<no_copy>); // as required by the C++ standard
static_assert(!thrust::is_trivially_relocatable_v<no_copy>); // thrust uses this check internally

// Requires use of placement new
struct no_construct
{
  uint32_t a = 1337;
};

static_assert(!std::is_trivially_default_constructible_v<no_construct>);
static_assert(std::is_trivially_copyable_v<no_construct>); // as required by the C++ standard
static_assert(thrust::is_trivially_relocatable_v<no_construct>); // thrust uses this check internally

PRIMBENCH_REGISTER_TYPE(no_copy, "no_copy")
PRIMBENCH_REGISTER_TYPE(no_construct, "no_construct")

template <typename T>
struct uninitialized_copy_benchmark : public primbench::benchmark_interface
{
  uninitialized_copy_benchmark(size_t items)
      : m_items(items)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "uninitialized_copy")
      .add("subalgo", "basic")
      .add("input_type", primbench::name<T>())
      .add("elements", m_items);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    thrust::device_vector<T> input(m_items, T{0xAA});
    thrust::device_vector<T> output(m_items, thrust::default_init);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<T>(m_items);

    state.run([&] {
      thrust::uninitialized_copy(policy(alloc).on(state.stream), input.cbegin(), input.cend(), output.begin());
    });
  }

private:
  size_t m_items;
};

#define QUEUE(T)                                        \
  for (size_t size : bench_utils::sizes(2 * sizeof(T))) \
    executor.queue<uninitialized_copy_benchmark<T>>(size);

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 100;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  QUEUE(uint8_t)
  QUEUE(uint16_t)
  QUEUE(uint32_t)
  QUEUE(uint64_t)
  QUEUE(no_copy)
  QUEUE(no_construct)

  executor.run();
}
