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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#include "bench_utils.hpp"

template <typename T, typename K>
inline bool is_valid(const size_t size, const size_t maxi)
{
  try
  {
    constexpr std::size_t min_segment_size = 1;
    thrust::device_vector<K> in_keys = bench_utils::generate.uniform.key_segments(size, 123, min_segment_size, maxi);
    thrust::device_vector<K> out_keys(size);
    thrust::device_vector<T> in_vals(size);
    thrust::device_vector<T> out_vals(size);
    return true;
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    return false;
  }
}

template <typename T, typename K>
struct unique_benchmark : public primbench::benchmark_interface
{
  unique_benchmark(size_t items, const size_t max_segment_size)
      : m_items(items)
      , max_segment_size(max_segment_size)
  {}

  primbench::json meta() const override
  {
    return primbench::json{}
      .add("algo", "unique")
      .add("subalgo", "by_key")
      .add("key_type", primbench::name<K>())
      .add("value_type", primbench::name<T>())
      .add("elements", m_items)
      .add("max_segment_size", max_segment_size);
  }

  void run(primbench::state& state) override
  {
    bench_utils::caching_allocator_t alloc{};
    thrust::detail::device_t policy{};

    constexpr std::size_t min_segment_size = 1;
    thrust::device_vector<K> in_keys =
      bench_utils::generate.uniform.key_segments(m_items, state.seed, min_segment_size, max_segment_size);

    thrust::device_vector<K> out_keys(m_items);
    thrust::device_vector<T> in_vals(m_items);
    thrust::device_vector<T> out_vals(m_items);

    // not a warm-up run, we need to run once to determine the size of the output
    const auto [new_key_end, new_val_end] = thrust::unique_by_key_copy(
      policy(alloc).on(state.stream),
      in_keys.cbegin(),
      in_keys.cend(),
      in_vals.cbegin(),
      out_keys.begin(),
      out_vals.begin());

    const size_t unique_elements = thrust::distance(out_keys.begin(), new_key_end);

    state.set_items(m_items);
    state.add_reads<T>(m_items);
    state.add_writes<K>(m_items);

    state.add_writes<T>(unique_elements);
    state.add_writes<K>(unique_elements);

    state.run([&] {
      thrust::unique_by_key_copy(
        policy(alloc).on(state.stream),
        in_keys.cbegin(),
        in_keys.cend(),
        in_vals.cbegin(),
        out_keys.begin(),
        out_vals.begin());
    });
  }

private:
  size_t m_items;
  const size_t max_segment_size;
};

#define QUEUE_KEY(K, T, M)                                            \
  for (size_t size : bench_utils::sizes(2 * (sizeof(T) + sizeof(K)))) \
  {                                                                   \
    if (is_valid<T, K>(size, M))                                      \
      executor.queue<unique_benchmark<T, K>>(size, M);                \
  }

#ifndef _MSC_VER
#  define QUEUE(K, M)          \
    QUEUE_KEY(K, int8_t, M)    \
    QUEUE_KEY(K, uint8_t, M)   \
    QUEUE_KEY(K, int16_t, M)   \
    QUEUE_KEY(K, uint16_t, M)  \
    QUEUE_KEY(K, int32_t, M)   \
    QUEUE_KEY(K, uint32_t, M)  \
    QUEUE_KEY(K, int64_t, M)   \
    QUEUE_KEY(K, uint64_t, M)  \
    QUEUE_KEY(K, int128_t, M)  \
    QUEUE_KEY(K, uint128_t, M) \
    QUEUE_KEY(K, float, M)     \
    QUEUE_KEY(K, double, M)
#else
#  define QUEUE(K, M)         \
    QUEUE_KEY(K, int8_t, M)   \
    QUEUE_KEY(K, uint8_t, M)  \
    QUEUE_KEY(K, int16_t, M)  \
    QUEUE_KEY(K, uint16_t, M) \
    QUEUE_KEY(K, int32_t, M)  \
    QUEUE_KEY(K, uint32_t, M) \
    QUEUE_KEY(K, int64_t, M)  \
    QUEUE_KEY(K, uint64_t, M) \
    QUEUE_KEY(K, float, M)    \
    QUEUE_KEY(K, double, M)
#endif

int main(int argc, char* argv[])
{
  primbench::settings settings;
  settings.size                 = 1; // bench_utils::sizes() calculates it later.
  settings.min_gpu_ms_per_batch = 150;
  settings.batch_window_size    = 3;
  primbench::executor executor(argc, argv, settings, primbench::flags::sync);

  constexpr size_t max_segment_sizes[] = {1, 8};

  for (size_t m : max_segment_sizes)
  {
    QUEUE(int8_t, m)
    QUEUE(int16_t, m)
    QUEUE(int32_t, m)
    QUEUE(int64_t, m)

#ifndef _MSC_VER
    QUEUE(int128_t, m)
#endif
  }

  executor.run();
}
