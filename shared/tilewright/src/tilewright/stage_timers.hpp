// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "tilewright/stage_timers.hpp"

#if TILEWRIGHT_STAGE_TIMERS

#include <chrono>
#include <cstdint>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define TILEWRIGHT_STAGE_TSC 1
#include <x86intrin.h>
#else
#define TILEWRIGHT_STAGE_TSC 0
#endif

namespace tilewright {
namespace detail {

inline std::uint64_t stage_now() noexcept {
#if TILEWRIGHT_STAGE_TSC
  return __rdtsc();
#else
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
#endif
}

// Per thread: the totals and the time of the last phase boundary.
struct StageClock {
  StageTimes times;
  std::uint64_t last = 0;
};

inline StageClock& stage_clock() noexcept {
  thread_local StageClock c;
  return c;
}

inline void stage_begin() noexcept { stage_clock().last = stage_now(); }

inline void stage_lap(Stage stage) noexcept {
  StageClock& c         = stage_clock();
  const std::uint64_t t = stage_now();
  c.times.ticks[static_cast<std::size_t>(stage)] += t - c.last;
  c.last = t;
}

}  // namespace detail
}  // namespace tilewright

#define TILEWRIGHT_STAGE_BEGIN() ::tilewright::detail::stage_begin()
#define TILEWRIGHT_STAGE_LAP(stage) ::tilewright::detail::stage_lap(::tilewright::Stage::stage)
#define TILEWRIGHT_STAGE_END(stage)                              \
  do {                                                           \
    ::tilewright::detail::stage_lap(::tilewright::Stage::stage); \
    ++::tilewright::detail::stage_clock().times.calls;           \
  } while (false)

#else

#define TILEWRIGHT_STAGE_BEGIN() ((void)0)
#define TILEWRIGHT_STAGE_LAP(stage) ((void)0)
#define TILEWRIGHT_STAGE_END(stage) ((void)0)

#endif
