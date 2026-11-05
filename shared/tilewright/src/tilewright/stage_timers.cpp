// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "stage_timers.hpp"

#include <chrono>

namespace tilewright {

const char* stage_name(Stage stage) noexcept {
  switch (stage) {
    case Stage::Setup: return "setup";
    case Stage::Feasibility: return "feasibility";
    case Stage::QueryTower: return "query_tower";
    case Stage::Stage1: return "stage1_embedding";
    case Stage::TopW: return "top_w";
    case Stage::Launch: return "launch_plan";
    case Stage::InterFeatures: return "inter_features";
    case Stage::FullScore: return "full_score";
    case Stage::Results: return "results";
    case Stage::Count: break;
  }
  return "";
}

#if TILEWRIGHT_STAGE_TIMERS

bool stage_timers_enabled() noexcept { return true; }

StageTimes stage_times() noexcept { return detail::stage_clock().times; }

void reset_stage_times() noexcept { detail::stage_clock().times = StageTimes{}; }

double stage_tick_ns() noexcept {
  static const double ns = [] {
    using clock            = std::chrono::steady_clock;
    const auto t0          = clock::now();
    const std::uint64_t c0 = detail::stage_now();
    while (clock::now() - t0 < std::chrono::milliseconds(50)) {}
    const auto t1          = clock::now();
    const std::uint64_t c1 = detail::stage_now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(c1 - c0);
  }();
  return ns;
}

#else

bool stage_timers_enabled() noexcept { return false; }

StageTimes stage_times() noexcept { return StageTimes{}; }

void reset_stage_times() noexcept {}

double stage_tick_ns() noexcept { return 0.0; }

#endif

}  // namespace tilewright
