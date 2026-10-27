// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace tilewright {

// Phases of one ranking call, timed per thread by builds with the CMake option
// TILEWRIGHT_STAGE_TIMERS (off by default). Other builds compile no timing
// into the ranking; the functions below then report nothing.
enum class Stage : std::uint8_t {
  Setup,          // routing, cell weights and pool caches, per-call state
  Feasibility,    // the feasibility rules over the pool's view keys
  QueryTower,     // query features and the query embedding
  Stage1,         // the cascade's embedding term over its tier's keys
  TopW,           // keeping the cascade's best keys
  Launch,         // LaunchModel::plan of the fully scored keys
  InterFeatures,  // interaction features of the fully scored keys
  FullScore,      // embedding term and interaction MLP of the fully scored keys
  Results,        // ordering the scored keys and writing the results
  Count
};

constexpr std::size_t kStageCount = static_cast<std::size_t>(Stage::Count);

const char* stage_name(Stage stage) noexcept;

struct StageTimes {
  std::uint64_t ticks[kStageCount] = {};
  std::uint64_t calls              = 0;  // ranking calls that reached their results
};

// Whether this build times stages.
bool stage_timers_enabled() noexcept;

// The calling thread's totals since its last reset_stage_times.
StageTimes stage_times() noexcept;
void reset_stage_times() noexcept;

// Nanoseconds per tick of StageTimes::ticks.
double stage_tick_ns() noexcept;

}  // namespace tilewright
