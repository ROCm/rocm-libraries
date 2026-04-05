// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Util.h"
#include <algorithm>
#include <vector>

namespace raceemulator {

/// Identifies a wave (SIMD execution unit) within a workgroup. Wave 0 runs
/// lanes [0, waveSize), wave 1 runs [waveSize, 2*waveSize), and so on.
/// Strongly typed to prevent accidental mixing with lane indices, register
/// indices, or event IDs.
struct WaveId {
  int value;
  bool operator==(WaveId o) const { return value == o.value; }
  bool operator!=(WaveId o) const { return value != o.value; }
  bool operator<(WaveId o) const { return value < o.value; }
};

/// Number of lanes per wave (32 for RDNA, 64 for CDNA).
/// Implicitly convertible to int for use in arithmetic and loop bounds.
struct WaveSize {
  int value;
  operator int() const { return value; }
};

/// Identifies a memory event within a workgroup. Each memory instruction
/// (LDS read/write, global load/store) creates an event via
/// Workgroup::allocateEventId(). The event ID is used to track the event
/// through its lifecycle: registration, s_waitcnt completion, and barrier
/// retirement. Strongly typed to prevent accidental mixing with wave IDs,
/// register indices, or byte addresses.
struct EventId {
  int value;
  bool operator==(EventId o) const { return value == o.value; }
  bool operator!=(EventId o) const { return value != o.value; }
  bool operator<(EventId o) const { return value < o.value; }
};

static inline void removeFromUnorderedList(std::vector<EventId> &list,
                                           EventId eventId) {
  auto it = std::find(list.begin(), list.end(), eventId);
  if (it != list.end()) {
    std::swap(*it, list.back());
    list.pop_back();
  }
}

/// Status of a memory event in the race detection lifecycle.
enum class EventStatus {
  ACTIVE,       // Pending. Unsafe for everyone.
  WAVE_COMPLETE // s_waitcnt passed. Safe for owning wave, unsafe for others.
};

/// Controls the order in which waves execute instructions in the emulator.
/// Race detection is independent of the scheduling order: all schedules
/// detect the same races because event retirement only occurs when all waves
/// have reached a barrier, not when each wave individually arrives.
enum class WaveSchedule {
  Sequential, ///< Wave 0 runs to barrier/endpgm, then wave 1, etc.
  RoundRobin, ///< Rotate which wave runs after each instruction.
};

/// Per-run configuration for the emulator. Controls race detection,
/// instruction completeness, profiling, and wave scheduling. Passed as a
/// single value to Emulator::run(), so all options are visible at the call
/// site and the emulator is never in a partially-configured state.
struct RunConfig {
  /// Check for data races (LDS and VGPR hazards).
  bool raceChecks = false;
  /// When true, global memory loads copy real data into VGPRs. When false,
  /// loads are no-ops (useful for race-only checks without valid kernargs).
  bool completeEmulation = true;
  /// Enable per-instruction profiling. Zero overhead when false.
  bool profiling = false;
  /// Order in which waves execute instructions within a workgroup.
  WaveSchedule waveSchedule = WaveSchedule::Sequential;
};

/// Describes a detected race condition. Used by the race detection layer
/// to report violations without depending on any exception type.
struct RaceViolation {
  enum class Space { VGPR, SGPR, LDS };
  Space space;
  int index;    ///< Register index (VGPR/SGPR) or byte address (LDS).
  int wave;     ///< Wave that triggered the violation.
  int lane;     ///< Lane within the wave, or -1 for scalar.
  bool isWrite; ///< True if the violating access was a write.
  Dim3d workgroupId;
};

} // namespace raceemulator
