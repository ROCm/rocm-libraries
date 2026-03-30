// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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

} // namespace raceemulator
