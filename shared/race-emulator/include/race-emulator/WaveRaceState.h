// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "CommonRegister.h"
#include "IntervalSet.h"
#include "Profiler.h"
#include "Types.h"
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace raceemulator {

class RaceDetector;

/// Per-wave race detection state. Owns VGPR event lists, wave-level event
/// queues, and provides event registration, waitcnt resolution, and VGPR
/// race checking. Holds a pointer to the shared RaceDetector for event
/// allocation and lifecycle transitions.
///
/// This class is self-contained: it has no dependency on Wave, Workgroup,
/// Emulator, or any instruction code.
class WaveRaceState {
public:
  WaveRaceState(int vgprCount, WaveId waveId, RaceDetector *detector);

  /// Register an in-flight memory event. execMask is passed explicitly (no
  /// dependency on Wave).
  void registerEvent(int pc, MemoryEventType type,
                     const std::vector<uint32_t> &registers, uint64_t execMask,
                     uint8_t byteMask = 0xF, IntervalSet ldsIntervals = {});

  /// Retire global memory events until at most vmcnt remain outstanding.
  void sWaitCntVmcnt(int vmcnt);

  /// Retire LDS events until at most lgkmcnt remain outstanding.
  void sWaitCntLgkmcnt(int lgkmcnt);

  /// Discard all wave-complete events (called when all waves reach barrier).
  void flushWaveCompleteMemoryEvents();

  /// Check a full VGPR read for races. Calls the RaceHandler on violation.
  void checkVgprRead(int reg, int lane, uint8_t byteMask) const;

  /// Check all lanes of a VGPR for races (used by getVgprs bulk read).
  void checkVgprReadAllLanes(int reg) const;

  /// True if any outstanding store reads from the given VGPR lane.
  bool isOutstandingFromVgpr(int lane, int reg) const;

  /// Number of outstanding events of a given type on a register.
  int getRegEventCount(MemoryEventType type, int reg) const {
    return regEventCount[static_cast<int>(type)][reg];
  }

  std::vector<EventId> &getVgprMemoryEvents(int reg) {
    return vgprMemoryEvents[reg];
  }

  const std::vector<EventId> &getWaveMemoryEvents() const {
    return waveMemoryEvents;
  }

  const std::vector<EventId> &getWaveCompleteMemoryEvents() const {
    return waveCompleteMemoryEvents;
  }

  RaceDetector *getDetector() { return detector; }
  const RaceDetector *getDetector() const { return detector; }

  void setProfiler(Profiler *p) { profiler = p; }

private:
  void retireEventRegisters(EventId);
  void resolveWaitCnt(int limit,
                      std::function<bool(MemoryEventType)> isTargetType);

  void regEventCountInc(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]++;
  }
  void regEventCountDec(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]--;
  }

  Profiler::ScopedStopwatch profileScope(std::string_view key);

  std::vector<std::vector<EventId>> vgprMemoryEvents;

  static constexpr int kNumEventTypes = static_cast<int>(MemoryEventType::N);
  std::array<std::vector<int>, kNumEventTypes> regEventCount;

  std::vector<EventId> waveMemoryEvents;
  std::vector<EventId> waveCompleteMemoryEvents;

  WaveId waveId;
  RaceDetector *detector;
  Profiler *profiler = nullptr;
};

} // namespace raceemulator
