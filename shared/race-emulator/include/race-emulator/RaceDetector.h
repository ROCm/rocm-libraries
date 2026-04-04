// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "CommonRegister.h"
#include "EmulatorException.h"
#include "IntervalSet.h"
#include "Types.h"
#include "Util.h"
#include "WaveRaceState.h"
#include <cstdint>
#include <vector>

namespace raceemulator {

class Profiler;

/// Workgroup-level race detection state. Owns the event registry, live LDS
/// event lists, per-byte counters, and per-wave WaveRaceStates.
///
/// Event lifecycle:
///   1. allocateEventId() — registers a new event (ACTIVE).
///   2. markEventWaveComplete() — transitions to WAVE_COMPLETE (s_waitcnt).
///   3. retireEvent() — removes from live lists, decrements byte counts
///      (s_barrier).
///
/// Race validation uses a two-level approach:
///   - Fast path: per-byte counters (byteWriteCounts / byteReadCounts)
///     provide O(1) checks.
///   - Slow path: when counts are non-zero, scans live event intervals
///     with binary search (IntervalSet::overlapsRange).
class RaceDetector {
  friend class Workgroup;

  /// Per-event metadata, stored once in the event registry (not per byte).
  struct EventInfo {
    WaveId waveId;
    int pc;
    MemoryEventType type;
    EventStatus status;
    uint8_t byteMask;
    uint64_t execMask;
    std::vector<uint32_t> registers;
    IntervalSet ldsIntervals;
  };

public:
  RaceDetector(int ldsSize, int nWaves, int vgprCount);

  /// Allocate a workgroup-global event ID and record its metadata.
  EventId allocateEventId(WaveId waveId, int pc, MemoryEventType type,
                          const std::vector<uint32_t> &registers,
                          uint64_t execMask, uint8_t byteMask = 0xF,
                          IntervalSet ldsIntervals = {});

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId);

  /// Retire an event at s_barrier. For LDS-touching events, removes from
  /// live lists and decrements per-byte counts. Non-LDS events are no-ops.
  void retireEvent(EventId);

  /// Check for RAW hazards: no outstanding LDS writes overlap the range.
  void validateRead(int addr, WaveId, int lane, int nBytes) const;

  /// Check for WAR hazards: no outstanding LDS reads overlap the range.
  void validateWrite(int addr, WaveId, int lane, int nBytes) const;

  // --- Event accessors ---

  const IntervalSet &getEventIntervals(EventId eventId) const {
    return eventRegistry[eventId.value].ldsIntervals;
  }

  WaveId getEventWaveId(EventId eventId) const {
    return eventRegistry[eventId.value].waveId;
  }

  MemoryEventType getEventType(EventId eventId) const {
    return eventRegistry[eventId.value].type;
  }

  EventStatus getEventStatus(EventId eventId) const {
    return eventRegistry[eventId.value].status;
  }

  uint8_t getEventByteMask(EventId eventId) const {
    return eventRegistry[eventId.value].byteMask;
  }

  int getEventPc(EventId eventId) const {
    return eventRegistry[eventId.value].pc;
  }

  uint64_t getEventExecMask(EventId eventId) const {
    return eventRegistry[eventId.value].execMask;
  }

  static inline bool isLaneActive(uint64_t execMask, int lane) {
    return (execMask >> lane) & 1;
  }

  bool isEventActiveForLane(EventId eventId, int lane) const {
    return isLaneActive(eventRegistry[eventId.value].execMask, lane);
  }

  const std::vector<uint32_t> &getEventRegisters(EventId eventId) const {
    return eventRegistry[eventId.value].registers;
  }

  const std::vector<EventId> &getLdsWriteEvents() const {
    return ldsWriteEvents;
  }
  const std::vector<EventId> &getLdsReadEvents() const {
    return ldsReadEvents;
  }

  void setWorkgroupId(Dim3d id) { workgroupId = id; }
  Dim3d getWorkgroupId() const { return workgroupId; }

private:
  /// Propagate profiler to all owned WaveRaceStates.
  void setProfiler(Profiler *p);

  WaveRaceState &getWaveRaceState(int waveIndex);

  static void adjustByteCounts(const IntervalSet &ivs, std::vector<int> &counts,
                                int delta);

  std::vector<EventId> ldsWriteEvents;
  std::vector<EventId> ldsReadEvents;
  std::vector<int> byteWriteCounts;
  std::vector<int> byteReadCounts;
  std::vector<EventInfo> eventRegistry;
  Dim3d workgroupId{0, 0, 0};
  std::vector<WaveRaceState> waveRaceStates;
};

} // namespace raceemulator
