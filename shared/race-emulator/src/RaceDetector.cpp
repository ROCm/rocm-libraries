// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/RaceDetector.h"
#include <cassert>

namespace raceemulator {

RaceDetector::RaceDetector(int ldsSize, int nWaves, int vgprCount)
    : byteWriteCounts(ldsSize, 0), byteReadCounts(ldsSize, 0) {
  waveRaceStates.reserve(nWaves);
  for (int i = 0; i < nWaves; ++i) {
    waveRaceStates.emplace_back(vgprCount, WaveId{i}, this);
  }
}

WaveRaceState &RaceDetector::getWaveRaceState(int waveIndex) {
  assert(waveIndex >= 0 &&
         waveIndex < static_cast<int>(waveRaceStates.size()));
  return waveRaceStates[waveIndex];
}

void RaceDetector::setProfiler(Profiler *p) {
  for (auto &wrs : waveRaceStates) {
    wrs.setProfiler(p);
  }
}

EventId RaceDetector::allocateEventId(WaveId waveId, int pc,
                                      MemoryEventType type,
                                      const std::vector<uint32_t> &registers,
                                      uint64_t execMask, uint8_t byteMask,
                                      IntervalSet ldsIntervals) {
  int id = static_cast<int>(eventRegistry.size());
  bool hasLds = !ldsIntervals.empty();
  eventRegistry.push_back({waveId, pc, type, EventStatus::ACTIVE, byteMask,
                           execMask, registers, std::move(ldsIntervals)});
  EventId eid{id};
  if (hasLds) {
    const auto &ivs = eventRegistry[id].ldsIntervals;
    if (isToLds(type)) {
      ldsWriteEvents.push_back(eid);
      adjustByteCounts(ivs, byteWriteCounts, +1);
    } else if (type == MemoryEventType::LDS_TO_VGPR) {
      ldsReadEvents.push_back(eid);
      adjustByteCounts(ivs, byteReadCounts, +1);
    }
  }
  return eid;
}

void RaceDetector::markEventWaveComplete(EventId eventId) {
  assert(eventId.value >= 0 &&
         eventId.value < static_cast<int64_t>(eventRegistry.size()));
  assert(eventRegistry[eventId.value].status == EventStatus::ACTIVE);
  eventRegistry[eventId.value].status = EventStatus::WAVE_COMPLETE;
}

void RaceDetector::retireEvent(EventId eventId) {
  const auto &info = eventRegistry[eventId.value];
  if (isToLds(info.type)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteWriteCounts, -1);
  } else if (info.type == MemoryEventType::LDS_TO_VGPR) {
    removeFromUnorderedList(ldsReadEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteReadCounts, -1);
  }
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane,
                                int nBytes) const {
  bool anyWrites = false;
  for (int i = 0; i < nBytes; ++i) {
    if (byteWriteCounts[addr + i] > 0) {
      anyWrites = true;
      break;
    }
  }
  if (!anyWrites) {
    return;
  }

  for (EventId eventId : ldsWriteEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      throw RaceConditionException::Lds(addr, wave.value, lane, false,
                                        workgroupId);
    }
  }
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane,
                                 int nBytes) const {
  bool anyReads = false;
  for (int i = 0; i < nBytes; ++i) {
    if (byteReadCounts[addr + i] > 0) {
      anyReads = true;
      break;
    }
  }
  if (!anyReads) {
    return;
  }

  for (EventId eventId : ldsReadEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      throw RaceConditionException::Lds(addr, wave.value, lane, true,
                                        workgroupId);
    }
  }
}

void RaceDetector::adjustByteCounts(const IntervalSet &ivs,
                                     std::vector<int> &counts, int delta) {
  for (const auto &iv : ivs) {
    for (int b = iv.start; b < iv.end; ++b) {
      counts[b] += delta;
    }
  }
}

} // namespace raceemulator
