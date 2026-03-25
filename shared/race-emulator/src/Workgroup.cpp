// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Workgroup implementation: hybrid LDS race validation.
// Per-byte counts provide O(1) fast-path checks. When counts are non-zero,
// falls back to scanning live event intervals with binary search.

#include "race-emulator/Workgroup.h"
#include <algorithm>
#include <cassert>

namespace raceemulator {

void Workgroup::markEventWaveComplete(EventId eventId) {
  assert(eventId.value >= 0 &&
         eventId.value < static_cast<int64_t>(eventRegistry.size()));
  assert(eventRegistry[eventId.value].status == EventStatus::ACTIVE);
  eventRegistry[eventId.value].status = EventStatus::WAVE_COMPLETE;
}

static void removeFromUnorderedList(std::vector<EventId> &list,
                                    EventId eventId) {
  auto it = std::find(list.begin(), list.end(), eventId);
  if (it != list.end()) {
    std::swap(*it, list.back());
    list.pop_back();
  }
}

void Workgroup::retireLdsEvent(EventId eventId) {
  const auto &info = eventRegistry[eventId.value];
  if (isToLds(info.type)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteWriteCounts, -1);
  } else if (info.type == MemoryEventType::LDS_TO_VGPR) {
    removeFromUnorderedList(ldsReadEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteReadCounts, -1);
  }
}

void Workgroup::validateRead(int addr, WaveId wave, int lane,
                             int nBytes) const {
  // Fast path: if no write events touch any byte in the range, no race.
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

  // Slow path: scan live write events for overlap.
  for (EventId eventId : ldsWriteEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      throw RaceConditionException::Lds(addr, wave.value, lane, false);
    }
  }
}

void Workgroup::validateWrite(int addr, WaveId wave, int lane,
                              int nBytes) const {
  // Fast path: if no read events touch any byte in the range, no race.
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

  // Slow path: scan live read events for overlap.
  for (EventId eventId : ldsReadEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      throw RaceConditionException::Lds(addr, wave.value, lane, true);
    }
  }
}

void Workgroup::resizeLds(int size) {
  lds.resize(size);
  byteWriteCounts.resize(size, 0);
  byteReadCounts.resize(size, 0);
}

void Workgroup::clear() {
  lds.clear();
  ldsWriteEvents.clear();
  ldsReadEvents.clear();
  byteWriteCounts.clear();
  byteReadCounts.clear();
  eventRegistry.clear();
  nextEventId = 0;
}

} // namespace raceemulator
