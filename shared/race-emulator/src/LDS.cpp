#include "race-emulator/Emulator.h"
#include "race-emulator/Util.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

void LDS::removeEvent(int byteAddr, uint32_t waveId, uint32_t eventId) {
  assert(byteAddr >= 0);
  assert(byteAddr < static_cast<int64_t>(byteMemoryEvents.size()));
  auto &events = byteMemoryEvents[byteAddr];
  // We expect the event to be present exactly once. Assert this
  // to be the case
  auto it = std::find_if(events.begin(), events.end(),
                         [eventId, waveId](const LDSMemoryEvent &e) {
                           return e.eventId == eventId && e.waveId == waveId;
                         });
  if (it == events.end()) {
    throw std::runtime_error("LDS Tracker Inconsistency");
  }
  events.erase(it);
}

void LDS::markEventWaveComplete(int byteAddr, uint32_t waveId,
                                uint32_t eventId) {
  assert(byteAddr >= 0);
  assert(byteAddr < static_cast<int64_t>(byteMemoryEvents.size()));
  auto &events = byteMemoryEvents[byteAddr];
  // We expect the event to be present exactly once. Assert this
  // to be the case
  auto it = std::find_if(events.begin(), events.end(),
                         [eventId, waveId](const LDSMemoryEvent &e) {
                           return e.eventId == eventId && e.waveId == waveId;
                         });
  if (it == events.end()) {
    throw std::runtime_error("LDS Tracker Inconsistency");
  }
  it->status = LDSMemoryEvent::EventStatus::WAVE_COMPLETE;
}

void LDS::validateRead(int addr, int wave, int lane, int nBytes) const {

  // TODO(newling) use wave (and lane for error)
  for (int i = 0; i < nBytes; ++i) {

    int currentByte = addr + i;

    // Tracker bounds check (safe guard)
    if (currentByte >= static_cast<int>(byteMemoryEvents.size())) {
      throw std::runtime_error("LDS read address out of bounds");
    }

    const auto &events = byteMemoryEvents[currentByte];
    for (const auto &event : events) {
      // READ-AFTER-WRITE (RAW) Check:
      // If there is a pending WRITE (VGPR_TO_LDS), it's a hazard.
      // (Reads overlapping with other Reads are safe).
      if (event.type == MemoryEventType::VGPR_TO_LDS) {

        if (static_cast<uint32_t>(wave) == event.waveId &&
            event.status == LDSMemoryEvent::EventStatus::WAVE_COMPLETE) {
          continue;
        }

        throw RaceConditionException::Lds(currentByte, wave, lane,
                                          false /* isWrite=false */);
      }
    }
  }
}

void LDS::validateWrite(int addr, int wave, int lane, int nBytes) const {
  // TODO(newling) use wave (and lane for error)
  for (int i = 0; i < nBytes; ++i) {
    int currentByte = addr + i;

    if (currentByte >= static_cast<int>(byteMemoryEvents.size())) {
      throw std::runtime_error("LDS write address out of bounds");
    }

    const auto &events = byteMemoryEvents[currentByte];
    for (const auto &e : events) {

      if (e.type == MemoryEventType::LDS_TO_VGPR) {
        if (static_cast<uint32_t>(wave) == e.waveId &&
            e.status == LDSMemoryEvent::EventStatus::WAVE_COMPLETE) {
          continue;
        }
        throw RaceConditionException::Lds(currentByte, wave, lane,
                                          true /* isWrite=true */);
      }
    }
  }
}

void LDS::resize(int size) {
  memory.resize(size, unset1);
  byteMemoryEvents.resize(size);
}

void LDS::clear() {
  memory.clear();
  byteMemoryEvents.clear();
}

} // namespace raceemulator
