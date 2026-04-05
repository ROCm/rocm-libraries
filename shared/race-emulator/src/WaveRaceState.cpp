// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/WaveRaceState.h"
#include "race-emulator/IntervalSet.h"
#include "race-emulator/RaceDetector.h"
#include "race-emulator/Util.h"
#include <bit>
#include <span>

namespace raceemulator {

WaveRaceState::WaveRaceState(int vgprCount, WaveId waveId,
                             RaceDetector *detector)
    : waveId(waveId), detector(detector) {
  vgprMemoryEvents.resize(vgprCount);
  for (auto &counts : regEventCount) {
    counts.resize(vgprCount, 0);
  }
}

void WaveRaceState::registerEvent(int pc, MemoryEventType type,
                                  std::vector<uint32_t> regIds,
                                  uint64_t execMask, uint8_t byteMask) {
  registerEventWithIntervals(pc, type, std::move(regIds), execMask, byteMask,
                             {});
}

void WaveRaceState::registerEventWithIntervals(int pc, MemoryEventType type,
                                               std::vector<uint32_t> regIds,
                                               uint64_t execMask,
                                               uint8_t byteMask,
                                               IntervalSet ldsIntervals) {
  auto sw = profileScope("registerEvent");
  for (auto reg : regIds) {
    regEventCountInc(type, reg);
  }
  auto eventId =
      detector->allocateEventId(waveId, pc, type, std::move(regIds), execMask,
                                byteMask, std::move(ldsIntervals));
  for (uint32_t reg : detector->getEventRegisters(eventId)) {
    vgprMemoryEvents[reg].push_back(eventId);
  }
  waveMemoryEvents.push_back(eventId);
}

void WaveRaceState::registerLdsEvent(
    int pc, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    uint8_t byteMask) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    int addr = static_cast<int>(laneBaseAddresses[lane]);
    intervals.append(addr, addr + bytesPerLane);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, byteMask,
                             std::move(intervals));
}

void WaveRaceState::registerDualOffsetLdsEvent(
    int pc, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
    int32_t offset1) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    int intAddr0 = static_cast<int>(vAddr + static_cast<uint32_t>(offset0) * 8);
    intervals.append(intAddr0, intAddr0 + 8);
    int intAddr1 = static_cast<int>(vAddr + static_cast<uint32_t>(offset1) * 8);
    intervals.append(intAddr1, intAddr1 + 8);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, 0xF,
                             std::move(intervals));
}

void WaveRaceState::retireEventRegisters(EventId eventId) {
  auto sw = profileScope("retireEventRegisters");
  auto eventType = detector->getEventType(eventId);
  for (uint32_t regId : detector->getEventRegisters(eventId)) {
    auto &eventsForReg = getVgprMemoryEvents(regId);
    removeFromUnorderedList(eventsForReg, eventId);
    regEventCountDec(eventType, regId);
  }
}

void WaveRaceState::resolveWaitCnt(
    int limit, std::function<bool(MemoryEventType)> isTargetType) {
  int seen = 0;
  std::vector<int> indicesToRemove;

  for (int i = waveMemoryEvents.size() - 1; i >= 0; --i) {
    EventId eventId = waveMemoryEvents[i];
    if (isTargetType(detector->getEventType(eventId))) {
      seen++;
      if (seen > limit) {
        indicesToRemove.push_back(i);
      }
    }
  }

  for (int idx : indicesToRemove) {
    EventId eventId = waveMemoryEvents[idx];
    retireEventRegisters(eventId);
    detector->markEventWaveComplete(eventId);
    waveCompleteMemoryEvents.push_back(eventId);
    waveMemoryEvents.erase(waveMemoryEvents.begin() + idx);
  }
}

void WaveRaceState::sWaitCntVmcnt(int vmcnt) {
  resolveWaitCnt(vmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::GLOBAL_TO_VGPR ||
           type == MemoryEventType::VGPR_TO_GLOBAL ||
           type == MemoryEventType::GLOBAL_TO_LDS;
  });
}

void WaveRaceState::sWaitCntLgkmcnt(int lgkmcnt) {
  resolveWaitCnt(lgkmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::LDS_TO_VGPR ||
           type == MemoryEventType::VGPR_TO_LDS;
  });
}

void WaveRaceState::flushWaveCompleteMemoryEvents() {
  auto sw = profileScope("removeEvents");
  for (EventId eventId : waveCompleteMemoryEvents) {
    detector->retireEvent(eventId);
  }
  waveCompleteMemoryEvents.clear();
}

void WaveRaceState::checkVgprRead(int reg, int lane, uint8_t byteMask) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    if (isToVgpr(detector->getEventType(eid)) &&
        (detector->getEventByteMask(eid) & byteMask) != 0 &&
        detector->isEventActiveForLane(eid, lane)) {
      detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value,
                                  lane, false, detector->getWorkgroupId()});
    }
  }
}

void WaveRaceState::checkVgprReadAllLanes(int reg) const {
  if (getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, reg) != 0 ||
      getRegEventCount(MemoryEventType::LDS_TO_VGPR, reg) != 0) {
    for (EventId eid : vgprMemoryEvents[reg]) {
      if (isToVgpr(detector->getEventType(eid)) &&
          (detector->getEventByteMask(eid) & 0xF) != 0) {
        int lane = std::countr_zero(detector->getEventExecMask(eid));
        detector->getRaceHandler()({RaceViolation::Space::VGPR, reg,
                                    waveId.value, lane, false,
                                    detector->getWorkgroupId()});
      }
    }
  }
}

bool WaveRaceState::isOutstandingFromVgpr(int lane, int reg) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    if (isFromVgpr(detector->getEventType(eid)) &&
        detector->isEventActiveForLane(eid, lane)) {
      return true;
    }
  }
  return false;
}

Profiler::ScopedStopwatch WaveRaceState::profileScope(std::string_view key) {
  return profiler ? profiler->scopedStopwatch(key)
                  : Profiler::ScopedStopwatch{};
}

} // namespace raceemulator
