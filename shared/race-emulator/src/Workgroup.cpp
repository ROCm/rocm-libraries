// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Workgroup implementation: wave execution loop, barrier synchronization,
// and hybrid LDS race validation. Per-byte counts provide O(1) fast-path
// checks. When counts are non-zero, falls back to scanning live event
// intervals with binary search.

#include "race-emulator/Workgroup.h"
#include "race-emulator/Parsing.h"
#include "race-emulator/Profiler.h"
#include "race-emulator/Util.h"
#include <algorithm>
#include <cassert>
#include <numeric>

namespace raceemulator {

void Workgroup::markEventWaveComplete(EventId eventId) {
  assert(eventId.value >= 0 &&
         eventId.value < static_cast<int64_t>(eventRegistry.size()));
  assert(eventRegistry[eventId.value].status == EventStatus::ACTIVE);
  eventRegistry[eventId.value].status = EventStatus::WAVE_COMPLETE;
}

void Workgroup::retireEvent(EventId eventId) {
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
      throw RaceConditionException::Lds(addr, wave.value, lane, false,
                                        workgroupId);
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
      throw RaceConditionException::Lds(addr, wave.value, lane, true,
                                        workgroupId);
    }
  }
}

void Workgroup::addWave(Wave &&wave) { waves.push_back(std::move(wave)); }

void Workgroup::run(const std::vector<ParsedLine> &tokens) {
  int nWaves = static_cast<int>(waves.size());
  int nActiveWaves = nWaves;
  int nWaitingWaves = 0;

  std::vector<int> preferenceOrder(nWaves);
  std::iota(preferenceOrder.begin(), preferenceOrder.end(), 0);

  auto getNextWaveToRun = [&]() -> int {
    for (int waveId : preferenceOrder) {
      if (waves[waveId].isActive() && !waves[waveId].isWaiting()) {
        return waveId;
      }
    }
    throw std::runtime_error(
        "didn't expect to fail to get wave in this function");
  };

  auto tryReleaseBarrier = [&]() {
    if (nWaitingWaves == nActiveWaves) {
      auto sw = profiler ? profiler->scopedStopwatch("tryReleaseBarrier")
                         : Profiler::ScopedStopwatch{};
      for (auto &wave : waves) {
        if (wave.isWaiting()) {
          wave.flushWaveCompleteMemoryEvents();
          wave.setWaiting(false);
        }
      }
      nWaitingWaves = 0;
    }
  };

  bool roundRobin = (waveSchedule == WaveSchedule::RoundRobin);

  while (nActiveWaves != 0) {
    int waveId = getNextWaveToRun();
    auto &wave = waves[waveId];

    const auto &token = tokens[wave.getPc()];
    std::string_view trimmedAndCommentFree = trim(token.commentFreeLine);

    auto sw = [&]() -> Profiler::ScopedStopwatch {
      if (!profiler || trimmedAndCommentFree.empty()) {
        return {};
      }
      return profiler->scopedStopwatch([&]() -> std::string_view {
        auto space = trimmedAndCommentFree.find(' ');
        auto mnemonic = trimmedAndCommentFree.substr(0, space);
        if (auto pos = mnemonic.find("_e32"); pos != std::string_view::npos) {
          mnemonic = mnemonic.substr(0, pos);
        } else if (auto pos = mnemonic.find("_e64");
                   pos != std::string_view::npos) {
          mnemonic = mnemonic.substr(0, pos);
        }
        return mnemonic;
      });
    }();

    wave.tryExecute(token.commentFreeLine, true);

    // Check if the instruction changed wave state (s_endpgm / s_barrier).
    if (!wave.isActive()) {
      nActiveWaves--;
      tryReleaseBarrier();
    } else if (wave.isWaiting()) {
      nWaitingWaves++;
      tryReleaseBarrier();
    }

    if (roundRobin && preferenceOrder.size() > 1) {
      std::rotate(preferenceOrder.begin(), preferenceOrder.begin() + 1,
                  preferenceOrder.end());
    }
  }
}

void Workgroup::resizeLds(int size) {
  lds.resize(size);
  byteWriteCounts.resize(size, 0);
  byteReadCounts.resize(size, 0);
}

void Workgroup::clear() {
  waves.clear();
  lds.clear();
  ldsWriteEvents.clear();
  ldsReadEvents.clear();
  byteWriteCounts.clear();
  byteReadCounts.clear();
  eventRegistry.clear();
}

} // namespace raceemulator
