// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Workgroup.h"
#include "race-emulator/Parsing.h"
#include "race-emulator/Profiler.h"
#include "race-emulator/Util.h"
#include "race-emulator/WaveRaceState.h"
#include <algorithm>
#include <numeric>

namespace raceemulator {

namespace {
const std::map<std::string, int> emptyLabels = {};
const std::map<std::string, Macro> emptyMacros = {};
} // namespace

Workgroup::~Workgroup() = default;
Workgroup::Workgroup(Workgroup &&) noexcept = default;
Workgroup &Workgroup::operator=(Workgroup &&) noexcept = default;

Workgroup::Workgroup(const WorkgroupConfig &config)
    : profiler(config.profiler), waveSchedule(config.waveSchedule),
      completeEmulation(config.completeEmulation),
      workgroupId(config.workgroupId) {
  if (config.ldsSize > 0) {
    lds.resize(config.ldsSize);
  }

  if (config.raceChecks) {
    raceDetector = std::make_unique<RaceDetector>(
        lds.getSize(), config.nWaves, config.vgprCount, config.workgroupId,
        config.raceHandler);
    if (profiler) {
      raceDetector->setProfiler(profiler);
    }
  }

  int agprOffset = config.agprOffset < 0 ? config.vgprCount : config.agprOffset;
  const auto *labels = config.labels ? config.labels : &emptyLabels;
  const auto *macros = config.macros ? config.macros : &emptyMacros;

  for (int i = 0; i < config.nWaves; ++i) {
    waves.push_back(Wave(config.vgprCount, agprOffset, config.sgprCount,
                         config.waveSize, WaveId{i}, *this, labels, macros));
    if (raceDetector) {
      waves.back().raceState = &raceDetector->getWaveRaceState(i);
    }
  }
}

void Workgroup::dispatchPendingRaceEvents(WaveId waveId) {
  auto &wave = waves.at(waveId.value);
  if (auto *rs = wave.raceState) {
    if (wave.pendingWaitCount) {
      rs->dispatch(*wave.pendingWaitCount);
    }
    if (wave.pendingMemoryEvent) {
      rs->dispatch(std::move(*wave.pendingMemoryEvent));
    }
  }
  wave.pendingWaitCount.reset();
  wave.pendingMemoryEvent.reset();
}

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
    dispatchPendingRaceEvents(WaveId{waveId});

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

} // namespace raceemulator
