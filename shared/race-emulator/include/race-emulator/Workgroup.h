// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "LDS.h"
#include "RaceDetector.h"
#include "Types.h"
#include "Wave.h"
#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace raceemulator {

class Profiler;
struct ParsedLine;

/// All structural configuration needed to construct a Workgroup. Once a
/// Workgroup is constructed from this, its wave count, register layout, and
/// LDS size are fixed — matching GPU hardware where a workgroup launches
/// with a known thread count.
struct WorkgroupConfig {
  int nWaves = 1;
  int vgprCount;
  int agprOffset = -1; // -1 → same as vgprCount
  int sgprCount;
  WaveSize waveSize;
  int ldsSize = 0;
  bool raceChecks = false;
  bool completeEmulation = true;
  WaveSchedule waveSchedule = WaveSchedule::Sequential;
  Dim3d workgroupId{0, 0, 0};
  std::function<void(RaceViolation)> raceHandler = {};
  const std::map<std::string, int> *labels = nullptr;
  const std::map<std::string, Macro> *macros = nullptr;
};

/// Per-workgroup state: owns waves, LDS memory, the execution loop, and an
/// optional RaceDetector for race checking. Mirrors the GPU hardware model
/// where a workgroup is a self-contained unit of execution.
class Workgroup {
public:
  explicit Workgroup(const WorkgroupConfig &config);
  ~Workgroup();
  Workgroup(Workgroup &&) noexcept;
  Workgroup &operator=(Workgroup &&) noexcept;
  Workgroup() = delete;
  Workgroup(const Workgroup &) = delete;
  Workgroup &operator=(const Workgroup &) = delete;

  // --- LDS access with race checking ---

  template <typename T> T readLds(int addr, WaveId wave, int lane) const {
    if (raceDetector) {
      raceDetector->validateRead(addr, wave, lane, sizeof(T));
    }
    return lds.read<T>(addr);
  }

  template <typename T>
  void writeLds(int addr, WaveId wave, int lane, T value) {
    if (raceDetector) {
      raceDetector->validateWrite(addr, wave, lane, sizeof(T));
    }
    lds.write<T>(addr, value);
  }

  // Bulk LDS read: validates the full range once, then copies count elements.
  template <typename T>
  void readLds(int addr, WaveId wave, int lane, T *out, int count) const {
    int nBytes = count * sizeof(T);
    if (raceDetector) {
      raceDetector->validateRead(addr, wave, lane, nBytes);
    }
    lds.read(addr, out, nBytes);
  }

  // --- Direct LDS access (for test scaffolding) ---
  LDS &getLds() { return lds; }
  const LDS &getLds() const { return lds; }

  int getLdsSize() const { return lds.getSize(); }

  bool isRaceChecks() const { return raceDetector != nullptr; }
  bool isCompleteEmulation() const { return completeEmulation; }

  Profiler *getProfiler() const { return profiler; }

  // --- Wave management ---

  void run(const std::vector<ParsedLine> &tokens);

  const Wave &getWave(int waveId) const { return waves.at(waveId); }
  Wave &getWave(int waveId) { return waves.at(waveId); }
  int getNumWaves() const { return static_cast<int>(waves.size()); }

  void setProfiler(Profiler *p);

  Dim3d getWorkgroupId() const { return workgroupId; }

  // --- Race detector access ---

  RaceDetector *getRaceDetector() { return raceDetector.get(); }
  const RaceDetector *getRaceDetector() const { return raceDetector.get(); }

private:
  std::vector<Wave> waves;
  Profiler *profiler = nullptr;
  WaveSchedule waveSchedule{WaveSchedule::Sequential};

  LDS lds;

  std::unique_ptr<RaceDetector> raceDetector;

  bool completeEmulation{true};
  Dim3d workgroupId;
};

} // namespace raceemulator
