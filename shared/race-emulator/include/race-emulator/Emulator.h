#pragma once
#include "Arch.h"
#include "Profiler.h"
#include "Wave.h"
#include "Workgroup.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>

namespace raceemulator {

struct ParsedAsm;

// Controls the order in which waves execute instructions in the emulator.
// Race detection is independent of the scheduling order: all schedules
// detect the same races because event retirement
// (flushWaveCompleteMemoryEvents) only occurs when ALL waves have reached a
// barrier, not when each wave individually arrives.
enum class WaveSchedule {
  // Wave 0 runs until it hits a barrier or terminates, then wave 1, etc.
  Sequential,
  // Rotate which wave runs after each instruction (round-robin).
  RoundRobin,
};

class Emulator {

public:
  // TODO(newling) handle case of multiple kernels in single assembly file.
  Emulator(std::string_view assembly, std::shared_ptr<Architecture> arch);

  static Emulator createGfx942(std::string_view assembly);
  static Emulator createGfx950(std::string_view assembly);
  static Emulator createGfx1151(std::string_view assembly);

  Emulator(const Emulator &other);
  Emulator &operator=(const Emulator &other);
  Emulator(Emulator &&other) noexcept;
  Emulator &operator=(Emulator &&other) = delete;
  ~Emulator();

  // If enabled, race condition checks will be performed during execution.
  // Currently, by default they are disabled.
  void setRaceChecks(bool);

  void setCompleteEmulation(bool enable) { completeEmulation = enable; }

  // Enable per-instruction profiling. When enabled, each instruction
  // execution is timed and results are available via profileReport().
  // Disabled by default (zero overhead when disabled).
  void setProfiling(bool enable) { profiler.setEnabled(enable); }

  // Return a formatted profiling report as a string. Entries below
  // minPercentage of wall-clock time are aggregated into an "other" row.
  std::string getProfileReport(double minPercentage = 1.0) const;

  void setWaveSchedule(WaveSchedule s) { waveSchedule = s; }

  // Write a kernel argument into this emulator's kernarg segment.
  void addKernarg(int argNumber, const void *argValue);

  void addAllKernargs(const void *args);

  // Run the kernel on a single workgroup.
  // completeEmulation = false: don't do buffer loads or stores.
  void run(Dim3d wgId, Dim3d blockDim);

  // The name of the kernel being emulated.
  std::string getName() const;
  int getKernargSegmentSize() const;
  int getNumKernargs() const;
  int getKernargOffset(int argNumber) const;
  int getKernargSize(int argNumber) const;
  std::string getKernargValueKind(int argNumber) const;
  std::string getKernargAddressSpace(int argNumber) const;
  std::string getKernargName(int argNumber) const;

  const Architecture &getArch() const { return *arch; }
  const Wave &getWave(int waveId) const { return waves.at(waveId); }

  void appendStr(std::ostream &) const;
  std::string str() const;

private:
  void initializeForRun(Dim3d wgId, Dim3d blockDim, int nWaves);
  std::shared_ptr<Architecture> arch;
  std::unique_ptr<ParsedAsm> parsedAsm;
  std::vector<Wave> waves;
  std::vector<char> kernargSegment;
  std::vector<bool> kernargIsSet;
  Workgroup workgroup;
  Profiler profiler;
  bool raceChecks{false};
  bool completeEmulation{true};
  WaveSchedule waveSchedule{WaveSchedule::Sequential};
};

} // namespace raceemulator
