// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace raceemulator {

/// Controls the order in which waves execute instructions in the emulator.
/// Race detection is independent of the scheduling order: all schedules
/// detect the same races because event retirement only occurs when all waves
/// have reached a barrier, not when each wave individually arrives.
enum class WaveSchedule {
  Sequential, ///< Wave 0 runs to barrier/endpgm, then wave 1, etc.
  RoundRobin, ///< Rotate which wave runs after each instruction.
};

/// Per-run configuration for the emulator. Controls race detection,
/// instruction completeness, profiling, and wave scheduling. Passed as a
/// single value to Emulator::run(), so all options are visible at the call
/// site and the emulator is never in a partially-configured state.
struct RunConfig {
  /// Check for data races (LDS and VGPR hazards).
  bool raceChecks = false;
  /// When true, global memory loads copy real data into VGPRs. When false,
  /// loads are no-ops (useful for race-only checks without valid kernargs).
  bool completeEmulation = true;
  /// Enable per-instruction profiling. Zero overhead when false.
  bool profiling = false;
  /// Order in which waves execute instructions within a workgroup.
  WaveSchedule waveSchedule = WaveSchedule::Sequential;
};

} // namespace raceemulator
