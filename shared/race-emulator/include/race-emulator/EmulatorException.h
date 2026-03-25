// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace raceemulator {

/// Base exception for emulator errors (unimplemented instructions, etc.).
class EmulatorException : public std::runtime_error {
public:
  explicit EmulatorException(const std::string &msg)
      : std::runtime_error(msg) {}
};

/// Thrown when the emulator detects a data race. Carries the address space,
/// register/byte index, wave, lane, and direction (read vs write) of the
/// racing access. Construct via the Vgpr/Sgpr/Lds factory methods.
class RaceConditionException : public EmulatorException {
public:
  enum class Space { VGPR, SGPR, LDS };

  Space space;
  int index;    ///< Register index (VGPR/SGPR) or byte address (LDS).
  int wave;     ///< Wave ID (only used for LDS).
  int lane;     ///< Lane ID (0-63) or -1 if scalar.
  bool isWrite; ///< True = write, false = read.

  RaceConditionException(const std::string &msg, Space s, int idx, int w, int l,
                         bool isWrite)
      : EmulatorException(msg), space(s), index(idx), wave(w), lane(l),
        isWrite(isWrite) {}

  RaceConditionException(Space s, int idx, int w, int l, bool isWrite)
      : RaceConditionException("Race Condition", s, idx, w, l, isWrite) {}

  static RaceConditionException Vgpr(int regIdx, int wave, int lane,
                                     bool isWrite) {
    return {Space::VGPR, regIdx, wave, lane, isWrite};
  }

  static RaceConditionException Sgpr(int regIdx, int wave, bool isWrite) {
    return {Space::SGPR, regIdx, wave, -1, isWrite};
  }

  /// The lane identifies who triggered the fault, even though LDS tracking
  /// is per-byte (not per-lane).
  static RaceConditionException Lds(int byteAddr, int wave, int lane,
                                    bool isWrite) {
    return {Space::LDS, byteAddr, wave, lane, isWrite};
  }
};

} // namespace raceemulator
