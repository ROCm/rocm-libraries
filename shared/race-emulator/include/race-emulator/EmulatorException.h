// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace raceemulator {

class EmulatorException : public std::runtime_error {
public:
  explicit EmulatorException(const std::string &msg)
      : std::runtime_error(msg) {}
};

class RaceConditionException : public EmulatorException {
public:
  enum class Space { VGPR, SGPR, LDS };

  Space space;
  int index;    // Register Index (VGPR/SGPR) or Byte Address (LDS)
  int wave;     // Wave ID (only used for LDS)
  int lane;     // Lane ID (0-63) or -1 if scalar
  bool isWrite; // True = Write, False = Read

  // Private constructor to force use of factories
  RaceConditionException(const std::string &msg, Space s, int idx, int w, int l,
                         bool isWrite)
      : EmulatorException(msg), space(s), index(idx), wave(w), lane(l),
        isWrite(isWrite) {}

  RaceConditionException(Space s, int idx, int w, int l, bool isWrite)
      : RaceConditionException("Race Condition", s, idx, w, l, isWrite) {}

  // Factory 1: VGPR Race (Specific Lane)
  static RaceConditionException Vgpr(int regIdx, int wave, int lane,
                                     bool isWrite) {

    return {Space::VGPR, regIdx, wave, lane, isWrite};
  }

  // Factory 2: SGPR Race (Scalar - No Lane)
  static RaceConditionException Sgpr(int regIdx, int wave, bool isWrite) {
    return {Space::SGPR, regIdx, wave, -1, isWrite};
  }

  // Factory 3: LDS Race (Byte Address)
  // We include 'lane' to identify WHO triggered the fault,
  // even though the tracker is per-byte.
  static RaceConditionException Lds(int byteAddr, int wave, int lane,
                                    bool isWrite) {

    return {Space::LDS, byteAddr, wave, lane, isWrite};
  }
};

} // namespace raceemulator
