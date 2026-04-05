// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "Types.h"
#include <stdexcept>
#include <string>

namespace raceemulator {

/// Base exception for emulator errors (unimplemented instructions, etc.).
class EmulatorException : public std::runtime_error {
public:
  explicit EmulatorException(const std::string &msg)
      : std::runtime_error(msg) {}
};

/// Thrown when the emulator detects a data race. Wraps a RaceViolation
/// with an optional diagnostic message.
class RaceConditionException : public EmulatorException {
public:
  RaceViolation violation;

  explicit RaceConditionException(RaceViolation v)
      : EmulatorException("Race Condition"), violation(v) {}

  RaceConditionException(const std::string &msg, RaceViolation v)
      : EmulatorException(msg), violation(v) {}
};

} // namespace raceemulator
