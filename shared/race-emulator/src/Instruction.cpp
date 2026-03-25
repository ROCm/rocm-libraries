// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Instruction.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>

namespace raceemulator {

InstructionRegistry::InstructionRegistry() = default;

const std::map<std::string, std::unique_ptr<Instruction>> &getInstructions() {
  return InstructionRegistry::instance().getInstructions();
}

} // namespace raceemulator
