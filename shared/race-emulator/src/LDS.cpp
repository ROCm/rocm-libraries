// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/LDS.h"

namespace raceemulator {

void LDS::resize(int size) { memory.resize(size, unset1); }

void LDS::clear() { memory.clear(); }

} // namespace raceemulator
