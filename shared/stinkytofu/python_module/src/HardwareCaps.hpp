// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace stinkytofu {

struct HardwareCapsResult {
    std::map<std::string, int> asmCaps;
    std::map<std::string, int> archCaps;
    std::map<std::string, int> regCaps;
    std::map<std::string, bool> asmBugs;
};

class HardwareCaps {
   public:
    static HardwareCapsResult query(uint32_t major, uint32_t minor, uint32_t stepping);

   private:
    HardwareCaps() = delete;
};

}  // namespace stinkytofu
