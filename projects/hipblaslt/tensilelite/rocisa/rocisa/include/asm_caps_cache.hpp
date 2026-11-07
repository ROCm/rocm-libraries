// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <map>
#include <optional>
#include <string>

#include "helper.hpp"

namespace rocisa
{
    // Probing an ISA's assembler capabilities runs the assembler hundreds of
    // times, but the answers only change with the assembler or with the probes,
    // so they are cached on disk under a key covering both. The cache lives in
    // TENSILE_ASM_CAPS_CACHE_DIR (default ~/.tensile/asm_caps_cache), and
    // TENSILE_DISABLE_ASM_CAPS_CACHE=1 turns it off. Both calls are best effort:
    // any failure is a miss, or a store that does nothing.
    std::optional<std::map<std::string, int>> loadAsmCaps(const IsaVersion&  isaVersion,
                                                          const std::string& assemblerPath);

    void storeAsmCaps(const IsaVersion&                 isaVersion,
                      const std::string&                assemblerPath,
                      const std::map<std::string, int>& asmCaps);
} // namespace rocisa
