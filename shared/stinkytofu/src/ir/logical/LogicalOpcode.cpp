/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "stinkytofu/ir/logical/LogicalOpcode.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

namespace stinkytofu {
namespace logical {

// Helper: Convert snake_case to CamelCase
// "v_max_f32" -> "VMaxF32"
// "v_cmp_gt_f32" -> "VCmpGTF32"
// "s_mov_b32" -> "SMovB32"
static std::string snakeToCamel(const char* snake) {
    std::string result;
    bool capitalizeNext = true;

    for (const char* p = snake; *p; ++p) {
        if (*p == '_') {
            capitalizeNext = true;
        } else if (capitalizeNext) {
            result += std::toupper(*p);
            capitalizeNext = false;
        } else {
            result += *p;
        }
    }

    return result;
}

// Lazy-initialized lookup table (built once on first call)
static const std::unordered_map<std::string, Opcode>& getOpcodeMap() {
    static std::unordered_map<std::string, Opcode> opcodeMap;

    // Build table on first call
    if (opcodeMap.empty()) {
        // Iterate through all opcodes and register both CamelCase and snake_case
        for (uint16_t i = 0; i < static_cast<uint16_t>(NUM_OPCODES); ++i) {
            Opcode opcode = static_cast<Opcode>(i);
            const char* name = getOpcodeName(opcode);
            const char* mnemonic = getOpcodeMnemonic(opcode);

            if (name && name[0] != '\0') {
                // Register CamelCase name (e.g., "VMaxF32")
                opcodeMap[name] = opcode;

                // Register snake_case mnemonic (e.g., "v_max_f32")
                if (mnemonic && mnemonic[0] != '\0') {
                    opcodeMap[mnemonic] = opcode;
                }
            }
        }
    }

    return opcodeMap;
}

Opcode parseOpcode(const char* name) {
    if (!name || !name[0]) {
        return UNKNOWN;
    }

    const auto& opcodeMap = getOpcodeMap();

    // Try direct lookup first
    auto it = opcodeMap.find(name);
    if (it != opcodeMap.end()) {
        return it->second;
    }

    // If not found and looks like snake_case, try converting to CamelCase
    if (strchr(name, '_')) {
        std::string camelName = snakeToCamel(name);
        it = opcodeMap.find(camelName);
        if (it != opcodeMap.end()) {
            return it->second;
        }
    }

    return UNKNOWN;
}

}  // namespace logical
}  // namespace stinkytofu
