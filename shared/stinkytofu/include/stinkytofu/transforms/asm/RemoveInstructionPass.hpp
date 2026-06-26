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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {
class Pass;

/// Default factory: disabled, returns nullptr (no instructions removed).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveInstructionPass();

/// Removes every instruction whose unified opcode is in \p opcodes. Returns nullptr when empty.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveInstructionPass(
    std::vector<UnifiedOpcode> opcodes);

/// Resolves each mnemonic against the target arch at run() time. Returns nullptr when empty.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveInstructionPass(
    std::vector<std::string> mnemonics);

/// Comma-separated mnemonics (e.g. @c "tensor_load_to_lds,s_nop"). Returns nullptr when empty.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveInstructionPass(
    const std::string& mnemonicsCsv);

}  // namespace stinkytofu
