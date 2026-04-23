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
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/support/Diagnostic.hpp"

namespace stinkytofu {

/// Result of parsing a raw GPU assembly string.
struct RawAsmParseResult {
    /// Parsed function (single flat block "entry").
    std::unique_ptr<ParsedFunction> parsedFunction;

    /// Kernel signature extracted from .amdhsa_kernel / .amdgpu_metadata blocks.
    /// Null if the input had no recognisable metadata header.
    std::shared_ptr<SignatureBase> signature;

    /// Diagnostics (errors and warnings) from parsing.
    std::vector<Diagnostic> diagnostics;

    bool hasErrors() const {
        for (const auto& d : diagnostics)
            if (d.getLevel() == Diagnostic::Level::Error) return true;
        return false;
    }
};

/// Parse a raw GPU assembly string (as emitted by StinkyAsmEmitter) into a ParsedFunction.
///
/// The returned ParsedFunction is in flat format (single block "entry") and can be
/// passed directly to StinkyIRConverter::populateFunctionFromParsed().
///
/// Supported input elements:
///   - Real GPU instructions (mnemonic + register operands + inline modifiers)
///   - Labels (e.g. "kernel_name:")
///   - Assembler directives (.set, and arbitrary text passed through as TEXTBLOCK)
///   - Comments (// and ; style) are stripped
///
/// @param asmText  Raw GPU assembly source text.
/// @param arch     Target architecture for instruction lookup.
/// @return         RawAsmParseResult with parsed function and any diagnostics.
STINKYTOFU_EXPORT RawAsmParseResult parseRawAsmString(const std::string& asmText, GfxArchID arch);

}  // namespace stinkytofu
