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
// Internal use only: modifier print/parse (serializer). Not part of the public API.

#pragma once

#include <ostream>

#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// Utility class: serialize (print) and deserialize (parse) modifiers.
class ModifierSerializer {
   public:
    /// Serialize (print) a modifier to the stream. Returns false if modifier type is unknown.
    static bool serialize(const Modifier& mod, std::ostream& os);

    /// Deserialize (parse): apply all entries in modifiers to inst (dispatches by attrKey).
    static void deserialize(StinkyInstruction* inst, const ParsedModifierDict& modifiers);
};

}  // namespace stinkytofu
