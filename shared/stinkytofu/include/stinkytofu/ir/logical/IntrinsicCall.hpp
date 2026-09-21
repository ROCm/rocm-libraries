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

#include <string>
#include <vector>

#include "stinkytofu/ir/logical/LogicalInstructions.hpp"

namespace stinkytofu {
/**
 * @brief High-level IR instruction for intrinsic function calls
 *
 * This instruction represents a call to a pre-compiled intrinsic from
 * the IntrinsicLibrary. It will be expanded by IntrinsicExpansionPass
 * into concrete high-level IR instructions.
 *
 * Example:
 *   IntrinsicCall("ReluF32", {v0}, {v1, v2})
 *   - functionName = "ReluF32"
 *   - dests = [v0 (dest), v2 (temp)]
 *   - srcs = [v1 (src)]
 *
 * Note: All registers (including temporaries) are part of the signature.
 */
class IntrinsicCall : public LogicalInstruction {
   private:
    friend class IRBase;

    IntrinsicCall(const std::string& name, const std::vector<StinkyRegister>& allRegs)
        : LogicalInstruction(logical::IntrinsicCall), functionName(name) {
        // Store all registers - IntrinsicExpansionPass will map them
        // to the intrinsic's argument list
        dests = allRegs;
    }

    ~IntrinsicCall() override = default;

   public:
    std::string functionName;

    const char* getLogicalName() const override {
        return "IntrinsicCall";
    }

    const std::string& getFunctionName() const {
        return functionName;
    }

    bool isComposite() const {
        return true;  // Needs expansion by IntrinsicExpansionPass
    }

    void dump(std::ostream& out) const override {
        out << "IntrinsicCall(" << functionName << ", " << dests.size() << " args)";
    }
};

}  // namespace stinkytofu
