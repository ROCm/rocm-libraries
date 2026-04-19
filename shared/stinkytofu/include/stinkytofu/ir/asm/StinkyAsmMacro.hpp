/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/core/IRBase.hpp"

namespace stinkytofu {
// Forward declaration
class IRListModule;

/**
 * @brief Assembly-level macro definition (.macro ... .endm)
 *
 * Represents a reusable assembly template that appears in the final .s file.
 *
 * Example:
 *   .macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
 *       s_set_vgpr_msb 0
 *       v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
 *       v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
 *       v_add_nc_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
 *       v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
 *   .endm
 */
struct AsmMacroDefinition : public IRBase {
    friend class IRBase;

    std::string name;                     // Macro name (e.g., "V_MAGIC_DIV")
    std::vector<std::string> parameters;  // Parameter names (e.g., ["vgprDstIdx", "dividend", ...])
    std::shared_ptr<IRListModule> body;   // Instructions inside the macro
    std::string comment;

   private:
    AsmMacroDefinition() : IRBase(IRType::StinkyTofu) {}

    AsmMacroDefinition(const std::string& name, const std::vector<std::string>& parameters,
                       std::shared_ptr<IRListModule> body, const std::string& comment = "")
        : IRBase(IRType::StinkyTofu),
          name(name),
          parameters(parameters),
          body(body),
          comment(comment) {}

    ~AsmMacroDefinition() = default;

   public:
    // Implement IRBase::dump()
    void dump(std::ostream& out) const override {
        out << ".macro " << name << "(";
        for (size_t i = 0; i < parameters.size(); ++i) {
            if (i > 0) out << ", ";
            out << parameters[i];
        }
        out << ")";
        if (!comment.empty()) out << "  // " << comment;
    }
};

/**
 * @brief Assembly-level macro invocation
 *
 * Represents a call to a defined macro with specific arguments.
 *
 * Example:
 *   V_MAGIC_DIV 0, v1, s2, 8, 1
 */
struct AsmMacroInvocation : public IRBase {
    friend class IRBase;

    std::string name;                    // Macro name to invoke
    std::vector<std::string> arguments;  // Actual argument values
    std::string comment;

   private:
    AsmMacroInvocation() : IRBase(IRType::StinkyTofu) {}

    AsmMacroInvocation(const std::string& name, const std::vector<std::string>& arguments,
                       const std::string& comment = "")
        : IRBase(IRType::StinkyTofu), name(name), arguments(arguments), comment(comment) {}

    ~AsmMacroInvocation() = default;

   public:
    // Implement IRBase::dump()
    void dump(std::ostream& out) const override {
        out << name << "(";
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (i > 0) out << ", ";
            out << arguments[i];
        }
        out << ")";
        if (!comment.empty()) out << "  // " << comment;
    }
};

}  // namespace stinkytofu
