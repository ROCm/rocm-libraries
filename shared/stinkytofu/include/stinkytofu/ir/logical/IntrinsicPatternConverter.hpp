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

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"

namespace stinkytofu {
/**
 * @brief Generic IR instruction for intrinsic body operations
 *
 * This is a lightweight IR instruction that holds text-based operation
 * information. It serves as an intermediate representation before full
 * IR instruction types (VAddF32, VMaxF32, etc.) are implemented.
 *
 * Future: Replace with concrete LogicalInstruction subclasses.
 */
class GenericIRInstruction : public LogicalInstruction {
   private:
    friend class IRBase;

    GenericIRInstruction(const std::string& dest, const std::string& op,
                         const std::vector<IntrinsicOperand>& ops)
        : LogicalInstruction(), destReg(dest), operation(op), operands(ops) {}

    ~GenericIRInstruction() override = default;

   public:
    std::string destReg;                     ///< Destination register name
    std::string operation;                   ///< Operation name (e.g., "v_add_f32")
    std::vector<IntrinsicOperand> operands;  ///< Typed operands

    const char* getLogicalName() const override {
        return operation.c_str();
    }

    void dump(std::ostream& out) const override {
        out << destReg << " = " << operation << "(";
        for (size_t i = 0; i < operands.size(); ++i) {
            if (i > 0) out << ", ";

            const auto& op = operands[i];
            if (op.type == IntrinsicOperand::Register) {
                out << op.registerName;
            } else if (op.type == IntrinsicOperand::IntLiteral) {
                out << op.intValue;
            } else if (op.type == IntrinsicOperand::FloatLiteral) {
                out << op.floatValue;
            } else if (op.type == IntrinsicOperand::HexLiteral) {
                // Output hex literal in hex format
                float floatVal = static_cast<float>(op.floatValue);
                uint32_t bits = *reinterpret_cast<uint32_t*>(&floatVal);
                out << "0x" << std::hex << bits << std::dec;
            }
        }
        out << ")";
    }
};

/**
 * @brief Metadata container for intrinsic definitions
 *
 * Holds a vector of LogicalInstructions with additional metadata (arguments, comment,
 * python_binding). This is for the intrinsic compiler (C++ tool), not Python.
 */
struct IntrinsicIRModule {
    std::string name;
    std::vector<IntrinsicArgument> arguments;
    std::vector<std::shared_ptr<LogicalInstruction>> instructions;  // Intrinsic body
    std::string comment;
    bool pythonBinding;

    IntrinsicIRModule(const std::string& n) : name(n), pythonBinding(false) {}
};

/**
 * @brief Converts between text patterns and IR modules
 *
 * This converter transforms:
 *   Text Pattern (IntrinsicPattern) <-> IR Module (IntrinsicIRModule)
 *
 * This enables the pipeline:
 *   Text -> IR -> Optimization -> Serialization
 */
class STINKYTOFU_EXPORT IntrinsicPatternConverter {
   public:
    /**
     * @brief Convert text pattern to IR module
     *
     * @param pattern Input text pattern
     * @return IR module with metadata
     */
    static IntrinsicIRModule patternToIR(const Pattern& pattern);

    /**
     * @brief Convert IR module back to text pattern
     *
     * @param irModule IR module with metadata
     * @return Text pattern
     */
    static Pattern irToPattern(const IntrinsicIRModule& irModule);

    /**
     * @brief Convert multiple patterns to IR modules
     *
     * @param patterns Vector of text patterns
     * @return Vector of IR modules
     */
    static std::vector<IntrinsicIRModule> patternsToIR(const std::vector<Pattern>& patterns);

    /**
     * @brief Convert multiple IR modules to patterns
     *
     * @param irModules Vector of IR modules
     * @return Vector of text patterns
     */
    static std::vector<Pattern> irToPatterns(const std::vector<IntrinsicIRModule>& irModules);

   private:
    /**
     * @brief Parse a register/literal operand string
     *
     * Handles:
     *   - Register references: "v[name]", "src", "dest"
     *   - Literals: "0.0", "1.0", "-1.0"
     *
     * @param operand Operand string
     * @return Parsed register or literal
     */
    static StinkyRegister parseOperand(const std::string& operand);
};

}  // namespace stinkytofu
