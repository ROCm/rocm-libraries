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

/**
 * @file IntrinsicFlowTest.cpp
 * @brief Test the intrinsic library loading and usage
 *
 * This test verifies:
 *   1. Loading intrinsics from .st.bc file
 *   2. Querying intrinsic definitions
 *   3. Accessing intrinsic body instructions
 *   4. Verifying intrinsic metadata
 */

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "stinkytofu/ir/logical/IntrinsicCall.hpp"
#include "stinkytofu/ir/logical/IntrinsicLibrary.hpp"

using namespace stinkytofu;

class IntrinsicFlowTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Load intrinsic library from .st.bc file
        // Try multiple paths since test working directory varies
        std::vector<std::string> paths = {
            "intrinsics.st.bc",          // Same directory
            "../intrinsics.st.bc",       // Parent directory
            "../../intrinsics.st.bc",    // Two levels up
            "../../../intrinsics.st.bc"  // Three levels up
        };

        for (const auto& bcFilePath : paths) {
            library = IntrinsicLibrary::loadFromFile(bcFilePath);
            if (library) {
                break;
            }
        }

        ASSERT_NE(library, nullptr) << "Failed to load intrinsics.st.bc from any path";
        ASSERT_GT(library->size(), 0) << "Intrinsic library is empty";
    }

    std::shared_ptr<IntrinsicLibrary> library;
};

TEST_F(IntrinsicFlowTest, LoadIntrinsicLibrary) {
    // Verify library loaded correctly
    ASSERT_TRUE(library->hasIntrinsic("ReluF32"));
    ASSERT_TRUE(library->hasIntrinsic("ReluF16"));
    ASSERT_TRUE(library->hasIntrinsic("ClampF32"));
    ASSERT_TRUE(library->hasIntrinsic("SigmoidF32"));

    // Print library stats
    std::cout << "\n=== Intrinsic Library Loaded ===\n";
    library->printStats();
}

TEST_F(IntrinsicFlowTest, IntrinsicDefinitionStructure) {
    // Test ReluF32 structure
    const Pattern* relu = library->lookup("ReluF32");
    ASSERT_NE(relu, nullptr);

    EXPECT_EQ(relu->name, "ReluF32");
    EXPECT_EQ(relu->arguments.size(), 2);  // dest, src
    EXPECT_EQ(relu->body.size(), 1);       // 1 instruction: v_max_f32(dest, src, 0.0)
    EXPECT_TRUE(relu->pythonBinding);

    // Verify arguments
    EXPECT_EQ(relu->arguments[0].name, "dest");
    EXPECT_EQ(relu->arguments[1].name, "src");

    // Verify body: dest = v_max_f32(src, 0.0)
    EXPECT_EQ(relu->body[0].destReg, "dest");
    EXPECT_EQ(relu->body[0].operation, "v_max_f32");
    EXPECT_EQ(relu->body[0].operands.size(), 2);
    EXPECT_EQ(relu->body[0].operands[0].type, IntrinsicOperand::Register);
    EXPECT_EQ(relu->body[0].operands[0].registerName, "src");
    EXPECT_EQ(relu->body[0].operands[1].type, IntrinsicOperand::FloatLiteral);
    EXPECT_DOUBLE_EQ(relu->body[0].operands[1].floatValue, 0.0);
}

TEST_F(IntrinsicFlowTest, CreateIntrinsicCall) {
    // Create registers for ReluF32(dest=v0, src=v1)
    StinkyRegister v0(RegType::V, 0, 1);
    StinkyRegister v1(RegType::V, 1, 1);

    // Create IntrinsicCall
    std::vector<StinkyRegister> args = {v0, v1};
    IntrinsicCall* call = IRBase::createIR<IntrinsicCall>("ReluF32", args);

    EXPECT_EQ(call->getFunctionName(), "ReluF32");
    EXPECT_TRUE(call->isComposite());
    EXPECT_EQ(strcmp(call->getLogicalName(), "IntrinsicCall"), 0);
    EXPECT_EQ(call->dests.size(), 2);

    call->safeErase();
}

TEST_F(IntrinsicFlowTest, IntrinsicUsageExample_ReluF32) {
    std::cout << "\n=== Example: Using ReluF32 Intrinsic ===\n\n";

    // Step 1: Look up the intrinsic
    const Pattern* relu = library->lookup("ReluF32");
    ASSERT_NE(relu, nullptr);

    std::cout << "Intrinsic: " << relu->name << "\n";
    std::cout << "Comment: " << relu->comment << "\n";
    std::cout << "Python Binding: " << (relu->pythonBinding ? "yes" : "no") << "\n\n";

    // Step 2: Show the arguments
    std::cout << "Arguments:\n";
    for (const auto& arg : relu->arguments) {
        std::cout << "  " << arg.name << ": " << arg.regType << "\n";
    }
    std::cout << "\n";

    // Step 3: Show the body instructions
    std::cout << "Body (expansion template):\n";
    for (const auto& inst : relu->body) {
        std::cout << "  " << inst.destReg << " = " << inst.operation << "(";
        for (size_t i = 0; i < inst.operands.size(); ++i) {
            if (i > 0) std::cout << ", ";

            const auto& op = inst.operands[i];
            if (op.type == IntrinsicOperand::Register) {
                std::cout << op.registerName;
            } else if (op.type == IntrinsicOperand::IntLiteral) {
                std::cout << op.intValue;
            } else if (op.type == IntrinsicOperand::FloatLiteral) {
                std::cout << op.floatValue;
            } else if (op.type == IntrinsicOperand::HexLiteral) {
                float floatVal = static_cast<float>(op.floatValue);
                uint32_t bits = std::bit_cast<uint32_t>(floatVal);
                std::cout << "0x" << std::hex << bits << std::dec;
            }
        }
        std::cout << ")\n";
    }
    std::cout << "\n";

    // Step 4: This is what TensileLite would do
    std::cout << "In TensileLite, this would expand to actual IR instructions:\n";
    std::cout << "  v2 = v_cmp_gt_f32(v1, 0.0)\n";
    std::cout << "  v0 = v_select_f32(v2, v1, 0.0)\n\n";

    std::cout << "=== Example Complete ? ===\n";
}

TEST_F(IntrinsicFlowTest, IntrinsicUsageExample_ClampF32) {
    std::cout << "\n=== Example: Using ClampF32 Intrinsic ===\n\n";

    const Pattern* clamp = library->lookup("ClampF32");
    ASSERT_NE(clamp, nullptr);

    std::cout << "Intrinsic: " << clamp->name << "\n";
    std::cout << "Arguments: " << clamp->arguments.size() << " (";
    for (size_t i = 0; i < clamp->arguments.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << clamp->arguments[i].name;
    }
    std::cout << ")\n\n";

    std::cout << "Expansion:\n";
    for (const auto& inst : clamp->body) {
        std::cout << "  " << inst.destReg << " = " << inst.operation << "(";
        for (size_t i = 0; i < inst.operands.size(); ++i) {
            if (i > 0) std::cout << ", ";

            const auto& op = inst.operands[i];
            if (op.type == IntrinsicOperand::Register) {
                std::cout << op.registerName;
            } else if (op.type == IntrinsicOperand::IntLiteral) {
                std::cout << op.intValue;
            } else if (op.type == IntrinsicOperand::FloatLiteral) {
                std::cout << op.floatValue;
            } else if (op.type == IntrinsicOperand::HexLiteral) {
                float floatVal = static_cast<float>(op.floatValue);
                uint32_t bits = std::bit_cast<uint32_t>(floatVal);
                std::cout << "0x" << std::hex << bits << std::dec;
            }
        }
        std::cout << ")\n";
    }

    std::cout << "\n=== Example Complete ? ===\n";
}
