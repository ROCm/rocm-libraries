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
#include <gtest/gtest.h>

#include <optional>
#include <sstream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"

using namespace stinkytofu;

class RedundantMovEliminationPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize GEMM config for gfx1250
        gemmConfig.arch = {12, 5, 0};
        gemmConfig.NumWaves = 2;
        gemmConfig.TileA0 = 16;
        gemmConfig.TileB0 = 16;
        gemmConfig.TileM0 = 16;
        gemmConfig.NumGRA = 4;
        gemmConfig.NumGRB = 4;
        gemmConfig.NumGRM = 4;
    }

    // Parse IR and return a non-owning pointer
    // The converter must be kept alive for the lifetime of the Function
    Function* parseIR(const std::string& irString, StinkyIRConverter& converter) {
        Function* func = converter.convertToFunction(irString);
        if (!func) {
            std::cerr << "Failed to parse IR\n";
            return nullptr;
        }
        return func;
    }

    std::string getFunctionIR(Function& func) {
        std::ostringstream oss;
        AsmPrinter printer(oss);
        for (BasicBlock& bb : func) {
            for (IRBase& ir : bb) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) {
                    auto* inst = static_cast<StinkyInstruction*>(&ir);
                    printer.print(*inst);
                    oss << "\n";
                }
            }
        }
        return oss.str();
    }

    int countOccurrences(const std::string& str, const std::string& substr) {
        int count = 0;
        std::string::size_type pos = 0;
        while ((pos = str.find(substr, pos)) != std::string::npos) {
            count++;
            pos++;
        }
        return count;
    }

    GemmTileConfig gemmConfig;
    AnalysisManager am;
};

// Test 1: Eliminate redundant mov to same destination
TEST_F(RedundantMovEliminationPassTest, EliminateSimpleDuplicate) {
    std::string irString = R"(
        v[0] = "st.v_mov_b32"(v[1])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[0] = "st.v_mov_b32"(v[1])
        v[7] = "st.v_sub_f32"(v[0], v[3])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_mov_b32 instruction (second duplicate to same dest eliminated)
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 1);
}

// Test 2: Multiple redundant movs to same destination
TEST_F(RedundantMovEliminationPassTest, MultipleDuplicates) {
    std::string irString = R"(
        v[0] = "st.v_mov_b32"(v[1])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[0] = "st.v_mov_b32"(v[1])
        v[6] = "st.v_sub_f32"(v[7], v[8])
        v[0] = "st.v_mov_b32"(v[1])
        v[9] = "st.v_fma_f32"(v[0], v[3], v[6])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_mov_b32 instruction (all redundant assignments eliminated)
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 1);
}

// Test 3: Don't eliminate when operand is modified
TEST_F(RedundantMovEliminationPassTest, DontEliminateWithModifiedOperand) {
    std::string irString = R"(
        v[0] = "st.v_mov_b32"(v[1])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[1] = "st.v_sub_f32"(v[6], v[7])
        v[8] = "st.v_mov_b32"(v[1])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // Should have 2 v_mov_b32 instructions (v1 was modified between them)
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 2);
}

// Test 4: Different operands - not duplicates
TEST_F(RedundantMovEliminationPassTest, DifferentOperands) {
    std::string irString = R"(
        v[0] = "st.v_mov_b32"(v[1])
        v[3] = "st.v_mov_b32"(v[4])
        v[6] = "st.v_mov_b32"(v[5])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // Should have 3 v_mov_b32 instructions (all have different operands)
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 3);
}

// Test 8: Empty IR
TEST_F(RedundantMovEliminationPassTest, EmptyIR) {
    std::string irString = "";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);
    // Empty IR should not cause any errors
}

// Test 9: No duplicates (all different sources)
TEST_F(RedundantMovEliminationPassTest, NoDuplicates) {
    std::string irString = R"(
        v[0] = "st.v_mov_b32"(v[1])
        v[3] = "st.v_mov_b32"(v[4])
        v[6] = "st.v_mov_b32"(v[7])
        v[9] = "st.v_mov_b32"(v[10])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createRedundantMovEliminationPass();
    dupElimPass->run(*func, passCtx, am);

    std::string after = getFunctionIR(*func);

    // Should be unchanged (all have different sources)
    EXPECT_EQ(countOccurrences(after, "v_mov_b32"), 4);
}
