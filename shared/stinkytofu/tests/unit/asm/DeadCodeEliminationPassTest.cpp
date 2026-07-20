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

using namespace stinkytofu;

class DeadCodeEliminationPassTest : public ::testing::Test {
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

    GemmTileConfig gemmConfig;
    AnalysisManager am;
};

// Test 1: Remove dead store (register overwritten before use)
TEST_F(DeadCodeEliminationPassTest, RemoveSimpleDeadCode) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[3] = "st.v_add_f32"(v[4], v[5])
v[0] = "st.v_mov_b32"(v[7])
v[6] = "st.v_sub_f32"(v[0], v[3])
"st.buffer_store_b32"(v[40], v[6])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // First v0 definition is overwritten by second v0 definition -> removed
    EXPECT_EQ(result.find("st.v_mul_f32"), std::string::npos);

    // All other instructions are live
    EXPECT_NE(result.find("st.v_add_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_mov_b32"), std::string::npos);
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
    EXPECT_NE(result.find("buffer_store_b32"), std::string::npos);
}

// Test 2: Chain of dead stores (multiple overwrites)
TEST_F(DeadCodeEliminationPassTest, ChainOfDeadCode) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[0] = "st.v_add_f32"(v[3], v[4])
v[0] = "st.v_fma_f32"(v[5], v[6], 1.0)
v[0] = "st.v_sub_f32"(v[10], v[11])
"st.buffer_store_b32"(v[40], v[0])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // First three v0 definitions are overwritten -> removed
    EXPECT_EQ(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_EQ(result.find("st.v_add_f32"), std::string::npos);
    EXPECT_EQ(result.find("st.v_fma_f32"), std::string::npos);

    // Last v0 definition is used by store -> kept
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
    EXPECT_NE(result.find("buffer_store_b32"), std::string::npos);
}

// Test 3: Don't remove live code
TEST_F(DeadCodeEliminationPassTest, DontRemoveLiveCode) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[3] = "st.v_add_f32"(v[0], v[4])
v[5] = "st.v_fma_f32"(v[3], v[0], 1.0)
"st.buffer_store_b32"(v[40], v[5])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // All instructions are live (v5 is stored, v3 used by v5, v0 used by both v3 and v5)
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_add_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_fma_f32"), std::string::npos);
    EXPECT_NE(result.find("buffer_store_b32"), std::string::npos);
}

// Test 4: Multiple redefinitions, some with uses in between
TEST_F(DeadCodeEliminationPassTest, MultipleUsesKeepAlive) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[3] = "st.v_add_f32"(v[0], v[4])
v[3] = "st.v_sub_f32"(v[5], v[6])
v[7] = "st.v_fma_f32"(v[3], v[1], 1.0)
"st.buffer_store_b32"(v[40], v[7])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // v0, second v3, and v7 are live
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_fma_f32"), std::string::npos);

    // First v3 definition is overwritten by second v3 -> removed
    EXPECT_EQ(result.find("st.v_add_f32"), std::string::npos);
}

// Test 5: Don't remove memory operations
TEST_F(DeadCodeEliminationPassTest, PreserveMemoryOperations) {
    std::string irString = R"(
v[0] = "st.buffer_load_b32"(v[40])
v[4] = "st.v_mul_f32"(v[1], v[2])
"st.buffer_store_b32"(v[41], v[4])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // Memory operations should always be preserved
    EXPECT_NE(result.find("buffer_load_b32"), std::string::npos);
    EXPECT_NE(result.find("buffer_store_b32"), std::string::npos);

    // v4 is used by store, so mul should also be preserved
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
}

// Test 6: Empty IR
TEST_F(DeadCodeEliminationPassTest, EmptyIR) {
    std::string irString = "";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);
    // Empty IR should not cause any errors
}

// Test 7: Multiple dead stores to same register
TEST_F(DeadCodeEliminationPassTest, AllDeadCode) {
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[0] = "st.v_add_f32"(v[4], v[5])
        v[0] = "st.v_sub_f32"(v[7], v[8])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // First two v0 definitions are overwritten -> removed
    EXPECT_EQ(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_EQ(result.find("st.v_add_f32"), std::string::npos);

    // Last v0 definition remains (not overwritten, even though not used)
    // The new DCE only removes overwrites, not all unused code
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
}

// Test 8: Interleaved live and dead code
TEST_F(DeadCodeEliminationPassTest, InterleavedLiveAndDead) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[3] = "st.v_add_f32"(v[4], v[5])
v[6] = "st.v_sub_f32"(v[3], v[0])
v[7] = "st.v_fma_f32"(v[8], v[9], 1.0)
v[10] = "st.v_add_f32"(v[6], v[7])
"st.buffer_store_b32"(v[40], v[10])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // All instructions are live (v10 is stored, all others used in the chain)
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_fma_f32"), std::string::npos);

    // Should have 2 add instructions (v3 and v10)
    std::string::size_type pos = 0;
    int addCount = 0;
    while ((pos = result.find("v_add_f32", pos)) != std::string::npos) {
        addCount++;
        pos++;
    }
    EXPECT_EQ(addCount, 2);
}

// Test 9: Constant operands
TEST_F(DeadCodeEliminationPassTest, ConstantOperands) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], 2.0)
v[3] = "st.v_add_f32"(v[4], 1.0)
v[6] = "st.v_fma_f32"(v[0], v[3], 3.0)
"st.buffer_store_b32"(v[40], v[6])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // All instructions are live (v6 is stored, v0 and v3 used by v6)
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_add_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_fma_f32"), std::string::npos);
}

// Test 10: In-place operations
TEST_F(DeadCodeEliminationPassTest, InPlaceOperations) {
    std::string irString = R"(
v[0] = "st.v_mul_f32"(v[1], v[2])
v[0] = "st.v_add_f32"(v[0], v[3])
v[4] = "st.v_sub_f32"(v[0], v[5])
"st.buffer_store_b32"(v[40], v[4])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);

    // All instructions are live (v4 is stored, v0 redefined and used by v4)
    EXPECT_NE(result.find("st.v_mul_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_add_f32"), std::string::npos);
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
}
