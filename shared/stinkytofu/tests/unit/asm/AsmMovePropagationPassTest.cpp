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

#include <sstream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"

using namespace stinkytofu;

class AsmMovePropagationPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        gemmConfig.arch = {12, 5, 0};
        gemmConfig.NumWaves = 2;
        gemmConfig.TileA0 = 16;
        gemmConfig.TileB0 = 16;
        gemmConfig.TileM0 = 16;
        gemmConfig.NumGRA = 4;
        gemmConfig.NumGRB = 4;
        gemmConfig.NumGRM = 4;
    }

    Function* parseIR(const std::string& irString, StinkyIRConverter& converter) {
        Function* func = converter.convertToFunction(irString);
        if (!func) return nullptr;
        return func;
    }

    std::string getFunctionIR(Function& func) {
        std::ostringstream oss;
        AsmPrinter printer(oss);
        for (BasicBlock& bb : func) {
            for (IRBase& ir : bb) {
                if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
                auto* inst = static_cast<StinkyInstruction*>(&ir);
                printer.print(*inst);
                oss << "\n";
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

TEST_F(AsmMovePropagationPassTest, PropagatesAndRemovesDeadMoveBeforeRedefinition) {
    std::string irString = R"(
v[0] = "st.v_mov_b32"(v[1])
v[2] = "st.v_add_f32"(v[0], v[3])
v[0] = "st.v_sub_f32"(v[4], v[5])
"st.buffer_store_b32"(v[40], v[2])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    createAsmMovePropagationPass()->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 0);
    EXPECT_NE(result.find("v_add_f32"), std::string::npos);
}

TEST_F(AsmMovePropagationPassTest, StopsPropagationAfterSourceIsRedefined) {
    std::string irString = R"(
v[0] = "st.v_mov_b32"(v[1])
v[2] = "st.v_add_f32"(v[0], v[3])
v[1] = "st.v_sub_f32"(v[6], v[7])
v[4] = "st.v_mul_f32"(v[0], v[5])
v[0] = "st.v_add_f32"(v[8], v[9])
"st.buffer_store_b32"(v[40], v[4])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    createAsmMovePropagationPass()->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 1);
    EXPECT_NE(result.find("v_add_f32"), std::string::npos);
    EXPECT_NE(result.find("v_mul_f32"), std::string::npos);
}

TEST_F(AsmMovePropagationPassTest, PropagatesThroughMoveChain) {
    std::string irString = R"(
v[0] = "st.v_mov_b32"(v[1])
v[2] = "st.v_mov_b32"(v[0])
v[3] = "st.v_add_f32"(v[2], v[4])
v[2] = "st.v_sub_f32"(v[5], v[6])
v[0] = "st.v_sub_f32"(v[7], v[8])
"st.buffer_store_b32"(v[40], v[3])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    createAsmMovePropagationPass()->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 0);
    EXPECT_NE(result.find("v_add_f32"), std::string::npos);
}

TEST_F(AsmMovePropagationPassTest, KeepsMoveWhenPotentiallyLiveOut) {
    std::string irString = R"(
v[0] = "st.v_mov_b32"(v[1])
v[2] = "st.v_add_f32"(v[3], v[4])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    createAsmMovePropagationPass()->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 1);
}

TEST_F(AsmMovePropagationPassTest, RemovesIdentityMove) {
    std::string irString = R"(
v[0] = "st.v_mov_b32"(v[0])
v[2] = "st.v_add_f32"(v[3], v[4])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    createAsmMovePropagationPass()->run(*func, passCtx, am);

    std::string result = getFunctionIR(*func);
    EXPECT_EQ(countOccurrences(result, "v_mov_b32"), 0);
}
