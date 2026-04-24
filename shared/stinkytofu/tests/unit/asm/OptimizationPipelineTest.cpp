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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/OptimizationPasses.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"

using namespace stinkytofu;

class OptimizationPipelineTest : public ::testing::Test {
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
            std::cerr << "Failed to parse IR" << std::endl;
            return nullptr;
        }
        return func;
    }

    int countInstructions(Function& func) {
        int count = 0;
        for (BasicBlock& bb : func) {
            for (IRBase& ir : bb) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
            }
        }
        return count;
    }

    GemmTileConfig gemmConfig;
};

// O0: no optimization passes, IR unchanged
TEST_F(OptimizationPipelineTest, OptLevelO0) {
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    int instCountBefore = countInstructions(*func);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    addPeepholeOptPasses(pm, OptLevel::O0);
    pm.run(*func);

    int instCountAfter = countInstructions(*func);

    // O0 should not change anything
    EXPECT_EQ(instCountBefore, instCountAfter);

    converter.cleanup();
}

// O1: peephole only, 1 iteration
TEST_F(OptimizationPipelineTest, OptLevelO1) {
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    addPeepholeOptPasses(pm, OptLevel::O1);

    // Should not crash
    pm.run(*func);
    EXPECT_GT(countInstructions(*func), 0);

    converter.cleanup();
}

// O2: peephole + DCE, 1 iteration
TEST_F(OptimizationPipelineTest, OptLevelO2) {
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    addPeepholeOptPasses(pm, OptLevel::O2);

    // Should not crash
    pm.run(*func);
    EXPECT_GT(countInstructions(*func), 0);

    converter.cleanup();
}

// O3: peephole + redundant mov elim + DCE, 3 iterations
TEST_F(OptimizationPipelineTest, OptLevelO3) {
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    addPeepholeOptPasses(pm, OptLevel::O3);

    // Should not crash
    pm.run(*func);
    EXPECT_GT(countInstructions(*func), 0);

    converter.cleanup();
}
