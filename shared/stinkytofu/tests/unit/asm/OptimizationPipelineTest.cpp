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

#include "ir/asm/OptimizationPipeline.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu.hpp"

#include <sstream>
#include <string>

using namespace stinkytofu;

class OptimizationPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize GEMM config for GFX94X (gfx942)
        gemmConfig.arch     = {9, 4, 2};
        gemmConfig.NumWaves = 2;
        gemmConfig.TileA0   = 16;
        gemmConfig.TileB0   = 16;
        gemmConfig.TileM0   = 16;
        gemmConfig.NumGRA   = 4;
        gemmConfig.NumGRB   = 4;
        gemmConfig.NumGRM   = 4;
    }

    // Parse IR and return a non-owning pointer
    // The converter must be kept alive for the lifetime of the Function
    Function* parseIR(const std::string& irString, StinkyIRConverter& converter)
    {
        Function* func = converter.convertToFunction(irString);
        if(!func)
        {
            std::cerr << "Failed to parse IR" << std::endl;
            return nullptr;
        }
        return func;
    }

    int countInstructions(Function& func)
    {
        int count = 0;
        for(BasicBlock& bb : func)
        {
            for(IRBase& ir : bb.getIR())
            {
                if(ir.getType() == IRBase::IRType::StinkyTofu)
                    count++;
            }
        }
        return count;
    }

    GemmTileConfig gemmConfig;
};

// Test OptimizationOnly profile with O0
TEST_F(OptimizationPipelineTest, OptimizationOnlyO0)
{
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function*         func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    int instCountBefore = countInstructions(*func);

    // Run OptimizationOnly with O0
    PipelineConfig config
        = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O0);
    config.gemmTileConfig = std::make_unique<GemmTileConfig>(gemmConfig);
    OptimizationPipeline::run(*func, config);

    int instCountAfter = countInstructions(*func);

    // O0 should not change anything
    EXPECT_EQ(instCountBefore, instCountAfter);

    converter.cleanup();
}

// Test custom config
TEST_F(OptimizationPipelineTest, CustomConfig)
{
    std::string irString = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;
    Function*         func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    // Custom config with verbose
    PipelineConfig config
        = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O2);
    config.verbose                = false; // Don't spam test output
    config.optimizationIterations = 10; // Custom iteration count
    config.gemmTileConfig         = std::make_unique<GemmTileConfig>(gemmConfig);

    // Should not crash
    OptimizationPipeline::run(*func, config);

    // Basic sanity check
    EXPECT_GT(countInstructions(*func), 0);

    converter.cleanup();
}

// Test pipeline profiles
TEST_F(OptimizationPipelineTest, PipelineProfiles)
{
    // OptimizationOnly profile (general)
    auto configOpt = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly);
    EXPECT_EQ(configOpt.profile, PipelineProfile::OptimizationOnly);
    EXPECT_TRUE(configOpt.enablePeephole); // O1+
    EXPECT_TRUE(configOpt.enableDCE); // O2+
    EXPECT_FALSE(configOpt.enableDuplicateElim); // O3 only
    EXPECT_FALSE(configOpt.enableDAGScheduler);
    EXPECT_FALSE(configOpt.enableWaitCnt);

    // FullPipeline profile (for production kernels)
    auto configFull = PipelineConfig::forProductionKernel();
    EXPECT_EQ(configFull.profile, PipelineProfile::FullPipeline);
    EXPECT_TRUE(configFull.enablePeephole); // Need pattern fusion
    EXPECT_TRUE(configFull.enableDCE);
    EXPECT_TRUE(configFull.enableDAGScheduler);
    EXPECT_TRUE(configFull.enableWaitCnt);
}

// Test OptimizationOnly profile with different optimization levels
TEST_F(OptimizationPipelineTest, OptimizationLevels)
{
    auto configO0 = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O0);
    auto configO1 = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O1);
    auto configO2 = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O2);
    auto configO3 = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly, OptLevel::O3);

    // O0: no optimization passes, 1 iteration
    EXPECT_FALSE(configO0.enablePeephole);
    EXPECT_FALSE(configO0.enableDCE);
    EXPECT_FALSE(configO0.enableDuplicateElim);
    EXPECT_EQ(configO0.optimizationIterations, 1);

    // O1: peephole only, 1 iteration
    EXPECT_TRUE(configO1.enablePeephole);
    EXPECT_FALSE(configO1.enableDCE);
    EXPECT_FALSE(configO1.enableDuplicateElim);
    EXPECT_EQ(configO1.optimizationIterations, 1);

    // O2: peephole + DCE, 3 iterations
    EXPECT_TRUE(configO2.enablePeephole);
    EXPECT_TRUE(configO2.enableDCE);
    EXPECT_FALSE(configO2.enableDuplicateElim);
    EXPECT_EQ(configO2.optimizationIterations, 3);

    // O3: all passes enabled, 5 iterations
    EXPECT_TRUE(configO3.enablePeephole);
    EXPECT_TRUE(configO3.enableDCE);
    EXPECT_TRUE(configO3.enableDuplicateElim);
    EXPECT_EQ(configO3.optimizationIterations, 5);
}
