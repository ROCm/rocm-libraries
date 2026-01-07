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

#include "ir/asm/DeadCodeEliminationPass.hpp"
#include "ir/asm/DuplicateEliminationPass.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu.hpp"

#include <optional>
#include <sstream>
#include <string>

using namespace stinkytofu;

class DuplicateEliminationPassTest : public ::testing::Test
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

    std::string getFunctionIR(Function& func)
    {
        std::ostringstream oss;
        AsmPrinter         printer(oss);
        for(BasicBlock& bb : func)
        {
            for(IRBase& ir : bb.getIR())
            {
                if(ir.getType() == IRBase::IRType::StinkyTofu)
                {
                    auto* inst = static_cast<StinkyInstruction*>(&ir);
                    printer.print(*inst);
                    oss << "\n";
                }
            }
        }
        return oss.str();
    }

    int countOccurrences(const std::string& str, const std::string& substr)
    {
        int                    count = 0;
        std::string::size_type pos   = 0;
        while((pos = str.find(substr, pos)) != std::string::npos)
        {
            count++;
            pos++;
        }
        return count;
    }

    GemmTileConfig gemmConfig;
};

// Test 1: Eliminate simple duplicate
TEST_F(DuplicateEliminationPassTest, EliminateSimpleDuplicate)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[6] = "st.v_mul_f32"(v[1], v[2])
        v[7] = "st.v_sub_f32"(v[6], v[0])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_mul_f32 instruction (duplicate eliminated)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);

    // v_sub should now use v0 instead of v6
    EXPECT_NE(result.find("st.v_sub_f32"), std::string::npos);
}

// Test 2: Multiple duplicates of same instruction
TEST_F(DuplicateEliminationPassTest, MultipleDuplicates)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_mul_f32"(v[1], v[2])
        v[5] = "st.v_mul_f32"(v[1], v[2])
        v[7] = "st.v_add_f32"(v[0], v[3])
        v[9] = "st.v_add_f32"(v[5], v[7])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_mul_f32 instruction (all duplicates eliminated)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);
}

// Test 3: Don't eliminate when operand is modified
TEST_F(DuplicateEliminationPassTest, DontEliminateWithModifiedOperand)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[1] = "st.v_sub_f32"(v[6], v[7])
        v[8] = "st.v_mul_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should have 2 v_mul_f32 instructions (v1 was modified between them)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 2);
}

// Test 4: Different operands - not duplicates
TEST_F(DuplicateEliminationPassTest, DifferentOperands)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_mul_f32"(v[4], v[5])
        v[6] = "st.v_mul_f32"(v[1], v[5])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should have 3 v_mul_f32 instructions (all have different operands)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 3);
}

// Test 5: Different opcodes - not duplicates
TEST_F(DuplicateEliminationPassTest, DifferentOpcodes)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[1], v[2])
        v[5] = "st.v_sub_f32"(v[1], v[2])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should have all 3 instructions (different opcodes)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);
    EXPECT_EQ(countOccurrences(result, "v_add_f32"), 1);
    EXPECT_EQ(countOccurrences(result, "v_sub_f32"), 1);
}

// Test 6: Duplicate with constants
TEST_F(DuplicateEliminationPassTest, DuplicateWithConstants)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], 2.0)
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[6] = "st.v_mul_f32"(v[1], 2.0)
        v[8] = "st.v_add_f32"(v[0], v[6])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_mul_f32 instruction
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);
}

// Test 7: Combination with DCE
TEST_F(DuplicateEliminationPassTest, CombinationWithDCE)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[8] = "st.v_fma_f32"(v[0], v[3], 1.0)
        "st.global_store_dword"(v[40], v[8])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    // First eliminate duplicates
    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    // Then remove dead code
    auto dcePass = createDeadCodeEliminationPass();
    dcePass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // After duplicate elimination: second v0 reuses first computation
    // After DCE: first v0 is overwritten by second v0 -> removed
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);
    EXPECT_EQ(countOccurrences(result, "v_add_f32"), 1);
    EXPECT_EQ(countOccurrences(result, "v_fma_f32"), 1);
}

// Test 8: Empty IR
TEST_F(DuplicateEliminationPassTest, EmptyIR)
{
    std::string irString = "";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);
    // Empty IR should not cause any errors
}

// Test 9: No duplicates
TEST_F(DuplicateEliminationPassTest, NoDuplicates)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[6] = "st.v_sub_f32"(v[7], v[8])
        v[9] = "st.v_fma_f32"(v[10], v[11], 1.0)
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    std::string before = getFunctionIR(*func);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string after = getFunctionIR(*func);

    // Should be unchanged
    EXPECT_EQ(countOccurrences(after, "v_mul_f32"), 1);
    EXPECT_EQ(countOccurrences(after, "v_add_f32"), 1);
    EXPECT_EQ(countOccurrences(after, "v_sub_f32"), 1);
    EXPECT_EQ(countOccurrences(after, "v_fma_f32"), 1);
}

// Test 10: Duplicate FMA instructions
TEST_F(DuplicateEliminationPassTest, DuplicateFMA)
{
    std::string irString = R"(
        v[0] = "st.v_fma_f32"(v[1], v[2], 1.0)
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[6] = "st.v_fma_f32"(v[1], v[2], 1.0)
        v[8] = "st.v_sub_f32"(v[0], v[6])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_fma_f32 instruction
    EXPECT_EQ(countOccurrences(result, "v_fma_f32"), 1);
}

// Test 11: Three-way duplicate
TEST_F(DuplicateEliminationPassTest, ThreeWayDuplicate)
{
    std::string irString = R"(
        v[0] = "st.v_add_f32"(v[1], v[2])
        v[3] = "st.v_mul_f32"(v[4], v[5])
        v[6] = "st.v_add_f32"(v[1], v[2])
        v[8] = "st.v_sub_f32"(v[9], v[10])
        v[11] = "st.v_add_f32"(v[1], v[2])
        v[12] = "st.v_fma_f32"(v[0], v[6], v[11])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should only have 1 v_add_f32 instruction (all three duplicates eliminated)
    EXPECT_EQ(countOccurrences(result, "v_add_f32"), 1);
}

// Test 12: Interleaved duplicates
TEST_F(DuplicateEliminationPassTest, InterleavedDuplicates)
{
    std::string irString = R"(
        v[0] = "st.v_mul_f32"(v[1], v[2])
        v[3] = "st.v_add_f32"(v[4], v[5])
        v[6] = "st.v_mul_f32"(v[1], v[2])
        v[8] = "st.v_add_f32"(v[4], v[5])
        v[10] = "st.v_sub_f32"(v[0], v[3])
        v[11] = "st.v_fma_f32"(v[6], v[8], v[10])
    )";

    StinkyIRConverter converter;

    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    PassContext passCtx;
    passCtx.setGemmTileConfig(gemmConfig);

    auto dupElimPass = createDuplicateEliminationPass();
    dupElimPass->run(*func, passCtx);

    std::string result = getFunctionIR(*func);

    // Should have 1 mul and 1 add (duplicates eliminated)
    EXPECT_EQ(countOccurrences(result, "v_mul_f32"), 1);
    EXPECT_EQ(countOccurrences(result, "v_add_f32"), 1);
}
