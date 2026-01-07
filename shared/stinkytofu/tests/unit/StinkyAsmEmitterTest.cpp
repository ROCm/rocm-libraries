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

#include "ir/asm/StinkyAsmEmitter.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"

using namespace stinkytofu;

// Helper class for building test IR
class AsmEmitterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        arch               = getGfxArchID(9, 4, 2); // GFX942
        gemmConfig.arch[0] = 9;
        gemmConfig.arch[1] = 4;
        gemmConfig.arch[2] = 2;

        gemmConfig.TileA0   = 0;
        gemmConfig.TileB0   = 0;
        gemmConfig.TileM0   = 0;
        gemmConfig.NumGRA   = 0;
        gemmConfig.NumGRB   = 0;
        gemmConfig.NumGRM   = 0;
        gemmConfig.NumWaves = 0;

        passCtx.setGemmTileConfig(gemmConfig);

        irBuilder = std::make_unique<StinkyInstIRBuilder>(insts, arch);
    }

    StinkyInstruction* createInstruction(const std::string& mnemonic)
    {
        auto              opcode     = getMnemonicToIsaOpcode(mnemonic, arch);
        const HwInstDesc* hwInstDesc = getMCIDByIsaOp(opcode, arch);

        if(!hwInstDesc)
        {
            return nullptr;
        }

        return irBuilder->createStinkyInstBefore(insts.end(), hwInstDesc);
    }

    IRList                               insts;
    PassContext                          passCtx;
    GemmTileConfig                       gemmConfig;
    GfxArchID                            arch;
    std::unique_ptr<StinkyInstIRBuilder> irBuilder;
};

// ============================================================================
// Basic Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitSingleInstruction)
{
    // Create a simple ds_read_b128 instruction
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    // Add destination and source registers
    inst->addDestReg(StinkyRegister("v", 0, 4)); // v[0:3]
    inst->addSrcReg(StinkyRegister("v", 40, 1)); // v[40]

    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    // Emit with default options (emitCycleInfo = false by default)
    StinkyAsmEmitter emitter;
    std::string      assembly = emitter.emit(*inst);

    // Compare exact assembly output (no cycle info with default options)
    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithCycleInfo)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitCycleInfo = true;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithoutCycleInfo)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Register Format Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitVectorRegisterRange)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 10, 4)); // v[10:13]

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    ds_read_b128 v[10:13]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitSingleVectorRegister)
{
    StinkyInstruction* inst = createInstruction("v_mov_b32");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 5, 1)); // v[5]
    inst->addSrcReg(StinkyRegister("v", 3, 1)); // v[3]

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    v_mov_b32 v[5], v[3]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitAccumulatorRegister)
{
    StinkyInstruction* inst = createInstruction("v_mfma_f32_16x16x16_f16");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("acc", 0, 16)); // acc[0:15]
    inst->addSrcReg(StinkyRegister("v", 6, 2)); // v[6:7]
    inst->addSrcReg(StinkyRegister("v", 22, 2)); // v[22:23]
    inst->addSrcReg(StinkyRegister("acc", 0, 16)); // acc[0:15]

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    v_mfma_f32_16x16x16_f16 acc[0:15], v[6:7], v[22:23], acc[0:15]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitScalarRegister)
{
    StinkyInstruction* inst = createInstruction("s_add_i32");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("s", 10, 1)); // s10

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    s_add_i32 s10\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitLiteralInt)
{
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);

    inst->addSrcReg(StinkyRegister(0)); // literal int 0

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    s_waitcnt 0\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Label Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitLabel)
{
    StinkyInstruction* label = irBuilder->createStinkyLabel(insts.end(), "loop_start");
    ASSERT_NE(label, nullptr);

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*label);

    // Label should not be indented and should end with colon
    std::string expected = "loop_start:\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// IRList Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitIRList)
{
    // Create a label
    irBuilder->createStinkyLabel(insts.end(), "loop_start");

    // Create two ds_read instructions
    StinkyInstruction* inst1 = createInstruction("ds_read_b128");
    ASSERT_NE(inst1, nullptr);
    inst1->addDestReg(StinkyRegister("v", 0, 4));
    inst1->addSrcReg(StinkyRegister("v", 40, 1));

    StinkyInstruction* inst2 = createInstruction("ds_read_b128");
    ASSERT_NE(inst2, nullptr);
    inst2->addDestReg(StinkyRegister("v", 4, 4));
    inst2->addSrcReg(StinkyRegister("v", 41, 1));

    ASSERT_EQ(insts.size(), 3);

    AsmEmitterOptions options;
    options.emitComments  = true;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(insts);

    std::string expected = "// ==================================================\n"
                           "// StinkyTofu Assembly Output\n"
                           "// Instructions: 3\n"
                           "// ==================================================\n"
                           "\n"
                           "loop_start:\n"
                           "    ds_read_b128 v[0:3], v[40]\n"
                           "    ds_read_b128 v[4:7], v[41]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitIRListWithoutComments)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(insts);

    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Indentation Tests
// ============================================================================

TEST_F(AsmEmitterTest, CustomIndentation)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));

    AsmEmitterOptions options;
    options.indent        = 8;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "        ds_read_b128 v[0:3]\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Stream Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitToStream)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    std::ostringstream oss;
    StinkyAsmEmitter   emitter;
    emitter.emit(oss, *inst);

    std::string assembly = oss.str();
    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitToStreamWithCycleInfo)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitCycleInfo = true;

    std::ostringstream oss;
    StinkyAsmEmitter   emitter(options);
    emitter.emit(oss, *inst);

    std::string assembly = oss.str();
    std::string expected = "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(AsmEmitterTest, ToAssemblyUtility)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments = false;

    std::string assembly = toAssembly(insts, options);
    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, ToAssemblyUtilityWithCycleInfo)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = true;

    std::string assembly = toAssembly(insts, options);
    std::string expected = "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, ToAssemblyUtilityWithOptions)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    std::string assembly = toAssembly(insts, options);
    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(AsmEmitterTest, GetSetOptions)
{
    StinkyAsmEmitter emitter;

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;
    options.indent        = 2;

    emitter.setOptions(options);

    const AsmEmitterOptions& retrievedOptions = emitter.getOptions();
    EXPECT_EQ(retrievedOptions.emitComments, false);
    EXPECT_EQ(retrievedOptions.emitCycleInfo, false);
    EXPECT_EQ(retrievedOptions.indent, 2);
}

// ============================================================================
// User Comment Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitWithUserComment)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments  = true;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    std::string expected = "    ds_read_b128 v[0:3], v[40] // load C\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithCycleInfoAndUserComment)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments  = true;
    options.emitCycleInfo = true;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Cycle info should come first, then user comment
    std::string expected = "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52, load C\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitUserCommentDisabled)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments  = false; // Comments disabled
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // User comment should not appear
    std::string expected = "    ds_read_b128 v[0:3], v[40]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitCycleInfoOnlyWithUserComment)
{
    StinkyInstruction* inst = createInstruction("ds_read_b128");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 0, 4)});
    inst->setSrcRegs({StinkyRegister("v", 40, 1)});
    inst->issueCycles   = 4;
    inst->latencyCycles = 52;

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments  = false; // User comments disabled
    options.emitCycleInfo = true; // But cycle info enabled

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Only cycle info should appear, no user comment
    std::string expected = "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierNegation)
{
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 0, 1)});
    inst->setSrcRegs({StinkyRegister("v", 1, 1), StinkyRegister("v", 2, 1)});

    // Add VOP3 modifier to negate src0
    VOP3Modifiers mod;
    mod.neg_src0 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v[0], -v[1], v[2]
    std::string expected = "    v_add_f32 v[0], -v[1], v[2]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierAbsoluteValue)
{
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 10, 1)});
    inst->setSrcRegs({StinkyRegister("v", 11, 1), StinkyRegister("v", 12, 1)});

    // Add VOP3 modifier for absolute value of src0
    VOP3Modifiers mod;
    mod.abs_src0 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v[10], abs(v[11]), v[12]
    std::string expected = "    v_add_f32 v[10], abs(v[11]), v[12]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierNegatedAbsoluteValue)
{
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 20, 1)});
    inst->setSrcRegs({StinkyRegister("v", 21, 1), StinkyRegister("v", 22, 1)});

    // Add VOP3 modifier for negated absolute value of src0
    VOP3Modifiers mod;
    mod.neg_src0 = true;
    mod.abs_src0 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v[20], -abs(v[21]), v[22]
    // This follows LLVM syntax: "-" before "abs()" is allowed
    std::string expected = "    v_add_f32 v[20], -abs(v[21]), v[22]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierMultipleSources)
{
    StinkyInstruction* inst = createInstruction("v_fma_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 30, 1)});
    inst->setSrcRegs(
        {StinkyRegister("v", 31, 1), StinkyRegister("v", 32, 1), StinkyRegister("v", 33, 1)});

    // Add VOP3 modifiers: neg src0, abs src1, neg+abs src2
    VOP3Modifiers mod;
    mod.neg_src0 = true;
    mod.abs_src1 = true;
    mod.neg_src2 = true;
    mod.abs_src2 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments  = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string      assembly = emitter.emit(*inst);

    // Should render with modifiers on each source according to LLVM syntax
    std::string expected = "    v_fma_f32 v[30], -v[31], abs(v[32]), -abs(v[33])\n";
    EXPECT_EQ(assembly, expected);
}
