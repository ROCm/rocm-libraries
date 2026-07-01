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

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"

using namespace stinkytofu;

// Helper class for building test IR
class AsmEmitterTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);  // GFX1250
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;

        gemmConfig.TileA0 = 0;
        gemmConfig.TileB0 = 0;
        gemmConfig.TileM0 = 0;
        gemmConfig.NumGRA = 0;
        gemmConfig.NumGRB = 0;
        gemmConfig.NumGRM = 0;
        gemmConfig.NumWaves = 0;

        passCtx.setGemmTileConfig(gemmConfig);

        func.setGemmTileConfig(gemmConfig);
        bb = func.createBasicBlock("entry");
        irBuilder = std::make_unique<AsmIRBuilder>(*bb, arch);
    }

    StinkyInstruction* createInstruction(const std::string& mnemonic) {
        auto opcode = getMnemonicToIsaOpcode(mnemonic, arch);
        const HwInstDesc* hwInstDesc = getMCIDByIsaOp(opcode, arch);

        if (!hwInstDesc) {
            return nullptr;
        }

        return irBuilder->create(hwInstDesc);
    }

    Function func;
    BasicBlock* bb = nullptr;
    PassContext passCtx;
    GemmTileConfig gemmConfig;
    GfxArchID arch;
    std::unique_ptr<AsmIRBuilder> irBuilder;
};

// ============================================================================
// Basic Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitSingleInstruction) {
    // Create a simple ds_load_b128 instruction
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    // Add destination and source registers
    inst->addDestReg(StinkyRegister("v", 0, 4));  // v[0:3]
    inst->addSrcReg(StinkyRegister("v", 40, 1));  // v[40]

    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    // Emit with default options (emitCycleInfo = false by default)
    StinkyAsmEmitter emitter;
    std::string assembly = emitter.emit(*inst);

    // Compare exact assembly output (no cycle info with default options)
    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithCycleInfo) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitCycleInfo = true;
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    ds_load_b128 v[0:3], v40 // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithoutCycleInfo) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Register Format Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitVectorRegisterRange) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 10, 4));  // v[10:13]

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    ds_load_b128 v[10:13]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitSingleVectorRegister) {
    StinkyInstruction* inst = createInstruction("v_mov_b32");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 5, 1));  // v5
    inst->addSrcReg(StinkyRegister("v", 3, 1));   // v3

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    v_mov_b32 v5, v3\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitAccumulatorRegister) {
    StinkyInstruction* inst = createInstruction("v_wmma_scale_f32_16x16x128_f8f6f4");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 8));  // v[0:7]
    inst->addSrcReg(StinkyRegister("v", 8, 8));   // v[8:15]
    inst->addSrcReg(StinkyRegister("v", 16, 8));  // v[16:23]
    inst->addSrcReg(StinkyRegister("v", 0, 8));   // v[0:7]

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected =
        "    v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7]\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitScalarRegister) {
    StinkyInstruction* inst = createInstruction("s_add_i32");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("s", 10, 1));  // s10

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    s_add_i32 s10\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitLiteralInt) {
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);

    inst->addSrcReg(StinkyRegister(0));  // literal int 0

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    s_waitcnt 0\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Label Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitLabel) {
    StinkyInstruction* label = irBuilder->createLabel("loop_start");
    ASSERT_NE(label, nullptr);

    AsmEmitterOptions options;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*label);

    // Label should not be indented and should end with colon
    std::string expected = "loop_start:\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// IRList Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitIRList) {
    // Create a label
    irBuilder->createLabel("loop_start");

    // Create two ds_read instructions
    StinkyInstruction* inst1 = createInstruction("ds_load_b128");
    ASSERT_NE(inst1, nullptr);
    inst1->addDestReg(StinkyRegister("v", 0, 4));
    inst1->addSrcReg(StinkyRegister("v", 40, 1));

    StinkyInstruction* inst2 = createInstruction("ds_load_b128");
    ASSERT_NE(inst2, nullptr);
    inst2->addDestReg(StinkyRegister("v", 4, 4));
    inst2->addSrcReg(StinkyRegister("v", 41, 1));

    ASSERT_EQ(bb->size(), 3);

    AsmEmitterOptions options;
    options.emitComments = true;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(func);

    std::string expected =
        "loop_start:\n"
        "    ds_load_b128 v[0:3], v40\n"
        "    ds_load_b128 v[4:7], v41\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitIRListWithoutComments) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(func);

    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Indentation Tests
// ============================================================================

TEST_F(AsmEmitterTest, CustomIndentation) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));

    AsmEmitterOptions options;
    options.indent = 8;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "        ds_load_b128 v[0:3]\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Stream Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitToStream) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    std::ostringstream oss;
    StinkyAsmEmitter emitter;
    emitter.emit(oss, *inst);

    std::string assembly = oss.str();
    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitToStreamWithCycleInfo) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitCycleInfo = true;
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    std::ostringstream oss;
    StinkyAsmEmitter emitter(options);
    emitter.emit(oss, *inst);

    std::string assembly = oss.str();
    std::string expected = "    ds_load_b128 v[0:3], v40 // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(AsmEmitterTest, ToAssemblyUtility) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments = false;

    std::string assembly = toAssembly(func, options);
    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, ToAssemblyUtilityWithCycleInfo) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = true;
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    std::string assembly = toAssembly(func, options);
    std::string expected = "    ds_load_b128 v[0:3], v40 // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, ToAssemblyUtilityWithOptions) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;

    std::string assembly = toAssembly(func, options);
    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(AsmEmitterTest, GetSetOptions) {
    StinkyAsmEmitter emitter;

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 2;

    emitter.setOptions(options);

    const AsmEmitterOptions& retrievedOptions = emitter.getOptions();
    EXPECT_EQ(retrievedOptions.emitComments, false);
    EXPECT_EQ(retrievedOptions.emitCycleInfo, false);
    EXPECT_EQ(retrievedOptions.indent, 2);
}

// ============================================================================
// User Comment Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitWithUserComment) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments = true;
    options.emitCycleInfo = false;
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "    ds_load_b128 v[0:3], v40 // load C\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitWithCycleInfoAndUserComment) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments = true;
    options.emitCycleInfo = true;
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Cycle info should come first, then user comment
    std::string expected = "    ds_load_b128 v[0:3], v40 // issue=4 latency=52, load C\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitUserCommentDisabled) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments = false;  // Comments disabled
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // User comment should not appear
    std::string expected = "    ds_load_b128 v[0:3], v40\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, EmitCycleInfoOnlyWithUserComment) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 0, 4)});
    inst->setSrcRegs({StinkyRegister("v", 40, 1)});
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    // Add a user comment
    inst->addModifier(CommentData("load C"));

    AsmEmitterOptions options;
    options.emitComments = false;    // User comments disabled
    options.emitCycleInfo = true;    // But cycle info enabled
    options.commentAlignColumn = 0;  // Disable comment alignment for this test

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Only cycle info should appear, no user comment
    std::string expected = "    ds_load_b128 v[0:3], v40 // issue=4 latency=52\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierNegation) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 0, 1)});
    inst->setSrcRegs({StinkyRegister("v", 1, 1), StinkyRegister("v", 2, 1)});

    // Add VOP3 modifier to negate src0
    VOP3Modifiers mod;
    mod.neg_src0 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v0, -v1, v2
    std::string expected = "    v_add_f32 v0, -v1, v2\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierAbsoluteValue) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);

    inst->setDestRegs({StinkyRegister("v", 10, 1)});
    inst->setSrcRegs({StinkyRegister("v", 11, 1), StinkyRegister("v", 12, 1)});

    // Add VOP3 modifier for absolute value of src0
    VOP3Modifiers mod;
    mod.abs_src0 = true;
    inst->addModifier(mod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v10, abs(v11), v12
    std::string expected = "    v_add_f32 v10, abs(v11), v12\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierNegatedAbsoluteValue) {
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
    options.emitComments = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Should render as: v_add_f32 v20, -abs(v21), v22
    // This follows LLVM syntax: "-" before "abs()" is allowed
    std::string expected = "    v_add_f32 v20, -abs(v21), v22\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, VOP3ModifierMultipleSources) {
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
    options.emitComments = false;
    options.emitCycleInfo = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Should render with modifiers on each source according to LLVM syntax
    std::string expected = "    v_fma_f32 v30, -v31, abs(v32), -abs(v33)\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// MUBUF "off" vaddr Tests
//
// When a BufferStore/Load is created with isOff=true (rocisa vgpr("off",
// isOff=True)), the conversion layer produces StinkyRegister("off") — a
// literal-string register — instead of a normal VGPR.  The emitter must
// write the keyword "off", not "v[vgproff]".
// ============================================================================

// buffer_store_b32 vdata=v12, vaddr=off, srsrc=s[60:63], soffset=s46
TEST_F(AsmEmitterTest, MUBUFStoreB32_OffVAddr) {
    StinkyInstruction* inst = createInstruction("buffer_store_b32");
    ASSERT_NE(inst, nullptr);

    // buffer_store_b32 has no dest — only sources: vdata, vaddr, rsrc, soffset
    inst->addSrcReg(StinkyRegister("v", 12, 1));  // vdata
    inst->addSrcReg(StinkyRegister("off"));       // vaddr = off keyword
    inst->addSrcReg(StinkyRegister("s", 60, 4));  // rsrc  s[60:63]
    inst->addSrcReg(StinkyRegister("s", 46, 1));  // soffset s46

    // glc+slc on an arch that uses sc0/sc1 naming (gfx1250, hasMUBUFConst=false)
    MUBUFModifiers mubufMod(/*offen=*/false, /*offset12=*/0, /*glc=*/true, /*slc=*/true,
                            /*nt=*/false, /*lds=*/false, /*isStore=*/true,
                            /*hasMUBUFConst=*/false, /*hasGLCModifier=*/false,
                            /*hasSC0Modifier=*/true);
    inst->addModifier(mubufMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "buffer_store_b32 v12, off, s[60:63], s46 sc0 sc1\n";
    EXPECT_EQ(assembly, expected);
}

// buffer_load_b32 vdst=v0, vaddr=off, srsrc=s[4:7], soffset=s3
TEST_F(AsmEmitterTest, MUBUFLoadB32_OffVAddr) {
    StinkyInstruction* inst = createInstruction("buffer_load_b32");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 1));  // vdst
    inst->addSrcReg(StinkyRegister("off"));       // vaddr = off keyword
    inst->addSrcReg(StinkyRegister("s", 4, 4));   // rsrc  s[4:7]
    inst->addSrcReg(StinkyRegister("s", 3, 1));   // soffset s3

    MUBUFModifiers mubufMod(/*offen=*/false, /*offset12=*/0, /*glc=*/false, /*slc=*/false,
                            /*nt=*/false, /*lds=*/false, /*isStore=*/false,
                            /*hasMUBUFConst=*/false, /*hasGLCModifier=*/false,
                            /*hasSC0Modifier=*/false);
    inst->addModifier(mubufMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected = "buffer_load_b32 v0, off, s[4:7], s3\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, MUBUFScopeModifier) {
    StinkyInstruction* inst = createInstruction("buffer_store_b32");
    ASSERT_NE(inst, nullptr);

    inst->addSrcReg(StinkyRegister("v", 12, 1));
    inst->addSrcReg(StinkyRegister("v", 32, 1));
    inst->addSrcReg(StinkyRegister("s", 60, 4));
    inst->addSrcReg(StinkyRegister("s", 46, 1));

    MUBUFModifiers mubufMod(/*offen=*/true, /*offset12=*/0, /*glc=*/false, /*slc=*/false,
                            /*nt=*/false, /*lds=*/false, /*isStore=*/true,
                            /*hasMUBUFConst=*/false, /*hasGLCModifier=*/false,
                            /*hasSC0Modifier=*/false, /*scope=*/MUBUFScope::SCOPE_DEV);
    inst->addModifier(mubufMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected =
        "buffer_store_b32 v12, v32, s[60:63], s46 offen offset:0 scope:SCOPE_DEV\n";
    EXPECT_EQ(assembly, expected);
}

TEST_F(AsmEmitterTest, MUBUFTemporalHintModifier) {
    StinkyInstruction* inst = createInstruction("buffer_load_b128");
    ASSERT_NE(inst, nullptr);

    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 8, 1));
    inst->addSrcReg(StinkyRegister("s", 0, 4));
    inst->addSrcReg(StinkyRegister("s", 0, 1));

    MUBUFModifiers mubufMod(/*offen=*/true, /*offset12=*/0, /*glc=*/false, /*slc=*/false,
                            /*nt=*/false, /*lds=*/false, /*isStore=*/false,
                            /*hasMUBUFConst=*/false, /*hasGLCModifier=*/false,
                            /*hasSC0Modifier=*/false, /*scope=*/MUBUFScope::SCOPE_CU,
                            /*th=*/TemporalHint::TH_RT);
    inst->addModifier(mubufMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    std::string expected =
        "buffer_load_b128 v[0:3], v8, s[0:3], s0 offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT\n";
    EXPECT_EQ(assembly, expected);
}

// ============================================================================
// DS Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, DSModifierSingleOffset) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->addModifier(DSModifiers(/*na=*/1, /*offset=*/128));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_EQ(assembly, "ds_load_b128 v[0:3], v40 offset:128\n");
}

TEST_F(AsmEmitterTest, DSModifierDualOffset) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    // na=2 triggers offset0/offset1 emission
    inst->addModifier(DSModifiers(/*na=*/2, /*offset=*/0, /*offset0=*/4, /*offset1=*/8));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_EQ(assembly, "ds_load_b128 v[0:3], v40 offset0:4 offset1:8\n");
}

TEST_F(AsmEmitterTest, DSModifierGds) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->addModifier(DSModifiers(/*na=*/1, /*offset=*/0, /*offset0=*/0, /*offset1=*/0, /*gds=*/true));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("gds"), std::string::npos);
}

// ============================================================================
// SDWA Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, SDWAModifierDstSel) {
    StinkyInstruction* inst = createInstruction("v_mov_b32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));

    SDWAModifiers sdwa(SDWAModifiers::SelectBit::BYTE_0, SDWAModifiers::UnusedBit::UNUSED_PAD,
                       SDWAModifiers::SelectBit::BYTE_1, SDWAModifiers::SelectBit::SEL_NONE);
    inst->addModifier(sdwa);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("dst_sel:BYTE_0"), std::string::npos);
    EXPECT_NE(assembly.find("dst_unused:UNUSED_PAD"), std::string::npos);
    EXPECT_NE(assembly.find("src0_sel:BYTE_1"), std::string::npos);
}

// ============================================================================
// DPP Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, DPPModifierRowShr) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));

    DPPModifiers dpp(dppRowShr(3));
    inst->addModifier(dpp);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("row_shr:3"), std::string::npos);
}

TEST_F(AsmEmitterTest, DPPModifierRowMaskAndBankMask) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));

    // rowMask != 0xF and bankMask != 0xF trigger emission
    DPPModifiers dpp(dppRowShl(1), /*rowMask=*/0xA, /*bankMask=*/0x5);
    inst->addModifier(dpp);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("row_mask:0xa"), std::string::npos);
    EXPECT_NE(assembly.find("bank_mask:0x5"), std::string::npos);
}

TEST_F(AsmEmitterTest, DPPModifierBoundCtrlAndFi) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));

    DPPModifiers dpp(dppRowShr(1), /*rowMask=*/0xF, /*bankMask=*/0xF, /*boundCtrl=*/1, /*fi=*/true);
    inst->addModifier(dpp);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("bound_ctrl:1"), std::string::npos);
    EXPECT_NE(assembly.find("fi:1"), std::string::npos);
}

TEST_F(AsmEmitterTest, DPPModifierDPP8) {
    StinkyInstruction* inst = createInstruction("v_add_f32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));

    DPPModifiers dpp;
    dpp.isDPP8 = true;
    dpp.dpp8 = {0, 1, 2, 3, 4, 5, 6, 7};
    inst->addModifier(dpp);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("dpp8:[0,1,2,3,4,5,6,7]"), std::string::npos);
}

// ============================================================================
// MatrixFmt Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, MatrixFmtModifier) {
    StinkyInstruction* inst = createInstruction("v_wmma_scale_f32_16x16x128_f8f6f4");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 8));
    inst->addSrcReg(StinkyRegister("v", 8, 8));
    inst->addSrcReg(StinkyRegister("v", 16, 8));
    inst->addSrcReg(StinkyRegister("v", 0, 8));

    MatrixFmtModifiers fmtMod;
    fmtMod.fmtA = MatrixFmt::FP8;
    fmtMod.fmtB = MatrixFmt::BF8;
    inst->addModifier(fmtMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("matrix_a_fmt:MATRIX_FMT_FP8"), std::string::npos);
    EXPECT_NE(assembly.find("matrix_b_fmt:MATRIX_FMT_BF8"), std::string::npos);
}

TEST_F(AsmEmitterTest, MatrixScaleFmtModifier) {
    StinkyInstruction* inst = createInstruction("v_wmma_scale_f32_16x16x128_f8f6f4");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 8));
    inst->addSrcReg(StinkyRegister("v", 8, 8));
    inst->addSrcReg(StinkyRegister("v", 16, 8));
    inst->addSrcReg(StinkyRegister("v", 0, 8));

    MatrixFmtModifiers fmtMod;
    fmtMod.scaleFmtA = MatrixScaleFmt::E8;
    fmtMod.scaleFmtB = MatrixScaleFmt::E4M3;
    inst->addModifier(fmtMod);

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Scale formats are emitted as raw integers
    EXPECT_NE(assembly.find("matrix_a_scale_fmt:0"), std::string::npos);
    EXPECT_NE(assembly.find("matrix_b_scale_fmt:2"), std::string::npos);
}

// ============================================================================
// s_delay_alu Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, DelayAluSingleDep) {
    StinkyInstruction* inst = createInstruction("s_delay_alu");
    ASSERT_NE(inst, nullptr);
    inst->addModifier(SDelayAluData(SDelayAluData::InstType::VALU, /*distance=*/1));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("s_delay_alu"), std::string::npos);
    EXPECT_NE(assembly.find("VALU"), std::string::npos);
}

TEST_F(AsmEmitterTest, DelayAluDualDep) {
    StinkyInstruction* inst = createInstruction("s_delay_alu");
    ASSERT_NE(inst, nullptr);
    inst->addModifier(SDelayAluData(SDelayAluData::InstType::VALU, /*id0Dist=*/1,
                                    /*skip=*/1,
                                    SDelayAluData::InstType::SALU, /*id1Dist=*/2));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("s_delay_alu"), std::string::npos);
    EXPECT_NE(assembly.find("VALU"), std::string::npos);
    EXPECT_NE(assembly.find("SALU"), std::string::npos);
}

// ============================================================================
// s_waitcnt with SWaitCntData Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, WaitCntBothZeroEmitsLiteral0) {
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);
    // vlcnt=0, dscnt=0 → lgkmcnt=0, vmcnt=0 → emits " 0"
    inst->addModifier(SWaitCntData(/*vlcnt=*/0, /*vscnt=*/-1, /*dlcnt=*/-1, /*dscnt=*/0));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_EQ(assembly, "s_waitcnt 0\n");
}

TEST_F(AsmEmitterTest, WaitCntLgkmcntOnly) {
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);
    // dscnt=2 → lgkmcnt=2, vmcnt=-1
    inst->addModifier(SWaitCntData(/*vlcnt=*/-1, /*vscnt=*/-1, /*dlcnt=*/-1, /*dscnt=*/2));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_EQ(assembly, "s_waitcnt lgkmcnt(2)\n");
}

TEST_F(AsmEmitterTest, WaitCntVmcntOnly) {
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);
    // vlcnt=4 → lgkmcnt=-1, vmcnt=4
    inst->addModifier(SWaitCntData(/*vlcnt=*/4));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_EQ(assembly, "s_waitcnt vmcnt(4)\n");
}

TEST_F(AsmEmitterTest, WaitCntBothCounts) {
    StinkyInstruction* inst = createInstruction("s_waitcnt");
    ASSERT_NE(inst, nullptr);
    // dscnt=1 → lgkmcnt=1, vlcnt=3 → vmcnt=3
    inst->addModifier(SWaitCntData(/*vlcnt=*/3, /*vscnt=*/-1, /*dlcnt=*/-1, /*dscnt=*/1));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("lgkmcnt(1)"), std::string::npos);
    EXPECT_NE(assembly.find("vmcnt(3)"), std::string::npos);
}

// ============================================================================
// GLOBAL Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, GLOBALModifierOffset) {
    // global_prefetch_b8 is the gfx1250 VGLOBAL instruction that carries GLOBALModifiers
    StinkyInstruction* inst = createInstruction("global_prefetch_b8");
    ASSERT_NE(inst, nullptr);
    inst->addSrcReg(StinkyRegister("v", 2, 2));  // vaddr
    inst->addSrcReg(StinkyRegister("s", 0, 2));  // saddr
    inst->addModifier(GLOBALModifiers(/*offset=*/256));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("offset:256"), std::string::npos);
}

TEST_F(AsmEmitterTest, GLOBALModifierScopeAndTh) {
    StinkyInstruction* inst = createInstruction("global_prefetch_b8");
    ASSERT_NE(inst, nullptr);
    inst->addSrcReg(StinkyRegister("v", 2, 2));
    inst->addSrcReg(StinkyRegister("s", 0, 2));
    inst->addModifier(GLOBALModifiers(/*offset=*/0, TemporalHint::TH_NT, MUBUFScope::SCOPE_SE));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("scope:SCOPE_SE"), std::string::npos);
}

// ============================================================================
// VOP3P Modifier Tests
// ============================================================================

TEST_F(AsmEmitterTest, VOP3PModifierOpSel) {
    StinkyInstruction* inst = createInstruction("v_wmma_f32_16x16x16_f16");
    if (!inst) GTEST_SKIP() << "v_wmma_f32_16x16x16_f16 not available on this arch";

    // Operands must be present so emitOperands() runs and emits the VOP3P modifier
    inst->addDestReg(StinkyRegister("v", 0, 8));
    inst->addSrcReg(StinkyRegister("v", 8, 8));
    inst->addSrcReg(StinkyRegister("v", 16, 8));
    inst->addSrcReg(StinkyRegister("v", 0, 8));
    inst->addModifier(VOP3PModifiers({0, 1}, {1, 0}, {}));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("op_sel:"), std::string::npos);
    EXPECT_NE(assembly.find("op_sel_hi:"), std::string::npos);
}

// ============================================================================
// Multiple Basic Blocks and Blank Lines
// ============================================================================

TEST_F(AsmEmitterTest, MultipleBasicBlocks) {
    BasicBlock* bb2 = func.createBasicBlock("exit");

    irBuilder->createLabel("entry");
    StinkyInstruction* inst1 = createInstruction("v_mov_b32");
    ASSERT_NE(inst1, nullptr);
    inst1->addDestReg(StinkyRegister("v", 0, 1));
    inst1->addSrcReg(StinkyRegister("v", 1, 1));

    AsmIRBuilder builder2(*bb2, arch);
    builder2.createLabel("exit");
    builder2.create(getMCIDByUOp(GFX::s_nop, arch));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(func);

    EXPECT_NE(assembly.find("entry:"), std::string::npos);
    EXPECT_NE(assembly.find("exit:"), std::string::npos);
    EXPECT_NE(assembly.find("v_mov_b32"), std::string::npos);
    EXPECT_NE(assembly.find("s_nop"), std::string::npos);
}

TEST_F(AsmEmitterTest, CommentAlignColumn) {
    StinkyInstruction* inst = createInstruction("ds_load_b128");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 0, 4));
    inst->addSrcReg(StinkyRegister("v", 40, 1));
    inst->issueCycles = 4;
    inst->latencyCycles = 52;

    AsmEmitterOptions options;
    options.emitCycleInfo = true;
    options.commentAlignColumn = 60;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    // Comment should appear after alignment padding
    auto commentPos = assembly.find("//");
    EXPECT_NE(commentPos, std::string::npos);
    EXPECT_GE(commentPos, 40u);
}

// ============================================================================
// Special Register Emission Tests
// ============================================================================

TEST_F(AsmEmitterTest, EmitExecRegister) {
    StinkyInstruction* inst = createInstruction("s_mov_b64");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister(RegType::EXEC, 0, 1));
    inst->addSrcReg(StinkyRegister("s", 0, 2));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("exec"), std::string::npos);
}

TEST_F(AsmEmitterTest, EmitVccRegister) {
    StinkyInstruction* inst = createInstruction("s_mov_b32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister(RegType::VCC_LO, 0, 1));
    inst->addSrcReg(StinkyRegister("s", 0, 1));

    AsmEmitterOptions options;
    options.emitComments = false;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;

    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(*inst);

    EXPECT_NE(assembly.find("vcc"), std::string::npos);
}
