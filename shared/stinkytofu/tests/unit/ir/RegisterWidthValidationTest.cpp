/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/IRBuilder.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::logical;
using namespace stinkytofu::test;

// ==============================================================================
// Register Width Validation Tests
// ==============================================================================

TEST(RegisterWidthValidationTest, TensorLoadToLds_CorrectWidths) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(
        static_cast<IRBase*>(stinkytofu::TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), &s2, &s3)));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass with correct register widths, got error: " << error;
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectSrc0Width) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(10, 4);
    StinkyRegister s3 = sgpr(14, 4);
    bb->appendIR(static_cast<IRBase*>(stinkytofu::TensorLoadToLds(
        sgpr(0, 2), sgpr(2, 8), &s2, &s3)  // group0=2 regs (WRONG!), group1=8 regs
                                      ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "Should fail with incorrect src0 width";
    EXPECT_NE(error.find("src[0]"), std::string::npos) << "Error should mention src[0]";
    EXPECT_NE(error.find("width 2"), std::string::npos) << "Error should mention actual width 2";
    EXPECT_NE(error.find("expected 4"), std::string::npos)
        << "Error should mention expected width 4";
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectSrc1Width) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(8, 4);
    StinkyRegister s3 = sgpr(12, 4);
    bb->appendIR(static_cast<IRBase*>(stinkytofu::TensorLoadToLds(
        sgpr(0, 4), sgpr(4, 4), &s2, &s3)  // group0=4 regs, group1=4 regs (WRONG!)
                                      ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "Should fail with incorrect src1 width";
    EXPECT_NE(error.find("src[1]"), std::string::npos) << "Error should mention src[1]";
    EXPECT_NE(error.find("width 4"), std::string::npos) << "Error should mention actual width 4";
    EXPECT_NE(error.find("expected 8"), std::string::npos)
        << "Error should mention expected width 8";
}

TEST(RegisterWidthValidationTest, OtherInstructions_NoWidthChecks) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    bb->appendIR(static_cast<IRBase*>(stinkytofu::VAddF32(vgpr(0), vgpr(1), vgpr(2))));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass for instructions without width requirements";
}

TEST(RegisterWidthValidationTest, DisableWidthChecks) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(2, 1);
    StinkyRegister s3 = sgpr(3, 1);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 1), sgpr(1, 1), &s2, &s3)  // All WRONG widths
        ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    AsmVerifierConfig verifierConfig;
    verifierConfig.checkRegisterWidths = false;

    std::string error = validateStinkyIR(func, verifierConfig);
    EXPECT_TRUE(error.empty()) << "Should pass when width checks are disabled";
}

// ==============================================================================
// Register Type Validation Tests
// ==============================================================================

TEST(RegisterWidthValidationTest, TensorLoadToLds_CorrectRegisterType) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(
        static_cast<IRBase*>(stinkytofu::TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), &s2, &s3)));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass with correct register types, got error: " << error;
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectRegisterType_Src0) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(stinkytofu::TensorLoadToLds(
        vgpr(0, 4), sgpr(4, 8), &s2, &s3)  // src0=VGPR (WRONG!), src1=SGPR
                                      ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "Should fail with incorrect src0 register type";
    EXPECT_NE(error.find("src[0]"), std::string::npos) << "Error should mention src[0]";
    EXPECT_NE(error.find("register type"), std::string::npos)
        << "Error should mention register type";
    EXPECT_NE(error.find("'v'"), std::string::npos) << "Error should mention actual type 'v'";
    EXPECT_NE(error.find("'s'"), std::string::npos) << "Error should mention expected type 's'";
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectRegisterType_Src1) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(stinkytofu::TensorLoadToLds(
        sgpr(0, 4), vgpr(4, 8), &s2, &s3)  // src0=SGPR, src1=VGPR (WRONG!)
                                      ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "Should fail with incorrect src1 register type";
    EXPECT_NE(error.find("src[1]"), std::string::npos) << "Error should mention src[1]";
    EXPECT_NE(error.find("register type"), std::string::npos)
        << "Error should mention register type";
    EXPECT_NE(error.find("'v'"), std::string::npos) << "Error should mention actual type 'v'";
    EXPECT_NE(error.find("'s'"), std::string::npos) << "Error should mention expected type 's'";
}

// ==============================================================================
// MUBUF Register Validation Tests
//
// buffer_store_b32 operands: S0=vdata(vgpr,32), S1=vaddr(vgpr,32), S2=rsrc(sreg,128), S3=soffset
// buffer_load_b32  operands: D0=vdst(vgpr,32),  S0=vaddr(vgpr,32), S1=rsrc(sreg,128), S2=soffset
// ==============================================================================

namespace {
/// Populate \p func with a single buffer_store_b32. The caller controls
/// register types/widths to probe the verifier.
void buildBufferStoreB32(Function& func, StinkyRegister vdata, StinkyRegister vaddr,
                         StinkyRegister rsrc, StinkyRegister soffset) {
    GfxArchID arch = getGfxArchID(12, 5, 0);
    setFunctionArch(func, arch);
    BasicBlock* bb = func.createBasicBlock("entry");
    AsmIRBuilder builder(*bb, arch);

    IsaOpcode opcode = getMnemonicToIsaOpcode("buffer_store_b32", arch);
    const HwInstDesc* desc = getMCIDByIsaOp(opcode, arch);
    assert(desc && "buffer_store_b32 not found for gfx1250");

    StinkyInstruction* inst = builder.create(desc);
    inst->addSrcReg(vdata);
    inst->addSrcReg(vaddr);
    inst->addSrcReg(rsrc);
    inst->addSrcReg(soffset);
    inst->addModifier(MUBUFModifiers());
}

/// Populate \p func with a single buffer_load_b32.
void buildBufferLoadB32(Function& func, StinkyRegister vdst, StinkyRegister vaddr,
                        StinkyRegister rsrc, StinkyRegister soffset) {
    GfxArchID arch = getGfxArchID(12, 5, 0);
    setFunctionArch(func, arch);
    BasicBlock* bb = func.createBasicBlock("entry");
    AsmIRBuilder builder(*bb, arch);

    IsaOpcode opcode = getMnemonicToIsaOpcode("buffer_load_b32", arch);
    const HwInstDesc* desc = getMCIDByIsaOp(opcode, arch);
    assert(desc && "buffer_load_b32 not found for gfx1250");

    StinkyInstruction* inst = builder.create(desc);
    inst->addDestReg(vdst);
    inst->addSrcReg(vaddr);
    inst->addSrcReg(rsrc);
    inst->addSrcReg(soffset);
    inst->addModifier(MUBUFModifiers());
}
}  // namespace

// buffer_store_b32 v12, off, s[60:63], s46
// "off" is a LiteralString after the isOff fix; the verifier must accept it.
TEST(MUBUFVerificationTest, BufferStoreB32_OffVaddr_Passes) {
    Function func("kernel");
    buildBufferStoreB32(func, vgpr(12), StinkyRegister("off"), sgpr(60, 4), sgpr(46));
    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "buffer_store_b32 with off vaddr should pass, got: " << error;
}

// buffer_load_b32 v0, off, s[4:7], s3
TEST(MUBUFVerificationTest, BufferLoadB32_OffVaddr_Passes) {
    Function func("kernel");
    buildBufferLoadB32(func, vgpr(0), StinkyRegister("off"), sgpr(4, 4), sgpr(3));
    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "buffer_load_b32 with off vaddr should pass, got: " << error;
}

// buffer_store_b32 with a VGPR vaddr (not off) must also pass.
TEST(MUBUFVerificationTest, BufferStoreB32_RegisterVaddr_Passes) {
    Function func("kernel");
    buildBufferStoreB32(func, vgpr(12), vgpr(1), sgpr(60, 4), sgpr(46));
    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "buffer_store_b32 with register vaddr should pass, got: "
                               << error;
}

// rsrc must be 4 SGPRs wide (128 bits). Passing 1 SGPR must be rejected.
TEST(MUBUFVerificationTest, BufferStoreB32_WrongRsrcWidth_Fails) {
    Function func("kernel");
    buildBufferStoreB32(func, vgpr(12), StinkyRegister("off"), sgpr(60, 1), sgpr(46));
    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "buffer_store_b32 with 1-wide rsrc should fail";
    EXPECT_NE(error.find("src[2]"), std::string::npos) << "Error should mention src[2] (rsrc)";
    EXPECT_NE(error.find("expected 4"), std::string::npos)
        << "Error should mention expected width 4";
}

// vdata must be a VGPR. Passing an SGPR must be rejected (type check for 32-bit field).
TEST(MUBUFVerificationTest, BufferStoreB32_WrongVdataType_Fails) {
    Function func("kernel");
    buildBufferStoreB32(func, sgpr(12), StinkyRegister("off"), sgpr(60, 4), sgpr(46));
    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "buffer_store_b32 with SGPR vdata should fail";
    EXPECT_NE(error.find("src[0]"), std::string::npos) << "Error should mention src[0] (vdata)";
    EXPECT_NE(error.find("register type"), std::string::npos)
        << "Error should mention register type";
    EXPECT_NE(error.find("'s'"), std::string::npos) << "Error should mention actual type 's'";
    EXPECT_NE(error.find("'v'"), std::string::npos) << "Error should mention expected type 'v'";
}

// soffset must be a SGPR/M0. Passing a VGPR must be rejected (type check for 32-bit field).
TEST(MUBUFVerificationTest, BufferStoreB32_WrongSoffsetType_Fails) {
    Function func("kernel");
    buildBufferStoreB32(func, vgpr(12), StinkyRegister("off"), sgpr(60, 4), vgpr(46));
    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "buffer_store_b32 with VGPR soffset should fail";
    EXPECT_NE(error.find("src[3]"), std::string::npos) << "Error should mention src[3] (soffset)";
    EXPECT_NE(error.find("register type"), std::string::npos)
        << "Error should mention register type";
    EXPECT_NE(error.find("'v'"), std::string::npos) << "Error should mention actual type 'v'";
    EXPECT_NE(error.find("'s'"), std::string::npos) << "Error should mention expected type 's'";
}

// vdst of a load must be a VGPR. Passing an SGPR must be rejected.
TEST(MUBUFVerificationTest, BufferLoadB32_WrongVdstType_Fails) {
    Function func("kernel");
    buildBufferLoadB32(func, sgpr(0), StinkyRegister("off"), sgpr(4, 4), sgpr(3));
    std::string error = validateStinkyIR(func);
    EXPECT_FALSE(error.empty()) << "buffer_load_b32 with SGPR vdst should fail";
    EXPECT_NE(error.find("dest[0]"), std::string::npos) << "Error should mention dest[0] (vdst)";
    EXPECT_NE(error.find("register type"), std::string::npos)
        << "Error should mention register type";
    EXPECT_NE(error.find("'s'"), std::string::npos) << "Error should mention actual type 's'";
    EXPECT_NE(error.find("'v'"), std::string::npos) << "Error should mention expected type 'v'";
}
