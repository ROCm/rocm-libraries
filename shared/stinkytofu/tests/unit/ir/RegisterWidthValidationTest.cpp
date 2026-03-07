/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

#include <gtest/gtest.h>

using namespace stinkytofu;
using namespace stinkytofu::logical;
using namespace stinkytofu::test;

// ==============================================================================
// Register Width Validation Tests
// ==============================================================================

TEST(RegisterWidthValidationTest, TensorLoadToLds_CorrectWidths)
{
    Function     func("kernel");
    BasicBlock*  bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), &s2, &s3)
        ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass with correct register widths, got error: " << error;
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectSrc0Width)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(10, 4);
    StinkyRegister s3 = sgpr(14, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 2), sgpr(2, 8), &s2, &s3) // group0=2 regs (WRONG!), group1=8 regs
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

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectSrc1Width)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(8, 4);
    StinkyRegister s3 = sgpr(12, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 4), sgpr(4, 4), &s2, &s3) // group0=4 regs, group1=4 regs (WRONG!)
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

TEST(RegisterWidthValidationTest, OtherInstructions_NoWidthChecks)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {9, 4, 2};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    bb->appendIR(static_cast<IRBase*>(stinkytofu::VAddF32(vgpr(0), vgpr(1), vgpr(2))));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass for instructions without width requirements";
}

TEST(RegisterWidthValidationTest, DisableWidthChecks)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(2, 1);
    StinkyRegister s3 = sgpr(3, 1);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 1), sgpr(1, 1), &s2, &s3) // All WRONG widths
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

TEST(RegisterWidthValidationTest, TensorLoadToLds_CorrectRegisterType)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), &s2, &s3)
        ));

    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    std::string error = validateStinkyIR(func);
    EXPECT_TRUE(error.empty()) << "Should pass with correct register types, got error: " << error;
}

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectRegisterType_Src0)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(vgpr(0, 4), sgpr(4, 8), &s2, &s3) // src0=VGPR (WRONG!), src1=SGPR
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

TEST(RegisterWidthValidationTest, TensorLoadToLds_IncorrectRegisterType_Src1)
{
    Function    func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    PassManager pm;
    GemmTileConfig config;
    config.arch     = {12, 5, 0};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    StinkyRegister s2 = sgpr(12, 4);
    StinkyRegister s3 = sgpr(16, 4);
    bb->appendIR(static_cast<IRBase*>(
        stinkytofu::TensorLoadToLds(sgpr(0, 4), vgpr(4, 8), &s2, &s3) // src0=SGPR, src1=VGPR (WRONG!)
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
