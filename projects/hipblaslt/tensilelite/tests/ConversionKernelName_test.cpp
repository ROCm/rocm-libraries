// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <Tensile/Activation.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    std::string conversionKernelName(bool useBeta,
                                     bool bespokeNarrowing,
                                     bool grouped = false,
                                     size_t vw = 1,
                                     size_t gsu = 16)
    {
        ContractionSolution solution;
        solution.bespokeNarrowing             = bespokeNarrowing;
        solution.problemType.useBeta          = useBeta;
        solution.problemType.groupedGemm      = grouped;
        solution.problemType.useBias          = 1;
        solution.problemType.activationType   = ActivationType::Gelu;
        solution.problemType.useScaleCD       = true;
        solution.problemType.useScaleAlphaVec = 1;
        solution.sizeMapping.globalSplitUPGR  = 16;

        auto problem = dummyProblem();
        problem.setUseBias(1);
        problem.setBias(rocisa::DataType::Float, problem.d().sizes()[0], problem.d().sizes()[0]);
        ContractionInputs inputs;
        return solution.outputConversionKernelName(problem, inputs, vw, gsu);
    }
}

// Feature: the launch symbol must match the beta-free device helper of a
// BespokeNarrowing solution, including token position after both optional
// scale suffixes. Removing/moving the host Beta0 token breaks this test while
// Python helper-name tests can still pass.
TEST(ConversionKernelName, BetaOffMatchesDeviceHelper)
{
    EXPECT_EQ("Cijk_SS_BiasS_GELUS_ScaleCD_ScaleAlphaVec_Beta0_PostGSU16_VW1",
              conversionKernelName(false, true));
}

// Feature: a UseBeta:false solution without BespokeNarrowing launches the
// develop helper (same symbol as beta-on). Adding Beta0 for keyless solutions
// breaks this test.
TEST(ConversionKernelName, KeylessBetaOffKeepsDevelopHelperName)
{
    EXPECT_EQ("Cijk_SS_BiasS_GELUS_ScaleCD_ScaleAlphaVec_PostGSU16_VW1",
              conversionKernelName(false, false));
}

// Preservation: beta-on keeps the stock device-helper symbol. Applying the
// Beta0 token unconditionally makes this control fail.
TEST(ConversionKernelName, BetaOnKeepsStockHelperName)
{
    EXPECT_EQ("Cijk_SS_BiasS_GELUS_ScaleCD_ScaleAlphaVec_PostGSU16_VW1",
              conversionKernelName(true, false));
}

// Feature: the beta-kept bespoke symbol occupies its own namespace after the
// epilogue tokens. Keeping the host's old betaFree-only token gate breaks it.
TEST(ConversionKernelName, BetaOnBespokeMatchesDeviceHelper)
{
    EXPECT_EQ("Cijk_SS_BiasS_GELUS_ScaleCD_ScaleAlphaVec_Bespoke_PostGSU16_VW1",
              conversionKernelName(true, true));
}

// Feature: resolve the emitted helper of each role without sharing the stock
// name or combining Bespoke and Beta0. Dropping the new host token fails this;
// applying it to stock also fails the stock positive controls.
TEST(ConversionKernelName, BespokeAndStockResolveSeparateConversionHelpers)
{
    for(bool grouped : {false, true})
        for(size_t gsu : {1, 2, 4, 8, 16})
            for(size_t vw : {1, 2, 4})
            {
                SCOPED_TRACE(::testing::Message() << "grouped=" << grouped
                                                 << " gsu=" << gsu << " vw=" << vw);
                std::string prefix = grouped ? "Cijk_SS_GG" : "Cijk_SS";
                prefix += "_BiasS_GELUS_ScaleCD_ScaleAlphaVec";
                auto suffix = "_PostGSU" + std::to_string(gsu) + "_VW" + std::to_string(vw);
                EXPECT_EQ(prefix + suffix, conversionKernelName(true, false, grouped, vw, gsu));
                EXPECT_EQ(prefix + suffix, conversionKernelName(false, false, grouped, vw, gsu));
                EXPECT_EQ(prefix + "_Beta0" + suffix,
                          conversionKernelName(false, true, grouped, vw, gsu));
                EXPECT_EQ(prefix + "_Bespoke" + suffix,
                          conversionKernelName(true, true, grouped, vw, gsu));
            }
}

namespace
{
    // The helper families whose emitted (device) names the TensileLite codegen test
    // test_bespoke_codegen_slowdown.py pins with the same literal spelling.
    struct HelperFamily
    {
        const char*      name;
        rocisa::DataType type;
        int              useBias;
        ActivationType   activation;
        int              useScaleAlphaVec;
    };
    constexpr HelperFamily families[] = {
        {"SB", rocisa::DataType::BFloat16, 0, ActivationType::None, 0},
        {"SB_BiasB", rocisa::DataType::BFloat16, 1, ActivationType::None, 0},
        {"SH_SIGMOIDS_ScaleAlphaVec", rocisa::DataType::Half, 0, ActivationType::Sigmoid, 1},
        {"SS", rocisa::DataType::Float, 0, ActivationType::None, 0},
        {"SS_SIGMOIDS", rocisa::DataType::Float, 0, ActivationType::Sigmoid, 0},
    };

    std::string familyHelperName(HelperFamily const& family, bool useBeta, bool bespoke,
                                 bool grouped, size_t vw, size_t gsu)
    {
        ContractionSolution solution;
        solution.bespokeNarrowing                       = bespoke;
        solution.problemType.useBeta                    = useBeta;
        solution.problemType.groupedGemm                = grouped;
        solution.problemType.useBias                    = family.useBias;
        solution.problemType.activationType             = family.activation;
        solution.problemType.activationComputeDataType  = rocisa::DataType::Float;
        solution.problemType.useScaleAlphaVec           = family.useScaleAlphaVec;
        solution.sizeMapping.globalSplitUPGR            = 16;

        TensorDescriptor a("a", family.type, {128, 256, 1}, {1, 128, 128 * 256});
        TensorDescriptor b("b", family.type, {256, 64, 1}, {1, 256, 64 * 256});
        TensorDescriptor c("c", family.type, {128, 64, 1}, {1, 128, 128 * 64});
        TensorDescriptor d("d", family.type, {128, 64, 1}, {1, 128, 128 * 64});
        auto problem = ContractionProblemGemm::GEMM(false, false, a, {}, b, {}, c, {}, d, {},
                                                    useBeta ? 1.0 : 0.0);
        problem.setUseBias(family.useBias);
        if(family.useBias)
            problem.setBias(family.type, 128, 128);
        problem.setUseScaleAlphaVec(family.useScaleAlphaVec);
        ContractionInputs inputs;
        return solution.outputConversionKernelName(problem, inputs, vw, gsu);
    }
}

// Contract: for every helper family, role, marker and epilogue the host requests
// exactly the name the code generator emits (pinned literally on the device side).
// Dropping or adding the bespoke token for any one family (for example only
// activation none) fails this test.
TEST(ConversionKernelName, HelperFamiliesMatchEmittedNames)
{
    for(auto const& family : families)
        for(bool useBeta : {true, false})
            for(bool bespoke : {false, true})
                for(bool grouped : {false, true})
                    for(size_t gsu : {1, 2, 4, 8, 16})
                        for(size_t vw : {1, 2, 4})
                        {
                            std::string const name = family.name;
                            auto const        split = name.find('_');
                            std::string expected
                                = "Cijk_" + name.substr(0, split) + (grouped ? "_GG" : "")
                                  + (split == std::string::npos ? "" : name.substr(split));
                            if(bespoke)
                                expected += useBeta ? "_Bespoke" : "_Beta0";
                            expected += "_PostGSU" + std::to_string(gsu) + "_VW" + std::to_string(vw);
                            EXPECT_EQ(expected,
                                      familyHelperName(family, useBeta, bespoke, grouped, vw, gsu))
                                << "useBeta=" << useBeta << " bespoke=" << bespoke;
                        }
}
