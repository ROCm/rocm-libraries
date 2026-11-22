// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Consumer-visible ActivationCheck decisions and helper launch names.
// These checks need no new production helper symbols to fail without the change.
#include <gtest/gtest.h>
#include <array>
#include <sstream>
#include <Tensile/Activation.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include "FallbackTestUtils.hpp"
using namespace TensileLite;
using namespace TensileLite::testing;
namespace
{
    constexpr std::array<ActivationType, 5> singles = {ActivationType::Gelu,
        ActivationType::Relu, ActivationType::Sigmoid, ActivationType::Silu, ActivationType::Clamp};
    ContractionProblemGemm makeProblem(ActivationType type, ActivationType activationEnum)
    {
        auto problem = dummyProblem();
        problem.setActivationType(type);
        problem.setParams().setActivationEnum(activationEnum);
        return problem;
    }

    std::string describe(ActivationType value, ContractionProblemGemm const& problem)
    {
        return "sol=" + ToString(value) + " prob=" + ToString(problem.activationType())
               + " enum=" + ToString(problem.getParams().activationEnum());
    }

    std::string conversionKernelName(ActivationType type)
    {
        ContractionSolution solution;
        solution.problemType.useBias          = 1;
        solution.problemType.useScaleAlphaVec = 1;
        solution.problemType.activationType   = type;
        solution.sizeMapping.globalSplitUPGR  = 16;

        auto problem = dummyProblem();
        problem.setUseBias(1);
        problem.setBias(rocisa::DataType::Float, problem.d().sizes()[0], problem.d().sizes()[0]);
        ContractionInputs inputs;
        return solution.outputConversionKernelName(problem, inputs, 1, 16);
    }

    std::vector<ActivationType> allTypes()
    {
        std::vector<ActivationType> rv;
        for(uint32_t i = 0; i < static_cast<uint32_t>(ActivationType::Count); ++i)
            rv.push_back(static_cast<ActivationType>(i));
        return rv;
    }
}

// Feature: enum-aware single-type equality, including refusal of plain calls.
// Removing that branch makes both the matching Hipblaslt_all and plain cells fail.
TEST(SingleActivation, ActivationCheckSingleRowsFullTable)
{
    for(auto value : singles)
    {
        Predicates::Contraction::ActivationCheck check(value);
        for(auto type : allTypes())
            for(auto activationEnum : allTypes())
            {
                auto problem = makeProblem(type, activationEnum);
                bool expected = type == value
                    || ((type == ActivationType::All || type == ActivationType::Hipblaslt_all)
                        && activationEnum == value);
                EXPECT_EQ(expected, check(problem)) << describe(value, problem);
            }
    }
}

TEST(SingleActivation, ActivationCheckHipblasltCalls)
{
    Predicates::Contraction::ActivationCheck gelu(ActivationType::Gelu);
    EXPECT_TRUE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Gelu)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Relu)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Silu)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Sigmoid)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Clamp)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Hipblaslt_all, ActivationType::None)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::None, ActivationType::None)));
    EXPECT_TRUE(gelu(makeProblem(ActivationType::Gelu, ActivationType::None)));
    EXPECT_FALSE(gelu(makeProblem(ActivationType::Relu, ActivationType::None)));
}

// Feature: host name must match the generated single-activation reduction helper.
TEST(SingleActivation, ConversionKernelName)
{
    EXPECT_EQ("Cijk_SS_BiasS_GELUS_ScaleAlphaVec_PostGSU16_VW1", conversionKernelName(ActivationType::Gelu));
    EXPECT_EQ("Cijk_SS_BiasS_CLAMPS_ScaleAlphaVec_PostGSU16_VW1", conversionKernelName(ActivationType::Clamp));
}

// Preservation: widening an already-general row must not be lost in this cutover.
TEST(SingleActivation, RunTimeRowsUnchanged)
{
    Predicates::Contraction::ActivationCheck all(ActivationType::All);
    Predicates::Contraction::ActivationCheck ha(ActivationType::Hipblaslt_all);
    Predicates::Contraction::ActivationCheck none(ActivationType::None);
    for(auto type : {ActivationType::None, ActivationType::Gelu, ActivationType::Relu,
                     ActivationType::Silu, ActivationType::Clamp, ActivationType::Hipblaslt_all})
    {
        EXPECT_TRUE(all(makeProblem(type, ActivationType::None)));
        EXPECT_TRUE(ha(makeProblem(type, ActivationType::None)));
        EXPECT_EQ(type == ActivationType::None, none(makeProblem(type, ActivationType::None)));
    }
}

// Preservation: whitelist alone is deliberately not the plain-call guard.
TEST(SingleActivation, EnumWhiteListAloneDoesNotRejectPlainCalls)
{
    Predicates::Contraction::ActivationEnumWhiteList whiteList;
    whiteList.value = {ActivationType::Gelu};
    EXPECT_TRUE(whiteList(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Gelu)));
    EXPECT_FALSE(whiteList(makeProblem(ActivationType::Hipblaslt_all, ActivationType::Relu)));
    EXPECT_TRUE(whiteList(makeProblem(ActivationType::None, ActivationType::None)));
}
