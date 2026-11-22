// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Bias presence as a narrowed fact of BespokeNarrowing solutions, evaluated by the
// real collectNarrowedFactPredicate(). A bespoke solution with nonzero UseBias must
// not serve a no-bias problem (UseBiasCheck alone accepts it); stock solutions keep
// no narrowed-fact guard. Dropping the bias-presence conjunct fails the first case.
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include "TestData.hpp"
using namespace TensileLite;
namespace
{
    namespace PC = Predicates::Contraction;
    using Pred   = Predicates::Predicate<ContractionProblemGemm>;

    ContractionProblemGemm problem()
    {
        return ContractionProblemGemm::GEMM(false, false, 32, 32, 128, 32, 128, 32, 0.0, false, 1);
    }

    class BiasPresentDerivedFacts : public ::testing::TestWithParam<int>
    {
    };

    TEST_P(BiasPresentDerivedFacts, nonzero_usebias_requires_problem_bias)
    {
        ContractionSolution solution;
        solution.bespokeNarrowing     = true;
        solution.problemType.useBias  = GetParam();
        solution.problemPredicate     = std::make_shared<Predicates::And<ContractionProblemGemm>>(
            std::vector<std::shared_ptr<Pred>>{std::make_shared<PC::ActivationCheck>(ActivationType::None),
                                               std::make_shared<PC::UseBiasCheck>(GetParam())});
        solution.collectNarrowedFactPredicate();
        ASSERT_NE(solution.narrowedFactPredicate, nullptr);
        auto gemm = problem();
        gemm.setActivationType(ActivationType::None);
        gemm.setUseBias(1);
        EXPECT_TRUE((*solution.narrowedFactPredicate)(gemm));
        EXPECT_TRUE((*solution.problemPredicate)(gemm));
        gemm.setUseBias(0);
        EXPECT_FALSE((*solution.narrowedFactPredicate)(gemm));
        EXPECT_FALSE((*solution.problemPredicate)(gemm)) << "selection must not pick it either";
    }
    INSTANTIATE_TEST_SUITE_P(BiasPresent, BiasPresentDerivedFacts, ::testing::Values(1, 2, 3));

    TEST(BiasPresentDerivedFacts, stock_solutions_have_no_narrowed_guard)
    {
        for(int useBias : {0, 1, 2, 3})
        {
            SCOPED_TRACE(useBias);
            ContractionSolution stock;
            stock.problemType.useBias = useBias;
            stock.problemPredicate    = std::make_shared<PC::UseBiasCheck>(useBias);
            stock.collectNarrowedFactPredicate();
            EXPECT_EQ(stock.narrowedFactPredicate, nullptr);
        }
    }

    TEST(BiasPresentDerivedFacts, nonbias_bespoke_retains_bias_off_requirement)
    {
        ContractionSolution nonbias;
        nonbias.bespokeNarrowing    = true;
        nonbias.problemType.useBias = 0;
        nonbias.problemPredicate    = std::make_shared<PC::UseBiasCheck>(0);
        nonbias.collectNarrowedFactPredicate();
        ASSERT_NE(nonbias.narrowedFactPredicate, nullptr);
        auto gemm = problem();
        gemm.setUseBias(0);
        EXPECT_TRUE((*nonbias.narrowedFactPredicate)(gemm));
        gemm.setUseBias(1);
        EXPECT_FALSE((*nonbias.narrowedFactPredicate)(gemm));
    }

    TEST(BiasPresentDerivedFacts, bias_source_whitelist_preserves_matching_source)
    {
        auto pred   = std::make_shared<PC::BiasSrcWhiteList>();
        pred->value = {static_cast<int>(ContractionProblemGemm::TENSOR::D)};
        auto gemm   = problem();
        gemm.setUseBias(1);
        gemm.setBias(rocisa::DataType::BFloat16, 32, 32, false, ContractionProblemGemm::TENSOR::D);
        EXPECT_TRUE((*pred)(gemm));
        gemm.setBias(rocisa::DataType::BFloat16, 32, 32, false, ContractionProblemGemm::TENSOR::A);
        EXPECT_FALSE((*pred)(gemm));
    }

    TEST(BiasPresentDerivedFacts, bias_length_minimum_is_preserved)
    {
        auto pred   = std::make_shared<PC::BiasSrcWhiteList>();
        pred->value = {static_cast<int>(ContractionProblemGemm::TENSOR::D)};
        auto gemm   = problem();
        gemm.setUseBias(1);
        gemm.setBias(rocisa::DataType::BFloat16, 32, 32);
        EXPECT_TRUE((*pred)(gemm));
        gemm.setBias(rocisa::DataType::BFloat16, 31, 31);
        EXPECT_FALSE((*pred)(gemm));
        gemm.setBias(rocisa::DataType::BFloat16, 33, 33);
        EXPECT_TRUE((*pred)(gemm));
    }
    // Pass: load the real writer's serialized contract and derive its host guard.
    // Native predicate deserialization requires an AMD GPU; pure bias unit cases above do not.
    // Break: drop bias presence, bias-off or a narrowed bias-type whitelist at load.
    TEST(GPUBiasPresentContract, geko_output_contract)
    {
        int deviceCount = 0;
        auto status = hipGetDeviceCount(&deviceCount);
        if(status == hipErrorNoDevice || (status == hipSuccess && deviceCount == 0))
            GTEST_SKIP() << "requires a visible AMD GPU to deserialize hardware predicates";
        ASSERT_EQ(status, hipSuccess);
#ifdef TENSILE_MSGPACK
        const auto path = TestData::Instance().file("geko_bias_contract.dat");
#else
        const auto path = TestData::Instance().file("geko_bias_contract.yaml");
#endif
        auto library = std::dynamic_pointer_cast<
            MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>>(
                LoadLibraryFile<ContractionProblemGemm>(path.string()));
        ASSERT_NE(library, nullptr);
        ASSERT_EQ(library->solutions.size(), 3u);
        int off = 0, typed = 0, general = 0;
        for(const auto& [index, solution] : library->solutions)
        {
            SCOPED_TRACE(index);
            ASSERT_TRUE(solution->bespokeNarrowing);
            solution->collectNarrowedFactPredicate();
            ASSERT_NE(solution->narrowedFactPredicate, nullptr);
            auto gemm = ContractionProblemGemm::GEMM_Strides(false, false,
                rocisa::DataType::BFloat16, rocisa::DataType::BFloat16,
                rocisa::DataType::BFloat16, rocisa::DataType::BFloat16,
                32, 618, 120, 1, 32, 0, 120, 0, 32, 0, 32, 0, 0.0);
            gemm.setActivationType(ActivationType::None);
            gemm.setUseScaleAlphaVec(0);
            gemm.setComputeInputTypeA(rocisa::DataType::BFloat16);
            gemm.setComputeInputTypeB(rocisa::DataType::BFloat16);
            gemm.setUseBias(0);
            EXPECT_EQ((*solution->narrowedFactPredicate)(gemm), solution->problemType.useBias == 0);
            gemm.setUseBias(1);
            gemm.setBias(rocisa::DataType::BFloat16, 32, 618);
            EXPECT_EQ((*solution->narrowedFactPredicate)(gemm), solution->problemType.useBias == 1);
            if(solution->problemType.useBias == 0)
                ++off;
            else
            {
                const auto& types = solution->problemType.biasDataTypeWhiteList;
                ASSERT_TRUE((types == std::vector<rocisa::DataType>{rocisa::DataType::BFloat16}
                             || types == std::vector<rocisa::DataType>{rocisa::DataType::Float,
                                                                      rocisa::DataType::BFloat16}));
                const bool narrowed = types.size() == 1;
                (narrowed ? typed : general)++;
                gemm.setBias(rocisa::DataType::Float, 32, 618);
                EXPECT_EQ((*solution->narrowedFactPredicate)(gemm), !narrowed);
            }
        }
        EXPECT_EQ(off, 1);
        EXPECT_EQ(typed, 1);
        EXPECT_EQ(general, 1);
    }
}
