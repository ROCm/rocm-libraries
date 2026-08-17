// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include "TestContext.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>
#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <common/CommonGraphs.hpp>

TEST_CASE("identifyParallelDimensionSets works for MatrixMultiply",
          "[kernel-graph][graph-transforms]")
{
    using namespace rocRoller;
    auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Int32);

    auto kgraph = KernelGraph::translate(example.getCommand());

    SECTION("reachableNodes")
    {
        auto sameDimensionLoadTiledNodes
            = KernelGraph::loadNodesReachableWithoutDimensionModifyingNodes(kgraph.control, 9);
        std::set<int> expected;

        CHECK(sameDimensionLoadTiledNodes == expected);
    }

    SECTION("identifyParallelDimensionSets")
    {
        auto redundantArgs = KernelGraph::identifyParallelDimensionSets(kgraph);

        std::vector<std::set<int>> expected = {{3, 9}, {2, 18}, {10, 19}};

        CHECK(redundantArgs == expected);
    }
}

TEST_CASE("identifyParallelDimensionSets works for GEMM", "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto kgraph = KernelGraph::translate(example.getCommand());

    auto redundantArgs = KernelGraph::identifyParallelDimensionSets(kgraph);

    std::vector<std::set<int>> ra2 = {{3, 9}, {2, 37}, {10, 38}, {16, 37}, {17, 38}};

    CHECK(redundantArgs == ra2);
}

struct HasKernelArgMatcher : Catch::Matchers::MatcherGenericBase
{
    HasKernelArgMatcher(std::string name_)
        : name(std::move(name_))
    {
    }

    bool match(std::vector<rocRoller::AssemblyKernelArgument> const& kargs) const
    {
        for(auto const& arg : kargs)
        {
            if(arg.getName().starts_with(name))
                return true;
        }

        return false;
    }

    std::string describe() const override
    {
        return fmt::format("Has an argument starting with {}", name);
    }

    std::string name;
};

auto HasKernelArgNamed(std::string name)
{
    return HasKernelArgMatcher(std::move(name));
}

SCENARIO("IdentifyParallelDimensions transformation works for MatrixMultiply", "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Float);

    GIVEN("The initial kernel graph for a MatrixMultiply")
    {
        auto kgraph = KernelGraph::translate(example.getCommand());

        THEN("A and B tensor sizes are added as kernel args.")
        {
            auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

            kgraph = cleanArgs.apply(kgraph);

            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));

            AND_THEN("The HasKernelArgNamed() matcher works.")
            {
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("rosneT"));
            }
        }

        WHEN("The IdentifyParallelDimensions transformation is applied")
        {
            auto ipd = KernelGraph::IdentifyParallelDimensions();
            kgraph   = ipd.apply(kgraph);

            THEN("The K dimension size is no longer added twice.")
            {
                auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

                kgraph = cleanArgs.apply(kgraph);

                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_2_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));
            }
        }
    }
}

SCENARIO("IdentifyParallelDimensions transformation works for GEMM", "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    GIVEN("The initial kernel graph for a GEMM")
    {
        auto kgraph = KernelGraph::translate(example.getCommand());

        THEN("A, B, and C tensor sizes are all added as kernel args.")
        {
            auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

            kgraph = cleanArgs.apply(kgraph);

            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_4_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_4_size_1"));

            AND_THEN("The HasKernelArgNamed() matcher works.")
            {
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("rosneT"));
            }
        }

        WHEN("The IdentifyParallelDimensions transformation is applied")
        {
            auto ipd = KernelGraph::IdentifyParallelDimensions();
            kgraph   = ipd.apply(kgraph);

            THEN("Redundant tensor sizes between A, B, and C are no longer added as kernel args.")
            {
                auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

                kgraph = cleanArgs.apply(kgraph);

                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_2_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_4_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_4_size_1"));
            }
        }
    }
}

SCENARIO("IdentifyParallelDimensions transformation works for scaled MatrixMultiply",
         "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    // MatrixMultiply computes: D = A × B (no C tensor, only inputs A,B and output D)
    auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Float,
                                                         DataType::Float,
                                                         DataType::Float,
                                                         Operations::ScaleMode::Separate,
                                                         Operations::ScaleMode::Separate);

    GIVEN("The initial kernel graph for a scaled MatrixMultiply")
    {
        auto kgraph = KernelGraph::translate(example.getCommand());

        THEN("A, ScaleA, B, ScaleB, and D tensor sizes are added as kernel args.")
        {
            auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

            kgraph = cleanArgs.apply(kgraph);

            // A tensor dimensions (M, K)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
            // ScaleA dimensions (M, K/blockSize)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));
            // B tensor dimensions (K, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_1"));
            // ScaleB dimensions (K/blockSize, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_7_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_7_size_1"));
            // D (output) tensor dimensions (M, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_11_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_11_size_1"));
        }

        WHEN("The IdentifyParallelDimensions transformation is applied")
        {
            auto ipd = KernelGraph::IdentifyParallelDimensions();
            kgraph   = ipd.apply(kgraph);

            THEN("Redundant dimensions between A, ScaleA, B, ScaleB, and D are eliminated.")
            {
                auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

                kgraph = cleanArgs.apply(kgraph);

                // A tensor: M and K dimensions retained
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));

                // ScaleA: M matches A's M (redundant), K/blockSize unique (retained)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_2_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));

                // B tensor: K redundant with A, only N retained
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_5_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_1"));

                // ScaleB: K/blockSize matches ScaleA (redundant), N matches B (redundant)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_7_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_7_size_1"));

                // D (output): M matches A (redundant), N matches B (redundant)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_11_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_11_size_1"));
            }
        }
    }
}

SCENARIO("IdentifyParallelDimensions transformation works for scaled GEMM", "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    // GEMM computes: D = alpha*A*B + beta*C (has both C input and D output tensors)
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);
    example.setScaling(Operations::ScaleMode::Separate,
                       Operations::ScaleMode::Separate,
                       DataType::E8M0,
                       DataType::E8M0,
                       32);

    GIVEN("The initial kernel graph for a scaled GEMM")
    {
        auto kgraph = KernelGraph::translate(example.getCommand());

        THEN("A, ScaleA, B, ScaleB, C, and D tensor sizes are added as kernel args.")
        {
            auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

            kgraph = cleanArgs.apply(kgraph);

            // A tensor dimensions (M, K)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));
            // ScaleA dimensions (M, K/blockSize)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));
            // B tensor dimensions (K, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_1"));
            // ScaleB dimensions (K/blockSize, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_7_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_7_size_1"));
            // C (input) tensor dimensions (M, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_10_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_10_size_1"));
            // D (output) tensor dimensions (M, N)
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_21_size_0"));
            CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_21_size_1"));
        }

        WHEN("The IdentifyParallelDimensions transformation is applied")
        {
            auto ipd = KernelGraph::IdentifyParallelDimensions();
            kgraph   = ipd.apply(kgraph);

            THEN("Redundant dimensions between A, ScaleA, B, ScaleB, C, and D are eliminated.")
            {
                auto cleanArgs = KernelGraph::CleanArguments(ctx.get(), example.getCommand());

                kgraph = cleanArgs.apply(kgraph);

                // A tensor: M and K dimensions retained
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_0_size_1"));

                // ScaleA: M matches A's M (redundant), K/blockSize unique (retained)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_2_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_2_size_1"));

                // B tensor: K redundant with A, only N retained
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_5_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), HasKernelArgNamed("Tensor_5_size_1"));

                // ScaleB: K/blockSize matches ScaleA (redundant), N matches B (redundant)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_7_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_7_size_1"));

                // C (input): M matches A (redundant), N matches B (redundant)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_10_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_10_size_1"));

                // D (output): M matches A (redundant), N matches B (redundant)
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_21_size_0"));
                CHECK_THAT(ctx->kernel()->arguments(), !HasKernelArgNamed("Tensor_21_size_1"));
            }
        }
    }
}
