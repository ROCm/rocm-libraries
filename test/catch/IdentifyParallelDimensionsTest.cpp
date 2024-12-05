#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include "TestContext.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>
#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <common/CommonGraphs.hpp>

TEST_CASE("identifyParallelDimensionSets works for MatrixMultiply", "[kernel-graph]")
{
    using namespace rocRoller;
    auto example = rocRollerTest::Graphs::MatrixMultiply<int>();

    auto kgraph = KernelGraph::translate(example.getCommand());

    {
        std::ofstream file("graph_mm.dot");
        file << kgraph.toDOT(true);
    }

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

        std::vector<std::set<int>> expected = {{3, 9}, {2, 17}, {10, 18}};

        CHECK(redundantArgs == expected);
    }
}

TEST_CASE("identifyParallelDimensionSets works for GEMM", "[kernel-graph]")
{
    using namespace rocRoller;
    auto ctx = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::GEMM<float>();

    auto kgraph = KernelGraph::translate(example.getCommand());

    {
        std::ofstream file("graph_gemm.dot");
        file << kgraph.toDOT(true);
    }

    auto redundantArgs = KernelGraph::identifyParallelDimensionSets(kgraph);

    std::vector<std::set<int>> ra2 = {{3, 9}, {2, 36}, {10, 37}, {16, 36}, {17, 37}};

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
            if(arg.name.starts_with(name))
                return true;
        }

        return false;
    }

    std::string describe() const override
    {
        return std::format("Has an argument starting with {}", name);
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

    auto example = rocRollerTest::Graphs::MatrixMultiply<float>();

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

    auto example = rocRollerTest::Graphs::GEMM<float>();

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
