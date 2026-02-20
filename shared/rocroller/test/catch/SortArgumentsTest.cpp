// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestContext.hpp"

#include <catch2/catch_test_macros.hpp>

#include <common/CommonGraphs.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/SortArguments_detail.hpp>

#include <vector>

namespace
{
    using namespace rocRoller;
    namespace KG = KernelGraph;

    std::vector<std::string> argumentNames(std::vector<AssemblyKernelArgument> const& args)
    {
        std::vector<std::string> names;
        names.reserve(args.size());
        for(auto const& a : args)
            names.push_back(a.getName());
        return names;
    }

} // namespace

TEST_CASE("SortArguments transform works as expected", "[kernel-graph][SortArguments]")
{
    auto context = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto command           = example.getCommand();
    auto commandParameters = example.getCommandParameters();

    auto argumentName = [](auto const& arg) { return arg.name; };

    KG::KernelGraph kgraph;

    auto one = Expression::literal(1);
    context->kernel()->setWorkgroupSize({64, 1, 1});
    context->kernel()->setWorkitemCount({one, one, one});
    context->kernel()->addCommandArguments(command->getArguments());

    kgraph = KG::translate(command);
    kgraph = rocRollerTest::transform<KG::OrderMemory>(kgraph, false);
    kgraph = rocRollerTest::transform<KG::UpdateParameters>(kgraph, commandParameters);
    kgraph = rocRollerTest::transform<KG::AddLDS>(kgraph, commandParameters, context.get());
    kgraph = rocRollerTest::transform<KG::LowerLinear>(kgraph, context.get());
    kgraph = rocRollerTest::transform<KG::LowerTile>(kgraph, commandParameters, context.get());
    kgraph = rocRollerTest::transform<KG::LowerTensorContraction>(
        kgraph, commandParameters, context.get());
    kgraph = rocRollerTest::transform<KG::Simplify>(kgraph);
    kgraph = rocRollerTest::transform<KG::FuseExpressions>(kgraph);
    kgraph = rocRollerTest::transform<KG::ConnectWorkgroups>(kgraph, context.get());
    kgraph = rocRollerTest::transform<KG::WorkgroupRemapXCC>(
        kgraph, context.get(), commandParameters->workgroupRemapXCC);
    kgraph = rocRollerTest::transform<KG::UnrollLoops>(kgraph, commandParameters, context.get());
    kgraph = rocRollerTest::transform<KG::FuseLoops>(kgraph);
    kgraph = rocRollerTest::transform<KG::RemoveDuplicates>(kgraph);
    kgraph = rocRollerTest::transform<KG::OrderEpilogueBlocks>(kgraph);
    kgraph = rocRollerTest::transform<KG::CleanLoops>(kgraph);
    kgraph = rocRollerTest::transform<KG::AddPrefetch>(kgraph, commandParameters, context.get());
    kgraph = rocRollerTest::transform<KG::UpdateWavefrontParameters>(kgraph, commandParameters);
    kgraph = rocRollerTest::transform<KG::AssignIndexExpressions>(kgraph, context.get(), command);
    kgraph = rocRollerTest::transform<KG::CleanArguments>(kgraph, context.get(), command);
    kgraph = rocRollerTest::transform<KG::AddDeallocateArguments>(kgraph, context.get());

    std::vector<std::string> argNamesByUse = {"user_Float_Value_8",
                                              "user_Float_Value_6",
                                              "Tensor_0_stride_1",
                                              "Tensor_0_pointer",
                                              "Tensor_0_extent",
                                              "Tensor_2_stride_0",
                                              "Tensor_2_pointer",
                                              "Tensor_2_extent",
                                              "Tensor_0_size_1",
                                              "Tensor_4_stride_1",
                                              "Tensor_4_pointer",
                                              "Tensor_4_extent",
                                              "Tensor_15_stride_1",
                                              "Tensor_15_pointer",
                                              "Tensor_15_extent"};

    std::vector<std::string> argNamesByUseAndSize = {"Tensor_0_stride_1",
                                                     "Tensor_0_pointer",
                                                     "Tensor_0_extent",
                                                     "Tensor_2_stride_0",
                                                     "Tensor_2_pointer",
                                                     "Tensor_2_extent",
                                                     "user_Float_Value_8",
                                                     "user_Float_Value_6",
                                                     "Tensor_0_size_1",
                                                     "Tensor_4_stride_1",
                                                     "Tensor_4_pointer",
                                                     "Tensor_4_extent",
                                                     "Tensor_15_stride_1",
                                                     "Tensor_15_pointer",
                                                     "Tensor_15_extent"};

    SECTION("sortArgumentsByFirstUse function reorders kernel arguments by first use")
    {
        auto args = context->kernel()->arguments();

        KG::SortArguments_detail::sortArgumentsByFirstUse(kgraph, context->kernel(), args);

        CHECK(argumentNames(args) == argNamesByUse);

        SECTION("decidePreloadedKernargs function sorts partitions by size and decides preloaded "
                "kernargs")
        {
            context->argLoader()->decidePreloadedKernargs(args);

            CHECK(argumentNames(args) == argNamesByUseAndSize);

            int totalSize = 0;
            for(auto const& arg : args)
            {
                totalSize += arg.getSize();

                CAPTURE(arg, totalSize);

                if(totalSize <= 56)
                {
                    CHECK(arg.getPreloaded());
                }
                else
                {
                    CHECK_FALSE(arg.getPreloaded());
                }
            }
        }
    }

    SECTION("Applying sortArguments transform reorders kernel arguments by first use and by size")
    {
        kgraph    = rocRollerTest::transform<KG::SortArguments>(kgraph, context.get());
        auto args = context->kernel()->arguments();

        CHECK(argumentNames(args) == argNamesByUseAndSize);
    }
}
