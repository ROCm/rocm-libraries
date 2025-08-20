/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <compare>
#include <fstream>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/CommonGraphs.hpp>
#include <common/SourceMatcher.hpp>
#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

namespace ArgumentTracerTest
{
    using namespace rocRoller;
    namespace KG = KernelGraph;
    namespace CT = KG::CoordinateGraph;
    namespace CG = KG::ControlGraph;

    TEST_CASE("AddDeallocateArguments transform works.", "[kernel-graph]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto example           = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command           = example.getCommand();
        auto commandParameters = example.getCommandParameters();

        auto one = Expression::literal(1);
        context->kernel()->setWorkgroupSize({64, 1, 1});
        context->kernel()->setWorkitemCount({one, one, one});

        auto findDeallocate
            = [](KG::KernelGraph const& kg, std::string const& arg) -> std::optional<int> {
            for(auto nodeID : kg.control.getNodes<CG::Deallocate>())
            {
                auto const& node = kg.control.getNode<CG::Deallocate>(nodeID);
                if(std::find(node.arguments.begin(), node.arguments.end(), arg)
                   != node.arguments.end())
                    return nodeID;
            }

            return {};
        };

        auto findPtrComArg = [](CommandPtr               command,
                                KG::KernelGraph const&   kg,
                                Operations::OperationTag tag) -> CommandArgumentPtr {
            auto args = command->getArguments();

            auto match = fmt::format("{}_pointer", static_cast<int>(tag));

            for(auto const& arg : args)
            {
                if(arg->name().find(match) != std::string::npos)
                    return arg;
            }

            return nullptr;
        };

        auto findKernarg
            = [&context](CommandArgumentPtr const& comArg) -> AssemblyKernelArgumentPtr {
            auto expr   = std::make_shared<Expression::Expression>(comArg);
            auto argExp = context->kernel()->findArgumentForExpression(expr);
            if(argExp == nullptr)
                return nullptr;

            return std::get<AssemblyKernelArgumentPtr>(*argExp);
        };

        auto findDimForKernarg
            = [](KG::KernelGraph const& kg, std::string const& argName) -> std::optional<int> {
            for(auto dimId : kg.coordinates.getNodes<CT::User>())
            {
                auto dim = kg.coordinates.getNode<CT::User>(dimId);

                if(dim.argumentName == argName)
                    return dimId;
            }

            return {};
        };

        auto kgraph = KG::translate(command);

        auto lowerLinearTransform = std::make_shared<KG::LowerLinear>(context.get());
        kgraph = rocRollerTest::transform<KG::LowerLinear>(kgraph, context.get());
        kgraph = rocRollerTest::transform<KG::CleanArguments>(kgraph, context.get(), command);
        kgraph = rocRollerTest::transform<KG::UpdateWavefrontParameters>(kgraph, commandParameters);
        kgraph = rocRollerTest::transform<KG::SetWorkitemCount>(kgraph, context.get());
        kgraph = rocRollerTest::transform<KG::AddDeallocateArguments>(kgraph, context.get());
        kgraph = rocRollerTest::transform<KG::CleanArguments>(kgraph, context.get(), command);
        kgraph = rocRollerTest::transform<KG::SetWorkitemCount>(kgraph, context.get());
        kgraph = rocRollerTest::transform<KG::RemoveSetCoordinate>(kgraph);

        auto hasUserMapping = [](KG::KernelGraph const& kg, int userDim) {
            auto pred
                = [&kg, userDim](int node) { return kg.mapper.get<CT::User>(node) == userDim; };

            return pred;
        };

        SECTION("A pointer access")
        {
            auto aPtrArg = findKernarg(findPtrComArg(command, kgraph, example.aTag));
            REQUIRE(aPtrArg != nullptr);

            auto aDim = findDimForKernarg(kgraph, aPtrArg->name);
            REQUIRE(aDim != std::nullopt);
            auto aPtrDeallocate = findDeallocate(kgraph, aPtrArg->name);
            REQUIRE(aPtrDeallocate != std::nullopt);

            auto aLoad = kgraph.control.getNodes<CG::LoadVGPR>()
                             .filter(hasUserMapping(kgraph, *aDim))
                             .only();
            REQUIRE(aLoad != std::nullopt);
            CHECK(CG::NodeOrdering::LeftFirst
                  == kgraph.control.compareNodes(UpdateCache, *aLoad, *aPtrDeallocate));
        }

        SECTION("B pointer access")
        {
            auto bPtrArg = findKernarg(findPtrComArg(command, kgraph, example.bTag));
            REQUIRE(bPtrArg != nullptr);

            auto bDim = findDimForKernarg(kgraph, bPtrArg->name);
            REQUIRE(bDim != std::nullopt);
            auto bPtrDeallocate = findDeallocate(kgraph, bPtrArg->name);
            REQUIRE(bPtrDeallocate != std::nullopt);

            auto bLoad = kgraph.control.getNodes<CG::LoadVGPR>()
                             .filter(hasUserMapping(kgraph, *bDim))
                             .only();
            REQUIRE(bLoad != std::nullopt);
            CHECK(CG::NodeOrdering::LeftFirst
                  == kgraph.control.compareNodes(UpdateCache, *bLoad, *bPtrDeallocate));
        }

        SECTION("D pointer access")
        {
            auto dPtrArg = findKernarg(findPtrComArg(command, kgraph, example.resultTag));
            REQUIRE(dPtrArg != nullptr);

            auto dDim = findDimForKernarg(kgraph, dPtrArg->name);
            REQUIRE(dDim != std::nullopt);
            auto dPtrDeallocate = findDeallocate(kgraph, dPtrArg->name);
            REQUIRE(dPtrDeallocate != std::nullopt);

            auto dLoad = kgraph.control.getNodes<CG::StoreVGPR>()
                             .filter(hasUserMapping(kgraph, *dDim))
                             .only();
            REQUIRE(dLoad != std::nullopt);
            CHECK(CG::NodeOrdering::LeftFirst
                  == kgraph.control.compareNodes(UpdateCache, *dLoad, *dPtrDeallocate));
        }

        SECTION("Regular codegen")
        {
            KG::ControlFlowArgumentTracer argTracer(kgraph, context->kernel());

            auto generate = [&]() -> Generator<Instruction> {
                co_yield context->kernel()->preamble();
                co_yield context->kernel()->prolog();

                co_yield KernelGraph::generate(kgraph, context->kernel(), std::move(argTracer));

                co_yield context->kernel()->postamble();
                co_yield context->kernel()->amdgpu_metadata();
            };

            CHECK_NOTHROW(context->schedule(generate()));
        }

        SECTION("Codegen with added kernel argument access: LowerFromKernelGraph's auditing of "
                "ControlFlowArgumentTracer.")
        {
            KG::ControlFlowArgumentTracer argTracer(kgraph, context->kernel());

            auto intArg = [&]() -> AssemblyKernelArgumentPtr {
                for(auto const& arg : context->kernel()->arguments())
                {
                    if(arg.variableType == DataType::Int64)
                        return std::make_shared<AssemblyKernelArgument>(arg);
                }

                return nullptr;
            }();
            REQUIRE(intArg != nullptr);

            // Modify a random Assign node to access an additional kernel
            // argument after we've made the arg tracer.
            {
                auto assignId = kgraph.control.getNodes<CG::Assign>().take(1).only().value();
                auto assign   = kgraph.control.getNode<CG::Assign>(assignId);

                REQUIRE_FALSE(argTracer.referencedArguments(assignId).contains(intArg->name));
                assign.expression
                    = assign.expression
                      + convert(DataType::Int32, std::make_shared<Expression::Expression>(intArg));

                CAPTURE(assignId, assign.expression, (argTracer.referencedArguments(assignId)));

                kgraph.control.setElement(assignId, std::move(assign));
            }

            auto generate = [&]() -> Generator<Instruction> {
                co_yield context->kernel()->preamble();
                co_yield context->kernel()->prolog();

                co_yield KernelGraph::generate(kgraph, context->kernel(), std::move(argTracer));

                co_yield context->kernel()->postamble();
                co_yield context->kernel()->amdgpu_metadata();
            };

            namespace m = Catch::Matchers;

            CHECK_THROWS_MATCHES(context->schedule(generate()),
                                 FatalError,
                                 m::MessageMatches(m::ContainsSubstring(intArg->name)));
        }
    }
}
