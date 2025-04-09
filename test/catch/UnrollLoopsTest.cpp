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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <fstream>

#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"

TEST_CASE("Test getForLoopName", "[kernel-graph][unroll]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;

    kg::KernelGraph kgraph;

    auto [forDim, forOp]   = kg::rangeFor(kgraph, Expression::literal(4), KLOOP);
    auto [forDim2, forOp2] = kg::rangeFor(kgraph, Expression::literal(16), "Another loop");

    CHECK(kg::getForLoopName(kgraph, forOp) == KLOOP);
    CHECK(kg::getForLoopName(kgraph, forOp2) == "Another loop");
}

TEST_CASE("Test getUnrollAmount", "[kernel-graph][unroll]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;

    kg::KernelGraph kgraph;

    auto params     = std::make_shared<CommandParameters>();
    params->unrollK = 5;
    params->unrollX = 2;

    auto [forDim, forOp]   = kg::rangeFor(kgraph, Expression::literal(4), KLOOP);
    auto [forDim2, forOp2] = kg::rangeFor(kgraph, Expression::literal(16), "Another loop");
    auto [forDim3, forOp3] = kg::rangeFor(kgraph, Expression::literal(4), XLOOP);

    CHECK(kg::getUnrollAmount(kgraph, forOp, params) == 5);
    CHECK(kg::getUnrollAmount(kgraph, forOp2, params) == 1);
    CHECK(kg::getUnrollAmount(kgraph, forOp3, params) == 4);

    params->unrollK = 7;
    CHECK(kg::getUnrollAmount(kgraph, forOp, params) == 7);
}

TEST_CASE("UnrollLoops simple test", "[kernel-graph][unroll][graph-transforms]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;

    Settings::getInstance()->set(Settings::LogConsole, true);
    Settings::getInstance()->set(Settings::LogLvl, LogLevel::Debug);
    Settings::getInstance()->set(Settings::LogGraphs, false);

    auto ctx = TestContext::ForDefaultTarget();

    auto command = std::make_shared<Command>();

    kg::KernelGraph kgraph;

    auto [user, wg, wi, vgpr] = kgraph.coordinates.addElements( //
        kg::CoordinateGraph::User{},
        kg::CoordinateGraph::Workgroup{0, 0},
        kg::CoordinateGraph::Workitem{0, Expression::literal(64)},
        kg::CoordinateGraph::VGPR{});

    auto argTag = command->allocateTag();
    auto arg    = command->allocateArgument(DataType::Int32, argTag, ArgumentType::Limit);

    auto [forDim, forOp] = kg::rangeFor(kgraph, arg->expression(), rocRoller::KLOOP);

    kgraph.coordinates.addElement(kg::CoordinateGraph::Tile{}, {user}, {wg, wi, forDim});
    kgraph.coordinates.addElement(kg::CoordinateGraph::Forget{}, {wg, wi, forDim}, {vgpr});

    auto x = std::make_shared<Expression::Expression>(
        Expression::DataFlowTag{forDim, Register::Type::Vector, DataType::Int32});

    auto [kernel, calculate, store] = kgraph.control.addElements( //
        kg::ControlGraph::Kernel{},
        kg::ControlGraph::Assign{Register::Type::Vector, Expression::literal(5) * x},
        kg::ControlGraph::StoreLinear{});

    kgraph.mapper.connect<kg::CoordinateGraph::VGPR>(calculate, vgpr);

    kgraph.control.chain<kg::ControlGraph::Body>(kernel, forOp, calculate);

    kgraph.control.chain<kg::ControlGraph::Sequence>(forOp, store);

    auto params     = std::make_shared<CommandParameters>();
    params->unrollK = 4;

    SECTION("Only one for loop initially.")
    {
        auto ops
            = kgraph.control.findElements(kgraph.control.isElemType<kg::ControlGraph::ForLoopOp>())
                  .to<std::set>();

        REQUIRE(ops == std::set{forOp});
    }

    SECTION("Create tail loop manually.")
    {
        int forLoopDimension = kgraph.coordinates.addElement(kg::CoordinateGraph::ForLoop());
        int unrollDimension  = kgraph.coordinates.addElement(kg::CoordinateGraph::Unroll(4));

        kg::UnrollLoops unroll(params, ctx.get());

        auto tail = unroll.createTailLoop(kgraph, forOp, 4, unrollDimension, forLoopDimension);

        CHECK(tail != std::nullopt);
        kgraph = kgraph.transform(std::make_shared<kg::Simplify>());

        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, forOp, *tail)
              == rocRoller::KernelGraph::ControlGraph::NodeOrdering::LeftFirst);
        CHECK(kgraph.control.compareNodes(rocRoller::CacheOnly, forOp, *tail)
              == rocRoller::KernelGraph::ControlGraph::NodeOrdering::LeftFirst);
        CHECK(kgraph.control.compareNodes(rocRoller::UseCacheIfAvailable, forOp, *tail)
              == rocRoller::KernelGraph::ControlGraph::NodeOrdering::LeftFirst);
        CHECK(kgraph.control.compareNodes(rocRoller::IgnoreCache, forOp, *tail)
              == rocRoller::KernelGraph::ControlGraph::NodeOrdering::LeftFirst);

        auto ops
            = kgraph.control.findElements(kgraph.control.isElemType<kg::ControlGraph::ForLoopOp>())
                  .to<std::set>();

        REQUIRE(ops == std::set{forOp, *tail});

        auto tailOp = kgraph.control.getNode<kg::ControlGraph::ForLoopOp>(*tail);

        auto [_, tailOpSize] = split<Expression::LessThan>(tailOp.condition);
        CHECK_THAT(tailOpSize, IdenticalTo(arg->expression()));

        auto unrollInitTag
            = kgraph.control.getOutputNodeIndices<kg::ControlGraph::Initialize>(*tail)
                  .only()
                  .value();

        auto unrollInit = kgraph.control.getNode<kg::ControlGraph::Assign>(unrollInitTag);

        auto origOp           = kgraph.control.getNode<kg::ControlGraph::ForLoopOp>(forOp);
        auto [__, origOpSize] = split<Expression::LessThan>(origOp.condition);

        CHECK_THAT(origOpSize, IdenticalTo(unrollInit.expression));
    }

    SECTION("Create tail loop automatically in UnrollLoops xform.")
    {
        kgraph = kgraph.transform(std::make_shared<kg::UnrollLoops>(params, ctx.get()));
        kgraph = kgraph.transform(std::make_shared<kg::Simplify>());

        auto ops
            = kgraph.control.findElements(kgraph.control.isElemType<kg::ControlGraph::ForLoopOp>())
                  .to<std::set>();

        REQUIRE(ops.size() == 2);
        REQUIRE_THAT(ops, Catch::Matchers::Contains(forOp));

        // remove the first loop, the other one will be the new tail loop.
        ops.erase(forOp);
        auto tailOpTag = *ops.begin();
        auto tailOp    = kgraph.control.getNode<kg::ControlGraph::ForLoopOp>(tailOpTag);

        auto [_, tailOpSize] = split<Expression::LessThan>(tailOp.condition);
        CHECK_THAT(tailOpSize, IdenticalTo(arg->expression()));

        auto unrollInitTag
            = kgraph.control.getOutputNodeIndices<kg::ControlGraph::Initialize>(tailOpTag)
                  .only()
                  .value();

        auto unrollInit = kgraph.control.getNode<kg::ControlGraph::Assign>(unrollInitTag);

        auto origOp           = kgraph.control.getNode<kg::ControlGraph::ForLoopOp>(forOp);
        auto [__, origOpSize] = split<Expression::LessThan>(origOp.condition);

        CHECK_THAT(origOpSize, IdenticalTo(unrollInit.expression));
    }
}
