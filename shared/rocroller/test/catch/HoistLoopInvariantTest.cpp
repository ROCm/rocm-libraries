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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/HoistLoopInvariant.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>

#include <fstream>
#include <iostream>

#include "TestContext.hpp"

void writeDotToFile(const std::string& dotContent, const std::string& filename)
{
    std::ofstream outFile(filename);
    if(outFile.is_open())
    {
        outFile << dotContent;
        outFile.close();
        std::cout << filename << std::endl;
    }
}

TEST_CASE("HoistLoopInvariant independent",
          "[kernel-graph][hoist-loop-invariant][graph-transforms]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;
    using namespace kg::ControlGraph;
    using namespace kg::CoordinateGraph;

    auto ctx = TestContext::ForDefaultTarget();

    kg::KernelGraph graph0;

    auto head = graph0.control.addElement(NOP{});

    auto loopSize                  = Expression::literal(10);
    auto [loopIndexCoord, forLoop] = kg::rangeFor(graph0, loopSize, "TestLoop");

    auto loopBody     = graph0.control.addElement(NOP{});
    auto loadConstant = graph0.control.addElement(LoadVGPR{});

    auto constantCoord = graph0.coordinates.addElement(VGPR{});
    auto resultCoord   = graph0.coordinates.addElement(VGPR{});

    auto constantDF = graph0.coordinates.addElement(DataFlow{}, {constantCoord}, {resultCoord});

    graph0.mapper.connect<VGPR>(loadConstant, constantDF);

    auto DF = [](int tag) {
        return std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{tag, Register::Type::Vector, DataType::Float});
    };

    auto loopInvariantExpr = Expression::literal(2.0f) * DF(constantDF);
    auto assignInvariant
        = graph0.control.addElement(Assign{Register::Type::Vector, loopInvariantExpr});

    graph0.mapper.connect(assignInvariant, resultCoord, NaryArgument::DEST);

    graph0.control.addElement(Sequence{}, {head}, {loadConstant});
    graph0.control.addElement(Sequence{}, {loadConstant}, {forLoop});
    graph0.control.addElement(Body{}, {forLoop}, {loopBody});
    graph0.control.addElement(Sequence{}, {loopBody}, {assignInvariant});

    std::string dotOutputBefore = graph0.toDOT(true);
    writeDotToFile(dotOutputBefore, "hoist_loop_invariant_independent_before.dot");

    // Apply the HoistLoopInvariant transformation
    kg::HoistLoopInvariant transform;
    auto                   graph1 = transform.apply(graph0);

    std::string dotOutputAfter = graph1.toDOT(true);
    writeDotToFile(dotOutputAfter, "hoist_loop_invariant_independent_after.dot");

    // Verify that the assign node was hoisted before the loop
    // The assignInvariant node should now have a sequence edge to the forLoop
    auto assignOutputs
        = graph1.control.getOutputNodeIndices<Sequence>(assignInvariant).to<std::vector>();
    REQUIRE(std::find(assignOutputs.begin(), assignOutputs.end(), forLoop) != assignOutputs.end());

    // Verify that assignInvariant is no longer inside the loop body
    auto loopBodyChildren
        = graph1.control.depthFirstVisit(loopBody, Graph::Direction::Downstream).to<std::vector>();
    REQUIRE(std::find(loopBodyChildren.begin(), loopBodyChildren.end(), assignInvariant)
            == loopBodyChildren.end());
}

TEST_CASE("HoistLoopInvariant dependent", "[kernel-graph][hoist-loop-invariant][graph-transforms]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;
    using namespace kg::ControlGraph;
    using namespace kg::CoordinateGraph;

    auto ctx = TestContext::ForDefaultTarget();

    kg::KernelGraph graph0;

    auto head = graph0.control.addElement(NOP{});

    auto loopSize                  = Expression::literal(10);
    auto [loopIndexCoord, forLoop] = kg::rangeFor(graph0, loopSize, "TestLoop");

    auto loopBody  = graph0.control.addElement(NOP{});
    auto destCoord = graph0.coordinates.addElement(VGPR{});

    auto rangeCoord = graph0.mapper.get(forLoop, NaryArgument::DEST);

    auto DF = [](int tag) {
        return std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{tag, Register::Type::Scalar, DataType::Int32});
    };

    auto loopDependentExpr = DF(rangeCoord) * Expression::literal(2);

    auto assignDependent
        = graph0.control.addElement(Assign{Register::Type::Vector, loopDependentExpr});

    graph0.mapper.connect(assignDependent, destCoord, NaryArgument::DEST);

    graph0.control.addElement(Sequence{}, {head}, {forLoop});
    graph0.control.addElement(Body{}, {forLoop}, {loopBody});
    graph0.control.addElement(Sequence{}, {loopBody}, {assignDependent});

    std::string dotOutputBefore = graph0.toDOT(true);
    writeDotToFile(dotOutputBefore, "hoist_loop_invariant_dependent_before.dot");

    // Apply the HoistLoopInvariant transformation
    kg::HoistLoopInvariant transform;
    auto                   graph1 = transform.apply(graph0);

    std::string dotOutputAfter = graph1.toDOT(true);
    writeDotToFile(dotOutputAfter, "hoist_loop_invariant_dependent_after.dot");

    // Verify that the assign node was NOT hoisted because it depends on the loop variable
    // The assignDependent node should still be inside the loop body
    auto loopBodyChildren
        = graph1.control.depthFirstVisit(loopBody, Graph::Direction::Downstream).to<std::vector>();
    REQUIRE(std::find(loopBodyChildren.begin(), loopBodyChildren.end(), assignDependent)
            != loopBodyChildren.end());

    // Verify that assignDependent does NOT have a direct sequence edge to the forLoop
    auto assignOutputs
        = graph1.control.getOutputNodeIndices<Sequence>(assignDependent).to<std::vector>();
    REQUIRE(std::find(assignOutputs.begin(), assignOutputs.end(), forLoop) == assignOutputs.end());
}

TEST_CASE("extractDataFlowTags", "[kernel-graph][hoist-loop-invariant][expression]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;

    SECTION("Binary operation with DataFlowTags")
    {
        // Create two DataFlowTag expressions with different tag IDs
        Expression::DataFlowTag tag1{42, Register::Type::Vector, DataType::Float};
        Expression::DataFlowTag tag2{77, Register::Type::Vector, DataType::Float};

        auto tag1Ptr = std::make_shared<Expression::Expression>(tag1);
        auto tag2Ptr = std::make_shared<Expression::Expression>(tag2);

        // Combine them with a binary operation (addition)
        auto binaryExpr = Expression::Add{{tag1Ptr, tag2Ptr}};

        // Extract the DataFlowTags
        auto extractedTags = kg::extractDataFlowTags(binaryExpr);

        // Verify that both tags were extracted
        REQUIRE(extractedTags.size() == 2);
        REQUIRE(extractedTags.count(42) == 1);
        REQUIRE(extractedTags.count(77) == 1);
    }
}

TEST_CASE("hoistNodeBeforeLoop", "[kernel-graph][hoist-loop-invariant][helper]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;
    using namespace kg::ControlGraph;
    using namespace kg::CoordinateGraph;

    auto ctx = TestContext::ForDefaultTarget();

    kg::KernelGraph graph;

    auto predecessor = graph.control.addElement(NOP{});

    auto loopSize                  = Expression::literal(10);
    auto [loopIndexCoord, forLoop] = kg::rangeFor(graph, loopSize, "TestLoop");

    auto loopBody = graph.control.addElement(NOP{});
    graph.control.addElement(Body{}, {forLoop}, {loopBody});
    auto sequenceEdge = graph.control.addElement(Sequence{}, {predecessor}, {forLoop});
    auto nodeToHoist  = graph.control.addElement(NOP{});
    graph.control.addElement(Sequence{}, {loopBody}, {nodeToHoist});

    std::string dotOutputBefore = graph.toDOT(true);
    writeDotToFile(dotOutputBefore, "hoist_node_before_loop_helper_before.dot");

    int result = kg::HoistLoopInvariant::hoistNodeBeforeLoop(
        graph, nodeToHoist, forLoop, predecessor, sequenceEdge);

    std::string dotOutputAfter = graph.toDOT(true);
    writeDotToFile(dotOutputAfter, "hoist_node_before_loop_helper_after.dot");

    // Verify that the node was hoisted
    // The nodeToHoist should now have a sequence edge to the forLoop
    auto nodeOutputs = graph.control.getOutputNodeIndices<Sequence>(nodeToHoist).to<std::vector>();
    REQUIRE(std::find(nodeOutputs.begin(), nodeOutputs.end(), forLoop) != nodeOutputs.end());

    // Verify that nodeToHoist is no longer inside the loop body
    auto loopBodyChildren
        = graph.control.depthFirstVisit(loopBody, Graph::Direction::Downstream).to<std::vector>();
    REQUIRE(std::find(loopBodyChildren.begin(), loopBodyChildren.end(), nodeToHoist)
            == loopBodyChildren.end());

    REQUIRE(result == nodeToHoist);
}
