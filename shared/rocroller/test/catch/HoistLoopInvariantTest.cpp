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

    std::string dotOutput = graph0.toDOT(true);
    writeDotToFile(dotOutput, "hoist_loop_invariant_independent.dot");
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

    std::string dotOutput = graph0.toDOT(true);
    writeDotToFile(dotOutput, "hoist_loop_invariant_dependent.dot");
}
