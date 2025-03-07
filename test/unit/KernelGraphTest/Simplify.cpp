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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

#include "../GenericContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace KernelGraphTest
{
    class KernelGraphSimplifyTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphSimplifyTest, BasicRedundantSequence)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Sequence(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Sequence(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }

    TEST_F(KernelGraphSimplifyTest, BasicRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }

    TEST_F(KernelGraphSimplifyTest, DoubleRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 4);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }
}
