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

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "../GenericContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace KernelGraphTest
{
    class KernelGraphSimplifyTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphSimplifyTest, RemoveRedundantEdges)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        auto b0 = graph0.control.addElement(Body(), {A}, {B});
        auto b1 = graph0.control.addElement(Body(), {B}, {C});

        auto b2 = graph0.control.addElement(Body(), {A}, {C});

        auto s0 = graph0.control.addElement(Sequence(), {A}, {C});

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 4);

        auto isBody = graph0.control.isElemType<Body>();
        Graph::removeRedundantEdges(graph0.control, isBody);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);

        EXPECT_ANY_THROW(graph0.control.getEdge(b2));
    }

    TEST_F(KernelGraphSimplifyTest, RemoveRedundantEdgesFullyDescribed)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());
        auto D = graph0.control.addElement(NOP());
        auto E = graph0.control.addElement(NOP());

        // Input: Graph with the following sequence but with every possible edge
        //  A -> C -> D
        //  B /    \ E
        // i.e. A->D, A->E, B->D, B->E

        std::vector<int> keep;
        std::vector<int> toss;

        keep.push_back(graph0.control.addElement(Sequence(), {A}, {C}));
        toss.push_back(graph0.control.addElement(Sequence(), {A}, {D}));
        toss.push_back(graph0.control.addElement(Sequence(), {A}, {E}));
        keep.push_back(graph0.control.addElement(Sequence(), {B}, {C}));
        toss.push_back(graph0.control.addElement(Sequence(), {B}, {D}));
        toss.push_back(graph0.control.addElement(Sequence(), {B}, {E}));
        keep.push_back(graph0.control.addElement(Sequence(), {C}, {D}));
        keep.push_back(graph0.control.addElement(Sequence(), {C}, {E}));

        // Add a node that shouldn't be considered at all.
        keep.push_back(graph0.control.addElement(Body(), {A}, {C}));

        for(auto idx : keep)
            EXPECT_NO_THROW(graph0.control.getEdge(idx)) << idx;
        for(auto idx : toss)
            EXPECT_NO_THROW(graph0.control.getEdge(idx)) << idx;

        auto isSequence = graph0.control.isElemType<Sequence>();

        EXPECT_EQ(Graph::findRedundantEdges(graph0.control, isSequence).to<std::vector>(), toss);

        Graph::removeRedundantEdges(graph0.control, isSequence);

        for(auto idx : keep)
            EXPECT_NO_THROW(graph0.control.getEdge(idx)) << idx;
        for(auto idx : toss)
            EXPECT_ANY_THROW(graph0.control.getEdge(idx)) << idx;
    }

    TEST_F(KernelGraphSimplifyTest, BasicRedundantSequence)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        auto s0 = graph0.control.addElement(Sequence(), {A}, {B});
        auto s1 = graph0.control.addElement(Sequence(), {B}, {C});

        auto s2 = graph0.control.addElement(Sequence(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);

        EXPECT_ANY_THROW(graph1.control.getEdge(s2));
    }

    TEST_F(KernelGraphSimplifyTest, BasicRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        auto b0 = graph0.control.addElement(Body(), {A}, {B});
        auto s0 = graph0.control.addElement(Sequence(), {B}, {C});

        auto b1 = graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);

        EXPECT_ANY_THROW(graph1.control.getEdge(b1));
    }

    TEST_F(KernelGraphSimplifyTest, DoubleRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        auto b0 = graph0.control.addElement(Body(), {A}, {B});
        auto b1 = graph0.control.addElement(Body(), {A}, {B});
        auto s0 = graph0.control.addElement(Sequence(), {B}, {C});

        auto b2 = graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 4);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);

        // Either b0 or b1 could be removed.
        EXPECT_ANY_THROW({
            graph1.control.getEdge(b0);
            graph1.control.getEdge(b1);
        });
        EXPECT_ANY_THROW(graph1.control.getEdge(b2));
    }

    TEST_F(KernelGraphSimplifyTest, MultipleRedundantBody)
    {
        //
        //  graph0's control graph is like this:
        //
        //                  A
        //                  | (body)
        //  ------------------------------
        //  |         |        |         |
        //  |         |        |         | ...
        //  v         v        v         v
        // NOP ---> NOP --->  NOP ---> NOP
        //    (seq)    (seq)     (seq)
        //

        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());

        std::vector<int> bodyNodes;
        std::vector<int> bodyEdges;
        for(int i = 0; i < 10; i++)
        {
            auto nop = graph0.control.addElement(NOP());
            bodyEdges.push_back(graph0.control.addElement(Body(), {A}, {nop}));
            bodyNodes.push_back(nop);
        }

        //
        // Chain the Body nodes with Sequence edges
        //
        for(int i = 1; i < bodyNodes.size(); i++)
            graph0.control.addElement(Sequence(), {bodyNodes[i - 1]}, {bodyNodes[i]});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        //
        // Use the result of baseline method as verification
        //
        rocRoller::KernelGraph::removeRedundantBodyEdgesBaselineMethod(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>(),
                  graph1.control.getEdges().to<std::vector>());

        //
        // Verify only the first Body edge remains
        //
        for(int i = 1; i < bodyEdges.size(); i++)
            EXPECT_ANY_THROW(graph1.control.getEdge(bodyEdges[i]));
    }
}
