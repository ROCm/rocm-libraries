// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

TEST_CASE("Simplify redundant Sequence edges", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    KernelGraph graph;

    auto kernel = graph.control.addElement(Kernel());
    auto loadA  = graph.control.addElement(LoadTiled());
    auto loadB  = graph.control.addElement(LoadTiled());
    auto assign = graph.control.addElement(Assign());

    graph.control.addElement(Sequence(), {kernel}, {loadA});
    graph.control.addElement(Sequence(), {kernel}, {loadB});
    graph.control.addElement(Sequence(), {loadA}, {loadB});
    graph.control.addElement(Sequence(), {loadA}, {assign});
    graph.control.addElement(Sequence(), {loadB}, {assign});

    /* graph is:
     *
     *          Kernel
     *         /   |
     *     LoadA   |
     *       |  \  |
     *       |   LoadB
     *       \    /
     *       Assign
     */
    CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 5);

    removeRedundantSequenceEdges(graph);

    /* graph should be
     *
     *          Kernel
     *         /
     *     LoadA
     *          \
     *           LoadB
     *            /
     *       Assign
     */

    CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 3);
}

TEST_CASE("Simplify redundant NOPs", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    SECTION("Simple redundant NOP")
    {
        KernelGraph graph;

        auto kernel = graph.control.addElement(Kernel());
        auto nop0   = graph.control.addElement(NOP());
        auto loadA  = graph.control.addElement(LoadTiled());
        auto loadB  = graph.control.addElement(LoadTiled());
        auto assign = graph.control.addElement(Assign());

        graph.control.addElement(Sequence(), {kernel}, {nop0});
        graph.control.addElement(Sequence(), {nop0}, {loadA});
        graph.control.addElement(Sequence(), {nop0}, {loadB});
        graph.control.addElement(Sequence(), {loadA}, {assign});
        graph.control.addElement(Sequence(), {loadB}, {assign});

        /* graph is:
         *
         *          Kernel
         *            |
         *           NOP
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 1);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 5);

        removeRedundantNOPs(graph);

        /* graph should be:
         *
         *          Kernel
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 0);

        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 4);
    }

    SECTION("Simple double NOP")
    {
        KernelGraph graph;

        auto kernel = graph.control.addElement(Kernel());
        auto nop0   = graph.control.addElement(NOP());
        auto nop1   = graph.control.addElement(NOP());
        auto assign = graph.control.addElement(Assign());

        graph.control.addElement(Sequence(), {kernel}, {nop0});
        graph.control.addElement(Sequence(), {kernel}, {nop1});
        graph.control.addElement(Sequence(), {nop0}, {assign});
        graph.control.addElement(Sequence(), {nop1}, {assign});

        /* graph is:
         *
         *          Kernel
         *          /   \
         *        NOP   NOP
         *          \   /
         *         Assign
         */

        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 2);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 4);

        removeRedundantNOPs(graph);

        /* graph should be:
         *
         *          Kernel
         *            |
         *          Assign
         */

        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 0);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 1);
    }

    SECTION("Scheduling redundant NOP")
    {
        KernelGraph graph;

        auto kernel = graph.control.addElement(Kernel());
        auto nop0   = graph.control.addElement(NOP());
        auto loadA  = graph.control.addElement(LoadTiled());
        auto loadB  = graph.control.addElement(LoadTiled());
        auto assign = graph.control.addElement(Assign());

        graph.control.addElement(Sequence(), {kernel}, {loadA});
        graph.control.addElement(Sequence(), {kernel}, {loadB});
        graph.control.addElement(Sequence(), {loadA}, {nop0});
        graph.control.addElement(Sequence(), {loadB}, {nop0});
        graph.control.addElement(Sequence(), {nop0}, {assign});

        /* graph is:
         *
         *          Kernel
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *           NOP
         *            |
         *         Assign
         *
         * Some transforms use NOPs as aids for scheduling.  As an
         * example, the NOP here ensures the loads happen before
         * assign.
         */

        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 5);
        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 1);

        removeRedundantNOPs(graph);

        /* graph should be:
         *
         *          Kernel
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 4);
        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 0);
    }

    SECTION("Scheduling NOPs from a ForLoop")
    {
        KernelGraph graph;

        auto forLoop = graph.control.addElement(ForLoopOp());
        auto nop0    = graph.control.addElement(NOP());
        auto nop1    = graph.control.addElement(NOP());
        auto loadA   = graph.control.addElement(LoadTiled());
        auto loadB   = graph.control.addElement(LoadTiled());

        graph.control.addElement(Body(), {forLoop}, {nop0});
        graph.control.addElement(Body(), {forLoop}, {nop1});
        graph.control.addElement(Sequence(), {nop0}, {loadA});
        graph.control.addElement(Sequence(), {nop1}, {loadB});

        /* graph is:
         *
         *         ForLoop
         *          /   \
         *        NOP   NOP
         *         |     |
         *       LoadA  LoadB
         *
         * where the edges out of the ForLoop are bodies.
         */
        CHECK(graph.control.getElements<Body>().to<std::vector>().size() == 2);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 2);
        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 2);

        removeRedundantNOPs(graph);

        /* graph should be:
         *
         *
         *         ForLoop
         *          /   \
         *       LoadA  LoadB
         *
         * where the edges out of the ForLoop are bodies.
         */

        CHECK(graph.control.getElements<Body>().to<std::vector>().size() == 2);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 0);
        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 0);
    }

    SECTION("Nasty NOP cluster")
    {
        KernelGraph graph;

        auto kernel = graph.control.addElement(Kernel());
        auto nop0   = graph.control.addElement(NOP());
        auto nop1   = graph.control.addElement(NOP());
        auto nop2   = graph.control.addElement(NOP());
        auto nop3   = graph.control.addElement(NOP());
        auto assign = graph.control.addElement(Assign());

        graph.control.addElement(Sequence(), {kernel}, {nop0});
        graph.control.addElement(Sequence(), {kernel}, {nop1});
        graph.control.addElement(Sequence(), {nop0}, {nop2});
        graph.control.addElement(Sequence(), {nop1}, {nop2});
        graph.control.addElement(Sequence(), {nop1}, {nop3});
        graph.control.addElement(Sequence(), {nop2}, {assign});
        graph.control.addElement(Sequence(), {nop3}, {assign});

        /* graph is:
         *
         *          Kernel
         *          /   \
         *        NOP   NOP
         *          \   / \
         *           NOP  NOP
         *             \  /
         *            Assign
         */
        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 4);
        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 7);

        removeRedundantNOPs(graph);

        /* graph should be:
         *
         *          Kernel
         *            |
         *          Assign
         */

        CHECK(graph.control.getElements<NOP>().to<std::vector>().size() == 0);

        CHECK(graph.control.getElements<Sequence>().to<std::vector>().size() == 1);
    }
}

TEST_CASE("Remove redundant edges", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());
    auto B = graph0.control.addElement(Assign());
    auto C = graph0.control.addElement(Assign());

    auto b0 = graph0.control.addElement(Body(), {A}, {B});
    auto b1 = graph0.control.addElement(Body(), {B}, {C});

    auto b2 = graph0.control.addElement(Body(), {A}, {C});

    auto s0 = graph0.control.addElement(Sequence(), {A}, {C});

    CHECK(graph0.control.getEdges().to<std::vector>().size() == 4);

    auto isBody = graph0.control.isElemType<Body>();
    rocRoller::Graph::removeRedundantEdges(graph0.control, isBody);

    CHECK(graph0.control.getEdges().to<std::vector>().size() == 3);

    CHECK_THROWS(graph0.control.getEdge(b2));
}

TEST_CASE("Remove redundant edges fully described", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());
    auto B = graph0.control.addElement(Assign());
    auto C = graph0.control.addElement(Assign());
    auto D = graph0.control.addElement(Assign());
    auto E = graph0.control.addElement(Assign());

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
        CHECK_NOTHROW(graph0.control.getEdge(idx));
    for(auto idx : toss)
        CHECK_NOTHROW(graph0.control.getEdge(idx));

    auto isSequence = graph0.control.isElemType<Sequence>();

    CHECK(rocRoller::Graph::findRedundantEdges(graph0.control, isSequence).to<std::vector>()
          == toss);

    rocRoller::Graph::removeRedundantEdges(graph0.control, isSequence);

    for(auto idx : keep)
        CHECK_NOTHROW(graph0.control.getEdge(idx));
    for(auto idx : toss)
        CHECK_THROWS(graph0.control.getEdge(idx));
}

TEST_CASE("Basic redundant Sequence", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());
    auto B = graph0.control.addElement(Assign());
    auto C = graph0.control.addElement(Assign());

    auto s0 = graph0.control.addElement(Sequence(), {A}, {B});
    auto s1 = graph0.control.addElement(Sequence(), {B}, {C});

    auto s2 = graph0.control.addElement(Sequence(), {A}, {C});

    auto graph1 = Simplify().apply(graph0);

    CHECK(graph0.control.getEdges().to<std::vector>().size() == 3);
    CHECK(graph1.control.getEdges().to<std::vector>().size() == 2);

    CHECK_THROWS(graph1.control.getEdge(s2));
}

TEST_CASE("Basic redundant Body", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());
    auto B = graph0.control.addElement(Assign());
    auto C = graph0.control.addElement(Assign());

    auto b0 = graph0.control.addElement(Body(), {A}, {B});
    auto s0 = graph0.control.addElement(Sequence(), {B}, {C});

    auto b1 = graph0.control.addElement(Body(), {A}, {C});

    auto graph1 = Simplify().apply(graph0);

    CHECK(graph0.control.getEdges().to<std::vector>().size() == 3);
    CHECK(graph1.control.getEdges().to<std::vector>().size() == 2);

    CHECK_THROWS(graph1.control.getEdge(b1));
}

TEST_CASE("Double redundant Body", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());
    auto B = graph0.control.addElement(Assign());
    auto C = graph0.control.addElement(Assign());

    auto b0 = graph0.control.addElement(Body(), {A}, {B});
    auto b1 = graph0.control.addElement(Body(), {A}, {B});
    auto s0 = graph0.control.addElement(Sequence(), {B}, {C});

    auto b2 = graph0.control.addElement(Body(), {A}, {C});

    auto graph1 = Simplify().apply(graph0);

    CHECK(graph0.control.getEdges().to<std::vector>().size() == 4);
    CHECK(graph1.control.getEdges().to<std::vector>().size() == 2);

    // Either b0 or b1 could be removed.
    CHECK_THROWS([&]() {
        graph1.control.getEdge(b0);
        graph1.control.getEdge(b1);
    }());
    CHECK_THROWS(graph1.control.getEdge(b2));
}

TEST_CASE("Multiple redundant Body", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    //
    //  graph0's control graph is like this:
    //
    //                  A
    //                  | (body)
    //  ------------------------------
    //  |         |        |         |
    //  |         |        |         | ...
    //  v         v        v         v
    // Assign -> Assign -> Assign -> Assign
    //    (seq)    (seq)     (seq)
    //

    auto graph0 = KernelGraph();

    auto A = graph0.control.addElement(Assign());

    std::vector<int> bodyNodes;
    std::vector<int> bodyEdges;
    for(int i = 0; i < 10; i++)
    {
        auto assign = graph0.control.addElement(Assign());
        bodyEdges.push_back(graph0.control.addElement(Body(), {A}, {assign}));
        bodyNodes.push_back(assign);
    }

    //
    // Chain the Body nodes with Sequence edges
    //
    for(int i = 1; i < bodyNodes.size(); i++)
        graph0.control.addElement(Sequence(), {bodyNodes[i - 1]}, {bodyNodes[i]});

    auto graph1 = Simplify().apply(graph0);

    //
    // Use the result of baseline method as verification
    //
    removeRedundantBodyEdgesBaselineMethod(graph0);

    CHECK(graph0.control.getEdges().to<std::vector>()
          == graph1.control.getEdges().to<std::vector>());

    //
    // Verify only the first Body edge remains
    //
    for(int i = 1; i < bodyEdges.size(); i++)
        CHECK_THROWS(graph1.control.getEdge(bodyEdges[i]));
}
