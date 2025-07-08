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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

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
