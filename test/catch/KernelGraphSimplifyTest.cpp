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

    KernelGraph graph0;

    auto kernel = graph0.control.addElement(Kernel());
    auto loadA  = graph0.control.addElement(LoadTiled());
    auto loadB  = graph0.control.addElement(LoadTiled());
    auto assign = graph0.control.addElement(Assign());

    graph0.control.addElement(Sequence(), {kernel}, {loadA});
    graph0.control.addElement(Sequence(), {kernel}, {loadB});
    graph0.control.addElement(Sequence(), {loadA}, {loadB});
    graph0.control.addElement(Sequence(), {loadA}, {assign});
    graph0.control.addElement(Sequence(), {loadB}, {assign});

    /* graph0 is:
     *
     *          Kernel
     *         /   |
     *     LoadA   |
     *       |  \  |
     *       |   LoadB
     *       \    /
     *       Assign
     */

    auto graph1 = removeRedundantSequenceEdges(graph0);

    /* graph1 should be
     *
     *          Kernel
     *         /
     *     LoadA
     *          \
     *           LoadB
     *            /
     *       Assign
     */

    CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 5);
    CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 3);
}

TEST_CASE("Simplify redundant NOPs", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    SECTION("Simple redundant NOP")
    {
        KernelGraph graph0;

        auto kernel = graph0.control.addElement(Kernel());
        auto nop0   = graph0.control.addElement(NOP());
        auto loadA  = graph0.control.addElement(LoadTiled());
        auto loadB  = graph0.control.addElement(LoadTiled());
        auto assign = graph0.control.addElement(Assign());

        graph0.control.addElement(Sequence(), {kernel}, {nop0});
        graph0.control.addElement(Sequence(), {nop0}, {loadA});
        graph0.control.addElement(Sequence(), {nop0}, {loadB});
        graph0.control.addElement(Sequence(), {loadA}, {assign});
        graph0.control.addElement(Sequence(), {loadB}, {assign});

        /* graph0 is:
         *
         *          Kernel
         *            |
         *           NOP
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        auto graph1 = removeRedundantNOPs(graph0);

        /* graph1 should be:
         *
         *          Kernel
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        CHECK(graph0.control.getElements<NOP>().to<std::vector>().size() == 1);
        CHECK(graph1.control.getElements<NOP>().to<std::vector>().size() == 0);

        CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 5);
        CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 4);
    }

    SECTION("Simple double NOP")
    {
        KernelGraph graph0;

        auto kernel = graph0.control.addElement(Kernel());
        auto nop0   = graph0.control.addElement(NOP());
        auto nop1   = graph0.control.addElement(NOP());
        auto assign = graph0.control.addElement(Assign());

        graph0.control.addElement(Sequence(), {kernel}, {nop0});
        graph0.control.addElement(Sequence(), {kernel}, {nop1});
        graph0.control.addElement(Sequence(), {nop0}, {assign});
        graph0.control.addElement(Sequence(), {nop1}, {assign});

        /* graph0 is:
         *
         *          Kernel
         *          /   \
         *        NOP   NOP
         *          \   /
         *         Assign
         */

        auto graph1 = removeRedundantNOPs(graph0);

        /* graph1 should be:
         *
         *          Kernel
         *            |
         *          Assign
         */

        CHECK(graph0.control.getElements<NOP>().to<std::vector>().size() == 2);
        CHECK(graph1.control.getElements<NOP>().to<std::vector>().size() == 0);

        CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 4);
        CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 1);
    }

    SECTION("Scheduling redundant NOP")
    {
        KernelGraph graph0;

        auto kernel = graph0.control.addElement(Kernel());
        auto nop0   = graph0.control.addElement(NOP());
        auto loadA  = graph0.control.addElement(LoadTiled());
        auto loadB  = graph0.control.addElement(LoadTiled());
        auto assign = graph0.control.addElement(Assign());

        graph0.control.addElement(Sequence(), {kernel}, {loadA});
        graph0.control.addElement(Sequence(), {kernel}, {loadB});
        graph0.control.addElement(Sequence(), {loadA}, {nop0});
        graph0.control.addElement(Sequence(), {loadB}, {nop0});
        graph0.control.addElement(Sequence(), {nop0}, {assign});

        /* graph0 is:
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

        auto graph1 = removeRedundantNOPs(graph0);

        /* graph1 should be:
         *
         *          Kernel
         *          /   \
         *      LoadA   LoadB
         *          \   /
         *         Assign
         */

        CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 5);
        CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 4);

        CHECK(graph0.control.getElements<NOP>().to<std::vector>().size() == 1);
        CHECK(graph1.control.getElements<NOP>().to<std::vector>().size() == 0);
    }

    SECTION("Scheduling NOPs from a ForLoop")
    {
        KernelGraph graph0;

        auto forLoop = graph0.control.addElement(ForLoopOp());
        auto nop0    = graph0.control.addElement(NOP());
        auto nop1    = graph0.control.addElement(NOP());
        auto loadA   = graph0.control.addElement(LoadTiled());
        auto loadB   = graph0.control.addElement(LoadTiled());

        graph0.control.addElement(Body(), {forLoop}, {nop0});
        graph0.control.addElement(Body(), {forLoop}, {nop1});
        graph0.control.addElement(Sequence(), {nop0}, {loadA});
        graph0.control.addElement(Sequence(), {nop1}, {loadB});

        /* graph0 is:
         *
         *         ForLoop
         *          /   \
         *        NOP   NOP
         *         |     |
	 *       LoadA  LoadB
	 *
	 * where the edges out of the ForLoop are bodies.
         */

        auto graph1 = removeRedundantNOPs(graph0);

        /* graph1 should be:
         *
         *
         *         ForLoop
         *          /   \
	 *       LoadA  LoadB
	 *
	 * where the edges out of the ForLoop are bodies.
         */

        CHECK(graph0.control.getElements<Body>().to<std::vector>().size() == 2);
        CHECK(graph1.control.getElements<Body>().to<std::vector>().size() == 2);

        CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 2);
        CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 0);

        CHECK(graph0.control.getElements<NOP>().to<std::vector>().size() == 2);
        CHECK(graph1.control.getElements<NOP>().to<std::vector>().size() == 0);
    }

    SECTION("Nasty NOP cluster")
    {
        KernelGraph graph0;

        auto kernel = graph0.control.addElement(Kernel());
        auto nop0   = graph0.control.addElement(NOP());
        auto nop1   = graph0.control.addElement(NOP());
        auto nop2   = graph0.control.addElement(NOP());
        auto nop3   = graph0.control.addElement(NOP());
        auto assign = graph0.control.addElement(Assign());

        graph0.control.addElement(Sequence(), {kernel}, {nop0});
        graph0.control.addElement(Sequence(), {kernel}, {nop1});
        graph0.control.addElement(Sequence(), {nop0}, {nop2});
        graph0.control.addElement(Sequence(), {nop1}, {nop2});
        graph0.control.addElement(Sequence(), {nop1}, {nop3});
        graph0.control.addElement(Sequence(), {nop2}, {assign});
        graph0.control.addElement(Sequence(), {nop3}, {assign});

        /* graph0 is:
         *
         *          Kernel
         *          /   \
         *        NOP   NOP
         *          \   / \
         *           NOP  NOP
         *             \  /
         *            Assign
         */

        auto graph1 = removeRedundantNOPs(graph0);

        /* graph1 should be:
         *
         *          Kernel
         *            |
         *          Assign
         */

        CHECK(graph0.control.getElements<NOP>().to<std::vector>().size() == 4);
        CHECK(graph1.control.getElements<NOP>().to<std::vector>().size() == 0);

        CHECK(graph0.control.getElements<Sequence>().to<std::vector>().size() == 7);
        CHECK(graph1.control.getElements<Sequence>().to<std::vector>().size() == 1);
    }
}
