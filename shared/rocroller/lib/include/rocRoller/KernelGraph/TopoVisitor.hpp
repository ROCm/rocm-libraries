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

#pragma once

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>

namespace rocRoller::KernelGraph
{
    /**
     * Utility base class that facilitates visiting each node in a control
     * graph in topological (program) order.
     * Note that this will only walk in one of potentially many valid
     * topological orderings of the control graph; most kernels have some
     * ambiguity here.
     *
     * If MyClass is derived from TopoControlGraphVisitor<MyClass>, it must
     * implement the call operator for the following signature:
     * void (operator()(int node, Operation op));
     *
     * A FatalError will be thrown if the entire graph cannot be walked in
     * program order.  This might be because:
     *  - There is a cycle in the graph
     *  - There is a Sequence edge that spans two Body scopes: either between
     *    two nodes which have different immediate Body parents, or between two
     *    separate logical scopes of the same Body parent (such as from the
     *    Initialize section of a For loop to the Body section of that same
     *    For loop.)
     */
    template <typename Derived>
    class TopoControlGraphVisitor
    {
    public:
        /**
         * Walks the entire control graph in topological order.
         */
        void walk();

        /**
         * Walks `nodes` and their descendents (both by Body and Sequence
         * relationships) in topological order.
         *
         * `nodes` must all have the same logical scope.
         */
        void walk(std::set<int> nodes);

        /**
         * Walks `node` and its Body descendents in topological order.
         */
        void walk(int node);

        /**
         * Resets the internal set of which nodes have been visited.  This
         * must be called in between calls in order to reuse this object.
         */
        void reset();

        TopoControlGraphVisitor(KernelGraph const& kgraph);

    protected:
        KernelGraph const& m_graph;

        /**
         * Default error handling.  Base implementation will throw an error.
         * Derived classes can override this in order to report an error in a
         * different way.
         */
        virtual void errorCondition(std::string const& message);

    private:
        std::unordered_map<int, std::unordered_set<int>> m_nodeDependencies;
        std::unordered_set<int>                          m_visitedNodes;

        std::unordered_set<int>& nodeDependencies(int node);

        bool hasWalkedInputs(int node);

        constexpr Derived* derived();

        auto call(int node);
    };
}

#include "TopoVisitor_impl.hpp"
