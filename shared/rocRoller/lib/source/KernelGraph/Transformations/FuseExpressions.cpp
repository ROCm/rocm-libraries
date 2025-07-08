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

#include <optional>
#include <tuple>
#include <unordered_set>
#include <variant>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;

    /**
     * @brief Look for {Assign Multiply(., .)} --Sequence--> {Assign Add(., .)}.
     *
     * Look for
     *
     *   Assign Multiply(., .) -- Sequence --> Assign Add(., .)
     *
     * Make sure only one DF edge out of the result of the multiply.
     */
    std::optional<std::tuple<int, int>> findMultiplyAdd(KernelGraph const&             kgraph,
                                                        std::unordered_set<int> const& exclude)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        for(auto const& parent : kgraph.control.getNodes<Assign>())
        {
            if(exclude.contains(parent))
                continue;

            // find Multiply
            auto multiplyAssign = kgraph.control.get<Assign>(parent);
            if(!std::holds_alternative<Expression::Multiply>(*multiplyAssign->expression))
                continue;

            auto child = only(kgraph.control.getOutputNodeIndices<Sequence>(parent));
            if(!child)
                continue;

            auto addAssign = kgraph.control.get<Assign>(*child);
            if(!addAssign)
                continue;

            auto dst = kgraph.mapper.get(parent, NaryArgument::DEST);
            AssertFatal(dst != -1, "Invalid connection.");
            auto dfs = only(kgraph.coordinates.getOutputNodeIndices(dst, CT::isEdge<CT::DataFlow>));
            if(!dfs)
                continue;

            if(!std::holds_alternative<Expression::Add>(*addAssign->expression))
                continue;

            return {{parent, *child}};
        }

        return {};
    }

    /**
     * @brief Reconnect incoming and outgoing DataFlow edges from dimension.
     *
     * Inputs coming into the dimension are augmented with `other_inputs`.
     */
    void reflow(KernelGraph& graph, int dim, std::vector<int> const& other_inputs)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        std::vector<int> inputs, outputs;
        for(auto const& tag : graph.coordinates.getNeighbours<Graph::Direction::Upstream>(dim))
        {
            auto df = graph.coordinates.get<CT::DataFlow>(tag);
            if(!df)
                continue;
            auto parents = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(tag);
            for(auto const parent : parents)
                inputs.push_back(parent);
            graph.coordinates.deleteElement(tag);
        }

        for(auto const& tag : graph.coordinates.getNeighbours<Graph::Direction::Downstream>(dim))
        {
            auto df = graph.coordinates.get<CT::DataFlow>(tag);
            if(!df)
                continue;
            auto children = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(tag);
            for(auto const child : children)
                outputs.push_back(child);
            graph.coordinates.deleteElement(tag);
        }

        std::copy(other_inputs.cbegin(), other_inputs.cend(), std::back_inserter(inputs));
        graph.coordinates.addElement(CT::DataFlow(), inputs, outputs);
    }

    /**
     * @brief Fuse Add(Multiply(., .), .) into MultiplyAdd(., ., .).
     */
    KernelGraph fuseMultiplyAdd(KernelGraph const& original)
    {
        using Expression::multiplyAdd;

        auto kgraph = original;

        std::unordered_set<int> excluded;

        for(;;)
        {
            auto candidate = findMultiplyAdd(kgraph, excluded);
            if(!candidate)
                break;

            auto const& [multiplyTag, addTag] = *candidate;
            excluded.emplace(multiplyTag);

            auto [addLHS, addLHSDF] = getBinaryLHS<Expression::Add>(kgraph, addTag);
            auto mulDST             = getDEST(kgraph, multiplyTag);

            if(addLHS != mulDST)
                continue;

            auto addDST             = getDEST(kgraph, addTag);
            auto [addRHS, addRHSDF] = getBinaryRHS<Expression::Add>(kgraph, addTag);
            auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, multiplyTag);
            auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, multiplyTag);
            auto fma                = multiplyAdd(mulLHSDF, mulRHSDF, addRHSDF);

            // Reuse register type and count from Multiply
            auto fmaAssign       = *kgraph.control.get<Assign>(multiplyTag);
            fmaAssign.expression = fma;

            auto fmaTag = kgraph.control.addElement(fmaAssign);

            // Connect FMA; delete Multiply and Add operations
            reconnect<Graph::Direction::Downstream>(kgraph, -1, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, addTag);
            reconnect<Graph::Direction::Downstream>(kgraph, fmaTag, addTag);

            kgraph.control.deleteElement(multiplyTag);
            kgraph.control.deleteElement(addTag);

            // Tidy up coordinate graph
            reflow(kgraph, addLHS, {addRHS});
            kgraph.coordinates.deleteElement(addLHS);

            // Tidy up connections
            kgraph.mapper.purge(multiplyTag);
            kgraph.mapper.purge(addTag);

            // Connect FMA
            kgraph.mapper.connect(fmaTag, addDST, NaryArgument::DEST);
        }

        return kgraph;
    }

    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::fuseExpressions");
        return fuseMultiplyAdd(original);
    }
}
