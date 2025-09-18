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
#include <rocRoller/KernelGraph/Transforms/ConstantPropagation.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <queue>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    /**
     * @brief Find the <scalarDFTag, scalarTag, tensorTag> when scalar * tensor
     */
    std::vector<std::tuple<int, int, int>> findScalarTensorTag(KernelGraph const& kgraph)
    {
        std::vector<std::pair<int, int>>       loadVGPRTag;
        std::vector<std::tuple<int, int, int>> candidates;

        auto loadVGPRNodes = kgraph.control.getNodes<LoadVGPR>().to<std::vector>();

        for(auto nodeTag : loadVGPRNodes)
        {
            auto DFtag = kgraph.mapper.get<VGPR>(nodeTag);
            loadVGPRTag.push_back({DFtag, nodeTag});
        }

        for(auto assign : kgraph.control.getNodes<Assign>().to<std::vector>())
        {
            auto node = kgraph.control.get<Assign>(assign);
            if(!std::holds_alternative<Expression::Multiply>(*node->expression))
                continue;
            auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, assign);
            auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, assign);
            for(const auto& tag1 : loadVGPRTag)
            {
                for(const auto& nodeTag : kgraph.control.getNodes<LoadTiled>().to<std::vector>())
                {
                    auto tag2 = kgraph.mapper.get<MacroTile>(nodeTag);
                    if(((tag1.first == mulLHS) && (tag2 == mulRHS))
                       || ((tag1.first == mulRHS) && (tag2 == mulLHS)))
                    {
                        candidates.push_back({tag1.first, tag1.second, nodeTag});
                    }
                }
            }
        }
        if(candidates.empty())
            return {{-1, -1, -1}};
        return candidates;
    }

    /**
     * @brief Find the operations (i.e., Assign, Multiply, LoadTiled) that the downstream destination matches the DataFlowTag
     */
    std::vector<int> findDest(KernelGraph const& kgraph, int DataFlowTag, int head)
    {
        std::vector<int> nodes;
        auto             dfs
            = kgraph.control.depthFirstVisit(head, Graph::Direction::Downstream).to<std::vector>();
        for(auto const& tag : dfs)
        {
            auto element = kgraph.control.getElement(tag);
            if(std::holds_alternative<ControlEdge>(element))
                continue;

            if(!std::holds_alternative<Operation>(element))
                continue;

            auto op   = std::get<Operation>(element);
            auto dest = -1;

            if(std::holds_alternative<Assign>(op))
                dest = getDEST(kgraph, tag);
            else if(std::holds_alternative<Multiply>(op))
                dest = kgraph.mapper.get(tag, NaryArgument::DEST);
            else if(std::holds_alternative<LoadTiled>(op))
                dest = kgraph.mapper.get<MacroTile>(tag);
            else
                continue;
            if(dest == DataFlowTag)
                nodes.push_back(tag);
        }
        return nodes;
    }

    /**
     * @brief Find the nodes to propagate zero DataFlowTag
     */

    /*
    for each DataFlowTag of in zero container
        for each Assign node
            if it is AssignMultiply(DataFlowTag1 * 0) node
                find the operations that their destination Dest == DataFlowTag1
                then replace these operations with NOP
                replace AssignMultiply with NOP
                push AssignMultiplyDest to the zero container
            if it is AssignAdd (DataFlowTag2 + 0)
                find the operations that their destination Dest == DataFlowTag2
                change the Dest to AssignAddDest
                replace AssignAdd with NOP
    */
    std::vector<std::pair<int, int>>
        zeroPropagation(KernelGraph const& original, int DFTag, int head)
    {
        auto                             kgraph = original;
        std::vector<std::pair<int, int>> candidates;
        std::queue<int>                  zeros;

        zeros.push(DFTag);

        while(!zeros.empty())
        {
            int zero = zeros.front();
            zeros.pop();

            auto assigns = kgraph.control.getNodes<Assign>().to<std::vector>();
            for(auto const& tag : assigns)
            {
                auto node = kgraph.control.get<Assign>(tag);
                if(std::holds_alternative<Expression::Multiply>(*node->expression))
                {
                    auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, tag);
                    auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, tag);
                    int mulDest             = getDEST(kgraph, tag);

                    if((mulLHS == zero) || (mulRHS == zero))
                    {
                        auto target = (mulLHS == zero) ? mulRHS : mulLHS;
                        auto nodes  = findDest(kgraph, target, head);
                        candidates.push_back({tag, -1});
                        zeros.push(mulDest);
                        for(auto const& nodeTag : nodes)
                            candidates.push_back({nodeTag, -1});
                    }
                }

                if(std::holds_alternative<Expression::Add>(*node->expression))
                {
                    auto [addLHS, addLHSDF] = getBinaryLHS<Expression::Add>(kgraph, tag);
                    auto [addRHS, addRHSDF] = getBinaryRHS<Expression::Add>(kgraph, tag);
                    int addDest             = getDEST(kgraph, tag);

                    if((addLHS == zero) || (addRHS == zero))
                    {
                        auto target = (addLHS == zero) ? addRHS : addLHS;
                        auto nodes  = findDest(kgraph, target, head);
                        candidates.push_back({tag, -1});
                        for(auto const& nodeTag : nodes)
                            candidates.push_back({nodeTag, addDest});
                    }
                }
            }
        }
        return candidates;
    }

    KernelGraph scalarIsZero(KernelGraph const& original)
    {
        auto kgraph = original;

        auto [scalarDFTag, scalarTag, tensorTag] = *only(findScalarTensorTag(kgraph));
        if(scalarDFTag == -1)
            return kgraph;

        // find the topmost for-loop that its body contains load tensor operation
        auto tmpTag = findContainingOperation<ForLoopOp>(tensorTag, kgraph);
        auto head   = 0;
        while(tmpTag)
        {
            head   = *tmpTag;
            tmpTag = findContainingOperation<ForLoopOp>(head, kgraph);
        }

        //  TODO: a more general way to find the head, so that it could be an arbitrary operation
        AssertFatal(head != 0, "Unable to perform ConstantPropagation: containing loop not found.");

        auto candidates = zeroPropagation(kgraph, scalarDFTag, head);

        auto         load          = kgraph.control.get<LoadVGPR>(scalarTag);
        VariableType scalarVarType = load->varType;
        auto         zero          = Expression::literal(0, scalarVarType);
        auto         DF            = [scalarVarType](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Scalar, scalarVarType});
        };

        auto condOp
            = kgraph.control.addElement(ConditionalOp{zero == DF(scalarDFTag), "0 == beta"});

        // find (parent --e--> head --body--> node)
        auto e           = *only(kgraph.control.getNeighbours<Graph::Direction::Upstream>(head));
        auto parent      = *only(kgraph.control.getInputNodeIndices<Sequence>(head));
        auto forLoopBody = *only(kgraph.control.getOutputNodeIndices<Body>(head));

        // change it to (parent  --head--> condOp)
        auto elem = kgraph.control.getElement(e);
        kgraph.control.deleteElement(e);
        kgraph.control.addElement(elem, {parent}, {condOp});

        // connect the true and else with conditional operation node
        auto elseTag
            = *only(duplicateControlNodes(kgraph, nullptr, {head}, [](int x) { return false; }));
        kgraph.control.addElement(Body(), {condOp}, {head});
        kgraph.control.addElement(Else(), {condOp}, {elseTag});

        // change the graph
        auto replaceWithNOP = [](KernelGraph& kgraph, auto tag) {
            auto nopTag = kgraph.control.addElement(NOP());
            replaceWith(kgraph, tag, nopTag, false);
            purgeNodes(kgraph, {tag});
        };
        for(auto const& node : candidates)
        {
            if(node.second == -1)
                replaceWithNOP(kgraph, node.first);
            else
                kgraph.mapper.connect(node.first, node.second, NaryArgument::DEST);
        }

        return kgraph;
    }

    KernelGraph ConstantPropagation::apply(KernelGraph const& original)
    {
        auto kgraph = scalarIsZero(original);

        return kgraph;
    }

    std::string ConstantPropagation::name() const
    {
        return "ConstantPropagation";
    }
}
