#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ConstantPropagation.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <queue>

namespace rocRoller::KernelGraph
{

    /**
     * @brief Find the DataFlowTag for VGPR that multiplies LoadTiled
     */
    int findVGPRTag(KernelGraph const& kgraph)
    {
        std::vector<int> loadVGPRdest;
        std::vector<int> loadTiledDest;
        std::vector<int> candidates;

        auto load = kgraph.control.getNodes<LoadVGPR>().to<std::vector>();

        // The condition prevents performing zero propagation for the command that only has one scalar multiplies a user provided tensor:
        // 1. vector add/multiply (i.e., scalar + UserTensor or scalar * UserTensor) where there is only one scalar, and beta is equal to zero
        // 2. GEMMleakyRELU that computes scalar * UserTensor and beta is zero
        // It can be removed when we add conditonal node for both conditions (i.e., scalar==0 and scalar!=0)
        if(load.size() != 2)
            return -1;

        for(auto tag : load)
        {
            auto dest = kgraph.mapper.get<VGPR>(tag);
            loadVGPRdest.push_back(dest);
        }

        for(auto tag : kgraph.control.getNodes<LoadTiled>().to<std::vector>())
        {
            auto dest = kgraph.mapper.get<MacroTile>(tag);
            loadTiledDest.push_back(dest);
        }

        for(auto assign : kgraph.control.getNodes<Assign>().to<std::vector>())
        {
            auto node = kgraph.control.get<Assign>(assign);
            if(!std::holds_alternative<Expression::Multiply>(*node->expression))
                continue;
            auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, assign);
            auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, assign);
            for(const auto& tag1 : loadVGPRdest)
            {
                for(const auto& tag2 : loadTiledDest)
                {
                    if(((tag1 == mulLHS) && (tag2 == mulRHS))
                       || ((tag1 == mulRHS) && (tag2 == mulLHS)))
                    {
                        candidates.push_back(tag1);
                    }
                }
            }
        }

        if(candidates.size() != 1)
            return -1;
        return candidates[0];
    }

    /**
     * @brief Find the operations (i.e., Assign, Multiply, LoadTiled) that the downstream destination matches the DataFlowTag
     */
    std::vector<int> findDest(KernelGraph const& kgraph, int DataFlowTag)
    {
        std::vector<int> nodes;
        auto             dfs
            = kgraph.control.depthFirstVisit(1, Graph::Direction::Downstream).to<std::vector>();
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
     * @brief Propagate zero DataFlowTag
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
    KernelGraph zeroPropagation(KernelGraph const& original)
    {
        auto kgraph = original;
        auto DFtag  = findVGPRTag(kgraph);
        if(DFtag == -1)
            return kgraph;
        std::queue<int> zeros;
        zeros.push(DFtag);

        // replace the tag with NOP operation
        auto replaceWithNOP = [](KernelGraph& kgraph, auto tag) {
            auto nopTag = kgraph.control.addElement(NOP());
            replaceWith(kgraph, tag, nopTag, false);
            purgeNodes(kgraph, {tag});
        };

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
                        auto nodes  = findDest(kgraph, target);
                        replaceWithNOP(kgraph, tag);
                        zeros.push(mulDest);
                        for(auto const& nodeTag : nodes)
                            replaceWithNOP(kgraph, nodeTag);
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
                        auto nodes  = findDest(kgraph, target);
                        replaceWithNOP(kgraph, tag);
                        for(auto const& nodeTag : nodes)
                            kgraph.mapper.connect(nodeTag, addDest, NaryArgument::DEST);
                    }
                }
            }
        }
        return kgraph;
    }

    KernelGraph ConstantPropagation::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::ConstantPropagation, only works for beta==0 now.");

        auto kgraph = zeroPropagation(original);

        return kgraph;
    }

    std::string ConstantPropagation::name() const
    {
        return "ConstantPropagation";
    }
}
