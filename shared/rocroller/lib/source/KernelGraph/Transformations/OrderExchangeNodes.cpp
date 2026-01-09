/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/Transforms/OrderExchangeNodes.hpp>

#include <rocRoller/KernelGraph/Transforms/OrderExchangeNodes_detail.hpp>
// #include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes_detail.hpp>
#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace OrderMultiplyNodesDetail
    {
        int ExchangeOrder::getMultiply(int exchange) const
        {
            auto isMultiply = [this](int idx) {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            };

            auto multiplies
                = graph.control.getOutputNodeIndices<ControlGraph::Sequence>(exchange).filter(
                    isMultiply);

            auto iter = multiplies.begin();
            AssertFatal(iter != multiplies.end(), "No Multiply nodes attached to ", exchange);

            auto rv = *iter;

            ++iter;

            for(; iter != multiplies.end(); ++iter)
            {
                auto candidate = *iter;

                if(candidate != rv
                   && graph.control.compareNodes(UpdateCache, rv, candidate)
                          == ControlGraph::NodeOrdering::RightFirst)
                    rv = candidate;
            }

            return rv;
        }

        bool ExchangeOrder::operator()(int a, int b) const
        {
            if(a == b)
                return false;

            auto aMultiply = getMultiply(a);
            auto bMultiply = getMultiply(b);

            if(aMultiply == bMultiply)
                return false;

            return TopologicalCompare(graph)(aMultiply, bMultiply);
        }
    }

    KernelGraph OrderExchangeNodes::apply(KernelGraph const& original)
    {
        auto rv = original;

        {
            auto exchangeWithConnectedMultiply = [&rv](int node) {
                if(!rv.control.get<rocRoller::KernelGraph::ControlGraph::Exchange>(node)
                        .has_value())
                    return false;

                auto isMultiply = [&rv](int idx) -> bool {
                    return rv.control.get<rocRoller::KernelGraph::ControlGraph::Multiply>(idx)
                        .has_value();
                };

                auto hasOutputMultiplies
                    = rv.control
                          .getOutputNodeIndices<rocRoller::KernelGraph::ControlGraph::Sequence>(
                              node)
                          .filter(isMultiply)
                          .take(1)
                          .only()
                          .has_value();

                return hasOutputMultiplies;
            };

            auto groupedExchangeNodes
                = NodeScheduling::getGroupedNodes(rv, exchangeWithConnectedMultiply);
            for(auto& [parent, nodes] : groupedExchangeNodes)
            {
                OrderMultiplyNodesDetail::ExchangeOrder comp{rv};
                NodeScheduling::orderNodes(rv, nodes, comp);

                for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
                {
                    rv.control.chain<ControlGraph::Sequence>(nodes[idx], nodes[idx + 1]);

                    {
                        auto idxMultiply = comp.getMultiply(nodes[idx]);
                        if(rv.control.compareNodes(UseCacheIfAvailable, idxMultiply, nodes[idx + 1])
                           == ControlGraph::NodeOrdering::Undefined)
                        {
                            rv.control.chain<ControlGraph::Sequence>(idxMultiply, nodes[idx + 1]);
                        }
                    }

                    {
                        auto idxMultiply = comp.getMultiply(nodes[idx + 1]);
                        if(rv.control.compareNodes(UseCacheIfAvailable, idxMultiply, nodes[idx])
                           == ControlGraph::NodeOrdering::Undefined)
                        {
                            rv.control.chain<ControlGraph::Sequence>(nodes[idx], idxMultiply);
                        }
                    }
                }
            }
        }

        return rv;
    }
}
