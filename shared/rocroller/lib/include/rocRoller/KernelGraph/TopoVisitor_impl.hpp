// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <variant>

#include "TopoVisitor.hpp"

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller::KernelGraph
{
    template <typename Derived>
    TopoControlGraphVisitor<Derived>::TopoControlGraphVisitor(KernelGraph const& kgraph)
        : m_graph(kgraph)
    {
    }

    template <typename Derived>
    void TopoControlGraphVisitor<Derived>::reset()
    {
        m_visitedNodes.clear();
    }

    template <typename Derived>
    std::unordered_set<int>& TopoControlGraphVisitor<Derived>::nodeDependencies(int node)
    {
        namespace CG = ControlGraph;

        auto iter = m_nodeDependencies.find(node);

        if(iter == m_nodeDependencies.end())
        {
            auto [newIter, wasInserted] = m_nodeDependencies.emplace(
                node,
                m_graph.control.getInputNodeIndices<CG::Sequence>(node).to<std::unordered_set>());
            AssertFatal(wasInserted);
            iter = newIter;
        }

        return iter->second;
    }

    template <typename Derived>
    bool TopoControlGraphVisitor<Derived>::hasWalkedInputs(int node)
    {
        auto& deps = nodeDependencies(node);

        for(auto iter = deps.begin(); iter != deps.end();)
        {
            if(m_visitedNodes.contains(*iter))
            {
                iter = deps.erase(iter);
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    template <typename Derived>
    void TopoControlGraphVisitor<Derived>::walk()
    {
        auto nodes = m_graph.control.roots().to<std::set>();
        walk(nodes);
    }

    template <typename Derived>
    void TopoControlGraphVisitor<Derived>::walk(std::set<int> nodes)
    {
        namespace CG = ControlGraph;
        nodes        = m_graph.control.followEdges<CG::Sequence>(nodes);

        while(!nodes.empty())
        {
            bool walkedAny = false;

            for(auto iter = nodes.begin(); iter != nodes.end();)
            {
                auto node = *iter;
                if(hasWalkedInputs(node))
                {
                    walk(node);
                    iter      = nodes.erase(iter);
                    walkedAny = true;
                }
                else
                {
                    ++iter;
                }
            }

            if(!walkedAny)
            {
                std::ostringstream msg;
                msg << "Graph cannot be completely walked! Node";
                if(nodes.size() != 1)
                    msg << "s";
                msg << " (";
                streamJoin(msg, nodes, ", ");
                msg << ") remain";
                if(nodes.size() == 1)
                    msg << "s";
                msg << ".";

                errorCondition(msg.str());
                break;
            }
        }
    }

    template <typename Derived>
    void TopoControlGraphVisitor<Derived>::errorCondition(std::string const& message)
    {
        Throw<FatalError>(message);
    }

    template <typename Derived>
    void TopoControlGraphVisitor<Derived>::walk(int node)
    {
        namespace CG = ControlGraph;
        call(node);

        walk(m_graph.control.getOutputNodeIndices<CG::Initialize>(node).to<std::set>());
        walk(m_graph.control.getOutputNodeIndices<CG::Body>(node).to<std::set>());
        walk(m_graph.control.getOutputNodeIndices<CG::Else>(node).to<std::set>());
        walk(m_graph.control.getOutputNodeIndices<CG::ForLoopIncrement>(node).to<std::set>());

        m_visitedNodes.insert(node);
    }

    template <typename Derived>
    constexpr Derived* TopoControlGraphVisitor<Derived>::derived()
    {
        return static_cast<Derived*>(this);
    }

    template <typename Derived>
    auto TopoControlGraphVisitor<Derived>::call(int node)
    {
        auto op = m_graph.control.getNode(node);

        return std::visit(*derived(), singleVariant(node), op);
    }

}
