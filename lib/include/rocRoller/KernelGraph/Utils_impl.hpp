
#pragma once

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{

    template <typename T>
    std::optional<T> only(std::vector<T> v)
    {
        if(v.size() == 1)
            return v[0];
        return {};
    }

    std::optional<int> only(Generator<int> g)
    {
        auto it = g.begin();

        if(it == g.end())
            return {};

        auto first = *it;

        it = std::next(it);
        if(it == g.end())
            return first;

        return {};
    }

    template <typename T>
    std::unordered_set<int> filterCoordinates(std::vector<int>   candidates,
                                              KernelGraph const& kgraph)
    {
        std::unordered_set<int> rv;
        for(auto candidate : candidates)
            if(kgraph.coordinates.get<T>(candidate))
                rv.insert(candidate);
        return rv;
    }

    template <typename T>
    std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph)
    {
        int lastTag = -1;
        for(auto parent : kgraph.control.depthFirstVisit(candidate, Graph::Direction::Upstream))
        {
            bool containing = lastTag != -1 && kgraph.control.get<ControlGraph::Body>(lastTag);
            lastTag         = parent;

            auto forLoop = kgraph.control.get<T>(parent);
            if(forLoop && containing)
                return parent;
        }
        return {};
    }

}
