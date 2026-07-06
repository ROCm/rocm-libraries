// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <ranges>

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::Graph
{
    size_t HypergraphIncidenceContainer::size() const
    {
        return std::accumulate(
            m_incidenceBySrc.begin(), m_incidenceBySrc.end(), 0, [](size_t sum, auto const& value) {
                return sum + value.second.size();
            });
    }

    void HypergraphIncidenceContainer::deleteTag(int tag)
    {
        auto srcs = getSrcs(tag);
        auto dsts = getDsts(tag);

        m_incidenceBySrc.erase(tag);
        m_incidenceByDst.erase(tag);

        for(auto src : srcs)
        {
            auto const erased = std::erase(m_incidenceBySrc.at(src), tag);
            AssertFatal(erased > 0);
        }

        for(auto dst : dsts)
        {
            auto const erased = std::erase(m_incidenceByDst.at(dst), tag);
            AssertFatal(erased > 0);
        }
    }

    std::vector<int> HypergraphIncidenceContainer::getSrcs(int tag) const
    {
        auto it = m_incidenceByDst.find(tag);
        if(it != m_incidenceByDst.end())
            return it->second;
        return {};
    }

    std::vector<int> HypergraphIncidenceContainer::getDsts(int tag) const
    {
        auto it = m_incidenceBySrc.find(tag);
        if(it != m_incidenceBySrc.end())
            return it->second;
        return {};
    }

    size_t HypergraphIncidenceContainer::getSrcCount(int tag) const
    {
        auto it = m_incidenceByDst.find(tag);
        if(it != m_incidenceByDst.end())
            return it->second.size();
        return 0;
    }

    size_t HypergraphIncidenceContainer::getDstCount(int tag) const
    {
        auto it = m_incidenceBySrc.find(tag);
        if(it != m_incidenceBySrc.end())
            return it->second.size();
        return 0;
    }

    std::string HypergraphIncidenceContainer::toDOTSection(std::string const& prefix) const
    {
        std::ostringstream s;

        for(auto const& connections : m_incidenceBySrc)
        {
            int src = connections.first;
            for(int dst : connections.second)
            {
                s << '"' << prefix << src << "\" -> \"" << prefix << dst << '"' << std::endl;
            }
        }

        return s.str();
    }
}
