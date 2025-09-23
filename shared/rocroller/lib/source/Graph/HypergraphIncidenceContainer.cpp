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

#include <iostream>
#include <numeric>
#include <ranges>

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::Graph
{
    HypergraphIncidenceContainer::HypergraphIncidenceContainer(
        std::vector<HypergraphIncidence> const& incidences)
    {
        for(auto const& incidence : incidences)
        {
            if(!m_incidencesBySrc.contains(incidence.src))
            {
                int                              src = incidence.src;
                std::vector<HypergraphIncidence> filtered;
                std::ranges::copy_if(incidences,
                                     std::back_inserter(filtered),
                                     [src](auto const& i) { return i.src == src; });
                std::ranges::sort(filtered, {}, &HypergraphIncidence::order);

                m_incidencesBySrc[src] = std::vector<int>{};
                m_incidencesBySrc[src].reserve(filtered.size());
                std::transform(filtered.begin(),
                               filtered.end(),
                               std::back_inserter(m_incidencesBySrc[src]),
                               [](auto const& i) { return i.dst; });
            }

            if(!m_incidencesByDst.contains(incidence.dst))
            {
                int                              dst = incidence.dst;
                std::vector<HypergraphIncidence> filtered;
                std::ranges::copy_if(incidences,
                                     std::back_inserter(filtered),
                                     [dst](auto const& i) { return i.dst == dst; });
                std::ranges::sort(filtered, {}, &HypergraphIncidence::order);

                m_incidencesByDst[dst] = std::vector<int>{};
                m_incidencesByDst[dst].reserve(filtered.size());
                std::transform(filtered.begin(),
                               filtered.end(),
                               std::back_inserter(m_incidencesByDst[dst]),
                               [](auto const& i) { return i.src; });
            }
        }
    }

    size_t HypergraphIncidenceContainer::size() const
    {
        return std::accumulate(
            m_incidencesBySrc.begin(),
            m_incidencesBySrc.end(),
            0,
            [](size_t sum, auto const& value) { return sum + value.second.size(); });
    }

    void HypergraphIncidenceContainer::deleteTag(int tag)
    {
        m_incidencesBySrc.erase(tag);
        m_incidencesByDst.erase(tag);

        for(auto& connections : m_incidencesBySrc)
            std::erase(connections.second, tag);

        for(auto& connections : m_incidencesByDst)
            std::erase(connections.second, tag);
    }

    std::vector<int> HypergraphIncidenceContainer::getSrcs(int tag) const
    {
        auto it = m_incidencesByDst.find(tag);
        if(it != m_incidencesByDst.end())
            return it->second;
        return {};
    }

    std::vector<int> HypergraphIncidenceContainer::getDsts(int tag) const
    {
        auto it = m_incidencesBySrc.find(tag);
        if(it != m_incidencesBySrc.end())
            return it->second;
        return {};
    }

    size_t HypergraphIncidenceContainer::getSrcCount(int tag) const
    {
        auto it = m_incidencesByDst.find(tag);
        if(it != m_incidencesByDst.end())
            return it->second.size();
        return 0;
    }

    size_t HypergraphIncidenceContainer::getDstCount(int tag) const
    {
        auto it = m_incidencesBySrc.find(tag);
        if(it != m_incidencesBySrc.end())
            return it->second.size();
        return 0;
    }

    std::vector<HypergraphIncidence> HypergraphIncidenceContainer::getAllIncidences() const
    {
        std::vector<HypergraphIncidence> rv;
        rv.reserve(this->size());
        for(auto const& connection : m_incidencesBySrc)
        {
            auto const& dsts = connection.second;
            for(auto it = dsts.begin(); it != dsts.end(); it++)
            {
                rv.push_back(HypergraphIncidence{
                    .src   = connection.first,
                    .dst   = *it,
                    .order = static_cast<int>(std::distance(dsts.begin(), it))});
            }
        }

        // Incidence ordering may be higher when assesed from other direction
        for(auto& incidence : rv)
        {
            auto const& srcs = m_incidencesByDst.at(incidence.dst);
            auto        it   = std::find(srcs.begin(), srcs.end(), incidence.src);
            AssertFatal(it != srcs.end(), "Mismatched internal incendence storage");
            incidence.order
                = std::max(incidence.order, static_cast<int>(std::distance(srcs.begin(), it)));
        }
        return rv;
    }

    std::string HypergraphIncidenceContainer::toDOTSection(std::string const& prefix) const
    {
        std::ostringstream s;
        for(auto const& incidence : getAllIncidences())
        {
            s << '"' << prefix << incidence.src << "\" -> \"" << prefix << incidence.dst << '"'
              << std::endl;
        }
        return s.str();
    }
}
