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

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::Graph
{
    size_t HypergraphIncidenceContainer::size() const
    {
        return std::accumulate(
            m_incidencesBySrc.begin(),
            m_incidencesBySrc.end(),
            0,
            [](size_t sum, auto const& value) { return sum + value.second.size(); });
    }

    void HypergraphIncidenceContainer::deleteIndex(int index)
    {
        m_incidencesBySrc.erase(index);
        for(auto& i : m_incidencesBySrc)
            std::erase_if(i.second, [index](auto const& d) { return d.second == index; });

        m_incidencesByDst.erase(index);
        for(auto& i : m_incidencesByDst)
            std::erase_if(i.second, [index](auto const& s) { return s.second == index; });
    }

    Generator<int> HypergraphIncidenceContainer::getSrcs(int index) const
    {
        if(!m_incidencesByDst.contains(index))
            co_return;

        for(auto const& connection : m_incidencesByDst.at(index))
        {
            co_yield connection.second;
        }
    }

    Generator<int> HypergraphIncidenceContainer::getDsts(int index) const
    {
        if(!m_incidencesBySrc.contains(index))
            co_return;

        for(auto const& connection : m_incidencesBySrc.at(index))
        {
            co_yield connection.second;
        }
    }

    size_t HypergraphIncidenceContainer::getSrcCount(int index) const
    {
        if(m_incidencesByDst.contains(index))
            return m_incidencesByDst.at(index).size();
        return 0;
    }

    size_t HypergraphIncidenceContainer::getDstCount(int index) const
    {
        if(m_incidencesBySrc.contains(index))
            return m_incidencesBySrc.at(index).size();
        return 0;
    }

    Generator<HypergraphIncidence> HypergraphIncidenceContainer::getAllIncidences() const
    {
        for(auto const& s : m_incidencesBySrc)
        {
            for(auto const& d : s.second)
            {
                HypergraphIncidence incidence{.src = s.first, .dst = d.second};
                co_yield incidence;
            }
        }
    }

    void HypergraphIncidenceContainer::accountForIndex(int index)
    {
        if(!m_incidencesBySrc.contains(index))
            m_incidencesBySrc.emplace(index, std::map<int, int>{});
        if(!m_incidencesByDst.contains(index))
            m_incidencesByDst.emplace(index, std::map<int, int>{});
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
