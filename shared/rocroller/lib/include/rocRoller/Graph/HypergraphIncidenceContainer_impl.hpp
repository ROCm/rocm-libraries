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

#pragma once

#include <algorithm>

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>

namespace rocRoller::Graph
{
    template <CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
    inline void HypergraphIncidenceContainer::addIncidentConnections(int              tag,
                                                                     T_Inputs const&  inputs,
                                                                     T_Outputs const& outputs)
    {
        // Adds incident connections to all internal incidence containers.
        auto addConnection = [this](int src, int dst) {
            auto& dsts = this->m_incidenceBySrc[src];
            // Only add connection if it doesn't already exist
            if(std::find(dsts.begin(), dsts.end(), dst) == dsts.end())
                dsts.push_back(dst);

            auto& srcs = this->m_incidenceByDst[dst];
            // Only add connection if it doesn't already exist
            if(std::find(srcs.begin(), srcs.end(), src) == srcs.end())
                srcs.push_back(src);
        };

        for(auto input : inputs)
        {
            addConnection(input, tag);
        }
        for(auto output : outputs)
        {
            addConnection(tag, output);
        }
    }
}
