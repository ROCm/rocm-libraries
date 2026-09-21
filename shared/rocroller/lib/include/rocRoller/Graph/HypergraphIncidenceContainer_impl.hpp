// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
