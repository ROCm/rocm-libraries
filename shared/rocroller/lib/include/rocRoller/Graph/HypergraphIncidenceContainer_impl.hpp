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

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>

namespace rocRoller::Graph
{
    template <CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
    inline void HypergraphIncidenceContainer::addIncidences(int              id,
                                                            T_Inputs const&  inputs,
                                                            T_Outputs const& outputs)
    {
        auto addIncidence = [](int target, auto& connections) {
            int incidenceOrder = 0;
            if(!connections.empty())
            {
                if(std::any_of(
                       connections.begin(), connections.end(), [target](auto const& connection) {
                           return connection.second == target;
                       }))
                {
                    // Don't add duplicate incidence
                    return;
                }
                incidenceOrder = connections.rbegin()->first + 1;
            }
            connections.emplace(incidenceOrder, target);
        };

        auto addIncidenceSrc = [this, &addIncidence](int src, int dst) {
            auto& connections = this->m_incidencesBySrc.at(src);
            addIncidence(dst, connections);
        };

        auto addIncidenceDst = [this, &addIncidence](int src, int dst) {
            auto& connections = this->m_incidencesByDst.at(dst);
            addIncidence(src, connections);
        };

        accountForId(id);

        for(int input : inputs)
        {
            accountForId(input);
            addIncidenceSrc(input, id);
            addIncidenceDst(input, id);
        }
        for(int output : outputs)
        {
            accountForId(output);
            addIncidenceSrc(id, output);
            addIncidenceDst(id, output);
        }
    }
}
