
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/GranularitySelectionLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<ExactSelectionTableEntry, IO>
        {
            using Entry = ExactSelectionTableEntry; //<Key, Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                iot::mapRequired(io, "key", entry.key);
                iot::mapRequired(io, "value", entry.value);
            }

            const static bool flow = true;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<GranularitySelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = GranularitySelectionLibrary<MyProblem, MySolution>;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "GranularitySelectionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }

                std::vector<int>                      mappingIndices;
                std::vector<ExactSelectionTableEntry> mapEntries;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solutions.size());

                    for(auto const& pair : lib.solutions)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "indices", mappingIndices);

                    for(auto it = lib.exactMap.begin(); it != lib.exactMap.end(); ++it)
                    {
                        ExactSelectionTableEntry newEntry;
                        newEntry.key   = it->first;
                        newEntry.value = it->second;
                        mapEntries.push_back(newEntry);
                    }
                    iot::mapRequired(io, "exact", mapEntries);
                }
                else
                {
                    iot::mapRequired(io, "indices", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "GranularitySelectionLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate(
                                    "[GranularitySelectionLibrary] Invalid solution index: ",
                                    index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutions.insert(std::make_pair(index, solution));
                        }
                    }

                    iot::mapRequired(io, "exact", mapEntries);

                    for(ExactSelectionTableEntry entry : mapEntries)
                    {
                        std::vector<size_t> key   = entry.key;
                        int                 value = entry.value;
                        lib.exactMap.insert(std::pair<std::vector<size_t>, int>(key, value));
                    }
                }
            }
        };
    } // namespace Serialization
} // namespace TensileLite
