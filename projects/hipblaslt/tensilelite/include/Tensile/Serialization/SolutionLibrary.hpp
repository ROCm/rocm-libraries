/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <limits>

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MapLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/Serialization/ExactLogicLibrary.hpp>
#include <Tensile/Serialization/GranularitySelectionLibrary.hpp>
#include <Tensile/Serialization/MLPClassificationLibrary.hpp>
#include <Tensile/Serialization/MapLibrary.hpp>
#include <Tensile/Serialization/MatchingLibrary.hpp>
#include <Tensile/Serialization/PlaceholderLibrary.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>, IO>
            : public BaseClassMappingTraits<SolutionLibrary<ContractionProblemGemm>, IO, false>
        {
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>,
                  SolutionLibrary<MyProblem, MySolution>,
                  IO>
        {
            using Self = SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>;
            using Base
                = DefaultSubclassMappingTraits<Self, SolutionLibrary<MyProblem, MySolution>, IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static typename Base::SubclassMap GetSubclasses()
            {
                return typename Base::SubclassMap(
                    {Base::template Pair<SingleSolutionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<HardwareSelectionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemSelectionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemMapLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemFreeSizeLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemPredictionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemMatchingLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<GranularitySelectionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<PlaceholderLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<MLPClassificationLibrary<MyProblem, MySolution>>()});
            }
        };

        template <typename MyProblem, typename MySolution, typename IO>
        using dsmt = SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>;

        template <typename MyProblem, typename MySolution, typename IO>
        const typename dsmt<MyProblem, MySolution, IO>::SubclassMap
            SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>::subclasses
            = dsmt<MyProblem, MySolution, IO>::GetSubclasses();

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<SingleSolutionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = SingleSolutionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr || (ctx->solutions == nullptr && ctx->blobCache == nullptr))
                {
                    iot::setError(io,
                                  "SingleSolutionLibrary requires that context be set to "
                                  "a SolutionMap.");
                }

                int index;

                if(iot::outputting(io))
                    index = lib.solution->index;

                iot::mapRequired(io, "index", index);

                if(!iot::outputting(io))
                {
                    if(ctx->blobCache)
                    {
                        // Checked here so a tree naming a solution the index
                        // table does not carry fails the load, as it would on
                        // the eager path below.
                        if(!ctx->blobCache->contains(index))
                        {
                            iot::setError(
                                io,
                                concatenate("[SolutionLibrary] Invalid solution index: ", index));
                        }
                        else
                        {
                            lib = Library(index, ctx->blobCache);
                        }
                    }
                    else
                    {
                        auto iter = ctx->solutions->find(index);
                        if(iter == ctx->solutions->end())
                        {
                            std::ostringstream msg;
                            msg << "[SolutionLibrary] Invalid solution index: " << index;
                            iot::setError(io, msg.str());
                        }
                        else
                        {
                            lib.solution = iter->second;
                        }
                    }
                }
            }

            const static bool flow = true;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<MasterSolutionLibrary<MyProblem, MySolution>, IO, EmptyContext>
        {
            using Library = MasterSolutionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;
            using Cache   = SolutionBlobCache<MySolution>;

            /// Highest on-disk layout this build can read. Must match
            /// INDEXED_FORMAT_VERSION in Tensile/LibraryIO.py.
            ///
            /// constexpr, not const: it is passed by reference into
            /// concatenate() below, which is an ODR-use and would otherwise
            /// need an out-of-class definition to link.
            static constexpr int IndexedFormatVersion = 2;

            /// Reads `solutions_index` + `solutions_blob` and installs a blob
            /// cache on `lib`. Returns false (after setError) on a malformed
            /// table, so a bad file fails at load exactly as the eager path
            /// does rather than surfacing later as a failed query.
            static bool readIndexedSolutions(IO& io, Library& lib)
            {
                std::vector<int64_t> table;
                iot::mapRequired(io, "solutions_index", table);

                if(table.size() % 3 != 0)
                {
                    iot::setError(io,
                                  concatenate("solutions_index length ",
                                              table.size(),
                                              " is not a multiple of 3"));
                    return false;
                }

                const uint8_t* blobPtr  = nullptr;
                size_t         blobSize = 0;
                if(!iot::mapRawBytes(io, "solutions_blob", blobPtr, blobSize))
                {
                    iot::setError(io,
                                  "format_version 2 requires a binary solutions_blob "
                                  "(indexed libraries are msgpack-only)");
                    return false;
                }

                std::unordered_map<int, typename Cache::Slice> slices;
                slices.reserve(table.size() / 3);

                for(size_t i = 0; i < table.size(); i += 3)
                {
                    const int64_t index  = table[i];
                    const int64_t offset = table[i + 1];
                    const int64_t length = table[i + 2];

                    if(index < 0 || offset < 0 || length < 0)
                    {
                        iot::setError(
                            io, concatenate("solutions_index has a negative field at entry ", i / 3));
                        return false;
                    }
                    // Solution indices are `int` everywhere else -- in the slice
                    // table, in the tree nodes that name them, and in the public
                    // lookup API -- so anything wider would be silently truncated
                    // into a key that aliases a different solution.
                    if(index > static_cast<int64_t>(std::numeric_limits<int>::max()))
                    {
                        iot::setError(io,
                                      concatenate("solutions_index entry ",
                                                  i / 3,
                                                  " has index ",
                                                  index,
                                                  ", which does not fit in an int"));
                        return false;
                    }
                    // Compared unsigned and by subtraction, never as a sum:
                    // offset + length overflows int64 for large values and wraps
                    // negative, which passes a sum comparison and leaves a slice
                    // pointing outside the blob. Both are known non-negative
                    // here, and the first clause makes the subtraction safe.
                    const uint64_t blobBytes = static_cast<uint64_t>(blobSize);
                    if(static_cast<uint64_t>(offset) > blobBytes
                       || static_cast<uint64_t>(length) > blobBytes - static_cast<uint64_t>(offset))
                    {
                        iot::setError(io,
                                      concatenate("solution ",
                                                  index,
                                                  " spans [",
                                                  offset,
                                                  ", ",
                                                  offset + length,
                                                  ") past the ",
                                                  blobSize,
                                                  "-byte solutions_blob"));
                        return false;
                    }
                    if(!slices
                            .emplace(static_cast<int>(index),
                                     typename Cache::Slice(static_cast<size_t>(offset),
                                                           static_cast<size_t>(length)))
                            .second)
                    {
                        iot::setError(io,
                                      concatenate("solutions_index lists index ", index, " twice"));
                        return false;
                    }
                }

                lib.blobCache
                    = std::make_shared<Cache>(std::vector<uint8_t>(blobPtr, blobPtr + blobSize),
                                              std::move(slices),
                                              iot::template solutionDeserializer<MySolution>(io));
                return true;
            }

            static void mapping(IO& io, Library& lib)
            {
                // Read above the format branch so both layouts pick it up, and
                // so an indexed file does not leave the key unconsumed for
                // checkUsedKeys() to flag under TENSILE_DB=0x1000.
                iot::mapOptional(io, "version", lib.version);

                // Absent (legacy) or 1 means the eager layout below. 2 is the
                // indexed layout. Anything higher is a format this build does
                // not know: fail rather than misread it as 2.
                int formatVersion = 0;
                iot::mapOptional(io, "format_version", formatVersion);

                if(!iot::outputting(io) && formatVersion > IndexedFormatVersion)
                {
                    iot::setError(io,
                                  concatenate("Unsupported library format_version: ",
                                              formatVersion,
                                              " (this build understands up to ",
                                              IndexedFormatVersion,
                                              ")"));
                    return;
                }

                const bool indexed = !iot::outputting(io) && formatVersion == IndexedFormatVersion;

                std::vector<std::shared_ptr<MySolution>> solutions;

                if(iot::outputting(io))
                {
                    solutions.reserve(lib.solutions.size());
                    for(auto const& pair : lib.solutions)
                        solutions.push_back(pair.second);
                }

                if(indexed)
                {
                    if(!readIndexedSolutions(io, lib))
                        return;
                }
                else
                {
                    iot::mapRequired(io, "solutions", solutions);
                }

                if(!iot::outputting(io))
                {
                    for(auto const& s : solutions)
                        lib.solutions[s->index] = s;

                    auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                    ctx->solutions            = &lib.solutions;
                    ctx->solutionsGuard       = &lib.solutionsGuard;
                    ctx->loadedFiles          = &lib.loadedFiles;
                    ctx->indexLoadedLibraries = (void*)&lib.indexLoadedLibraries;
                    // Published before the tree is read: leaf nodes capture the
                    // cache as they are constructed.
                    ctx->blobCache       = lib.blobCache;
                    ctx->solutionSources = &lib.solutionSources;
                }

                std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> innerLibrary;

                if(iot::outputting(io))
                {
                    auto cache = std::dynamic_pointer_cast<CachingLibrary<MyProblem, MySolution>>(
                        lib.library);
                    if(cache)
                    {
                        innerLibrary = cache->library();
                    }
                    else
                    {
                        innerLibrary = lib.library;
                    }
                }

                iot::mapRequired(io, "library", innerLibrary);

                if(!iot::outputting(io))
                {
                    auto cache
                        = std::make_shared<CachingLibrary<MyProblem, MySolution>>(innerLibrary);

                    lib.library = cache;
                }
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite

