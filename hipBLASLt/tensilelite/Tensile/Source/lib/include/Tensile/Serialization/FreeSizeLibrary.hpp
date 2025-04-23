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
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <Tensile/FreeSizeLibrary.hpp>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename Value, typename IO>
        struct MappingTraits<FreeSizeEntry<Value>, IO>
        {
            using Entry = FreeSizeEntry<Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                int32_t index = -1;
                iot::mapRequired(io, "index", index);

                using SSLibrary
                    = SingleSolutionLibrary<ContractionProblemGemm, ContractionSolution>;

                auto ctx = static_cast<LibraryIOContext<ContractionSolution>*>(iot::getContext(io));
                if(ctx == nullptr || ctx->solutions == nullptr)
                {
                    iot::setError(io,
                                  "SingleSolutionLibrary requires that context be set to "
                                  "a SolutionMap.");
                }

                auto iter = ctx->solutions->find(index);
                if(iter == ctx->solutions->end())
                {
                    std::ostringstream msg;
                    msg << "[FreeSizeLibrary] Invalid solution index: " << index;
                    iot::setError(io, msg.str());
                }
                else
                {
                    std::shared_ptr<ContractionSolution> solution = iter->second;
                    entry.value = std::make_shared<SSLibrary>(solution);
                }
            }

            const static bool flow = true;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemFreeSizeLibrary<MyProblem, MySolution>, IO>
        {
            using Library = ProblemFreeSizeLibrary<MyProblem, MySolution>;
            using Element = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                std::shared_ptr<typename Library::Table> table;

                if(iot::outputting(io))
                {
                    table = std::dynamic_pointer_cast<typename Library::Table>(lib.table);
                    if(!table)
                    {
                        std::ostringstream msg;
                        msg << "[FreeSizeLibrary] Empty table";
                        iot::setError(io, msg.str());
                    }
                }
                else
                {
                    table     = std::make_shared<typename Library::Table>();
                    lib.table = table;
                }

                iot::mapRequired(io, "table", *table);
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite
