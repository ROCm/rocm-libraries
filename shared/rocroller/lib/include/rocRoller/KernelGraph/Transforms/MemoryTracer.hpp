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
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace MemoryTracer
        {
            struct Summary
            {
                static constexpr bool echoBanks = true;

                struct Banks
                {
                    uint   bankIndex;
                    size_t workitemsAccessed;
                    bool   imbalanced;
                };
                struct Access
                {
                    int                           tag;
                    uint                          instruction;
                    int                           ldsTag;
                    std::vector<Banks>            accessedBanks;
                    std::vector<std::vector<int>> banksToWorkitems;
                };

                std::vector<Access>     accesses;
                std::unordered_set<int> imbalancedTags;

                std::string toString() const;
            };

            std::ostream& operator<<(std::ostream& stream, Summary const& summary);
        }
        /**
	 * @brief Memory tracer for the rocRoller kernel graph.
	 *
	 * This is a work-in-progress implementation of a memory
	 * access analysis tool that simulates memory accesses.
	 */

        MemoryTracer::Summary memoryTrace(KernelGraph const&      original,
                                          KernelInvocation const& invocation,
                                          KernelArguments const&  arguments);
    }
}
