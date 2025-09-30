/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

namespace rocRoller
{
    void combineCoexec(DisallowedCycles& dst, DisallowedCycles const& src, int offset)
    {
        auto bakSrc = toString(src);
        auto bakDst = toString(dst);
        for(auto srcIter = src.begin(); srcIter != src.end(); ++srcIter)
        {
            auto cycle = srcIter->first + offset;

            auto dstIter = dst.lower_bound(cycle);

            if(dstIter == dst.end() || dstIter->first > cycle)
            {
                dst.emplace_hint(dstIter, cycle, srcIter->second);
            }
            else
            {
                AssertFatal(dstIter->first == cycle,
                            ShowValue(bakSrc),
                            ShowValue(bakDst),
                            ShowValue(offset));
                dstIter->second |= srcIter->second;
            }
        }
    }

    std::string toString(DisallowedCycles const& vals)
    {
        std::string rv = "{";

        bool first = true;

        for(auto const& [cycle, categories] : vals)
        {
            if(!first)
                rv += ", ";

            rv += fmt::format("[+{}: {}]", cycle, shortString(categories));

            first = false;
        }

        return rv + "}";
    }

}
