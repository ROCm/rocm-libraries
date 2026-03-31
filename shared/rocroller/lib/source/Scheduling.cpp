// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
