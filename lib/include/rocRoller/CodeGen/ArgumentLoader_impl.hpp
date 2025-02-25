#pragma once

#include <rocRoller/CodeGen/ArgumentLoader.hpp>

namespace rocRoller
{

    template <typename Iter, typename End>
    int PickInstructionWidthBytes(int offset, int endOffset, Iter beginReg, End endReg)
    {
        if(offset >= endOffset)
        {
            return 0;
        }

        AssertFatal(beginReg < endReg, "Inconsistent register bounds");
        AssertFatal(offset % 4 == 0, "Must be 4-byte aligned.");

        static const std::vector<int> widths{16, 8, 4, 2, 1};

        for(auto width : widths)
        {
            auto widthBytes = width * 4;

            if((offset % widthBytes == 0) && (offset + widthBytes) <= endOffset
               && (*beginReg % width == 0) && IsContiguousRange(beginReg, beginReg + width))
            {
                return widthBytes;
            }
        }

        Throw<FatalError>("Should be impossible to get here!");
    }
}
