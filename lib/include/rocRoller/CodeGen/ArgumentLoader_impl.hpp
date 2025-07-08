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
