/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include <iostream>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace rocRoller
{
    struct Buffer
    {
        uint32_t desc0 = 0;
        uint32_t desc1 = 0;
        uint32_t desc2 = 0;
        uint32_t desc3 = 0;
    };

    inline std::ostream& operator<<(std::ostream& os, const Buffer& buf)
    {
        os << "Buffer(desc0=0x" << std::hex << buf.desc0 << ", desc1=0x" << buf.desc1
           << ", desc2=0x" << buf.desc2 << ", desc3=0x" << buf.desc3 << ")";
        return os;
    }

    inline bool operator!(Buffer const& a)
    {
        return a.desc0 == 0 && a.desc1 == 0 && a.desc2 == 0 && a.desc3 == 0;
    }

    inline bool operator==(Buffer const& a, Buffer const& b)
    {
        return a.desc0 == b.desc0 && a.desc1 == b.desc1 && a.desc2 == b.desc2 && a.desc3 == b.desc3;
    }

} // namespace rocRoller
