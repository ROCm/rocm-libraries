// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
