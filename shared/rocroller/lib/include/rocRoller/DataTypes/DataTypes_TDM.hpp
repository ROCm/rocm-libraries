// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <fmt/format.h>
#include <iostream>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    struct TDM
    {
        uint32_t parts[12];
    };

    inline bool IsTDMAllZeros(TDM const& tdm)
    {
        auto isZero = [](uint32_t val) { return val == 0; };
        return std::all_of(std::begin(tdm.parts), std::end(tdm.parts), isZero);
    }

    inline std::ostream& operator<<(std::ostream& os, TDM const& buf)
    {
        os << "TDM(";
        rocRoller::streamJoin(os, buf.parts, ", ");
        os << ")" << std::endl;
        return os;
    }

    inline bool operator!(TDM const& tdm)
    {
        return IsTDMAllZeros(tdm);
    }

    inline bool operator==(TDM const& a, TDM const& b)
    {
        return std::equal(std::begin(a.parts), std::end(a.parts), std::begin(b.parts));
    }

} // namespace rocRoller
