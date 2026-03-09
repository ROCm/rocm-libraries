// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     */
    struct Int8 : public DistinctType<int8_t, Int8>
    {
    };
} // namespace rocRoller

namespace std
{
    inline ostream& operator<<(ostream& stream, const rocRoller::Int8 val)
    {
        return stream << static_cast<int32_t>(static_cast<int8_t>(val));
    }
} // namespace std
