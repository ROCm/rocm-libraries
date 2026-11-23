// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_math.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{
    template <typename T>
    void check_fnuz_negation()
    {
        for(unsigned bits = 0; bits < 256; ++bits)
        {
            SCOPED_TRACE(bits);
            T value;
            value.__x      = static_cast<uint8_t>(bits);
            const T result = negate(value);

            // FNUZ has no negative zero: 0x00 is zero and 0x80 is NaN.
            if(bits == 0x00 || bits == 0x80)
                EXPECT_EQ(result.__x, value.__x);
            else
                // Every finite FP8 value is exactly representable in float.
                EXPECT_EQ(static_cast<float>(result), -static_cast<float>(value));
        }
    }

    TEST(Math_smoke, negate_f8_fnuz)
    {
        check_fnuz_negation<hipblaslt_f8_fnuz>();
    }

    TEST(Math_smoke, negate_bf8_fnuz)
    {
        check_fnuz_negation<hipblaslt_bf8_fnuz>();
    }
}
