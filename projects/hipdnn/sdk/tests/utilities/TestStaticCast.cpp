// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hipdnn_sdk/utilities/StaticCast.hpp>

using namespace hipdnn_sdk::utilities;

namespace
{

[[maybe_unused]] void testCompiles()
{
    std::ignore = staticCast<hip_bfloat16>(float());
    std::ignore = staticCast<hip_bfloat16>(double());
    std::ignore = staticCast<hip_bfloat16>(half());
    std::ignore = staticCast<hip_bfloat16>(hip_bfloat16());
    std::ignore = staticCast<hip_bfloat16>(int());
    std::ignore = staticCast<hip_bfloat16>(0U);
    std::ignore = staticCast<hip_bfloat16>(0UL);
    std::ignore = staticCast<hip_bfloat16>(0L);

    std::ignore = staticCast<half>(float());
    std::ignore = staticCast<half>(double());
    std::ignore = staticCast<half>(half());
    std::ignore = staticCast<half>(hip_bfloat16());
    std::ignore = staticCast<half>(int());
    std::ignore = staticCast<half>(0U);
    std::ignore = staticCast<half>(0UL);
    std::ignore = staticCast<half>(0L);
}

}
