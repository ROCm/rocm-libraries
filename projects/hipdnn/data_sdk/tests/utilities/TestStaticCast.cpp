// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hipdnn_data_sdk/utilities/StaticCast.hpp>

using namespace hipdnn_data_sdk::utilities;

namespace
{

template <class T, class S>
void testCastTo(S value)
{
    EXPECT_EQ(staticCast<T>(value), static_cast<T>(value));
}

template <class T, class S>
void testCastToWithFloatIntermediate(S value)
{
    EXPECT_EQ(staticCast<T>(value), static_cast<T>(static_cast<float>(value)));
}

TEST(TestStaticCast, Correctness)
{
    testCastTo<hip_bfloat16>(float());
    testCastToWithFloatIntermediate<hip_bfloat16>(double());
    testCastTo<hip_bfloat16>(half());
    testCastTo<hip_bfloat16>(hip_bfloat16());
    testCastToWithFloatIntermediate<hip_bfloat16>(int());
    testCastToWithFloatIntermediate<hip_bfloat16>(0U);
    testCastToWithFloatIntermediate<hip_bfloat16>(uint64_t{0});
    testCastToWithFloatIntermediate<hip_bfloat16>(int64_t{0});

    testCastTo<half>(float());
    testCastToWithFloatIntermediate<half>(double());
    testCastTo<half>(half());
    testCastTo<half>(hip_bfloat16());
    testCastToWithFloatIntermediate<half>(int());
    testCastToWithFloatIntermediate<half>(0U);
    testCastToWithFloatIntermediate<half>(uint64_t{0});
    testCastToWithFloatIntermediate<half>(int64_t{0});
}

}
