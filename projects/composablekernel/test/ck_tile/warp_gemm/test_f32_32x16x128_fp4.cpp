// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_util.hpp"
#include "gtest/gtest.h"

using WGDispatcherTypesList =
    ::testing::Types<ck_tile::test::warp_gemm::WGDispCase<ck_tile::pk_fp4_t,
                                                          ck_tile::pk_fp4_t,
                                                          float,
                                                          false,
                                                          false,
                                                          false,
                                                          ck_tile::WGAttrNumAccessEnum::Single>>;

template <typename T>
class WGRuntimeTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG_NonScaled)
{
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 32, 16, 128, false, false>();
}

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG_Scale16)
{
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 32, 32, 128, true, true>();
}

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG_Scale32)
{
#if CK_TILE_WORKAROUND_GFX1250_SCALE32_WMMA_BUG
    GTEST_SKIP() << "Scale32 (V_WMMA_SCALE_F32_32X16X128_F4) disabled on gfx1250 "
                    "due to hardware bug (incorrect results even for zero inputs)";
#else
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 32, 32, 128, true, false>();
#endif
}
