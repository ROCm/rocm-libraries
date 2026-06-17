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

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG)
{
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 16, 16, 128, true, false>();
}
