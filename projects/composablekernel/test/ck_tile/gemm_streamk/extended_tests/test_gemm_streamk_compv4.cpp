// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKCompV4 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKCompV4

TYPED_TEST_SUITE(TestCkTileStreamKCompV4, KernelTypesStreamKCompV4);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
