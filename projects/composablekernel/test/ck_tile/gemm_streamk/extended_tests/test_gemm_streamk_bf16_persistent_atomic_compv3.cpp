// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf16PersistentAtomicCompV3 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf16PersistentAtomicCompV3

TYPED_TEST_SUITE(TestCkTileStreamKBf16PersistentAtomicCompV3,
                 KernelTypesStreamKBf16PersistentAtomicCompV3);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
