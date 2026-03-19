// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp16NonPersistentAtomicCompV3 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp16NonPersistentAtomicCompV3

TYPED_TEST_SUITE(TestCkTileStreamKFp16NonPersistentAtomicCompV3,
                 KernelTypesStreamKFp16NonPersistentAtomicCompV3);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
