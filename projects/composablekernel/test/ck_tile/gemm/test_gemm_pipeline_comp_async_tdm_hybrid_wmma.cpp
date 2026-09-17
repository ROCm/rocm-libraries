// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncTDMHybridWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompAsyncTDMHybridWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncTDMHybridWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsyncTDMHybridWmma, KernelTypesCompAsyncTDMHybridWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
