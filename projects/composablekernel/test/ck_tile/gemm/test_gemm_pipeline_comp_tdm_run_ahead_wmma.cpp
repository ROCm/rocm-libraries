// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMRunAheadWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMRunAheadWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMRunAheadWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMRunAheadWmma, KernelTypesCompTDMRunAheadWmma);

#include "test_gemm_pipeline_ut_cases.inc"
#include "test_gemm_pipeline_comp_tdm_run_ahead_cases.inc"

#undef TEST_SUITE_NAME
