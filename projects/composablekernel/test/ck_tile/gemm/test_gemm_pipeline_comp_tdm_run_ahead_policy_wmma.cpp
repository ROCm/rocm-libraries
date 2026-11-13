// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMRunAheadPolicyWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMRunAheadPolicyWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMRunAheadPolicyWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMRunAheadPolicyWmma,
                 KernelTypesCompTDMRunAheadPolicyWmma);

#include "test_gemm_pipeline_ut_cases.inc"
#include "test_gemm_pipeline_comp_tdm_run_ahead_cases.inc"

#undef TEST_SUITE_NAME
