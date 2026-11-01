// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 bf16 gtest, 4096 x 4096 x 4096 (REGRESSION_TEST: too slow for the smoke set under the
// gfx1250 emulator). Instances (V3Bf16LargeConfigs), inputs and checks:
// test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMV3, V3Bf16LargeConfigs);

TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Int4096) { this->Run(4096, 4096, 4096, Init::Int); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Random4096)
{
    this->Run(4096, 4096, 4096, Init::Random);
}
// Ragged in M and K, with odd A/B strides (unpadded kernels clip A/B in TDM; a ragged N tile
// needs a padded instance, see the ragged smoke test).
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, IntRagged4111x4096x4105)
{
    this->Run(4111, 4096, 4105, Init::Int);
}

} // namespace
