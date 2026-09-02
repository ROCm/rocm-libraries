// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Large-tensor RowColQuant coverage: exercises the 64-bit global load/store path
// (GemmConfigLargeTensor sets LargeTensors=true) for every A/B layout combination.
// RowColQuant requires a RowMajor C, so the combinations are A in {Row, Col} x B in
// {Row, Col} with C fixed to RowMajor.
//
// Scoped to gfx1250: this is the architecture on which the large-tensor path is
// validated. CMake only registers this target when gfx1250 is a build target (which is
// also when CK_USE_GFX1250 is defined); the guard below keeps the translation unit empty
// as a defensive fallback if it is ever compiled for another target.
#if defined(CK_USE_GFX1250)

// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType,
// CDataType, QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using RowColQuantLargeTensorTypes = ::testing::Types<
    std::tuple<RowMajor,    RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<RowMajor,    ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<ColumnMajor, RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantLargeTensorTypes);

// Validated: builds the kernel with LargeTensors=true and checks numerics against the
// host reference for every layout combination at a modest size. This size is below the
// runtime large-tensor gate (2^31 bytes), so the executed branch is the normal
// small-tensor path -- the purpose here is to prove that enabling the compile-time
// LargeTensors flag (which widens offsets and compiles in the 64-bit branches) does not
// regress numerical correctness of the common path. The true >2GB 64-bit arithmetic is
// exercised (without a host reference) by LargeTensorPathLaunchOnly below; validating it
// numerically is infeasible because a host GEMM reference at that scale is prohibitive.
TYPED_TEST(TestCkTileGemmRowColQuant, LargeTensorPathValidated)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// Launch-only: C is M * N * sizeof(Half) = 32768 * 32896 * 2 bytes ~= 2.006 GiB, strictly
// above the 2 GiB single-buffer limit (2^31) with a one-N-tile margin so the runtime
// large-tensor gate stays engaged for every combo even if its comparison is ever
// tightened. A=ColumnMajor combos are accepted via the global load/store branch;
// A=RowMajor combos are accepted via the M base-shift branch (with a globalized C store).
// N is kept a multiple of N_Tile (128) so no padding is required. A host reference is
// infeasible at this scale, so this only asserts the argument is accepted and the kernel
// completes without a device fault. Requires a device with >~2.2 GiB of free memory.
TYPED_TEST(TestCkTileGemmRowColQuant, LargeTensorPathLaunchOnly)
{
    this->run_test_launch_only(32768, 32896, 512);
}

#endif // CK_USE_GFX1250
