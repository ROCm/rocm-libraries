// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// API-level applicability / compile probe for large 2D grouped convolutions in
// NHWC layout. Shapes span three regimes: (a) sub-INT_MAX element counts,
// (b) tensors above 2 GB whose element count still fits int32, and (c) tensors
// whose element count exceeds INT_MAX (needs the CK large-tensor instances).
// All geometries are 3x3 with SAME padding. Fast: CompileSolution only, no
// allocation or execution.
//
// fwd/bwd/wrw all assert a compilable CK solution, so any shape/direction the
// solver does not cover shows up as a FAILED case.

#include "conv_api_solution_count_large_stride_common.hpp"
#include <vector>

namespace {

using miopen_test_large_stride::Descriptors;
using miopen_test_large_stride::RunCompileBwdData;
using miopen_test_large_stride::RunCompileFwd;
using miopen_test_large_stride::RunCompileWrw;

// A 2D conv problem: input [N,C,H,W], weight [K,C,kH,kW], pad (pH,pW). Lengths
// are in logical [N,C,H,W] order; all descriptors are created NHWC.
struct Shape2D
{
    const char* tag;
    int x[4];
    int w[4];
    int pad[2];
};

std::vector<Shape2D> Shapes()
{
    return {
        // ---- sub-INT_MAX elements, < 2 GB ----
        {"small_hc", {48, 2048, 64, 64}, {2048, 2048, 3, 3}, {1, 1}}, // 4.0e8 elem
        // ---- > 2 GB bytes, element count still fits int32 ----
        {"gt2gb", {96, 1024, 112, 112}, {1024, 1024, 3, 3}, {1, 1}},    // 1.23e9 elem
        {"gt2gb_kexp", {96, 2048, 64, 64}, {4096, 2048, 3, 3}, {1, 1}}, // 1.61e9 elem, K=2C
        // ---- element count > INT_MAX (needs CK large-tensor instances) ----
        {"over_intmax", {160, 1024, 140, 100}, {1024, 1024, 3, 3}, {1, 1}},      // 2.29e9 elem
        {"over_intmax_swap", {160, 1024, 100, 140}, {1024, 1024, 3, 3}, {1, 1}}, // 2.29e9, H/W swap
        {"over_intmax_spatial", {200, 512, 200, 200}, {512, 512, 3, 3}, {1, 1}}, // 4.10e9 elem
        {"over_intmax_kexp", {160, 1024, 120, 120}, {2048, 1024, 3, 3}, {1, 1}}, // 4.72e9, K=2C
    };
}

::testing::AssertionResult SetupShape2D(const Shape2D& s, miopenDataType_t dtype, Descriptors& d)
{
    constexpr int rank = 4;

    if(miopenCreateWithStream(&d.handle, /*stream=*/nullptr) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "miopenCreateWithStream failed";

    if(miopenCreateTensorDescriptor(&d.xDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create xDesc failed";
    if(miopenSetNdTensorDescriptorWithLayout(d.xDesc, dtype, miopenTensorNHWC, s.x, rank) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set xDesc failed";

    if(miopenCreateTensorDescriptor(&d.wDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create wDesc failed";
    if(miopenSetNdTensorDescriptorWithLayout(d.wDesc, dtype, miopenTensorNHWC, s.w, rank) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set wDesc failed";

    if(miopenCreateConvolutionDescriptor(&d.convDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create convDesc failed";
    {
        int pads[2]    = {s.pad[0], s.pad[1]};
        int strides[2] = {1, 1};
        int dils[2]    = {1, 1};
        if(miopenInitConvolutionNdDescriptor(
               d.convDesc, 2, pads, strides, dils, miopenConvolution) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "init convDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.yDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create yDesc failed";
    {
        int yDim[rank] = {0};
        int yNbDims    = 0;
        if(miopenGetConvolutionNdForwardOutputDim(d.convDesc, d.xDesc, d.wDesc, &yNbDims, yDim) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "get yDim failed";
        if(yNbDims != rank)
            return ::testing::AssertionFailure() << "yNbDims != " << rank;
        if(miopenSetNdTensorDescriptorWithLayout(d.yDesc, dtype, miopenTensorNHWC, yDim, rank) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set yDesc failed";
    }
    return ::testing::AssertionSuccess();
}

void RunFwd(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileFwd(s, dtype, SetupShape2D, "ConvHipImplicitGemmGroupFwdXdlops");
}
void RunBwd(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileBwdData(s, dtype, SetupShape2D, "ConvHipImplicitGemmGroupBwdXdlops");
}
void RunWrw(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileWrw(s, dtype, SetupShape2D, "ConvHipImplicitGemmGroupWrwXdlops");
}

class GPU_ConvApi_Large2D_FP16 : public ::testing::TestWithParam<Shape2D>
{
};

} // namespace

TEST_P(GPU_ConvApi_Large2D_FP16, FwdIncludesCK) { RunFwd(GetParam(), miopenHalf); }
TEST_P(GPU_ConvApi_Large2D_FP16, BwdDataIncludesCK) { RunBwd(GetParam(), miopenHalf); }
TEST_P(GPU_ConvApi_Large2D_FP16, WrwIncludesCK) { RunWrw(GetParam(), miopenHalf); }

INSTANTIATE_TEST_SUITE_P(Large2D, GPU_ConvApi_Large2D_FP16, ::testing::ValuesIn(Shapes()));
