// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared scaffolding for the 2D and 3D large-stride API applicability sweeps
// (conv_api_solution_count_{2d,3d}_large_stride.cpp).
//
// The two sweeps differ only in shape rank and per-dimension data; the
// descriptor lifecycle, gtest plumbing, and the GTEST_SKIP pattern below are
// identical, so they live here.

#pragma once

#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest_common.hpp"

namespace miopen_test_large_stride {

struct Descriptors
{
    miopenHandle_t handle                  = nullptr;
    miopenTensorDescriptor_t xDesc         = nullptr;
    miopenTensorDescriptor_t wDesc         = nullptr;
    miopenTensorDescriptor_t yDesc         = nullptr;
    miopenConvolutionDescriptor_t convDesc = nullptr;

    ~Descriptors()
    {
        if(yDesc != nullptr)
            miopenDestroyTensorDescriptor(yDesc);
        if(convDesc != nullptr)
            miopenDestroyConvolutionDescriptor(convDesc);
        if(wDesc != nullptr)
            miopenDestroyTensorDescriptor(wDesc);
        if(xDesc != nullptr)
            miopenDestroyTensorDescriptor(xDesc);
        if(handle != nullptr)
            miopenDestroy(handle);
    }
};

inline uint64_t SolverIdFromName(const char* name) { return miopen::solver::Id(name).Value(); }

// DescriptorNeedsLargeTensorPath -- true if this tensor descriptor has any
// length or packed stride exceeding INT_MAX, i.e. it needs a CK "large-tensor"
// (int64-indexed) instance. Mirrors TensorDescriptor::AllDimsFitIntoInt
// (src/tensor.cpp::CheckDimsFitIntoInt). The C API returns strides as int,
// which would truncate the very >INT_MAX values we are looking for, so we
// recompute the packed row-major strides in 64-bit from the faithfully-returned
// lengths (matching CalculateStrides for the default NCHW/NCDHW layout these
// descriptors use: stride[i] = product(lengths[i+1 .. rank-1])).
inline bool DescriptorNeedsLargeTensorPath(miopenTensorDescriptor_t desc)
{
    int rank = 0;
    if(miopenGetTensorDescriptorSize(desc, &rank) != miopenStatusSuccess || rank <= 0)
        return false; // Can't determine; do not skip.

    std::vector<int> lens(rank);
    std::vector<int> strides_unused(rank);
    miopenDataType_t dtype;
    if(miopenGetTensorDescriptor(desc, &dtype, lens.data(), strides_unused.data()) !=
       miopenStatusSuccess)
        return false;

    constexpr std::int64_t int_max = std::numeric_limits<int>::max();

    // Walk from the innermost dim outward; `stride` holds stride[i] on entry.
    std::int64_t stride = 1;
    for(int i = rank - 1; i >= 0; --i)
    {
        if(static_cast<std::int64_t>(lens[i]) > int_max || stride > int_max)
            return true;
        stride *= static_cast<std::int64_t>(lens[i]);
    }
    return false;
}

// True if the problem exceeds int32 indexing on any of its x/w/y tensors --
// matches the solver's RequiresLargeTensorCKInstance gate, which requires all
// three to fit (conv::ProblemDescription::AllTensorsDimsFitIntoInt).
inline bool ProblemNeedsLargeTensorPath(const Descriptors& d)
{
    return DescriptorNeedsLargeTensorPath(d.xDesc) ||
           DescriptorNeedsLargeTensorPath(d.wDesc) || DescriptorNeedsLargeTensorPath(d.yDesc);
}

// SetupDescriptorsImpl -- shared rank-templated descriptor builder. The
// reproducer-family tests use uniform pad=1, stride=1, dilation=1 in all
// spatial dims, so those are hard-coded here.
template <int Ndim>
inline ::testing::AssertionResult
SetupDescriptorsImpl(const int* x_dims, const int* w_dims, miopenDataType_t dtype, Descriptors& d)
{
    static_assert(Ndim == 2 || Ndim == 3, "Only 2D/3D supported");
    constexpr int rank = Ndim + 2;

    if(miopenCreateWithStream(&d.handle, /*stream=*/nullptr) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "miopenCreateWithStream failed";

    if(miopenCreateTensorDescriptor(&d.xDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create xDesc failed";
    if(miopenSetTensorDescriptor(d.xDesc, dtype, rank, const_cast<int*>(x_dims), nullptr) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set xDesc failed";

    if(miopenCreateTensorDescriptor(&d.wDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create wDesc failed";
    if(miopenSetTensorDescriptor(d.wDesc, dtype, rank, const_cast<int*>(w_dims), nullptr) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set wDesc failed";

    if(miopenCreateConvolutionDescriptor(&d.convDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create convDesc failed";
    {
        int pads[Ndim];
        int strides[Ndim];
        int dils[Ndim];
        for(int i = 0; i < Ndim; ++i)
        {
            pads[i]    = 1;
            strides[i] = 1;
            dils[i]    = 1;
        }
        if(miopenInitConvolutionNdDescriptor(
               d.convDesc, Ndim, pads, strides, dils, miopenConvolution) != miopenStatusSuccess)
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
        if(miopenSetTensorDescriptor(d.yDesc, dtype, rank, yDim, nullptr) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set yDesc failed";
    }
    return ::testing::AssertionSuccess();
}

// CI-covered architectures for these sweeps. We only run on arches whose CK
// large-tensor tile coverage has been characterized in CI, so non-allowlisted
// GPUs SKIP instead of emitting stale FAILED lines. gfx115X (RDNA 3.5) is
// intentionally omitted: the CK *Xdlops solvers target CDNA MFMA, and a
// gfx1151 CI run produced 26 failures across sub- and >INT_MAX shapes -- re-add
// once gfx115X CK coverage is characterized.
inline bool IsArchInCiAllowlist()
{
    return IsTestSupportedByDevice(Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950);
}

// Run* helpers -- templated on Shape and SetupFn (per-rank descriptor wrapper).
// Direction-specific Compile API call is the only thing that varies between the
// three. Diagnostic note: the test framework's parameter machinery prints the
// shape, so we keep the skip/failure messages terse (solver name only).

template <typename Shape, typename SetupFn>
void RunCompileFwd(const Shape& s,
                   miopenDataType_t dtype,
                   SetupFn setup_fn,
                   const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    const auto status = miopenConvolutionForwardCompileSolution(
        d.handle, d.wDesc, d.xDesc, d.convDesc, d.yDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

template <typename Shape, typename SetupFn>
void RunCompileBwdData(const Shape& s,
                       miopenDataType_t dtype,
                       SetupFn setup_fn,
                       const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    // dyDesc has y's shape, dxDesc has x's shape.
    const auto status = miopenConvolutionBackwardDataCompileSolution(
        d.handle, d.yDesc, d.wDesc, d.convDesc, d.xDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

template <typename Shape, typename SetupFn>
void RunCompileWrw(const Shape& s,
                   miopenDataType_t dtype,
                   SetupFn setup_fn,
                   const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    // dyDesc has y's shape, xDesc has x's shape, dwDesc has w's shape.
    const auto status = miopenConvolutionBackwardWeightsCompileSolution(
        d.handle, d.yDesc, d.xDesc, d.convDesc, d.wDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

} // namespace miopen_test_large_stride
