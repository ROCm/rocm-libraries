// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ResampleShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_resample_fwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;
using namespace gpu_resample_ref_test;

template <class XDataType,
          class YDataType = XDataType,
          class ComputeDataType = float,
          class IndexDataType = int32_t>
void runGpuVsCpuResampleFwd(const std::vector<int64_t>& xDims,
                            const std::vector<int64_t>& yDims,
                            const TensorLayout& layout,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& window,
                            ResampleMode resampleMode,
                            PaddingMode paddingMode,
                            float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();

    auto xTensor = Tensor<XDataType>(xDims, layout);
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed);

    auto yCpu = Tensor<YDataType>(yDims, layout);
    auto yGpu = Tensor<YDataType>(yDims, layout);

    const auto includeIndex = (resampleMode == ResampleMode::MAXPOOL);
    auto indexCpu = includeIndex ? Tensor<IndexDataType>(yDims, layout) : Tensor<IndexDataType>({});
    auto indexGpu = includeIndex ? Tensor<IndexDataType>(yDims, layout) : Tensor<IndexDataType>({});

    CpuFpReferenceResampleFwd::forward<XDataType, YDataType, ComputeDataType, IndexDataType>(
        xTensor,
        yCpu,
        prePadding,
        stride,
        window,
        resampleMode,
        paddingMode,
        includeIndex ? &indexCpu : nullptr);
    GpuFpReferenceResample::forward<XDataType, YDataType, ComputeDataType, IndexDataType>(
        xTensor,
        yGpu,
        prePadding,
        postPadding,
        stride,
        window,
        resampleMode,
        paddingMode,
        includeIndex ? &indexGpu : nullptr);

    assertAllClose(yCpu, yGpu, resample::getTolerance<YDataType>());
    if(includeIndex)
    {
        assertAllExact(indexCpu, indexGpu);
    }
}

// =============================================================================
// ResampleFwdTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// =============================================================================

template <typename DataType>
class ResampleFwdTestSuite : public ::testing::TestWithParam<ResampleTestCase>
{
protected:
    void runResampleFwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuResampleFwd<DataType>(tc.xDims,
                                         tc.yDims,
                                         tc.layout,
                                         tc.prePadding,
                                         tc.postPadding,
                                         tc.stride,
                                         tc.window,
                                         tc.resampleMode,
                                         tc.paddingMode,
                                         1.0f);
    }
};

} // namespace gpu_resample_fwd_ref_test
