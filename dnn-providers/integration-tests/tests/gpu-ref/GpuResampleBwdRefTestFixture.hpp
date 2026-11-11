// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ResampleShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_resample_bwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;
using namespace gpu_resample_ref_test;

template <class DyDataType,
          class DxDataType = DyDataType,
          class ComputeDataType = float,
          class IndexDataType = int32_t>
void runGpuVsCpuResampleBwd(const std::vector<int64_t>& dyDims,
                            const std::vector<int64_t>& dxDims,
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

    auto dyTensor = Tensor<DyDataType>(dyDims, layout);
    fillWithRandomValues(
        dyTensor, static_cast<DyDataType>(-fillRange), static_cast<DyDataType>(fillRange), seed);
    dyTensor.memory().hostData();

    const auto includeIndex = (resampleMode == ResampleMode::MAXPOOL);
    auto indexTensor
        = includeIndex ? Tensor<IndexDataType>(dyDims, layout) : Tensor<IndexDataType>({});

    auto dxTensorCpu = Tensor<DxDataType>(dxDims, layout);
    auto dxTensorGpu = Tensor<DxDataType>(dxDims, layout);

    // Run GPU resample forward to compute sensible index values for the backward pass
    if(includeIndex)
    {
        auto xTensor = Tensor<DxDataType>(dxDims, layout);
        auto yTensor = Tensor<DyDataType>(dyDims, layout);
        fillWithRandomValues(xTensor,
                             static_cast<DxDataType>(-fillRange),
                             static_cast<DxDataType>(fillRange),
                             seed + 1);
        xTensor.memory().hostData();

        GpuFpReferenceResample::forward<DxDataType, DyDataType, ComputeDataType, IndexDataType>(
            xTensor,
            yTensor,
            prePadding,
            postPadding,
            stride,
            window,
            resampleMode,
            paddingMode,
            &indexTensor);
        indexTensor.memory().hostData();
    }

    CpuFpReferenceResampleBwd::backward<DyDataType, DxDataType, ComputeDataType, IndexDataType>(
        dyTensor,
        dxTensorCpu,
        prePadding,
        stride,
        window,
        resampleMode,
        paddingMode,
        includeIndex ? &indexTensor : nullptr);

    GpuFpReferenceResample::backward<DyDataType, DxDataType, ComputeDataType, IndexDataType>(
        dyTensor,
        dxTensorGpu,
        prePadding,
        postPadding,
        stride,
        window,
        resampleMode,
        paddingMode,
        includeIndex ? &indexTensor : nullptr);

    assertAllClose(dxTensorCpu, dxTensorGpu, resample::getTolerance<DxDataType>());
}

// =============================================================================
// ResampleBwdTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// =============================================================================

template <typename DataType>
class ResampleBwdTestSuite : public ::testing::TestWithParam<ResampleTestCase>
{
protected:
    void runResampleBwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuResampleBwd<DataType>(tc.yDims,
                                         tc.xDims,
                                         tc.layout,
                                         tc.prePadding,
                                         tc.postPadding,
                                         tc.stride,
                                         tc.window,
                                         tc.resampleMode,
                                         tc.paddingMode);
    }
};

} // namespace gpu_resample_bwd_ref_test
