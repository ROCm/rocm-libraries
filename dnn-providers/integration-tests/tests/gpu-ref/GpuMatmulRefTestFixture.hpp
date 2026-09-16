// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "MatmulShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceMatmul.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_matmul_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::matmul;
using namespace hipdnn_gpu_ref;
using namespace gpu_matmul_ref_test;

template <typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ComputeDataType = double>
void runGpuVsCpuMatmul(const std::vector<int64_t>& aDims,
                       const std::vector<int64_t>& bDims,
                       const std::vector<int64_t>& cDims,
                       const std::vector<int64_t>& aStrides,
                       const std::vector<int64_t>& bStrides,
                       const std::vector<int64_t>& cStrides,
                       const float tolerance,
                       const float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();

    auto aTensor = Tensor<ADataType>(aDims, aStrides);
    aTensor.fillWithRandomValues(
        static_cast<ADataType>(-fillRange), static_cast<ADataType>(fillRange), seed);
    auto bTensor = Tensor<BDataType>(bDims, bStrides);
    bTensor.fillWithRandomValues(
        static_cast<BDataType>(-fillRange), static_cast<BDataType>(fillRange), seed + 1);
    auto cGpu = Tensor<CDataType>(cDims, cStrides);
    auto cCpu = Tensor<CDataType>(cDims, cStrides);

    GpuFpReferenceMatmul::matmul<ADataType, BDataType, CDataType, ComputeDataType>(
        aTensor, bTensor, cGpu);
    cGpu.markDeviceModified();

    CpuFpReferenceMatmul::matmul<ADataType, BDataType, CDataType, ComputeDataType>(
        aTensor, bTensor, cCpu);
    cCpu.markHostModified();

    assertAllClose(cCpu, cGpu, tolerance, "C");
}

// =========================================================================
// MatmulShapeSuite - parameterized fixture for shape-based GPU-vs-CPU tests
// =========================================================================

template <typename ADataType, typename BDataType, typename CDataType>
class MatmulShapeSuite : public ::testing::TestWithParam<gpu_matmul_ref_test::MatmulTestCase>
{
protected:
    void runMatmulTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& testCase = GetParam();
        runGpuVsCpuMatmul<ADataType, BDataType, CDataType>(
            testCase.aDims,
            testCase.bDims,
            testCase.cDims,
            testCase.aStrides,
            testCase.bStrides,
            testCase.cStrides,
            hipdnn_test_sdk::utilities::matmul::getTolerance<CDataType>());
    }
};

// "Pure" = A, B and C all the same type.
template <typename DataType>
using MatmulPureShapeSuite = MatmulShapeSuite<DataType, DataType, DataType>;

// "Mixed" = A and C are FP16/BFP16, but B is FP32.
template <typename DataType>
using MatmulMixedShapeSuite = MatmulShapeSuite<DataType, float, DataType>;

// "Upcast" = A and B are FP16/BFP16, but C widens to FP32
template <typename DataType>
using MatmulUpcastShapeSuite = MatmulShapeSuite<DataType, DataType, float>;

} // namespace gpu_matmul_ref_test
