// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "IntegrationTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w
                  << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

struct Batchnorm3dTestCase
{
    int64_t n;
    int64_t c;
    int64_t d;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm3dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " d:" << tc.d << " h:" << tc.h
                  << " w:" << tc.w << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, d, h, w};
    }
};

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCase>
class BatchnormBackwardActivation : public GraphVerifierTest<DataType, TestCase>
{
protected:
    void runGraphTest([[maybe_unused]] DataType tolerance, const TensorLayout& layout) override
    {
        namespace fe = hipdnn_frontend;

        const TestCase& testCase = this->GetParam();
        auto dims = testCase.getDims();

        std::vector<int64_t> channelDims;
        if(dims.size() == 4)
        {
            channelDims = {1, dims[1], 1, 1};
        }
        else
        {
            channelDims = {1, dims[1], 1, 1, 1};
        }

        auto graphObj = std::make_shared<graph::Graph>();
        graphObj->set_name("BatchnormBackwardActivationTest");
        graphObj->set_compute_data_type(fe::DataType::FLOAT);

        int64_t uid = 1;
        auto nextUid = [&]() { return uid++; };

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = fe::DataType::FLOAT;

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(nextUid());
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        scaleAttr.set_uid(nextUid());
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = graph::makeTensorAttributes("bias",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        biasAttr.set_uid(nextUid());
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto meanAttr
            = graph::makeTensorAttributes("mean",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        meanAttr.set_uid(nextUid());
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarAttr
            = graph::makeTensorAttributes("inv_variance",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        invVarAttr.set_uid(nextUid());
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));

        // BN_Y = batchnorm_inference(X, mean, inv_variance, scale, bias)
        graph::BatchnormInferenceAttributes bnInfAttrs;
        bnInfAttrs.set_name("batchnorm_inference");

        auto bnY = graphObj->batchnorm_inference(xTensorAttr,
                                                 meanTensorAttr,
                                                 invVarianceTensorAttr,
                                                 scaleTensorAttr,
                                                 biasTensorAttr,
                                                 bnInfAttrs);

        bnY->set_name("BN_Y");
        bnY->set_data_type(dataType);
        bnY->set_dim(dims);
        bnY->set_stride(generateStrides(dims, layout.strideOrder));
        bnY->set_is_virtual(true);
        if(!bnY->has_uid())
        {
            bnY->set_uid(nextUid());
        }

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(nextUid());
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // DX_drelu = pointwise(DY, BN_Y, RELU_BWD)
        graph::PointwiseAttributes reluBwdAttrs;
        reluBwdAttrs.set_name("relu_bwd");
        reluBwdAttrs.set_mode(hipdnn_frontend::PointwiseMode::RELU_BWD);

        auto dxDrelu = graphObj->pointwise(dyTensorAttr, bnY, reluBwdAttrs);
        dxDrelu->set_name("DX_drelu");
        dxDrelu->set_data_type(dataType);
        dxDrelu->set_dim(dims);
        dxDrelu->set_stride(generateStrides(dims, layout.strideOrder));
        dxDrelu->set_is_virtual(true);
        if(!dxDrelu->has_uid())
        {
            dxDrelu->set_uid(nextUid());
        }

        graph::BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("batchnorm_backward");
        bnBwdAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        // [DX, dscale, dbias] = batchnorm_backward(DX_drelu, X, scale, saved_mean_inv_var)
        auto bnBwdOuts
            = graphObj->batchnorm_backward(dxDrelu, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

        auto& dxOut = bnBwdOuts[0];
        dxOut->set_name("dx");
        dxOut->set_data_type(dataType);
        dxOut->set_dim(dims);
        dxOut->set_stride(generateStrides(dims, layout.strideOrder));
        dxOut->set_is_virtual(false);
        dxOut->set_output(true);
        if(!dxOut->has_uid())
        {
            dxOut->set_uid(nextUid());
        }

        auto& dscaleOut = bnBwdOuts[1];
        dscaleOut->set_name("dscale");
        dscaleOut->set_data_type(intermediateDataType);
        dscaleOut->set_dim(channelDims);
        dscaleOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dscaleOut->set_is_virtual(false);
        dscaleOut->set_output(true);
        if(!dscaleOut->has_uid())
        {
            dscaleOut->set_uid(nextUid());
        }

        auto& dbiasOut = bnBwdOuts[2];
        dbiasOut->set_name("dbias");
        dbiasOut->set_data_type(intermediateDataType);
        dbiasOut->set_dim(channelDims);
        dbiasOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dbiasOut->set_is_virtual(false);
        dbiasOut->set_output(true);
        if(!dbiasOut->has_uid())
        {
            dbiasOut->set_uid(nextUid());
        }

        auto result = graphObj->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // todo: add registry
    }
};
// NOLINTEND (portability-template-virtual-member-function)

using IntegrationGpuBatchnormBackwardActivationNchwFp32
    = BatchnormBackwardActivation<float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNchwBfp16
    = BatchnormBackwardActivation<hip_bfloat16, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNchwFp16
    = BatchnormBackwardActivation<half, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp32
    = BatchnormBackwardActivation<float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNhwcBfp16
    = BatchnormBackwardActivation<hip_bfloat16, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp16
    = BatchnormBackwardActivation<half, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp32
    = BatchnormBackwardActivation<float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNcdhwBfp16
    = BatchnormBackwardActivation<hip_bfloat16, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp16
    = BatchnormBackwardActivation<half, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp32
    = BatchnormBackwardActivation<float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNdhwcBfp16
    = BatchnormBackwardActivation<hip_bfloat16, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp16
    = BatchnormBackwardActivation<half, Batchnorm3dTestCase>;

std::vector<Batchnorm2dTestCase> getBnBwdTestCases()
{
    unsigned int seed = std::random_device{}();

    return std::vector<Batchnorm2dTestCase>{
        {1, 3, 14, 14, seed},
        // MIOpen segfaults for this case, re-enable when fix is released:
        // https://github.com/ROCm/rocm-libraries/pull/1197
        // {1, 256, 1, 1, seed}, // Would produce near-zero variance in theory
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

std::vector<Batchnorm3dTestCase> getBnBwd3dTestCases()
{
    unsigned int seed = std::random_device{}();

    return std::vector<Batchnorm3dTestCase>{
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197}
TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));
