// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../tests/common/BatchnormCommon.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace test_bn_common;

namespace
{

struct UnhappyBnDtypeCase
{
    DataType io;
    DataType scale;
    DataType bias;
    const char* name;
};

inline std::vector<int64_t> makeDims()
{
    return {1, 4, 8, 8};
}

static Graph makeGraph(const UnhappyBnDtypeCase& tc,
                       const TensorLayout& layout = TensorLayout::NCHW)
{
    Graph g;
    g.set_name("IntegrationGpuBatchnormUnhappyDataTypes");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(tc.io);

    const auto dims = makeDims();
    const auto cDims = getDerivedShape(dims);

    auto xAttr = makeTensorAttributes("X", tc.io, dims, generateStrides(dims, layout.strideOrder));
    auto X = std::make_shared<TensorAttributes>(std::move(xAttr));

    auto meanAttr = makeTensorAttributes("mean", DataType::FLOAT, cDims, generateStrides(cDims));
    auto invVarAttr
        = makeTensorAttributes("inv_variance", DataType::FLOAT, cDims, generateStrides(cDims));
    auto mean = std::make_shared<TensorAttributes>(std::move(meanAttr));
    auto invVar = std::make_shared<TensorAttributes>(std::move(invVarAttr));

    auto scaleAttr = makeTensorAttributes("scale", tc.scale, cDims, generateStrides(cDims));
    auto biasAttr = makeTensorAttributes("bias", tc.bias, cDims, generateStrides(cDims));
    auto scale = std::make_shared<TensorAttributes>(std::move(scaleAttr));
    auto bias = std::make_shared<TensorAttributes>(std::move(biasAttr));

    BatchnormInferenceAttributes bn;
    g.batchnorm_inference(X, mean, invVar, scale, bias, bn);

    return g;
}

class IntegrationGpuBatchnormUnhappyDataTypes : public ::testing::TestWithParam<UnhappyBnDtypeCase>
{
};

} // namespace

TEST_P(IntegrationGpuBatchnormUnhappyDataTypes, RejectsUnsupportedDataTypes)
{
    const auto& tc = GetParam();
    auto g = makeGraph(tc, TensorLayout::NCHW);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    auto result = g.build(handle);

    EXPECT_NE(result.code, ErrorCode::OK);
    EXPECT_FALSE(result.err_msg.empty());
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormUnhappyDataTypes,
    ::testing::Values(
        UnhappyBnDtypeCase{DataType::UINT8, DataType::FLOAT, DataType::FLOAT, "Uint8IO"},
        UnhappyBnDtypeCase{DataType::FLOAT, DataType::HALF, DataType::HALF, "HalfScaleBias"},
        UnhappyBnDtypeCase{
            DataType::FLOAT, DataType::HALF, DataType::FLOAT, "MismatchedScaleHalfBiasFloat"}),
    [](const ::testing::TestParamInfo<UnhappyBnDtypeCase>& info) {
        return std::string(info.param.name);
    });
