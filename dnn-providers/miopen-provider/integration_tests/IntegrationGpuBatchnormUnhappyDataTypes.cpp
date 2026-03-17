// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include "../tests/common/BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
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

inline std::vector<int64_t> makeDims() { return {1, 4, 8, 8}; }

static Graph makeGraph(const UnhappyBnDtypeCase& tc, const TensorLayout& layout = TensorLayout::NCHW)
{
    Graph g;
    g.set_name("IntegrationGpuBatchnormUnhappyDataTypes");
    g.set_compute_data_type(DataType::FLOAT)
     .set_intermediate_data_type(DataType::FLOAT)
     .set_io_data_type(tc.io);

    const auto dims  = makeDims();
    const auto cDims = getDerivedShape(dims);

    auto xAttr = makeTensorAttributes("X", tc.io, dims, generateStrides(dims, layout.strideOrder));
    auto X = std::make_shared<TensorAttributes>(std::move(xAttr));

    auto meanAttr   = makeTensorAttributes("mean", DataType::FLOAT, cDims, generateStrides(cDims));
    auto invVarAttr = makeTensorAttributes("inv_variance", DataType::FLOAT, cDims, generateStrides(cDims));
    auto mean       = std::make_shared<TensorAttributes>(std::move(meanAttr));
    auto invVar     = std::make_shared<TensorAttributes>(std::move(invVarAttr));

    auto scaleAttr = makeTensorAttributes("scale", tc.scale, cDims, generateStrides(cDims));
    auto biasAttr  = makeTensorAttributes("bias",  tc.bias,  cDims, generateStrides(cDims));
    auto scale     = std::make_shared<TensorAttributes>(std::move(scaleAttr));
    auto bias      = std::make_shared<TensorAttributes>(std::move(biasAttr));

    BatchnormInferenceAttributes bn;
    g.batchnorm_inference(X, mean, invVar, scale, bias, bn);

    return g;
}

class IntegrationGpuBatchnormUnhappyDataTypes
    : public ::testing::TestWithParam<UnhappyBnDtypeCase>
{
};

TEST_P(IntegrationGpuBatchnormUnhappyDataTypes, BuildRejectsInvalidDTypes)
{
    const auto& tc = GetParam();
    auto g = makeGraph(tc, TensorLayout::NCHW);
    auto result = g.build();

    EXPECT_NE(result.code, ErrorCode::OK);
    EXPECT_NE(result.message.find("type"), std::string::npos);
}

static UnhappyBnDtypeCase kCases[] = {
    {DataType::UINT8, DataType::FLOAT, DataType::FLOAT, "Uint8IO"},
    {DataType::FLOAT, DataType::HALF,  DataType::HALF,  "HalfScaleBias"},
    {DataType::FLOAT, DataType::HALF,  DataType::FLOAT, "MismatchedScaleHalfBiasFloat"},
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormUnhappyDataTypes,
    ::testing::ValuesIn(kCases),
    [](const ::testing::TestParamInfo<UnhappyBnDtypeCase>& info) {
        return std::string(info.param.name);
    });

} // namespace
