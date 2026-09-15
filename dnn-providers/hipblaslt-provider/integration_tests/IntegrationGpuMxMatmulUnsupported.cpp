// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "utils/MxMatmulUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace test_matmul_common;
using namespace test_mx_matmul_common;

namespace
{

struct MxMatmulCase
{
    const char* name;
    MatmulTestCase params;
};

Graph buildMxMatmulGraph(const MatmulTestCase& tc)
{
    Graph g;
    g.set_name("IntegrationGpuMxMatmulUnsupported");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::HALF);

    const auto inType = DataType::FP8_E4M3;
    const auto& aDims = tc.aDims;
    const auto& bDims = tc.bDims;
    const int64_t scaleK = aDims[aDims.size() - 1] / 32;

    auto aAttr
        = makeTensorAttributes("a", inType, aDims, generateInputStrideOrder(aDims, tc.transA));
    auto aTensor = std::make_shared<TensorAttributes>(std::move(aAttr));

    std::vector<int64_t> scaleADims = aDims;
    scaleADims.back() = scaleK;
    auto scaleAAttr = makeTensorAttributes(
        "scale_a", DataType::FP8_E8M0, scaleADims, generateInputStrideOrder(scaleADims, false));
    auto scaleATensor = std::make_shared<TensorAttributes>(std::move(scaleAAttr));

    BlockScaleDequantizeAttributes deqA;
    deqA.set_block_size(32);
    auto yA = g.block_scale_dequantize(aTensor, scaleATensor, deqA);

    auto bAttr
        = makeTensorAttributes("b", inType, bDims, generateInputStrideOrder(bDims, tc.transB));
    auto bTensor = std::make_shared<TensorAttributes>(std::move(bAttr));

    std::vector<int64_t> scaleBDims = bDims;
    scaleBDims[scaleBDims.size() - 2] = scaleK;
    auto scaleBAttr = makeTensorAttributes(
        "scale_b", DataType::FP8_E8M0, scaleBDims, generateInputStrideOrder(scaleBDims, false));
    auto scaleBTensor = std::make_shared<TensorAttributes>(std::move(scaleBAttr));

    BlockScaleDequantizeAttributes deqB;
    deqB.set_block_size(std::vector<int32_t>{32, 1});
    auto yB = g.block_scale_dequantize(bTensor, scaleBTensor, deqB);

    const MatmulAttributes matmulAttrs;
    auto d = g.matmul(yA, yB, matmulAttrs);
    d->set_output(true);

    return g;
}

class IntegrationGpuMxMatmulUnsupported : public ::testing::TestWithParam<MxMatmulCase>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
};

class IntegrationGpuMxMatmulSupported : public IntegrationGpuMxMatmulUnsupported
{
protected:
    void SetUp() override
    {
        IntegrationGpuMxMatmulUnsupported::SetUp();
        if(::testing::Test::IsSkipped())
        {
            return;
        }

        const auto archName = hipdnn_test_sdk::utilities::currentDeviceArch();
        if(!isMxSupportedArch(archName))
        {
            GTEST_SKIP() << "MX block-scaled GEMM is not supported on " << archName;
        }
    }
};

} // namespace

// Positive control: a supported config must build, proving the negatives'
// GRAPH_NOT_SUPPORTED reflects the config and not a broken harness.
TEST_F(IntegrationGpuMxMatmulSupported, AcceptsSupportedConfig)
{
    auto g = buildMxMatmulGraph(MatmulTestCase{{32, 128}, {128, 32}, true, false, 0});

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::OK) << "err_msg: " << result.err_msg;
}

TEST_P(IntegrationGpuMxMatmulUnsupported, RejectsUnsupportedConfig)
{
    auto g = buildMxMatmulGraph(GetParam().params);

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxMatmulUnsupported,
                         ::testing::Values(
                             // batch != 1 (VEC32_UE8M0 requires a single batch)
                             MxMatmulCase{"BatchGreaterThanOne",
                                          {{2, 32, 128}, {2, 128, 32}, true, false, 0}},
                             // M not a multiple of 16
                             MxMatmulCase{"MisalignedM", {{8, 128}, {128, 32}, true, false, 0}},
                             // K not a multiple of 128
                             MxMatmulCase{"MisalignedK", {{32, 64}, {64, 32}, true, false, 0}}),
                         [](const ::testing::TestParamInfo<MxMatmulCase>& info) {
                             return std::string(info.param.name);
                         });
