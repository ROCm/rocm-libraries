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
    // K-contiguous A and B with K/32-contiguous scales and BF16 output (hipBLAS TN,
    // the layout gfx1250 accepts), so the case is declined for its shape there too.
    bool kContiguous = false;
};

// transScaleA stores scale_a [M, K/32] column-major (M-contiguous) and
// transScaleB stores scale_b [K/32, N] column-major (K/32-contiguous); both
// scales are row-major otherwise.
Graph buildMxMatmulGraph(const MatmulTestCase& tc,
                         bool transScaleA = false,
                         bool transScaleB = false,
                         DataType outType = DataType::HALF)
{
    Graph g;
    g.set_name("IntegrationGpuMxMatmulUnsupported");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(outType);

    const auto inType = DataType::FP8_E4M3;
    const auto& aDims = tc.aDims;
    const auto& bDims = tc.bDims;
    const int64_t scaleK = aDims[aDims.size() - 1] / 32;

    auto aAttr
        = makeTensorAttributes("a", inType, aDims, generateInputStrideOrder(aDims, tc.transA));
    auto aTensor = std::make_shared<TensorAttributes>(std::move(aAttr));

    std::vector<int64_t> scaleADims = aDims;
    scaleADims.back() = scaleK;
    auto scaleAAttr = makeTensorAttributes("scale_a",
                                           DataType::FP8_E8M0,
                                           scaleADims,
                                           generateInputStrideOrder(scaleADims, transScaleA));
    auto scaleATensor = std::make_shared<TensorAttributes>(std::move(scaleAAttr));

    BlockScaleDequantizeAttributes deqA;
    deqA.set_block_size(32);
    auto yA = g.block_scale_dequantize(aTensor, scaleATensor, deqA);

    auto bAttr
        = makeTensorAttributes("b", inType, bDims, generateInputStrideOrder(bDims, tc.transB));
    auto bTensor = std::make_shared<TensorAttributes>(std::move(bAttr));

    std::vector<int64_t> scaleBDims = bDims;
    scaleBDims[scaleBDims.size() - 2] = scaleK;
    auto scaleBAttr = makeTensorAttributes("scale_b",
                                           DataType::FP8_E8M0,
                                           scaleBDims,
                                           generateInputStrideOrder(scaleBDims, transScaleB));
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
// A [M, K] column-major (M-contiguous), B [K, N] row-major (N-contiguous), both
// scales row-major: hipBLASLt sees transA=N, transB=T.
TEST_F(IntegrationGpuMxMatmulSupported, AcceptsSupportedConfig)
{
    // WORKAROUND_ISSUE_13715 (see tests/TestWorkarounds.hpp): on gfx1250 hipBLASLt
    // has MX kernels only for transA=T, transB=N, so this layout is declined there.
    if(isGfx1250Target(hipdnn_test_sdk::utilities::currentDeviceArch()))
    {
        GTEST_SKIP() << "[#13715] hipBLASLt has no MX GEMM kernel for transA=N, transB=T "
                        "on gfx1250";
    }

    auto g = buildMxMatmulGraph(MatmulTestCase{{32, 128}, {128, 32}, true, false, 0});

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::OK) << "err_msg: " << result.err_msg;
}

// Positive control for the framework-natural layout, which every MX arch must
// accept (gfx1250 included): A [M, K] row-major and B [K, N] column-major (both
// K-contiguous), scale_a [M, K/32] row-major and scale_b [K/32, N] column-major
// (both K/32-contiguous). hipBLASLt sees transA=T, transB=N. BF16 output, which
// every gfx1250 hipBLASLt with MX kernels has (some lack FP16 output).
TEST_F(IntegrationGpuMxMatmulSupported, AcceptsKContiguousConfig)
{
    auto g = buildMxMatmulGraph(MatmulTestCase{{32, 128}, {128, 32}, false, true, 0},
                                /*transScaleA=*/false,
                                /*transScaleB=*/true,
                                DataType::BFLOAT16);

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::OK) << "err_msg: " << result.err_msg;
}

TEST_P(IntegrationGpuMxMatmulUnsupported, RejectsUnsupportedConfig)
{
    const auto& param = GetParam();
    auto g = param.kContiguous ? buildMxMatmulGraph(param.params,
                                                    /*transScaleA=*/false,
                                                    /*transScaleB=*/true,
                                                    DataType::BFLOAT16)
                               : buildMxMatmulGraph(param.params);

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

INSTANTIATE_TEST_SUITE_P(
    IntegrationGpuMxMatmul,
    IntegrationGpuMxMatmulUnsupported,
    ::testing::Values(
        // batch != 1 (VEC32_UE8M0 requires a single batch)
        MxMatmulCase{"BatchGreaterThanOne", {{2, 32, 128}, {2, 128, 32}, true, false, 0}},
        // M not a multiple of 16
        MxMatmulCase{"MisalignedM", {{8, 128}, {128, 32}, true, false, 0}},
        // K not a multiple of 128
        MxMatmulCase{"MisalignedK", {{32, 64}, {64, 32}, true, false, 0}},
        // The same shapes with K-contiguous operands
        MxMatmulCase{
            "BatchGreaterThanOneKContiguous", {{2, 32, 128}, {2, 128, 32}, false, true, 0}, true},
        MxMatmulCase{"MisalignedMKContiguous", {{8, 128}, {128, 32}, false, true, 0}, true},
        MxMatmulCase{"MisalignedKKContiguous", {{32, 64}, {64, 32}, false, true, 0}, true}),
    [](const ::testing::TestParamInfo<MxMatmulCase>& info) {
        return std::string(info.param.name);
    });
