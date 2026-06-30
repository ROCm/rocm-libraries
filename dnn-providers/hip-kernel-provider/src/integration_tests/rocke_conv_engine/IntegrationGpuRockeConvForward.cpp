// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;
using hipdnn_data_sdk::types::half;

namespace
{

// ---------------------------------------------------------------------------
// Test case descriptor
// ---------------------------------------------------------------------------

struct ConvFwdTestCase
{
    std::string name;

    // Problem shape (NHWC layout throughout)
    int N, Hi, Wi, C, K;
    int Y, X; // filter height, width
    int strideH = 1, strideW = 1;
    int padH = 0, padW = 0;
    int dilH = 1, dilW = 1;

    static std::string getName(const testing::TestParamInfo<ConvFwdTestCase>& info)
    {
        return info.param.name;
    }
};

// Smoke shapes: small enough that JIT compile + kernel launch complete quickly,
// covering 3x3/1x1/strided cases and both square and rectangular spatial dims.
static const ConvFwdTestCase kSmokeShapes[] = {
    // name                              N   Hi  Wi   C   K  Y  X  sH sW pH pW dH dW
    {"3x3_small", 1, 16, 16, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1},
    {"1x1_pointwise", 2, 14, 14, 64, 128, 1, 1, 1, 1, 0, 0, 1, 1},
    {"3x3_stride2", 1, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1},
    {"3x3_rect_spatial", 1, 32, 16, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1},
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class IntegrationGpuRockeConvFwdFp16
    : public IntegrationGraphVerificationHarness<half, ConvFwdTestCase>
{
protected:
    // NHWC strides for NCHW logical dims [N, C, H, W] (or [K, C, Y, X] for weights).
    // Physical layout: n*(C*H*W) + h*(W*C) + w*C + c
    // → strides[N]=C*H*W, strides[C]=1, strides[H]=W*C, strides[W]=C
    static std::vector<int64_t> nhwcStrides(const std::vector<int64_t>& dims)
    {
        // Only implemented for rank-4; extend if needed.
        const int64_t C = dims[1];
        const int64_t H = dims[2];
        const int64_t W = dims[3];
        return {C * H * W, 1, W * C, C};
    }

    void runGraphTest()
    {
        const ConvFwdTestCase& tc = this->GetParam();

        // Skip on archs the rocKE conv engine doesn't support
        std::string arch;
        try
        {
            arch = hip_kernel_provider_common::getDeviceString(this->stream());
        }
        catch(const std::exception& e)
        {
            GTEST_SKIP() << "Could not query arch: " << e.what();
        }
        if(arch != "gfx942" && arch != "gfx950" && arch != "gfx90a")
        {
            GTEST_SKIP() << "rocKE conv engine not supported on " << arch;
        }

        // Derive output spatial dims
        const int Ho = (tc.Hi + 2 * tc.padH - tc.dilH * (tc.Y - 1) - 1) / tc.strideH + 1;
        const int Wo = (tc.Wi + 2 * tc.padW - tc.dilW * (tc.X - 1) - 1) / tc.strideW + 1;

        // Tensor dims: NCHW logical order (required by frontend validation).
        // Strides are set to NHWC-contiguous so the engine sees NHWC layout.
        // x: [N, C, Hi, Wi], w: [K, C, Y, X], y: [N, K, Ho, Wo]
        const std::vector<int64_t> xDims = {tc.N, tc.C, tc.Hi, tc.Wi};
        const std::vector<int64_t> wDims = {tc.K, tc.C, tc.Y, tc.X};
        const std::vector<int64_t> yDims = {tc.N, tc.K, Ho, Wo};

        // Build the graph using the hipdnn frontend
        graph::Graph graphObj;
        graphObj.set_name("RockeConvFwdTest_" + tc.name)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::HALF);

        auto xAttr = std::make_shared<TensorAttributes>(TensorAttributes()
                                                            .set_name("x")
                                                            .set_uid(1)
                                                            .set_dim(xDims)
                                                            .set_stride(nhwcStrides(xDims))
                                                            .set_data_type(DataType::HALF));

        auto wAttr = std::make_shared<TensorAttributes>(TensorAttributes()
                                                            .set_name("w")
                                                            .set_uid(2)
                                                            .set_dim(wDims)
                                                            .set_stride(nhwcStrides(wDims))
                                                            .set_data_type(DataType::HALF));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding({tc.padH, tc.padW})
            .set_post_padding({tc.padH, tc.padW})
            .set_stride({tc.strideH, tc.strideW})
            .set_dilation({tc.dilH, tc.dilW});

        auto yAttr = graphObj.conv_fprop(xAttr, wAttr, convAttrs);
        yAttr->set_output(true);

        auto validateResult = graphObj.validate();
        ASSERT_TRUE(validateResult.is_good())
            << "Graph validation failed: " << validateResult.get_message();

        this->registerValidator(yAttr, /*tolerance=*/1e-2f);
        this->verifyGraph(graphObj, /*seed=*/42);
    }
};

} // namespace

TEST_P(IntegrationGpuRockeConvFwdFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRockeConvFwdFp16,
                         testing::ValuesIn(kSmokeShapes),
                         ConvFwdTestCase::getName);
