// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hip/HipKernelCompileOptions.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace hip_kernel_provider::tests
{

enum class TensorLayout
{
    NCHW,
    NHWC
};

class TestHipKernelCompileOptions : public ::testing::Test
{
protected:
    hipDeviceProp_t _deviceProps = {};
    const TensorAttributes* _inputTensorAttrs = nullptr;
    flatbuffers::FlatBufferBuilder _fbb;
    std::unique_ptr<GraphWrapper> _graph;

    void setUpTestCase(DataType dataType = DataType::FLOAT,
                       TensorLayout layout = TensorLayout::NCHW)
    {
        // Setup Device Properties
        _deviceProps.multiProcessorCount = 60;
        _deviceProps.warpSize = 64;
        std::snprintf(_deviceProps.gcnArchName, sizeof(_deviceProps.gcnArchName), "%s", "gfx942");

        // Set ROCM_PATH (to default path on linux)
        hipdnn_data_sdk::utilities::setEnv("ROCM_PATH", "/opt/rocm");

        std::vector<int64_t> dims = {1, 3, 224, 224};
        std::vector<int64_t> strides;

        // Setup input tensor attributes based on data type and layout
        if(layout == TensorLayout::NCHW)
        {
            strides = hipdnn_data_sdk::utilities::generateStrides(
                dims, hipdnn_data_sdk::utilities::TensorLayout::NCHW.strideOrder);
        }
        else if(layout == TensorLayout::NHWC)
        {
            strides = hipdnn_data_sdk::utilities::generateStrides(
                dims, hipdnn_data_sdk::utilities::TensorLayout::NHWC.strideOrder);
        }
        else
        {
            throw std::invalid_argument("Unsupported tensor layout");
        }

        std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttrs
            = {CreateTensorAttributesDirect(_fbb, 1, "tensor", dataType, &strides, &dims)};
        std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

        auto graphOffset
            = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(_fbb,
                                                                      "test",
                                                                      DataType::FLOAT,
                                                                      DataType::HALF,
                                                                      DataType::BFLOAT16,
                                                                      &tensorAttrs,
                                                                      &nodes);

        _fbb.Finish(graphOffset);
        _graph = std::make_unique<GraphWrapper>(_fbb.GetBufferPointer(), _fbb.GetSize());

        _inputTensorAttrs = _graph->getTensorMap().at(1);
    }

    auto static hasOption(const std::vector<std::string>& opts, const std::string& val)
    {
        return std::find(opts.begin(), opts.end(), val) != opts.end();
    }
};

TEST_F(TestHipKernelCompileOptions, VerifiesOptionsForFp32Nchw)
{
    setUpTestCase(DataType::FLOAT, TensorLayout::NCHW);

    HipKernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    EXPECT_TRUE(hasOption(options, "-I/opt/rocm/include"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_NHWC=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

TEST_F(TestHipKernelCompileOptions, VerifiesOptionsForBFp16Nhwc)
{
    setUpTestCase(DataType::BFLOAT16, TensorLayout::NHWC);

    HipKernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    EXPECT_TRUE(hasOption(options, "-I/opt/rocm/include"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_NHWC=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

TEST_F(TestHipKernelCompileOptions, VerifyAddCustomOptions)
{
    setUpTestCase();

    HipKernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    options.add("HIP_PLUGIN_TEST_INT", 42);
    options.add("HIP_PLUGIN_TEST_STRING", std::string("hello"));
    options.add("HIP_PLUGIN_TEST_BOOL", true);

    // Verify expected options
    EXPECT_TRUE(hasOption(options, "-I/opt/rocm/include"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_NHWC=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));

    // Verify custom options
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_INT=42"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_STRING=hello"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_BOOL=1"));
}

TEST_F(TestHipKernelCompileOptions, VerifiesActivationOption)
{
    setUpTestCase(DataType::HALF, TensorLayout::NHWC);

    HipKernelCompileOptions options(
        _inputTensorAttrs, _deviceProps, hip_kernel_utils::ActivationMode::RELU);

    EXPECT_TRUE(hasOption(options, "-I/opt/rocm/include"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_NHWC=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_NRN_OP_ID=3"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

};
