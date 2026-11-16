// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/KernelCompileOptions.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hip_kernel_provider::compilation;
using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::tests
{

enum class TensorLayout
{
    NCHW,
    NHWC
};

class TestKernelCompileOptions : public ::testing::Test
{
protected:
    hipDeviceProp_t _deviceProps = {};
    const TensorAttributes* _inputTensorAttrs = nullptr;
    flatbuffers::FlatBufferBuilder _fbb;
    std::unique_ptr<GraphWrapper> _graph;

    void setUpTestCase(DataType dataType = DataType::FLOAT,
                       TensorLayout layout = TensorLayout::NCHW,
                       const char* archName = "gfx942")
    {
        // Setup Device Properties
        _deviceProps.multiProcessorCount = 60;
        _deviceProps.warpSize = 64;
        std::snprintf(_deviceProps.gcnArchName, sizeof(_deviceProps.gcnArchName), "%s", archName);

        const std::vector<int64_t> dims = {1, 3, 224, 224};
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

        const std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttrs
            = {CreateTensorAttributesDirect(_fbb, 1, "tensor", dataType, &strides, &dims)};
        const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

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

class TestKernelCompileOptionsNchwFp32 : public TestKernelCompileOptions
{
};

TEST_F(TestKernelCompileOptionsNchwFp32, VerifiesOptions)
{
    setUpTestCase(DataType::FLOAT, TensorLayout::NCHW);

    const KernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_RNE_BFLOAT16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX103X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX110X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX115X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX120X=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

class TestKernelCompileOptionsNhwcBfp16 : public TestKernelCompileOptions
{
};

TEST_F(TestKernelCompileOptionsNhwcBfp16, VerifiesOptions)
{
    setUpTestCase(DataType::BFLOAT16, TensorLayout::NHWC);

    const KernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_RNE_BFLOAT16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX103X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX110X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX115X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX120X=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

TEST_F(TestKernelCompileOptions, VerifyAddCustomOptions)
{
    setUpTestCase();

    KernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    options.add("HIP_PLUGIN_TEST_INT", 42);
    options.add("HIP_PLUGIN_TEST_STRING", std::string("hello"));
    options.add("HIP_PLUGIN_TEST_BOOL", true);

    // Verify expected options
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_RNE_BFLOAT16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX103X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX110X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX115X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX120X=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));

    // Verify custom options
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_INT=42"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_STRING=hello"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_TEST_BOOL=1"));
}

TEST_F(TestKernelCompileOptions, VerifiesArchOptions)
{
    setUpTestCase(DataType::HALF, TensorLayout::NHWC, "gfx110");

    const KernelCompileOptions options(_inputTensorAttrs, _deviceProps);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_RNE_BFLOAT16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX103X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX110X=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX115X=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_GFX120X=0"));
    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx110"));
}

// Whether the defines after the two base options are in name order.
static bool definesSortedByName(const std::vector<std::string>& options)
{
    std::vector<std::string> names;
    for(size_t i = 2; i < options.size(); ++i)
    {
        names.push_back(options[i].substr(0, options[i].find('=')));
    }
    return std::is_sorted(names.begin(), names.end());
}

// The defines come out sorted by name, whatever order they went in: the option vector, and the
// code-object cache key digesting it, is a function of the define set alone.
TEST(TestKernelCompileOptionsOrder, SameDefinesInAnyOrderGiveTheSameOptions)
{
    KernelCompileOptions forward("gfx942");
    KernelCompileOptions backward("gfx942");
    const std::vector<std::pair<std::string, std::string>> defines
        = {{"CONV_BM", "128"}, {"CONV_BN", "64"}, {"A_FIRST", "1"}, {"Z_LAST", "0"}};
    for(const auto& [name, value] : defines)
    {
        forward.add(name, value);
    }
    for(auto it = defines.rbegin(); it != defines.rend(); ++it)
    {
        backward.add(it->first, it->second);
    }
    const std::vector<std::string> forwardOptions = forward;
    const std::vector<std::string> backwardOptions = backward;
    EXPECT_EQ(forwardOptions, backwardOptions);

    // The base options first, as added; then every define in name order.
    ASSERT_GE(forwardOptions.size(), 2u);
    EXPECT_EQ(forwardOptions[0], "-std=c++17");
    EXPECT_EQ(forwardOptions[1], "--offload-arch=gfx942");
    EXPECT_TRUE(definesSortedByName(forwardOptions));
}

TEST(TestKernelCompileOptionsOrder, OrderSurvivesRehashAndRemoval)
{
    // Enough defines to rehash the map several times, inserted in opposite orders, with
    // extras added and removed in between.
    constexpr int DEFINES = 64;
    KernelCompileOptions ascending("gfx1151");
    KernelCompileOptions descending("gfx1151");
    for(int i = 0; i < DEFINES; ++i)
    {
        ascending.add("DEFINE_" + std::to_string(i), i);
        descending.add("DEFINE_" + std::to_string(DEFINES - 1 - i), DEFINES - 1 - i);
        descending.add("TRANSIENT_" + std::to_string(i), i);
    }
    for(int i = 0; i < DEFINES; ++i)
    {
        descending.remove("TRANSIENT_" + std::to_string(i));
    }
    const std::vector<std::string> ascendingOptions = ascending;
    const std::vector<std::string> descendingOptions = descending;
    EXPECT_EQ(ascendingOptions, descendingOptions);
    EXPECT_TRUE(definesSortedByName(ascendingOptions));
}

};
