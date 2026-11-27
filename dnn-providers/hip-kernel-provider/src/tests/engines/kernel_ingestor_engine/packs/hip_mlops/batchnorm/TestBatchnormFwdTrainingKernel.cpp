// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/batchnorm/BatchnormFwdTrainingKernel.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "../TestPlanCommon.hpp"

namespace hip_kernel_provider::batchnorm::test
{

// ============================================================================
// BatchnormFwdTrainingParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdTrainingParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    EXPECT_NO_THROW(const BatchnormFwdTrainingParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdTrainingParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    const BatchnormFwdTrainingParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    const BatchnormFwdTrainingParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.mean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attr.inv_variance_tensor_uid());
}

TEST(TestBatchnormFwdTrainingParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());
    const BatchnormFwdTrainingParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(nullptr, 0), 1e-5, 1e-10);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingParams, LeavesSavedStatisticsUnsetWhenBothVirtual)
{
    auto base = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, true);
    auto builder = hipdnn_test_sdk::utilities::mutateGraph(
        base, [](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
            auto* attrs = graph.nodes.at(0)->attributes.AsBatchnormAttributes();
            for(auto& tensor : graph.tensors)
            {
                if(tensor->uid == attrs->mean_tensor_uid.value()
                   || tensor->uid == attrs->inv_variance_tensor_uid.value())
                {
                    tensor->virtual_ = true;
                }
            }
        });
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
    EXPECT_EQ(params.mean(), nullptr);
    EXPECT_EQ(params.invVariance(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}

TEST(TestBatchnormFwdTrainingParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdTrainingParams>);
}

// ============================================================================
// buildBatchnormFwdTrainingKernel - helpers
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, BatchnormFwdTrainingParams>
    createParamsFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                          const std::vector<int64_t>& dims = {1, 3, 224, 224},
                          bool withMeanVariance = true)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        strides, dims, withMeanVariance);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

// The kernel build for @p params on a device of @p deviceProps with the default configuration.
BatchnormKernelBuild defaultBuild(const BatchnormFwdTrainingParams& params,
                                  const hipDeviceProp_t& deviceProps)
{
    return buildBatchnormFwdTrainingKernel(
        params,
        std::string(deviceProps.gcnArchName),
        deviceProps.multiProcessorCount,
        deviceProps.warpSize,
        defaultBatchnormFwdTrainingConfig(params, deviceProps.multiProcessorCount));
}

// Whether @p build compiles with @p option.
bool hasBuildOption(const BatchnormKernelBuild& build, const std::string& option)
{
    const std::vector<std::string> options = build.options;
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

// ============================================================================
// buildBatchnormFwdTrainingKernel - default configuration
// ============================================================================

TEST(TestBatchnormFwdTrainingKernel, BuildLaunchesTheSingleKernelOfVariant1)
{
    auto [fbb, params] = createParamsFromGraph();
    const auto build = defaultBuild(params, createTestDeviceProps());

    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=1"));
    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormFwdTrainSpatial");
}

TEST(TestBatchnormFwdTrainingKernel, BuildLaunchesTheThreeKernelsOfVariant2)
{
    auto [fbb, params] = createParamsFromGraph({147456, 4608, 576, 1}, {32, 8, 24, 24}, true);
    const auto build = defaultBuild(params, createTestDeviceProps());

    ASSERT_EQ(build.launches.size(), 3u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormFwdTrainSpatialMeanVariance");
    EXPECT_STREQ(build.launches[1].entryPoint, "BatchNormFwdTrainSpatialFinalMeanVariance");
    EXPECT_STREQ(build.launches[2].entryPoint, "BatchNormFwdTrainSpatialNorm");
}

TEST(TestBatchnormFwdTrainingKernel, BuildIncludesOffloadArchOption)
{
    auto [fbb, params] = createParamsFromGraph();
    EXPECT_TRUE(hasBuildOption(defaultBuild(params, createTestDeviceProps("gfx942")),
                               "--offload-arch=gfx942"));
}

TEST(TestBatchnormFwdTrainingKernel, BuildDefaultSetsCorrectDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPUPCAST=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPUPCAST=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_RUNNING_RESULT=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCHES=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHANNELS=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SPATIAL_SIZE=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS0=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCH_ELEMENTS=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_BATCH=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
}

// NOTE: Missing tests for FP16 and BFP16 mix options since createValidBatchnormFwdTrainingGraph
// currently creates graphs only with FP32 input data type. We can add tests for those options
// once we have graph builders that can create graphs with FP16/BFP16 input tensors.

TEST(TestBatchnormFwdTrainingKernel, BuildLdsGcnSizeScalesWithWaveSize)
{
    auto [fbb, params] = createParamsFromGraph();
    auto deviceProps = createTestDeviceProps("gfx1201");
    deviceProps.warpSize = 32;
    const auto build = defaultBuild(params, deviceProps);

    // Variant 1 launches 1024 threads, so a wave32 device has twice as many waves as wave64.
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDSGCN_SIZE=32"));
}

TEST(TestBatchnormFwdTrainingKernel, BuildNhwcLdsGcnSizeScalesWithWaveSize)
{
    for(const int waveSize : {32, 64})
    {
        auto [fbb, params] = createParamsFromGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
        auto deviceProps = createTestDeviceProps();
        deviceProps.warpSize = waveSize;
        const auto build = defaultBuild(params, deviceProps);

        // NHWC selects variant 2, whose LDS sizing is a separate code path from variant 1.
        EXPECT_EQ(build.launches.size(), 3u);
        EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=2"));
        EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
        EXPECT_TRUE(
            hasBuildOption(build, "-DHIP_PLUGIN_BN_LDSGCN_SIZE=" + std::to_string(1024 / waveSize)))
            << "waveSize=" << waveSize;
    }
}

TEST(TestBatchnormFwdTrainingKernel, BuildNchwLayoutSetsCorrectDefine)
{
    // NCHW strides: N=C*H*W, C=H*W, H=W, W=1
    auto [fbb, params] = createParamsFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=0"));
}

TEST(TestBatchnormFwdTrainingKernel, BuildNhwcLayoutSetsCorrectDefine)
{
    // NHWC strides: N=H*W*C, H=W*C, W=C, C=1
    auto [fbb, params] = createParamsFromGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    const auto build = defaultBuild(params, createTestDeviceProps());

    EXPECT_EQ(build.launches.size(), 3u);
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=1"));
}

TEST(TestBatchnormFwdTrainingKernel, HasNoSaveStatsSetsCorrectDefines)
{
    // NCHW strides: N=C*H*W, C=H*W, H=W, W=1
    auto [fbb, params] = createParamsFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=0"));
}

TEST(TestBatchnormFwdTrainingKernel, BuildWithUnsupportedDimensionThrows)
{
    // 2D tensor is not supported
    auto [fbb, params] = createParamsFromGraph({4, 1}, {1, 4});

    EXPECT_THROW(static_cast<void>(defaultBuild(params, createTestDeviceProps())),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// buildBatchnormFwdTrainingKernel - explicit performance configuration
// ============================================================================

namespace
{

// A training graph and the params of its batchnorm node, which point into the graph's buffer.
struct TrainingGraph
{
    flatbuffers::FlatBufferBuilder builder;
    BatchnormFwdTrainingParams params;
};

// Every tensor of the graph is fp32 unless xDataType makes x another type, which upcasts it to
// the fp32 y.
TrainingGraph createTrainingGraph(const std::vector<int64_t>& strides,
                                  const std::vector<int64_t>& dims,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType xDataType
                                  = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(strides, dims, true);
    if(xDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        auto* mutableGraph
            = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
        const auto xUid
            = mutableGraph->nodes()->Get(0)->attributes_as_BatchnormAttributes()->x_tensor_uid();
        auto* tensors = mutableGraph->mutable_tensors();
        for(flatbuffers::uoffset_t i = 0; i < tensors->size(); ++i)
        {
            auto* tensor = tensors->GetMutableObject(i);
            if(tensor->uid() == xUid)
            {
                EXPECT_TRUE(tensor->mutate_data_type(xDataType));
            }
        }
    }
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    BatchnormFwdTrainingParams params(*graph.getNode(0).attributes_as_BatchnormAttributes(),
                                      graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

} // namespace

TEST(TestBatchnormFwdTrainingKernelConfig, DefaultConfigIsTheHeuristicChoice)
{
    // NCHW, one batch: spatial single variant 1 of 1024 threads
    const auto nchw = createTrainingGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    EXPECT_EQ(defaultBatchnormFwdTrainingConfig(nchw.params, 60),
              (BatchnormSpatialConfig{1, 1024, 1}));

    // NHWC with C = 3: spatial multiple with vector size 1
    const auto nhwc = createTrainingGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    EXPECT_EQ(defaultBatchnormFwdTrainingConfig(nhwc.params, 60),
              (BatchnormSpatialConfig{2, 0, 1}));

    // NCHW, 32 batches: spatial multiple keeps the smallest applicable vector size
    const auto multiple = createTrainingGraph({147456, 4608, 576, 1}, {32, 8, 24, 24});
    EXPECT_EQ(defaultBatchnormFwdTrainingConfig(multiple.params, 60),
              (BatchnormSpatialConfig{2, 0, 1}));
}

// Spatial multiple is not preferred for an fp16 x upcast to an fp32 y with N <= 768, nor for
// fp32 with N <= 2, and a spatial size above 1024 takes variant 1 even once a channel holds
// 2^25 values (N * H * W), where the spatial single fallback once picked variant 0, whose
// 1024 threads cover only 1024 of the spatial positions.
// IntegrationGpuBatchnormTrainingSpatialBeyondWorkgroup runs these graphs on a device.
TEST(TestBatchnormFwdTrainingKernelConfig, DefaultConfigForASpatialSizeBeyondTheWorkgroup)
{
    const auto upcast = createTrainingGraph({2097152, 1048576, 1024, 1},
                                            {32, 2, 1024, 1024},
                                            hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const auto upcastConfig = defaultBatchnormFwdTrainingConfig(upcast.params, 60);
    EXPECT_EQ(upcastConfig, (BatchnormSpatialConfig{1, 1024, 1}));
    EXPECT_TRUE(isBatchnormFwdTrainingConfigApplicable(upcast.params, 60, upcastConfig));

    const auto fp32 = createTrainingGraph({33554432, 16777216, 4096, 1}, {2, 2, 4096, 4096});
    const auto fp32Config = defaultBatchnormFwdTrainingConfig(fp32.params, 60);
    EXPECT_EQ(fp32Config, (BatchnormSpatialConfig{1, 1024, 1}));
    EXPECT_TRUE(isBatchnormFwdTrainingConfigApplicable(fp32.params, 60, fp32Config));
}

TEST(TestBatchnormFwdTrainingKernelConfig, BuildTakesTheWorkgroupSizeOfTheConfig)
{
    const auto graph = createTrainingGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});

    const auto build = buildBatchnormFwdTrainingKernel(graph.params, "gfx942", 60, 64, {1, 256, 1});

    EXPECT_EQ(build.variant, 1);
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRP0=256"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDS_SIZE=256"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDSGCN_SIZE=4"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRPS0=3"));
    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormFwdTrainSpatial");
    EXPECT_EQ(build.launches[0].blockSize, (std::array<unsigned int, 3>{256, 1, 1}));
    EXPECT_EQ(build.launches[0].gridSize, (std::array<unsigned int, 3>{3, 1, 1}));
}

TEST(TestBatchnormFwdTrainingKernelConfig, BuildTakesTheVectorSizeOfTheConfig)
{
    const auto graph = createTrainingGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});

    const auto build = buildBatchnormFwdTrainingKernel(graph.params, "gfx942", 60, 64, {2, 0, 4});

    // NCHW vectorizes the spatial dimension: 50176 / 4 values over workgroups of 1024
    EXPECT_EQ(build.variant, 2);
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=2"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRP0=1"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRP1=1024"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRPS1=13"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    ASSERT_EQ(build.launches.size(), 3u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormFwdTrainSpatialMeanVariance");
    EXPECT_STREQ(build.launches[1].entryPoint, "BatchNormFwdTrainSpatialFinalMeanVariance");
    EXPECT_STREQ(build.launches[2].entryPoint, "BatchNormFwdTrainSpatialNorm");
    EXPECT_EQ(build.launches[0].gridSize, (std::array<unsigned int, 3>{3, 13, 1}));
}

TEST(TestBatchnormFwdTrainingKernelConfig, ApplicabilityFollowsTheProblem)
{
    const auto graph = createTrainingGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});

    EXPECT_TRUE(isBatchnormFwdTrainingConfigApplicable(graph.params, 60, {1, 256, 1}));
    EXPECT_TRUE(isBatchnormFwdTrainingConfigApplicable(graph.params, 60, {2, 0, 4}));
    // Variants 0 and 3 give each thread one spatial position: 50176 do not fit 1024 threads
    EXPECT_FALSE(isBatchnormFwdTrainingConfigApplicable(graph.params, 60, {0, 1024, 1}));
    EXPECT_FALSE(isBatchnormFwdTrainingConfigApplicable(graph.params, 60, {3, 1024, 1}));
}

} // namespace hip_kernel_provider::batchnorm::test
