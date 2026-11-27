// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormApplicabilityChecks.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormBwdKernel.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormBwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(LayernormBwdParams(*attrs, graph.getTensorMap()));
}

TEST(TestLayernormBwdParams, HasCorrectTensorPointersForSingleNode)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormBwdParams params(*attrs, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_NE(params.epsilonValue(nullptr, 0), std::nullopt);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0).value(), 1e-5, 1e-10);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
}

TEST(TestLayernormBwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormBwdParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attrs->dy_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attrs->x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrs->scale_tensor_uid());
    EXPECT_EQ(params.mean()->uid(), attrs->mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attrs->inv_variance_tensor_uid());
    EXPECT_NE(params.epsilonValue(nullptr, 0), std::nullopt);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0).value(), 1e-5, 1e-10);
    EXPECT_EQ(params.dx()->uid(), attrs->dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attrs->dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attrs->dbias_tensor_uid());
}

TEST(TestLayernormBwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormBwdParams params(attr, graph.getTensorMap());
    const LayernormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.dx(), nullptr);
}

TEST(TestLayernormBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormBwdParams>);
}

namespace
{

/// The CU count the builds below assume: 4 * 1024 * 60 = 245760 work-items keep a device busy.
constexpr int MULTI_PROCESSOR_COUNT = 60;

std::pair<flatbuffers::FlatBufferBuilder, LayernormBwdParams>
    createParamsFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                          const std::vector<int64_t>& dims = {1, 3, 224, 224},
                          hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                          = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                          bool optionalTensors = true)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        strides, dims, optionalTensors, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormBwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

/// The default configuration on MULTI_PROCESSOR_COUNT CUs.
LayernormBwdConfig defaultConfig(const LayernormBwdParams& params)
{
    return defaultLayernormBwdConfig(params, MULTI_PROCESSOR_COUNT);
}

/// What buildLayernormBwdKernel() makes for @p params on gfx942 with MULTI_PROCESSOR_COUNT CUs,
/// in the default configuration.
LayernormKernelBuild defaultBuild(const LayernormBwdParams& params)
{
    return buildLayernormBwdKernel(params, "gfx942", MULTI_PROCESSOR_COUNT, defaultConfig(params));
}

/// The workspace the build of the default configuration takes on MULTI_PROCESSOR_COUNT CUs.
size_t defaultWorkspaceBytes(const LayernormBwdParams& params)
{
    return layernormBwdWorkspaceBytes(params, MULTI_PROCESSOR_COUNT, defaultConfig(params));
}

bool hasOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestLayernormBwdKernel, WorkspaceSizeIsZeroWithStatisticsAndOnePassReduction)
{
    auto [fbb, params] = createParamsFromGraph();
    EXPECT_EQ(defaultWorkspaceBytes(params), 0u);
}

TEST(TestLayernormBwdKernel, WorkspaceSizeParallelIsNonzero)
{
    // The answer depends on the number of multiprocessors
    constexpr auto multiProcessorCount = static_cast<int64_t>(MULTI_PROCESSOR_COUNT);
    auto [fbb, params]
        = createParamsFromGraph({3072 * multiProcessorCount, 1024 * multiProcessorCount, 32, 1},
                                {4, 3, 32 * multiProcessorCount, 32});

    // The answer is `sizeof(float) * 2 * innerSize * parallelSize` = `4 * 2 * (32 * 32 * 3 * multiProcessorCount) * 2` = `49152 * multiProcessorCount`
    EXPECT_EQ(defaultWorkspaceBytes(params), 49152u * multiProcessorCount);
}

TEST(TestLayernormBwdKernel, WorkspaceSizeWithoutOptionalTensorsIsNonzero)
{
    const std::vector<int64_t> strides{150528, 50176, 224, 1};
    const std::vector<int64_t> dims{2, 3, 224, 224};
    auto [fbbWithStats, paramsWithStats] = createParamsFromGraph(
        strides, dims, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, true);
    auto [fbbWithoutStats, paramsWithoutStats] = createParamsFromGraph(
        strides, dims, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, false);

    // Both graphs describe the same shape, so they share whatever parallel reduction buffer the
    // multiprocessor count calls for. Their difference is the fallback mean/rstd workspace on
    // its own: `sizeof(float) * 2 * outerSize * stride` = `4 * 2 * 2 * 1` = `16`
    EXPECT_EQ(defaultWorkspaceBytes(paramsWithoutStats) - defaultWorkspaceBytes(paramsWithStats),
              16u);
}

TEST(TestLayernormBwdKernel, BuildLaunchesTheLayernormBwdEntryPoints)
{
    auto [fbb, params] = createParamsFromGraph();

    const auto build = defaultBuild(params);
    ASSERT_EQ(build.launches.size(), 2u);
    EXPECT_STREQ(build.launches[0].entryPoint, "LayernormBwd");
    EXPECT_STREQ(build.launches[1].entryPoint, "LayernormBwdScaleBias");

    const auto shapes
        = layernormBwdLaunchShapes(params, MULTI_PROCESSOR_COUNT, defaultConfig(params));
    ASSERT_EQ(shapes.size(), 2u);
    EXPECT_STREQ(shapes[0].kernelName, "LayernormBwd");
    EXPECT_STREQ(shapes[1].kernelName, "LayernormBwdScaleBias");
}

TEST(TestLayernormBwdKernel, BuildIncludesOffloadArchOption)
{
    auto [fbb, params] = createParamsFromGraph();

    EXPECT_TRUE(hasOption(defaultBuild(params).options, "--offload-arch=gfx942"));
}

TEST(TestLayernormBwdKernelFp32, BuildSetsCorrectDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const std::vector<std::string> options = defaultBuild(params).options;

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdKernelFp32, BuildSetsCorrectDefinesParallel)
{
    auto [fbb, params] = createParamsFromGraph({768, 256, 16, 1}, {1024, 3, 16, 16});
    const auto build = defaultBuild(params);
    const std::vector<std::string> options = build.options;

    ASSERT_EQ(build.launches.size(), 3u);
    EXPECT_STREQ(build.launches[1].entryPoint, "LayernormBwdScaleBiasParallel");
    EXPECT_STREQ(build.launches[2].entryPoint, "LayernormBwdScaleBiasReduceSum");
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdKernelFp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const std::vector<std::string> options = defaultBuild(params).options;

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdKernelFp16, BuildSetsCorrectDefinesParallel)
{
    auto [fbb, params] = createParamsFromGraph(
        {768, 256, 16, 1}, {1024, 3, 16, 16}, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const std::vector<std::string> options = defaultBuild(params).options;

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdKernelBfp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const std::vector<std::string> options = defaultBuild(params).options;

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdKernelBfp16, BuildSetsCorrectDefinesParallel)
{
    auto [fbb, params]
        = createParamsFromGraph({768, 256, 16, 1},
                                {1024, 3, 16, 16},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const std::vector<std::string> options = defaultBuild(params).options;

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdKernel, BuildWithUnsupportedDimensionThrows)
{
    // 3D tensor is not supported
    auto [fbb, params] = createParamsFromGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    EXPECT_THROW(defaultBuild(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormBwdKernel, BuildWithThreadCountAtLimitThrows)
{
    // 4194304 blocks of 1024 threads reach 2^32 threads
    auto [fbb, params] = createParamsFromGraph({4, 1, 1, 1}, {4194304, 4, 1, 1});

    EXPECT_THROW(defaultBuild(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// buildLayernormBwdKernel - explicit performance configuration
// ============================================================================

namespace
{

// A layernorm backward graph with mean and inverse variance, and its params, which point into
// the graph's buffer.
struct LayernormBwdGraph
{
    flatbuffers::FlatBufferBuilder builder;
    LayernormBwdParams params;
};

LayernormBwdGraph createBwdGraph(const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dims)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        strides, dims, true, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    LayernormBwdParams params(*graph.getNode(0).attributes_as_LayernormBackwardAttributes(),
                              graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

bool hasBuildOption(const LayernormKernelBuild& build, const std::string& option)
{
    const std::vector<std::string> options = build.options;
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestLayernormBwdKernelConfig, DefaultConfigIsTheHeuristicChoice)
{
    // 60 CUs keep 4 * 1024 * 60 = 245760 work-items busy: an inner size of 768 over 1024 rows
    // splits its reduction, an inner size of 150528 over one row does not
    const auto parallel = createBwdGraph({768, 256, 16, 1}, {1024, 3, 16, 16});
    EXPECT_EQ(defaultLayernormBwdConfig(parallel.params, 60), (LayernormBwdConfig{true}));

    const auto onePass = createBwdGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    EXPECT_EQ(defaultLayernormBwdConfig(onePass.params, 60), (LayernormBwdConfig{false}));
}

TEST(TestLayernormBwdKernelConfig, BuildTakesTheReductionOfTheConfig)
{
    const auto graph = createBwdGraph({768, 256, 16, 1}, {1024, 3, 16, 16});

    const auto split = buildLayernormBwdKernel(graph.params, "gfx942", 60, {true});
    EXPECT_TRUE(hasBuildOption(split, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
    ASSERT_EQ(split.launches.size(), 3u);
    EXPECT_STREQ(split.launches[1].entryPoint, "LayernormBwdScaleBiasParallel");
    EXPECT_STREQ(split.launches[2].entryPoint, "LayernormBwdScaleBiasReduceSum");

    const auto onePass = buildLayernormBwdKernel(graph.params, "gfx942", 60, {false});
    EXPECT_TRUE(hasBuildOption(onePass, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
    ASSERT_EQ(onePass.launches.size(), 2u);
    EXPECT_STREQ(onePass.launches[1].entryPoint, "LayernormBwdScaleBias");
    EXPECT_EQ(onePass.launches[1].gridSize, 1u);
}

TEST(TestLayernormBwdKernelConfig, ParallelReductionAppliesToEveryShape)
{
    // One row: the parallel reduction runs with a parallel size of 1
    const auto graph = createBwdGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    EXPECT_TRUE(isBwdConfigApplicable(graph.params, 60, {true}));
    EXPECT_TRUE(isBwdConfigApplicable(graph.params, 60, {false}));

    const auto build = buildLayernormBwdKernel(graph.params, "gfx942", 60, {true});
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=1"));
    ASSERT_EQ(build.launches.size(), 3u);
}

TEST(TestLayernormBwdKernelConfig, ParallelReductionDeclinedWhenItsGridReachesThreadLimit)
{
    // 2^21 rows of 2^22: the main kernel launches 2^21 blocks of 1024 threads and the one-pass
    // reduction 4096, so the graph's launches fit
    const auto graph = createBwdGraph({4194304, 1, 1, 1}, {2097152, 4194304, 1, 1});
    EXPECT_TRUE(isBwdConfigApplicable(graph.params, 60, {false}));
    // On 60 CUs a row exceeds one parallel pass, so the parallel size is 1 and the reduction
    // launches 4096 blocks as well
    EXPECT_TRUE(isBwdConfigApplicable(graph.params, 60, {true}));

    // On 2^22 CUs a pass keeps 2^34 work-items busy and the parallel size grows to 2048, bounded
    // by the square root of the row count: 2048 * 2^22 = 2^33 threads
    EXPECT_TRUE(isBwdConfigApplicable(graph.params, 1 << 22, {false}));
    EXPECT_FALSE(isBwdConfigApplicable(graph.params, 1 << 22, {true}));
    EXPECT_THROW(buildLayernormBwdKernel(graph.params, "gfx942", 1 << 22, {true}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormBwdKernelConfig, WorkspaceFollowsTheConfig)
{
    const auto graph = createBwdGraph({768, 256, 16, 1}, {1024, 3, 16, 16});

    // The graph supplies mean and inverse variance, so only the partial sums take workspace:
    // `sizeof(float) * 2 * innerSize * parallelSize` = `4 * 2 * 768 * 32` = `196608`
    EXPECT_EQ(layernormBwdWorkspaceBytes(graph.params, 60, {true}), 196608u);
    EXPECT_EQ(layernormBwdWorkspaceBytes(graph.params, 60, {false}), 0u);
}
