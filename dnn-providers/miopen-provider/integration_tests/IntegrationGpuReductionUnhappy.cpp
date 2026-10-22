// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

// Provider-local unhappy paths of the MIOpen reduction route (legacy miopenReduceTensor), run
// end to end through the frontend with only the MIOpen plugin loaded. The gate under test is
// engines/plans/MiopenReductionChecks.cpp; the execute-time guard is MiopenReductionPlan.cpp.

struct ReductionCase
{
    std::string name;
    ReductionMode mode = ReductionMode::ADD;
    bool isDeterministic = false;
    DataType xType = DataType::FLOAT;
    DataType yType = DataType::FLOAT;
    DataType computeType = DataType::FLOAT;
    std::vector<int64_t> xDims = {2, 4, 8, 8};
    std::vector<int64_t> yDims = {2, 4, 1, 1};
    // Empty means packed.
    std::vector<int64_t> xStrides;
};

// fp32 ADD over the two spatial axes of a packed NCHW tensor: the default MIOpen engine
// serves it, so each decline case below differs from a supported graph in one property only.
ReductionCase plainAdd(const std::string& name)
{
    ReductionCase reductionCase;
    reductionCase.name = name;
    return reductionCase;
}

struct ReductionGraph
{
    Graph graph;
    std::shared_ptr<TensorAttributes> x;
    std::shared_ptr<TensorAttributes> y;
};

ReductionGraph makeReductionGraph(const ReductionCase& reductionCase)
{
    ReductionGraph result;
    result.graph.set_name("IntegrationGpuReductionUnhappy");
    result.graph.set_compute_data_type(reductionCase.computeType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(reductionCase.xType);

    const auto xStrides = reductionCase.xStrides.empty() ? generateStrides(reductionCase.xDims)
                                                         : reductionCase.xStrides;
    result.x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", reductionCase.xType, reductionCase.xDims, xStrides));
    auto y = std::make_shared<TensorAttributes>(makeTensorAttributes(
        "y", reductionCase.yType, reductionCase.yDims, generateStrides(reductionCase.yDims)));

    ReductionAttributes attrs;
    attrs.set_mode(reductionCase.mode).set_is_deterministic(reductionCase.isDeterministic);

    result.y = result.graph.reduction(result.x, y, attrs);
    result.y->set_output(true);
    return result;
}

std::vector<ReductionCase> getDeclinedCases()
{
    std::vector<ReductionCase> cases;

    // MIOPEN_REDUCE_TENSOR_MUL_NO_ZEROS is commented out of miopenReduceTensorOp_t.
    auto mulNoZeros = plainAdd("MulNoZeros");
    mulNoZeros.mode = ReductionMode::MUL_NO_ZEROS;
    cases.push_back(mulNoZeros);

    // MIOpen documents no determinism guarantee for miopenReduceTensor.
    auto deterministic = plainAdd("IsDeterministic");
    deterministic.isDeterministic = true;
    cases.push_back(deterministic);

    // The legacy CK reduce kernels have no bf16 instantiation; hipRTC would fail at execute.
    auto bfloat16Io = plainAdd("Bfloat16Io");
    bfloat16Io.xType = DataType::BFLOAT16;
    bfloat16Io.yType = DataType::BFLOAT16;
    cases.push_back(bfloat16Io);

    // MIOpen only validates reductions whose input and output types match.
    auto mixedTypes = plainAdd("HalfInputFloatOutput");
    mixedTypes.xType = DataType::HALF;
    mixedTypes.yType = DataType::FLOAT;
    cases.push_back(mixedTypes);

    // The plan always passes miopenFloat as the reduce compute type, so honouring a HALF
    // compute type would silently compute in fp32.
    auto halfCompute = plainAdd("HalfCompute");
    halfCompute.xType = DataType::HALF;
    halfCompute.yType = DataType::HALF;
    halfCompute.computeType = DataType::HALF;
    cases.push_back(halfCompute);

    // miopenReduceTensor takes at most six dimensions.
    auto rank7 = plainAdd("Rank7");
    rank7.xDims = {2, 2, 2, 2, 2, 2, 2};
    rank7.yDims = {1, 2, 2, 2, 2, 2, 2};
    cases.push_back(rank7);

    // A broadcast (stride 0) input has no MIOpen tensor descriptor representation.
    auto zeroStride = plainAdd("ZeroStrideInput");
    zeroStride.xStrides = {0, 64, 8, 1};
    cases.push_back(zeroStride);

    // Legacy reduce addresses memory through 32-bit byte offsets; this four-element input spans
    // about 4 GiB. Nothing is allocated: the graph is only built.
    auto int32Span = plainAdd("InputSpanAboveInt32Max");
    int32Span.xDims = {2, 2};
    int32Span.xStrides = {int64_t{1} << 30, 1};
    int32Span.yDims = {1, 2};
    cases.push_back(int32Span);

    return cases;
}

// Loads only the MIOpen plugin, so "no engine" can only mean the MIOpen provider declined.
class MiopenOnlyFixture : public ::testing::Test
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

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
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
};

class IntegrationGpuReductionUnhappy : public MiopenOnlyFixture,
                                       public ::testing::WithParamInterface<ReductionCase>
{
};

using IntegrationGpuReductionUnhappyEngine = MiopenOnlyFixture;

size_t elementCount(const std::vector<int64_t>& dims)
{
    return static_cast<size_t>(
        std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
}

bool containsEngine(const std::vector<int64_t>& engineIds, int64_t engineId)
{
    return std::find(engineIds.begin(), engineIds.end(), engineId) != engineIds.end();
}

} // namespace

TEST_P(IntegrationGpuReductionUnhappy, DeclinesGraph)
{
    auto reduction = makeReductionGraph(GetParam());

    // The frontend validates the graph and the backend accepts its descriptors, so the
    // GRAPH_NOT_SUPPORTED below comes from engine selection, not from validation.
    auto result = reduction.graph.build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = reduction.graph.build(_handle);
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED) << "err_msg: " << result.err_msg;
    EXPECT_FALSE(result.err_msg.empty()) << "err_msg is empty";
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuReductionUnhappy,
                         ::testing::ValuesIn(getDeclinedCases()),
                         [](const ::testing::TestParamInfo<ReductionCase>& info) {
                             return info.param.name;
                         });

// Control for the decline table: the unmodified graph builds on the default MIOpen engine.
TEST_F(IntegrationGpuReductionUnhappyEngine, PlainAddBuildsOnDefaultEngine)
{
    auto reduction = makeReductionGraph(plainAdd("PlainAdd"));

    auto result = reduction.graph.build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t engineId = 0;
    result = reduction.graph.get_execution_plan_engine_id(engineId);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(engineId, MIOPEN_ENGINE_ID);
}

// MiopenContainer registers no reduction plan builder on MIOPEN_ENGINE_DETERMINISTIC, so that
// engine is never applicable to a reduction, not even one the default engine serves.
TEST_F(IntegrationGpuReductionUnhappyEngine, DeterministicEngineIsNotApplicable)
{
    auto reduction = makeReductionGraph(plainAdd("PlainAdd"));
    reduction.graph.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

    auto result = reduction.graph.build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = reduction.graph.get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_TRUE(containsEngine(rankedEngineIds, MIOPEN_ENGINE_ID));
    EXPECT_FALSE(containsEngine(rankedEngineIds, MIOPEN_ENGINE_DETERMINISTIC_ID))
        << "MIOPEN_ENGINE_DETERMINISTIC must not claim a reduction graph";

    // Pinning the engine outright must fail as well.
    result = reduction.graph.create_execution_plan_ext(MIOPEN_ENGINE_DETERMINISTIC_ID, {});
    EXPECT_NE(result.code, ErrorCode::OK) << "err_msg: " << result.err_msg;
}

// The gate's same-uid check cannot be reached through the frontend: ReductionNode requires some
// Y dim to be smaller than X's, so X and Y can never be one tensor. Two distinct uids bound to
// overlapping memory are only visible at execute, where the plan must refuse them because the
// legacy kernels take A and C as __restrict__ pointers.
TEST_F(IntegrationGpuReductionUnhappyEngine, OverlappingBuffersRejectedAtExecute)
{
    const auto reductionCase = plainAdd("PlainAdd");
    auto reduction = makeReductionGraph(reductionCase);

    auto result = reduction.graph.build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t workspaceSize = 0;
    result = reduction.graph.get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace workspace(static_cast<size_t>(workspaceSize));

    // x is all ones, so every output element sums xCount / yCount ones.
    constexpr float INPUT_VALUE = 1.0f;
    const auto xCount = elementCount(reductionCase.xDims);
    const auto yCount = elementCount(reductionCase.yDims);
    const auto reducedCount = xCount / yCount;
    const auto expectedSum = static_cast<float>(reducedCount) * INPUT_VALUE;

    MigratableMemory<float> x(xCount);
    std::fill_n(x.hostData(), xCount, INPUT_VALUE);
    x.markHostModified();
    MigratableMemory<float> y(yCount);
    std::fill_n(y.hostData(), yCount, 0.0f);
    y.markHostModified();

    auto* xDevice = static_cast<float*>(x.deviceData());
    const auto xUid = reduction.x->get_uid();
    const auto yUid = reduction.y->get_uid();

    // Control: with separate buffers the same plan executes and reduces correctly, so the
    // failures below are the overlap guard and not a broken plan.
    std::unordered_map<int64_t, void*> variantPack = {{xUid, xDevice}, {yUid, y.deviceData()}};
    result = reduction.graph.execute(_handle, variantPack, workspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
    y.markDeviceModified();
    const auto* yHost = y.hostData();
    for(size_t i = 0; i < yCount; ++i)
    {
        EXPECT_FLOAT_EQ(yHost[i], expectedSum) << "y[" << i << "]";
    }

    // y bound to the start of x's buffer, then to its middle: both overlap x's span.
    for(const size_t yOffset : {size_t{0}, xCount / 2})
    {
        SCOPED_TRACE("y bound at element offset " + std::to_string(yOffset) + " of x's buffer");

        std::unordered_map<int64_t, void*> overlappingPack
            = {{xUid, xDevice}, {yUid, xDevice + yOffset}};
        result = reduction.graph.execute(_handle, overlappingPack, workspace.get());
        EXPECT_NE(result.code, ErrorCode::OK) << "err_msg: " << result.err_msg;
        EXPECT_NE(result.err_msg.find("overlap"), std::string::npos)
            << "err_msg: " << result.err_msg;

        // Rejected before launch: x, which y aliases, still holds its input.
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        x.markDeviceModified();
        const auto* xHost = x.hostData();
        const auto modified = std::count_if(
            xHost, xHost + xCount, [](float value) { return value != INPUT_VALUE; });
        EXPECT_EQ(modified, 0) << "elements of x were overwritten";
    }
}
