// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cuDNN v8 cudnnGetPlan / cudnnFindPlan on a graph built with the v0.x builders, across
// test plugins with one engine that fails at context creation and one that fails at
// execute, against the real hipDNN backend. Enumeration and timing need a device, so
// these tests skip on device-less hosts.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace plugin_constants = hipdnn_tests::plugin_constants;
using hipdnn_shim_v8_test::DeviceBuffer;
using hipdnn_shim_v8_test::jsonEngineId;
using hipdnn_shim_v8_test::lastError;

constexpr fe::CudnnFindSamplingTechnique K_SAMPLE_ONCE
    = fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE;

constexpr std::int64_t K_X_UID = 1;
constexpr std::int64_t K_Y_UID = 2;
const std::vector<std::int64_t> K_DIMS = {2, 4, 8, 8};
constexpr std::size_t K_TENSOR_BYTES = std::size_t{2} * 4 * 8 * 8 * sizeof(float);

fe::Tensor tensor(std::int64_t uid)
{
    const std::vector<std::int64_t> strides = hipdnn_shim_v8_test::packedStrides(K_DIMS);
    return fe::TensorBuilder()
        .setDataType(fe::DataType_t::FLOAT)
        .setDim(static_cast<std::int64_t>(K_DIMS.size()), K_DIMS.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .build();
}

fe::EngineConfigList instantHeuristics(fe::OperationGraph& opGraph)
{
    auto heuristics = fe::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
}

std::vector<std::string> engineIds(const fe::executionPlans_t& plans)
{
    std::vector<std::string> ids;
    ids.reserve(plans.size());
    for(const fe::ExecutionPlan& plan : plans)
    {
        ids.push_back(jsonEngineId(plan.getJsonRepresentation()));
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<std::string> sortedIds(const std::vector<std::int64_t>& ids)
{
    std::vector<std::string> result;
    result.reserve(ids.size());
    for(const std::int64_t id : ids)
    {
        result.push_back(std::to_string(id));
    }
    std::sort(result.begin(), result.end());
    return result;
}

class IntegrationCudnnShimV8FindPlan : public hipdnn_tests::IntegrationTestFixture
{
protected:
    static void SetUpTestSuite()
    {
        const std::array<const char*, 1> heuristicPaths
            = {plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        sPolicyOrderEnv.emplace("HIPDNN_HEUR_POLICY_ORDER",
                                plugin_constants::testGoodHeuristicPolicyName());
    }

    static void TearDownTestSuite()
    {
        sPolicyOrderEnv.reset();
        const std::array<const char*, 1> heuristicPaths
            = {plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testCudnnShimV8PluginPath()};
    }

    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        const fe::Tensor x = tensor(K_X_UID);
        const fe::Tensor y = tensor(K_Y_UID);
        const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                           .setMode(fe::PointwiseMode_t::RELU_FWD)
                                           .setComputeType(fe::DataType_t::FLOAT)
                                           .build();
        const fe::Operation op
            = fe::OperationBuilder(fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                  .setpwDesc(relu)
                  .setxDesc(x)
                  .setyDesc(y)
                  .build();
        std::array<const fe::Operation*, 1> ops = {&op};
        _opGraph.emplace(fe::OperationGraphBuilder()
                             .setHandle(_handle)
                             .setOperationGraph(static_cast<std::int64_t>(ops.size()), ops.data())
                             .build());
    }

    // Plans and the graph own plugin state, so they go before the handle that loaded the plugin.
    void TearDown() override
    {
        _variantPack.reset();
        _buffers.clear();
        _opGraph.reset();
        IntegrationTestFixture::TearDown();
    }

    fe::executionPlans_t getPlans()
    {
        return _generator.cudnnGetPlan(_handle, *_opGraph);
    }

    // Sized for the largest workspace any of the graph's plans asks for.
    const fe::VariantPack& variantPack()
    {
        std::int64_t workspaceBytes = 0;
        for(const fe::ExecutionPlan& plan : getPlans())
        {
            workspaceBytes = std::max(workspaceBytes, plan.getWorkspaceSize());
        }
        void* x = allocate(K_TENSOR_BYTES);
        void* y = allocate(K_TENSOR_BYTES);
        void* workspace = allocate(static_cast<std::size_t>(workspaceBytes));
        std::array<void*, 2> pointers = {x, y};
        const std::array<std::int64_t, 2> uids = {K_X_UID, K_Y_UID};
        _variantPack.emplace(
            fe::VariantPackBuilder()
                .setWorkspacePointer(workspace)
                .setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data())
                .setUids(static_cast<std::int64_t>(uids.size()), uids.data())
                .build());
        return *_variantPack;
    }

    void* allocate(std::size_t bytes)
    {
        _buffers.push_back(std::make_unique<DeviceBuffer>(bytes));
        return _buffers.back()->get();
    }

    static std::vector<std::string> runnableIds()
    {
        return sortedIds({plugin_constants::engineId<GoodPlugin>(),
                          plugin_constants::engineId<CudnnShimV8Plugin>()});
    }

    static inline std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>
        sPolicyOrderEnv;

    std::array<fe::GeneratorSource, 1> _sources = {instantHeuristics};
    fe::EngineConfigGenerator _generator{static_cast<int>(_sources.size()), _sources.data()};
    std::optional<fe::OperationGraph> _opGraph;
    std::vector<std::unique_ptr<DeviceBuffer>> _buffers;
    std::optional<fe::VariantPack> _variantPack;
};

TEST_F(IntegrationCudnnShimV8FindPlan, GetPlanSkipsContextFailure)
{
    // Engine -32 is offered and then skipped when its plan fails at context creation.
    ASSERT_EQ(_generator.generate_engine_config(*_opGraph).size(), 4U);
    const fe::executionPlans_t plans = getPlans();

    for(const fe::ExecutionPlan& plan : plans)
    {
        EXPECT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.getTag() << ": " << lastError();
    }
    EXPECT_EQ(engineIds(plans),
              sortedIds({plugin_constants::engineId<GoodPlugin>(),
                         plugin_constants::engineId<ExecuteFailsPlugin>(),
                         plugin_constants::engineId<CudnnShimV8Plugin>()}));
}

TEST_F(IntegrationCudnnShimV8FindPlan, TimeSortedPlanSkipsFailedWarmup)
{
    const fe::VariantPack& pack = variantPack();
    const fe::executionPlans_t sorted
        = fe::time_sorted_plan<K_SAMPLE_ONCE>(_handle, getPlans(), pack);

    EXPECT_EQ(engineIds(sorted), runnableIds());
    for(std::size_t i = 0; i < sorted.size(); ++i)
    {
        EXPECT_GE(sorted[i].getExecutionTime(), 0.0F) << sorted[i].getTag();
        if(i > 0)
        {
            EXPECT_LE(sorted[i - 1].getExecutionTime(), sorted[i].getExecutionTime());
        }
    }
}

TEST_F(IntegrationCudnnShimV8FindPlan, FindPlanAndCacheReusesPlan)
{
    const fe::VariantPack& pack = variantPack();
    fe::ExecutionPlanCache cache("find_plan_cache");
    const fe::ExecutionPlan* cached = nullptr;
    ASSERT_FALSE(cache.get_plan_from_cache(*_opGraph, cached));

    const fe::ExecutionPlan plan
        = _generator.cudnnFindPlanAndCache<K_SAMPLE_ONCE>(_handle, *_opGraph, pack, cache);
    const std::vector<std::string> runnable = runnableIds();
    EXPECT_NE(
        std::find(runnable.begin(), runnable.end(), jsonEngineId(plan.getJsonRepresentation())),
        runnable.end());

    ASSERT_TRUE(cache.get_plan_from_cache(*_opGraph, cached));
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(cached->getTag(), plan.getTag());
    EXPECT_EQ(cached->get_raw_desc(), plan.get_raw_desc());
}

TEST_F(IntegrationCudnnShimV8FindPlan, SortedFrontPlanExecutes)
{
    const fe::VariantPack& pack = variantPack();
    const fe::executionPlans_t sorted
        = _generator.cudnnFindPlan<K_SAMPLE_ONCE>(_handle, *_opGraph, pack);
    ASSERT_FALSE(sorted.empty());

    EXPECT_EQ(cudnnBackendExecute(_handle, sorted.front().get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_SUCCESS)
        << sorted.front().getTag() << ": " << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

} // namespace
