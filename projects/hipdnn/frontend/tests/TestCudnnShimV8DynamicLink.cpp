// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The cuDNN v8 shim in runtime-load mode: this TU links hipdnn_frontend_dynamic and
// not libhipdnn_backend (see CMakeLists.txt), so a shim code path that referenced a
// backend symbol directly would fail to link. exerciseShimSurface() is emitted but
// never called.

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

[[maybe_unused]] cudnnStatus_t exerciseShimSurface(cudnnHandle_t handle)
{
    const std::array<std::int64_t, 4> dims = {1, 1, 1, 4};
    const std::array<std::int64_t, 4> strides = {4, 4, 4, 1};
    auto tensor = [&](std::int64_t uid) {
        return fe::TensorBuilder()
            .setDataType(fe::DataType_t::FLOAT)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .build();
    };
    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(fe::PointwiseMode_t::RELU_FWD)
                                       .setComputeType(fe::DataType_t::FLOAT)
                                       .build();
    const fe::Operation op
        = fe::OperationBuilder(fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
              .setxDesc(tensor(1))
              .setyDesc(tensor(2))
              .setpwDesc(relu)
              .build();
    std::array<const fe::Operation*, 1> ops = {&op};
    fe::OperationGraph graph = fe::OperationGraphBuilder()
                                   .setHandle(handle)
                                   .setOperationGraph(ops.size(), ops.data())
                                   .build();
    fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                          .setOperationGraph(graph)
                                          .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                          .build();
    fe::EngineConfigList configs = heuristics.getEngineConfig(1);
    fe::EngineConfigList rejected;
    fe::filter(configs, rejected, [](cudnnBackendDescriptor_t config) {
        return fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config)
               || fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(config);
    });
    const fe::ExecutionPlan plan = fe::ExecutionPlanBuilder()
                                       .setHandle(handle)
                                       .setEngineConfig(configs.front(), graph.getTag())
                                       .build();
    std::array<void*, 2> pointers = {nullptr, nullptr};
    const std::array<std::int64_t, 2> uids = {1, 2};
    const fe::VariantPack pack = fe::VariantPackBuilder()
                                     .setDataPointers(2, pointers.data())
                                     .setUids(2, uids.data())
                                     .build();
    return cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc());
}

TEST(TestCudnnShimV8DynamicLink, LinksWithoutBackendLibrary)
{
    using SurfaceFn = cudnnStatus_t (*)(cudnnHandle_t);
    volatile SurfaceFn surface = &exerciseShimSurface;
    EXPECT_NE(surface, nullptr);
}

#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
// The runtime-loaded backend answers the passthroughs.
TEST(TestCudnnShimV8DynamicLink, PassthroughsReachLoadedBackend)
{
    EXPECT_EQ(cudnnBackendExecute(nullptr, nullptr, nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(nullptr));
}
#endif

} // namespace
