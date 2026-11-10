// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fused graphs on the GPU reference executor, each run on both executors from the same inputs and
// compared (the CPU side is TestFusedOperationsCpuGraphExecutor.cpp). FP32 throughout: datatypes
// are covered per op, these tests cover what a fused graph adds, the virtual intermediates between
// nodes and a plan reading what another one wrote.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "LayernormFwdGraphTestUtils.hpp"
#include "MatmulGraphTestUtils.hpp"
#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_integration_tests::CpuReferenceGraphExecutorAdapter;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_integration_tests::test_utils::createConvFwdGraph;
using hipdnn_integration_tests::test_utils::createLayernormFwdGraph;
using hipdnn_integration_tests::test_utils::createMatmulGraph;
using hipdnn_integration_tests::tolerance::defaultTolerance;
using hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph;
using hipdnn_test_sdk::utilities::createValidConvFwdBiasActivGraph;
using hipdnn_test_sdk::utilities::createValidRMSNormGraph;

namespace
{

// `base` with a pointwise node after the op that wrote `consumedUid`, which becomes virtual. The
// node writes the new tensor `newUid`; a non-empty `operandDims` adds a second input `operandUid`.
flatbuffers::FlatBufferBuilder appendPointwise(const flatbuffers::FlatBufferBuilder& base,
                                               PointwiseMode mode,
                                               int64_t consumedUid,
                                               int64_t newUid,
                                               int64_t operandUid = 0,
                                               const std::vector<int64_t>& operandDims = {})
{
    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);

    const auto consumed = std::find_if(graph.tensors.begin(),
                                       graph.tensors.end(),
                                       [consumedUid](const std::unique_ptr<TensorAttributesT>& t) {
                                           return t->uid == consumedUid;
                                       });
    if(consumed == graph.tensors.end())
    {
        throw std::logic_error("appendPointwise: the graph has no tensor with the consumed uid");
    }
    (*consumed)->virtual_ = true;

    // Read what the new tensors copy now: growing graph.tensors invalidates `consumed`.
    const DataType dataType = (*consumed)->data_type;
    const std::vector<int64_t> dims = (*consumed)->dims;
    const std::vector<int64_t> strides = (*consumed)->strides;

    PointwiseAttributesT attributes;
    attributes.operation = mode;
    attributes.in_0_tensor_uid = consumedUid;
    attributes.out_0_tensor_uid = newUid;

    if(!operandDims.empty())
    {
        auto operand = std::make_unique<TensorAttributesT>();
        operand->uid = operandUid;
        operand->name = "operand_" + std::to_string(operandUid);
        operand->data_type = dataType;
        operand->dims = operandDims;
        operand->strides = generateStrides(operandDims);
        attributes.in_1_tensor_uid = operandUid;
        graph.tensors.push_back(std::move(operand));
    }

    auto output = std::make_unique<TensorAttributesT>();
    output->uid = newUid;
    output->name = "fused_" + std::to_string(newUid);
    output->data_type = dataType;
    output->dims = dims;
    output->strides = strides;
    graph.tensors.push_back(std::move(output));

    auto node = std::make_unique<NodeT>();
    node->name = "fused_pointwise_" + std::to_string(newUid);
    node->compute_data_type = DataType::FLOAT;
    node->attributes.Set(attributes);
    graph.nodes.push_back(std::move(node));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

// Runs `graph` on both executors from the same inputs and compares the output of its last node, a
// pointwise one. Inputs are drawn from [-1, 1) unless `inputRanges` has their uid. Applicability
// needs no device, so it is checked before the skip.
void expectGpuMatchesCpu(const flatbuffers::FlatBufferBuilder& graph,
                         const std::map<int64_t, std::pair<float, float>>& inputRanges = {})
{
    auto* buffer = graph.GetBufferPointer();
    const size_t size = graph.GetSize();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(buffer, size);
    const auto* lastNode
        = wrapper.getNode(wrapper.nodeCount() - 1).attributes_as_PointwiseAttributes();
    ASSERT_NE(lastNode, nullptr);
    const int64_t outputUid = lastNode->out_0_tensor_uid();

    CpuReferenceGraphExecutorAdapter cpuExecutor;
    GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(cpuExecutor.isApplicable(buffer, size));
    ASSERT_TRUE(gpuExecutor.isApplicable(buffer, size));
    SKIP_IF_NO_DEVICES();

    std::unordered_map<int64_t, std::vector<float>> cpuValues;
    std::unordered_map<int64_t, void*> cpuPack;
    std::unordered_map<int64_t, void*> gpuPack;
    std::vector<Workspace<>> deviceBuffers;
    deviceBuffers.reserve(wrapper.getTensorMap().size());

    for(const auto& entry : wrapper.getTensorMap())
    {
        const int64_t uid = entry.first;
        const TensorAttributes& tensor = *entry.second;
        // The executors allocate virtual tensors themselves and read a baked scalar (the epsilon of
        // a normalization) from the graph, so neither belongs in the variant pack.
        if(tensor.virtual_() || tensor.value_type() != TensorValue::NONE)
        {
            continue;
        }
        ASSERT_EQ(tensor.data_type(), DataType::FLOAT) << "tensor " << uid;

        int64_t space = 1;
        for(flatbuffers::uoffset_t axis = 0; axis < tensor.dims()->size(); ++axis)
        {
            space += (tensor.dims()->Get(axis) - 1) * tensor.strides()->Get(axis);
        }
        std::vector<float> values(static_cast<size_t>(space), 0.0f);
        if(uid != outputUid)
        {
            float low = -1.0f;
            float high = 1.0f;
            const auto range = inputRanges.find(uid);
            if(range != inputRanges.end())
            {
                low = range->second.first;
                high = range->second.second;
            }
            std::mt19937 generator(1234U + static_cast<uint32_t>(uid));
            std::uniform_real_distribution<float> distribution(low, high);
            std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
        }

        const size_t bytes = values.size() * sizeof(float);
        deviceBuffers.emplace_back(bytes);
        void* const device = deviceBuffers.back().get();
        ASSERT_EQ(hipMemcpy(device, values.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        gpuPack[uid] = device;

        cpuValues[uid] = std::move(values);
        cpuPack[uid] = cpuValues[uid].data();
    }

    gpuExecutor.execute(buffer, size, gpuPack);
    cpuExecutor.execute(buffer, size, cpuPack);

    const std::vector<float>& expected = cpuValues.at(outputUid);
    std::vector<float> actual(expected.size());
    ASSERT_EQ(hipMemcpy(actual.data(),
                        gpuPack.at(outputUid),
                        actual.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    // Loosest per-op SDK tolerance across the graph's nodes, as reference validation uses
    const float tolerance = defaultTolerance(wrapper, DataType::FLOAT);
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "output element " << i;
    }
}

// conv x [2, 3, 6, 6] * w [4, 3, 3, 3] -> [2, 4, 6, 6], then `first` and `second` with per-channel
// operands, as in the CPU executor's ConvAddMulFusedGraph and ConvMaxMinFusedGraph.
flatbuffers::FlatBufferBuilder convThenTwoPointwise(PointwiseMode first, PointwiseMode second)
{
    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t CONV_UID = 3;
    constexpr int64_t FIRST_OPERAND_UID = 4;
    constexpr int64_t FIRST_OUTPUT_UID = 5;
    constexpr int64_t SECOND_OPERAND_UID = 6;
    constexpr int64_t SECOND_OUTPUT_UID = 7;
    const std::vector<int64_t> xDims = {2, 3, 6, 6};
    const std::vector<int64_t> wDims = {4, 3, 3, 3};
    const std::vector<int64_t> yDims = {2, 4, 6, 6};
    const std::vector<int64_t> perChannel = {1, 4, 1, 1};

    const auto conv = createConvFwdGraph(X_UID,
                                         W_UID,
                                         CONV_UID,
                                         xDims,
                                         wDims,
                                         yDims,
                                         generateStrides(xDims),
                                         generateStrides(wDims),
                                         generateStrides(yDims),
                                         {1, 1},
                                         {1, 1},
                                         {1, 1},
                                         DataType::FLOAT);
    const auto firstApplied
        = appendPointwise(conv, first, CONV_UID, FIRST_OUTPUT_UID, FIRST_OPERAND_UID, perChannel);
    return appendPointwise(
        firstApplied, second, FIRST_OUTPUT_UID, SECOND_OUTPUT_UID, SECOND_OPERAND_UID, perChannel);
}

} // namespace

TEST(TestGpuFusedOperationsGraphExecutor, ConvAddMulMatchesCpu)
{
    expectGpuMatchesCpu(convThenTwoPointwise(PointwiseMode::ADD, PointwiseMode::MUL));
}

// The GPU reference has no MAX_OP or MIN_OP kernel yet (see TestPointwiseReferenceParity), so it
// declines this graph; once it takes it, the executors are compared.
TEST(TestGpuFusedOperationsGraphExecutor, ConvMaxMinMatchesCpu)
{
    const auto graph = convThenTwoPointwise(PointwiseMode::MAX_OP, PointwiseMode::MIN_OP);

    GpuReferenceGraphExecutor gpuExecutor;
    if(!gpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()))
    {
        GTEST_SKIP() << "The GPU reference declines MAX_OP and MIN_OP (known pointwise gap)";
    }

    expectGpuMatchesCpu(graph);
}

// conv, then a bias add, then ReLU: the three-node graph the engines fuse.
TEST(TestGpuFusedOperationsGraphExecutor, ConvBiasReluMatchesCpu)
{
    const std::vector<int64_t> xDims = {2, 3, 6, 6};
    const std::vector<int64_t> wDims = {4, 3, 3, 3};
    const std::vector<int64_t> yDims = {2, 4, 6, 6};

    expectGpuMatchesCpu(createValidConvFwdBiasActivGraph(xDims,
                                                         generateStrides(xDims),
                                                         wDims,
                                                         generateStrides(wDims),
                                                         yDims,
                                                         generateStrides(yDims),
                                                         {1, 1},
                                                         {1, 1},
                                                         {1, 1},
                                                         {1, 1},
                                                         true,
                                                         PointwiseMode::RELU_FWD,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         DataType::FLOAT));
}

// Tensor 5 of the builder is the inverse variance, which has to be positive.
TEST(TestGpuFusedOperationsGraphExecutor, BatchnormReluMatchesCpu)
{
    constexpr int64_t INV_VARIANCE_UID = 5;
    const std::vector<int64_t> dims = {2, 3, 4, 4};

    expectGpuMatchesCpu(createValidBatchnormFwdInferActGraph(
                            generateStrides(dims), dims, DataType::FLOAT, DataType::FLOAT),
                        {{INV_VARIANCE_UID, {0.5f, 2.0f}}});
}

// Not the test SDK's layernorm builder: it gives scale and bias the normalized dims alone, which
// the GPU reference rejects at execution.
TEST(TestGpuFusedOperationsGraphExecutor, LayernormReluMatchesCpu)
{
    constexpr int64_t X_UID = 1;
    constexpr int64_t NORMALIZED_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t BIAS_UID = 4;
    constexpr int64_t EPSILON_UID = 5;
    constexpr int64_t ACTIVATED_UID = 6;
    constexpr int64_t NORMALIZED_DIM_COUNT = 3;
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    const std::vector<int64_t> dims = {2, 3, 5, 7};

    const auto layernorm = createLayernormFwdGraph(X_UID,
                                                   NORMALIZED_UID,
                                                   SCALE_UID,
                                                   BIAS_UID,
                                                   EPSILON_UID,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   dims,
                                                   TensorLayout::NCHW,
                                                   epsilon,
                                                   NORMALIZED_DIM_COUNT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   std::nullopt,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT);
    expectGpuMatchesCpu(
        appendPointwise(layernorm, PointwiseMode::RELU_FWD, NORMALIZED_UID, ACTIVATED_UID));
}

// Tensor 2 of the builder is the RMSNorm output.
TEST(TestGpuFusedOperationsGraphExecutor, RMSNormReluMatchesCpu)
{
    constexpr int64_t NORMALIZED_UID = 2;
    constexpr int64_t ACTIVATED_UID = 5;
    const std::vector<int64_t> dims = {2, 3, 4, 4};

    const auto rmsnorm
        = createValidRMSNormGraph(generateStrides(dims), dims, DataType::FLOAT, DataType::FLOAT);
    expectGpuMatchesCpu(
        appendPointwise(rmsnorm, PointwiseMode::RELU_FWD, NORMALIZED_UID, ACTIVATED_UID));
}

// matmul, then a bias add, then ReLU.
TEST(TestGpuFusedOperationsGraphExecutor, MatmulBiasReluMatchesCpu)
{
    constexpr int64_t A_UID = 1;
    constexpr int64_t B_UID = 2;
    constexpr int64_t C_UID = 3;
    constexpr int64_t BIAS_UID = 4;
    constexpr int64_t BIASED_UID = 5;
    constexpr int64_t ACTIVATED_UID = 6;
    const std::vector<int64_t> aDims = {3, 5, 7};
    const std::vector<int64_t> bDims = {3, 7, 4};
    const std::vector<int64_t> cDims = {3, 5, 4};

    const auto matmul = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          aDims,
                                          generateStrides(aDims),
                                          bDims,
                                          generateStrides(bDims),
                                          cDims,
                                          generateStrides(cDims),
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    const auto biased
        = appendPointwise(matmul, PointwiseMode::ADD, C_UID, BIASED_UID, BIAS_UID, {1, 1, 4});
    expectGpuMatchesCpu(
        appendPointwise(biased, PointwiseMode::RELU_FWD, BIASED_UID, ACTIVATED_UID));
}
