// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v8 core's graph lowering and engine index space, driven with records
// directly (no cuDNN backend descriptors) against the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/lowering.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"
#include "fake_backend/ShimV8TestSupport.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace v8 = hipdnn_frontend::compatibility::cudnn_frontend::detail::v8;

using hipdnn_frontend::DataType_t;
using hipdnn_frontend::PointwiseMode_t;
using hipdnn_shim_test::expectShimError;
using hipdnn_shim_test::packedStrides;
using hipdnn_shim_test::shimErrorOf;

std::string operationName(cudnnBackendDescriptorType_t /*kind*/)
{
    return "OP";
}

const v8::LoweringLabels LABELS{"H", "OPS", "DYN", "OVR", "CONSTANT", &operationName};

v8::TensorRecord::Fields tensorFields(std::int64_t uid)
{
    v8::TensorRecord::Fields fields;
    fields.dataType = DataType_t::FLOAT;
    fields.dimensions = {2, 3, 4, 5};
    fields.strides = packedStrides(fields.dimensions);
    fields.uniqueId = uid;
    fields.alignment = 16;
    return fields;
}

v8::TensorPtr tensor(v8::TensorRecord::Fields fields)
{
    return std::make_shared<const v8::TensorRecord>(std::move(fields));
}

v8::TensorPtr tensor(std::int64_t uid)
{
    return tensor(tensorFields(uid));
}

template <typename T>
v8::TensorPtr constant(std::int64_t uid, DataType_t dataType, T value)
{
    v8::TensorRecord::Fields fields = tensorFields(uid);
    fields.dataType = dataType;
    fields.dimensions = {1, 1, 1, 1};
    fields.strides = {1, 1, 1, 1};
    fields.isByValue = true;
    std::vector<std::byte> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    fields.constantValue = std::move(bytes);
    return tensor(std::move(fields));
}

std::shared_ptr<const v8::OperationRecord> relu(v8::TensorPtr x, v8::TensorPtr y)
{
    v8::PointwiseRecord::Fields pointwise;
    pointwise.mode = PointwiseMode_t::RELU_FWD;
    pointwise.computeType = DataType_t::FLOAT;
    v8::PointwiseOperationRecord::Fields fields;
    fields.pointwise = std::make_shared<const v8::PointwiseRecord>(pointwise);
    fields.traits = *v8::pointwiseModeTraits(PointwiseMode_t::RELU_FWD);
    fields.in0 = std::move(x);
    fields.out0 = std::move(y);
    return std::make_shared<const v8::PointwiseOperationRecord>(std::move(fields));
}

std::shared_ptr<const v8::OperationRecord> relu(std::int64_t xUid, std::int64_t yUid)
{
    return relu(tensor(xUid), tensor(yUid));
}

v8::OperationPorts ports(const std::vector<std::int64_t>& inputs,
                         const std::vector<std::int64_t>& outputs)
{
    v8::OperationPorts result;
    for(const std::int64_t uid : inputs)
    {
        result.inputs.push_back(tensor(uid));
    }
    for(const std::int64_t uid : outputs)
    {
        result.outputs.push_back(tensor(uid));
    }
    return result;
}

// Records built through the factories, as a builder builds them.
v8::TensorPtr makeTensor(std::int64_t uid,
                         const std::vector<std::int64_t>& dims,
                         cudnnDataType_t dataType = CUDNN_DATA_FLOAT,
                         bool isByValue = false)
{
    v8::TensorInputs inputs;
    inputs.dataType = dataType;
    inputs.dimensions = dims;
    inputs.strides = packedStrides(dims);
    inputs.uniqueId = uid;
    inputs.alignment = 16;
    inputs.isByValue = isByValue;
    return v8::makeTensorRecord(std::move(inputs), {});
}

v8::TensorPtr makeScalar(std::int64_t uid)
{
    return makeTensor(uid, {1, 1, 1, 1}, CUDNN_DATA_FLOAT, true);
}

std::shared_ptr<const v8::ConvolutionRecord>
    makeConvolution(std::vector<std::int64_t> prePaddings,
                    std::vector<std::int64_t> postPaddings,
                    std::vector<std::int64_t> strides,
                    std::vector<std::int64_t> dilations,
                    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
                    cudnnDataType_t computeType = CUDNN_DATA_FLOAT)
{
    v8::ConvolutionInputs inputs;
    inputs.computeType = computeType;
    inputs.mode = mode;
    inputs.spatialDims = static_cast<std::int64_t>(prePaddings.size());
    inputs.prePaddings = std::move(prePaddings);
    inputs.postPaddings = std::move(postPaddings);
    inputs.filterStrides = std::move(strides);
    inputs.dilations = std::move(dilations);
    return v8::makeConvRecord(std::move(inputs), {});
}

template <typename OperationRecordT>
std::shared_ptr<const v8::OperationRecord>
    makeConvolutionOperation(std::shared_ptr<const v8::ConvolutionRecord> convolution,
                             std::array<v8::TensorPtr, 3> tensors)
{
    v8::ConvolutionOperationInputs inputs;
    inputs.convolution = std::move(convolution);
    inputs.tensors = std::move(tensors);
    return v8::makeOperationRecord<OperationRecordT>(inputs, {});
}

std::shared_ptr<const v8::OperationRecord> makePointwise(cudnnPointwiseMode_t mode,
                                                         v8::PointwiseOperationInputs slots,
                                                         double lowerClip = 0.0)
{
    v8::PointwiseInputs parameters;
    parameters.mode = mode;
    parameters.computeType = CUDNN_DATA_FLOAT;
    // Upstream sends every ReLU parameter, defaults included.
    parameters.reluLowerClip = lowerClip;
    parameters.reluUpperClip = std::numeric_limits<double>::max();
    parameters.reluLowerClipSlope = 0.0;
    slots.pointwise = v8::makePointwiseRecord(parameters, {});
    return v8::makeOperationRecord(slots, {});
}

// 2x2 max pooling with stride 2, unpadded unless asked.
std::shared_ptr<const v8::ResampleRecord>
    makeMaxPool(std::vector<cudnnFraction_t> prePaddings = {{0, 1}, {0, 1}},
                std::vector<cudnnFraction_t> postPaddings = {{0, 1}, {0, 1}})
{
    v8::ResampleInputs inputs;
    inputs.mode = CUDNN_RESAMPLE_MAXPOOL;
    inputs.computeType = CUDNN_DATA_FLOAT;
    inputs.paddingMode = CUDNN_NEG_INF_PAD;
    inputs.spatialDims = 2;
    inputs.windowDims = {{2, 1}, {2, 1}};
    inputs.prePaddings = std::move(prePaddings);
    inputs.postPaddings = std::move(postPaddings);
    inputs.strides = {{2, 1}, {2, 1}};
    return v8::makeResampleRecord(inputs, {});
}

std::shared_ptr<const v8::OperationRecord> makeNormForward(cudnnBackendNormMode_t mode,
                                                           cudnnBackendNormFwdPhase_t phase,
                                                           v8::NormForwardOperationInputs inputs)
{
    inputs.mode = mode;
    inputs.phase = phase;
    return v8::makeOperationRecord(inputs, {});
}

std::shared_ptr<const v8::OperationRecord> makeNormBackward(cudnnBackendNormMode_t mode,
                                                            v8::NormBackwardOperationInputs inputs)
{
    inputs.mode = mode;
    return v8::makeOperationRecord(inputs, {});
}

// One lowered operation and the hipDNN slot each of its tensor UIDs must land in.
struct LoweredCase
{
    const char* label;
    std::function<std::shared_ptr<const v8::OperationRecord>()> build;
    hipdnnBackendDescriptorType_t operationType;
    hipdnnBackendAttributeName_t computeType;
    std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>> slots;
};

class TestCudnnShimV8CoreLowering : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    // The UID each hipDNN operation in the graph's OPS array writes through
    // POINTWISE_OUT_0, in graph order.
    std::vector<std::int64_t> outputUidsInGraphOrder() const
    {
        std::vector<std::int64_t> uids;
        for(const auto& call : setAttributesOn(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR))
        {
            if(call.name != HIPDNN_ATTR_OPERATIONGRAPH_OPS)
            {
                continue;
            }
            for(hipdnnBackendDescriptor_t op : call.values<hipdnnBackendDescriptor_t>())
            {
                uids.push_back(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT));
            }
        }
        return uids;
    }

    std::shared_ptr<const v8::GraphCore>
        lower(const std::vector<std::shared_ptr<const v8::OperationRecord>>& operations) const
    {
        return v8::lowerGraph(operations, _handle, {}, LABELS, {});
    }

    void expectLowered(const LoweredCase& item)
    {
        SCOPED_TRACE(item.label);
        lower({item.build()});
        const std::vector<hipdnnBackendDescriptor_t> ops = created(item.operationType);
        ASSERT_FALSE(ops.empty());
        const auto computeType = callOn(ops.back(), item.computeType);
        ASSERT_TRUE(computeType);
        EXPECT_EQ(computeType->value<hipdnnDataType_t>(), HIPDNN_DATA_FLOAT);
        for(const auto& [slotName, uid] : item.slots)
        {
            EXPECT_EQ(slotUid(ops.back(), slotName), uid) << "hipDNN attribute " << slotName;
        }
    }
};

TEST(TestCudnnShimV8CoreLoweringOrder, KeepsTopologicalOrderAndDisconnectedComponents)
{
    // 1 -> 2 -> 3, and a disconnected 10 -> 11 in between.
    const std::vector<v8::OperationPorts> graph
        = {ports({1}, {2}), ports({10}, {11}), ports({2}, {3})};
    EXPECT_EQ(v8::stableTopologicalOrder(graph, LABELS), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(TestCudnnShimV8CoreLoweringOrder, MovesConsumersAfterProducersSmallestIndexFirst)
{
    // Op 0 reads what op 2 writes; op 1 is disconnected; op 3 reads op 0.
    const std::vector<v8::OperationPorts> graph
        = {ports({2}, {3}), ports({10}, {11}), ports({1}, {2}), ports({3}, {4})};
    EXPECT_EQ(v8::stableTopologicalOrder(graph, LABELS), (std::vector<std::size_t>{1, 2, 0, 3}));
}

TEST(TestCudnnShimV8CoreLoweringOrder, ReleasedConsumerKeepsItsPlaceAheadOfReadyOperations)
{
    // Already topological: op 1 reads op 0 and is released while the
    // independent op 2 is already ready. A FIFO ready queue would yield 0, 2, 1.
    const std::vector<v8::OperationPorts> graph
        = {ports({1}, {2}), ports({2}, {3}), ports({10}, {11})};
    EXPECT_EQ(v8::stableTopologicalOrder(graph, LABELS), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(TestCudnnShimV8CoreLoweringOrder, RejectsDoubleWriter)
{
    const std::vector<v8::OperationPorts> graph
        = {ports({1}, {3}), ports({10}, {11}), ports({2}, {3})};
    expectShimError([&] { v8::stableTopologicalOrder(graph, LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "OPS: UID 3 is written by operations 0 and 2");
}

TEST(TestCudnnShimV8CoreLoweringOrder, RejectsCycle)
{
    const std::vector<std::vector<v8::OperationPorts>> graphs = {
        {ports({1}, {2}), ports({10}, {11}), ports({2}, {1})},
        // An in-place operation depends on itself.
        {ports({1}, {1})},
    };
    for(const std::vector<v8::OperationPorts>& graph : graphs)
    {
        SCOPED_TRACE(graph.size());
        expectShimError([&] { v8::stableTopologicalOrder(graph, LABELS); },
                        CUDNN_STATUS_BAD_PARAM,
                        "OPS: the operations form a cycle");
    }
}

TEST(TestCudnnShimV8CoreLoweringTensors, SharedUidMustAgreeOnEveryField)
{
    struct Case
    {
        const char* difference;
        void (*mutate)(v8::TensorRecord::Fields&);
    };
    const std::vector<Case> cases = {
        {"data type", [](v8::TensorRecord::Fields& f) { f.dataType = DataType_t::HALF; }},
        {"dimensions",
         [](v8::TensorRecord::Fields& f) {
             f.dimensions = {2, 3, 5, 4};
             f.strides = packedStrides(f.dimensions);
         }},
        {"strides", [](v8::TensorRecord::Fields& f) { f.strides = {60, 1, 15, 3}; }},
        {"virtualness", [](v8::TensorRecord::Fields& f) { f.isVirtual = true; }},
        {"by-value scalar", [](v8::TensorRecord::Fields& f) { f.isByValue = true; }},
        {"by-value scalar",
         [](v8::TensorRecord::Fields& f) {
             f.isByValue = true;
             f.constantValue = std::vector<std::byte>(4, std::byte{1});
         }},
        {"ragged offset", [](v8::TensorRecord::Fields& f) { f.raggedOffset = tensor(99); }},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.difference);
        v8::TensorRecord::Fields changed = tensorFields(7);
        c.mutate(changed);
        v8::CanonicalTensors tensors(LABELS);
        const v8::TensorPtr first = tensor(7);
        const v8::TensorPtr second = tensor(changed);
        tensors.add(first);
        expectShimError([&] { tensors.add(second); },
                        CUDNN_STATUS_BAD_PARAM,
                        std::string("OPS: two tensors with UID 7 differ in ") + c.difference);
    }
}

TEST(TestCudnnShimV8CoreLoweringTensors, SharedUidKeepsWeakestAlignment)
{
    v8::CanonicalTensors tensors(LABELS);
    const v8::TensorPtr first = tensor(7);
    tensors.add(first);
    EXPECT_EQ(tensors.at(first)->get_alignment(),
              hipdnn_frontend::detail::DEFAULT_TENSOR_ALIGNMENT);

    v8::TensorRecord::Fields weakerFields = tensorFields(7);
    weakerFields.alignment = 4;
    const v8::TensorPtr weaker = tensor(weakerFields);
    tensors.add(weaker);
    EXPECT_EQ(tensors.at(first)->get_alignment(), 4);

    v8::TensorRecord::Fields betweenFields = tensorFields(7);
    betweenFields.alignment = 8;
    const v8::TensorPtr between = tensor(betweenFields);
    tensors.add(between);
    EXPECT_EQ(tensors.at(first)->get_alignment(), 4);

    // The same record twice is not a second claim.
    tensors.add(first);
    EXPECT_EQ(tensors.at(first)->get_alignment(), 4);
}

TEST(TestCudnnShimV8CoreLoweringTensors, ConstantKeepsShapeAndValue)
{
    v8::CanonicalTensors tensors(LABELS);
    const v8::TensorPtr scalar = constant(5, DataType_t::FLOAT, 2.5F);
    tensors.add(scalar);
    const auto& lowered = tensors.at(scalar);
    // set_value resets dims and strides, so it must run before set_dim.
    EXPECT_EQ(lowered->get_dim(), (std::vector<std::int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(lowered->get_stride(), (std::vector<std::int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(lowered->get_compile_time_constant<float>(), std::optional<float>(2.5F));
    EXPECT_EQ(lowered->get_uid(), 5);
}

TEST(TestCudnnShimV8CoreLoweringTensors, ConstantKeepsItsElementType)
{
    // 1.5 in each 16-bit format; the two formats share a size, so only the
    // stored scalar type tells a swapped read apart.
    v8::CanonicalTensors tensors(LABELS);
    const v8::TensorPtr halfConstant = constant(1, DataType_t::HALF, std::uint16_t{0x3E00});
    const v8::TensorPtr bfloatConstant = constant(2, DataType_t::BFLOAT16, std::uint16_t{0x3FC0});
    const v8::TensorPtr byteConstant = constant(3, DataType_t::UINT8, std::uint8_t{200});
    // A 32-bit read would keep only the low word, 2.
    const v8::TensorPtr wideConstant = constant(4, DataType_t::INT64, std::int64_t{0x100000002});
    for(const v8::TensorPtr& constantTensor :
        {halfConstant, bfloatConstant, byteConstant, wideConstant})
    {
        tensors.add(constantTensor);
    }

    const auto halfValue
        = tensors.at(halfConstant)->get_compile_time_constant<v8::fe_graph::half>();
    ASSERT_TRUE(halfValue);
    EXPECT_EQ(halfValue->data, 0x3E00);
    const auto bfloatValue
        = tensors.at(bfloatConstant)->get_compile_time_constant<v8::fe_graph::bfloat16>();
    ASSERT_TRUE(bfloatValue);
    EXPECT_EQ(bfloatValue->data, 0x3FC0);
    EXPECT_EQ(tensors.at(byteConstant)->get_compile_time_constant<std::uint8_t>(),
              std::optional<std::uint8_t>(200));
    EXPECT_EQ(tensors.at(wideConstant)->get_compile_time_constant<std::int64_t>(),
              std::optional<std::int64_t>(0x100000002));
    EXPECT_EQ(tensors.at(halfConstant)->get_data_type(), DataType_t::HALF);
    EXPECT_EQ(tensors.at(bfloatConstant)->get_data_type(), DataType_t::BFLOAT16);
    EXPECT_EQ(tensors.at(byteConstant)->get_data_type(), DataType_t::UINT8);
    EXPECT_EQ(tensors.at(wideConstant)->get_data_type(), DataType_t::INT64);
}

TEST(TestCudnnShimV8CoreLoweringTensors, ConstantOfUncarriableTypeNotSupported)
{
    v8::CanonicalTensors tensors(LABELS);
    expectShimError([&] { tensors.add(constant(5, DataType_t::INT8, std::int8_t{3})); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "CONSTANT of UID 5: hipDNN cannot carry a constant of this data type");
}

TEST_F(TestCudnnShimV8CoreLowering, LowersInStableTopologicalOrder)
{
    // relu(2 -> 3) consumes relu(1 -> 2); relu(10 -> 11) is disconnected.
    const std::shared_ptr<const v8::GraphCore> core
        = v8::lowerGraph({relu(2, 3), relu(10, 11), relu(1, 2)}, _handle, {}, LABELS, "tag");

    EXPECT_EQ(outputUidsInGraphOrder(), (std::vector<std::int64_t>{11, 2, 3}));
    EXPECT_EQ(core->kind(), CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    EXPECT_EQ(core->handle(), _handle);
    EXPECT_EQ(core->tag(), "tag");
    EXPECT_EQ(core->hipdnnGraph(), created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).front());
    EXPECT_EQ(finalized().back(), core->hipdnnGraph());
    // Lowering alone runs no heuristics.
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());

    const std::shared_ptr<const v8::Record> record = core;
    EXPECT_EQ(v8::recordAs<v8::GraphCore>(record), core);
}

TEST_F(TestCudnnShimV8CoreLowering, SharedUidLowersOnce)
{
    v8::lowerGraph({relu(1, 2), relu(2, 3)}, _handle, {}, LABELS, {});
    std::size_t uid2 = 0;
    for(const auto& call : setAttributesOn(HIPDNN_BACKEND_TENSOR_DESCRIPTOR))
    {
        if(call.name == HIPDNN_ATTR_TENSOR_UNIQUE_ID && call.value<std::int64_t>() == 2)
        {
            ++uid2;
        }
    }
    EXPECT_EQ(uid2, 1U);
}

TEST_F(TestCudnnShimV8CoreLowering, GraphErrorsUseCallerLabels)
{
    expectShimError([&] { v8::lowerGraph({relu(1, 2), relu(2, 1)}, _handle, {}, LABELS, {}); },
                    CUDNN_STATUS_BAD_PARAM,
                    "OPS: the operations form a cycle");
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());

    // A norm record without a lowerable target names the operation through the
    // caller's callback.
    v8::NormForwardOperationRecord::Fields norm;
    norm.x = tensor(1);
    norm.y = tensor(2);
    expectShimError(
        [&] {
            v8::lowerGraph({std::make_shared<const v8::NormForwardOperationRecord>(
                               v8::OperationTarget::NOT_SET, std::move(norm))},
                           _handle,
                           {},
                           LABELS,
                           {});
        },
        CUDNN_STATUS_INTERNAL_ERROR,
        "OP has no lowering target");
}

TEST_F(TestCudnnShimV8CoreLowering, DenseIndexFollowsSignedId)
{
    installHeuristicResults({5, -2, 300, -40});
    const std::shared_ptr<const v8::GraphCore> core
        = v8::lowerGraph({relu(1, 2)}, _handle, {}, LABELS, {});

    EXPECT_EQ(core->engineCount(), 4);
    EXPECT_EQ(core->rankedEngineIds(), (std::vector<std::int64_t>{5, -2, 300, -40}));
    const std::vector<std::int64_t> byIndex = {-40, -2, 5, 300};
    for(std::size_t index = 0; index < byIndex.size(); ++index)
    {
        const auto signedIndex = static_cast<std::int64_t>(index);
        EXPECT_EQ(core->engineIdAt(signedIndex), byIndex[index]);
        EXPECT_EQ(core->engineIndexOf(byIndex[index]), signedIndex);
        EXPECT_NE(core->defaultEngineConfig(byIndex[index]), nullptr);
        EXPECT_EQ(typeOf(core->defaultEngine(byIndex[index])), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    }
    EXPECT_EQ(core->engineIdAt(-1), std::nullopt);
    EXPECT_EQ(core->engineIdAt(4), std::nullopt);
    EXPECT_EQ(core->engineIndexOf(7), std::nullopt);
    EXPECT_EQ(core->defaultEngineConfig(7), nullptr);
    EXPECT_EQ(core->defaultEngine(7), nullptr);
    // Memoized: one heuristic sweep however often the space is read.
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8CoreLowering, ConfigFinalizeFailureDropsEngine)
{
    installHeuristicResults({7, -1, 4, 7}, {-1});
    const std::shared_ptr<const v8::GraphCore> core
        = v8::lowerGraph({relu(1, 2)}, _handle, {}, LABELS, {});

    // -1 is dropped and the duplicate 7 is kept once, at its first rank.
    EXPECT_EQ(core->rankedEngineIds(), (std::vector<std::int64_t>{7, 4}));
    EXPECT_EQ(core->engineCount(), 2);
    EXPECT_EQ(core->engineIdAt(0), 4);
    EXPECT_EQ(core->engineIdAt(1), 7);
    EXPECT_EQ(core->engineIndexOf(-1), std::nullopt);
    EXPECT_EQ(core->defaultEngineConfig(-1), nullptr);

    // The dropped config's hipDNN error was drained, not left for the next caller.
    std::array<char, 256> message{};
    hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(message.data(), message.size());
    EXPECT_STREQ(message.data(), "");
}

TEST_F(TestCudnnShimV8CoreLowering, FailedEnumerationIsRetried)
{
    using ::testing::_;
    installHeuristicResults({-2});
    const std::shared_ptr<const v8::GraphCore> core = lower({relu(1, 2)});

    int attempts = 0;
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault(
            [this, &attempts](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
                if(attempts++ == 0)
                {
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                }
                *out = newMockDescriptor(type);
                return HIPDNN_STATUS_SUCCESS;
            });

    const v8::ShimError error = shimErrorOf([&] { static_cast<void>(core->engineCount()); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(std::string(error.what()).rfind("creating the hipDNN engine heuristic: ", 0), 0U)
        << error.what();
    EXPECT_EQ(core->engineCount(), 1);
    EXPECT_EQ(core->rankedEngineIds(), (std::vector<std::int64_t>{-2}));
    EXPECT_EQ(attempts, 2);
}

TEST_F(TestCudnnShimV8CoreLowering, PreflightRulesApplyInOrderBeforeAnyHipdnnCall)
{
    using ::testing::_;
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(0);

    struct Case
    {
        const char* label;
        std::vector<std::shared_ptr<const v8::OperationRecord>> operations;
        bool withHandle;
        v8::GraphFlags flags;
        cudnnStatus_t status;
        const char* message;
    };
    const std::shared_ptr<const v8::OperationRecord> op = relu(1, 2);
    const std::vector<Case> cases = {
        {"handle first", {}, false, {true, true}, CUDNN_STATUS_BAD_PARAM, "H is required"},
        {"operations next",
         {},
         true,
         {true, true},
         CUDNN_STATUS_BAD_PARAM,
         "OPS is required (non-empty)"},
        {"dynamic shape next",
         {nullptr},
         true,
         {true, true},
         CUDNN_STATUS_NOT_SUPPORTED,
         "DYN = true is not supported by the hipDNN cuDNN shim"},
        {"override shape next",
         {nullptr},
         true,
         {false, true},
         CUDNN_STATUS_NOT_SUPPORTED,
         "OVR = true is not supported by the hipDNN cuDNN shim"},
        {"null record last",
         {op, nullptr},
         true,
         {},
         CUDNN_STATUS_BAD_PARAM,
         "OPS: operation 1 is null"},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        hipdnnHandle_t handle = c.withHandle ? _handle : nullptr;
        expectShimError([&] { v8::lowerGraph(c.operations, handle, c.flags, LABELS, {}); },
                        c.status,
                        c.message);
    }
}

TEST_F(TestCudnnShimV8CoreLowering, OperationNameDefaultsToCudnnKindName)
{
    v8::LoweringLabels defaulted = LABELS;
    defaulted.operationName = v8::LoweringLabels{}.operationName;
    v8::LoweringLabels unset = LABELS;
    unset.operationName = nullptr;
    for(const v8::LoweringLabels* labels : {&defaulted, &unset})
    {
        v8::NormForwardOperationRecord::Fields norm;
        norm.x = tensor(1);
        norm.y = tensor(2);
        expectShimError(
            [&] {
                v8::lowerGraph({std::make_shared<const v8::NormForwardOperationRecord>(
                                   v8::OperationTarget::NOT_SET, std::move(norm))},
                               _handle,
                               {},
                               *labels,
                               {});
            },
            CUDNN_STATUS_INTERNAL_ERROR,
            "CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR has no lowering target");
    }

    // The node lifecycle names the operation the same way.
    v8::PointwiseOperationInputs add;
    add.x = makeTensor(1, {2, 3, 4, 5});
    add.b = makeTensor(2, {2, 7, 4, 5});
    add.y = makeTensor(3, {2, 3, 4, 5});
    const v8::ShimError error = shimErrorOf(
        [&] { v8::lowerGraph({makePointwise(CUDNN_POINTWISE_ADD, add)}, _handle, {}, unset, {}); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(std::string(error.what())
                  .rfind("lowering CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: ", 0),
              0U)
        << error.what();
}

TEST_F(TestCudnnShimV8CoreLowering, ConvolutionForwardCarriesGeometry)
{
    // Pre and post paddings differ per axis but keep the per-axis sums, so Y
    // is still [1,8,4,6].
    lower({makeConvolutionOperation<v8::ConvolutionForwardOperationRecord>(
        makeConvolution({1, 0}, {1, 2}, {2, 1}, {1, 2}),
        {makeTensor(1, {1, 4, 8, 8}), makeTensor(2, {8, 4, 3, 3}), makeTensor(3, {1, 8, 4, 6})})});

    hipdnnBackendDescriptor_t op
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS).values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 0}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS).values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES).values<std::int64_t>(),
              (std::vector<std::int64_t>{2, 1}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_DILATIONS).values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_CONV_MODE).value<hipdnnConvolutionMode_t>(),
              HIPDNN_CROSS_CORRELATION);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE).value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X), 1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W), 2);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y), 3);
    EXPECT_EQ(attributeOn(onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
                          HIPDNN_ATTR_OPERATIONGRAPH_HANDLE)
                  .value<hipdnnHandle_t>(),
              _handle);
}

TEST_F(TestCudnnShimV8CoreLowering, ConvolutionBackwardSlotsAndGeometry)
{
    // True convolution and HALF compute: neither is hipDNN's default.
    const std::shared_ptr<const v8::ConvolutionRecord> convolution
        = makeConvolution({1, 1}, {1, 1}, {1, 1}, {1, 1}, CUDNN_CONVOLUTION, CUDNN_DATA_HALF);
    lower({makeConvolutionOperation<v8::ConvolutionBackwardDataOperationRecord>(
               convolution,
               {makeTensor(1, {1, 8, 8, 8}),
                makeTensor(2, {8, 4, 3, 3}),
                makeTensor(3, {1, 4, 8, 8})}),
           makeConvolutionOperation<v8::ConvolutionBackwardFilterOperationRecord>(
               convolution,
               {makeTensor(5, {1, 8, 8, 8}),
                makeTensor(4, {1, 4, 8, 8}),
                makeTensor(6, {8, 4, 3, 3})})});

    hipdnnBackendDescriptor_t data
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY), 1);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W), 2);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX), 3);
    hipdnnBackendDescriptor_t filter
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X), 4);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY), 5);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW), 6);
    for(hipdnnBackendDescriptor_t op : {data, filter})
    {
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS).values<std::int64_t>(),
                  (std::vector<std::int64_t>{1, 1}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS).values<std::int64_t>(),
                  (std::vector<std::int64_t>{1, 1}));
        EXPECT_EQ(
            attributeOn(op, HIPDNN_ATTR_CONVOLUTION_CONV_MODE).value<hipdnnConvolutionMode_t>(),
            HIPDNN_CONVOLUTION);
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE).value<hipdnnDataType_t>(),
                  HIPDNN_DATA_HALF);
    }
}

TEST_F(TestCudnnShimV8CoreLowering, PointwiseSlotsMapByModeClass)
{
    v8::PointwiseOperationInputs add;
    add.x = makeTensor(10, {2, 3, 4, 5});
    add.b = makeTensor(11, {1, 3, 1, 1});
    add.y = makeTensor(12, {2, 3, 4, 5});
    v8::PointwiseOperationInputs reluBackward;
    reluBackward.dy = makeTensor(20, {2, 3, 4, 5});
    reluBackward.x = makeTensor(21, {2, 3, 4, 5});
    reluBackward.dx = makeTensor(22, {2, 3, 4, 5});
    lower({makePointwise(CUDNN_POINTWISE_ADD, add),
           makePointwise(CUDNN_POINTWISE_RELU_BWD, reluBackward)});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    EXPECT_EQ(attributeOn(ops[0], HIPDNN_ATTR_POINTWISE_MODE).value<hipdnnPointwiseMode_t>(),
              HIPDNN_POINTWISE_ADD);
    EXPECT_EQ(attributeOn(ops[0], HIPDNN_ATTR_POINTWISE_MATH_PREC).value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT), 10);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT), 11);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT), -1);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT), 12);
    EXPECT_EQ(attributeOn(ops[1], HIPDNN_ATTR_POINTWISE_MODE).value<hipdnnPointwiseMode_t>(),
              HIPDNN_POINTWISE_RELU_BWD);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT), 20);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT), 21);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT), 22);
}

TEST_F(TestCudnnShimV8CoreLowering, ReluClipNarrowsToFloatAndSaturates)
{
    v8::PointwiseOperationInputs plain;
    plain.x = makeTensor(1, {2, 3, 4, 5});
    plain.y = makeTensor(2, {2, 3, 4, 5});
    v8::PointwiseOperationInputs saturated;
    saturated.x = makeTensor(3, {2, 3, 4, 5});
    saturated.y = makeTensor(4, {2, 3, 4, 5});
    lower({makePointwise(CUDNN_POINTWISE_RELU_FWD, plain),
           makePointwise(CUDNN_POINTWISE_RELU_FWD, saturated, -1e300)});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    // Plain ReLU carries none of its defaults.
    EXPECT_FALSE(callOn(ops[0], HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP));
    const auto lowerClip = callOn(ops[1], HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP);
    ASSERT_TRUE(lowerClip);
    EXPECT_EQ(lowerClip->type, HIPDNN_TYPE_FLOAT);
    EXPECT_EQ(lowerClip->value<float>(), -std::numeric_limits<float>::max());
    for(auto* const op : ops)
    {
        EXPECT_FALSE(callOn(op, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP));
        EXPECT_FALSE(callOn(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE));
    }
}

TEST_F(TestCudnnShimV8CoreLowering, MatmulSlotsAndComputeType)
{
    v8::MatmulInputs parameters;
    parameters.computeType = CUDNN_DATA_FLOAT;
    v8::MatmulOperationInputs inputs;
    inputs.matmul = v8::makeMatmulRecord(parameters, {});
    inputs.a = makeTensor(1, {1, 4, 8});
    inputs.b = makeTensor(2, {1, 8, 5});
    inputs.c = makeTensor(3, {1, 4, 5});
    lower({v8::makeOperationRecord(inputs, {})});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_MATMUL_ADESC), 1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_MATMUL_BDESC), 2);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_MATMUL_CDESC), 3);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_MATMUL_COMP_TYPE).value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
}

TEST_F(TestCudnnShimV8CoreLowering, ReductionCarriesModeAndDeterminism)
{
    v8::ReductionInputs parameters;
    parameters.mode = CUDNN_REDUCE_TENSOR_ADD;
    parameters.computeType = CUDNN_DATA_FLOAT;
    parameters.isDeterministic = true;
    v8::ReductionOperationInputs inputs;
    inputs.reduction = v8::makeReductionRecord(parameters, {});
    inputs.x = makeTensor(1, {2, 3, 4, 5});
    inputs.y = makeTensor(2, {2, 3, 1, 1});
    lower({v8::makeOperationRecord(inputs, {})});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_REDUCTION_OPERATOR).value<hipdnnReduceTensorOp_t>(),
              HIPDNN_REDUCE_TENSOR_ADD);
    EXPECT_TRUE(attributeOn(op, HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC).value<bool>());
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_REDUCTION_COMP_TYPE).value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_REDUCTION_XDESC), 1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_REDUCTION_YDESC), 2);
}

TEST_F(TestCudnnShimV8CoreLowering, ResampleForwardCarriesWindowAndIndex)
{
    v8::ResampleForwardOperationInputs indexed;
    indexed.resample = makeMaxPool();
    indexed.x = makeTensor(1, {1, 4, 8, 8});
    indexed.y = makeTensor(2, {1, 4, 4, 4});
    indexed.index = makeTensor(3, {1, 4, 4, 4}, CUDNN_DATA_INT32);
    v8::ResampleForwardOperationInputs plain;
    plain.resample = makeMaxPool();
    plain.x = makeTensor(4, {1, 4, 8, 8});
    plain.y = makeTensor(5, {1, 4, 4, 4});
    lower({v8::makeOperationRecord(indexed, {}), v8::makeOperationRecord(plain, {})});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    for(hipdnnBackendDescriptor_t op : ops)
    {
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_MODE).value<hipdnnResampleMode_t>(),
                  HIPDNN_RESAMPLE_MAXPOOL);
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_PADDING_MODE).value<hipdnnPaddingMode_t>(),
                  HIPDNN_PADDING_NEG_INF_PAD);
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS).values<std::int64_t>(),
                  (std::vector<std::int64_t>{2, 2}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_STRIDES).values<std::int64_t>(),
                  (std::vector<std::int64_t>{2, 2}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS).values<std::int64_t>(),
                  (std::vector<std::int64_t>{0, 0}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS).values<std::int64_t>(),
                  (std::vector<std::int64_t>{0, 0}));
    }
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC), 1);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC), 2);
    EXPECT_TRUE(attributeOn(ops[0], HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT).value<bool>());
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC), 3);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC), 5);
    EXPECT_FALSE(callOn(ops[1], HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT));
    EXPECT_FALSE(callOn(ops[1], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC));
}

TEST_F(TestCudnnShimV8CoreLowering, ResampleBackwardSlotsAndWindow)
{
    v8::ResampleBackwardOperationInputs inputs;
    // Padding differs per axis and between pre and post; the output stays 4x4.
    inputs.resample = makeMaxPool({{1, 1}, {0, 1}}, {{0, 1}, {1, 1}});
    inputs.dy = makeTensor(1, {1, 4, 4, 4});
    inputs.index = makeTensor(2, {1, 4, 4, 4}, CUDNN_DATA_INT32);
    inputs.dx = makeTensor(3, {1, 4, 8, 8});
    lower({v8::makeOperationRecord(inputs, {})});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC), 1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC), 2);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC), 3);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_MODE).value<hipdnnResampleMode_t>(),
              HIPDNN_RESAMPLE_MAXPOOL);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS).values<std::int64_t>(),
              (std::vector<std::int64_t>{2, 2}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_STRIDES).values<std::int64_t>(),
              (std::vector<std::int64_t>{2, 2}));
    const auto paddingMode = callOn(op, HIPDNN_ATTR_RESAMPLE_PADDING_MODE);
    ASSERT_TRUE(paddingMode);
    EXPECT_EQ(paddingMode->value<hipdnnPaddingMode_t>(), HIPDNN_PADDING_NEG_INF_PAD);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS).values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 0}));
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS).values<std::int64_t>(),
              (std::vector<std::int64_t>{0, 1}));
}

TEST_F(TestCudnnShimV8CoreLowering, TensorKeepsCallerStrides)
{
    // NHWC strides for [2,3,4,5], not the packed NCHW {60,20,5,1}.
    const auto channelsLast = [](std::int64_t uid) {
        v8::TensorRecord::Fields fields = tensorFields(uid);
        fields.strides = {60, 1, 15, 3};
        return tensor(std::move(fields));
    };
    lower({relu(channelsLast(1), channelsLast(2))});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    for(const hipdnnBackendAttributeName_t name :
        {HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT})
    {
        SCOPED_TRACE(name);
        const auto strides = callOn(slotTensor(op, name), HIPDNN_ATTR_TENSOR_STRIDES);
        ASSERT_TRUE(strides);
        EXPECT_EQ(strides->values<std::int64_t>(), (std::vector<std::int64_t>{60, 1, 15, 3}));
    }
}

TEST_F(TestCudnnShimV8CoreLowering, NormForwardSlotsPerMode)
{
    const std::vector<LoweredCase> cases = {
        {"batch training",
         [] {
             v8::NormForwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.scale = makeTensor(2, {1, 3, 1, 1});
             in.bias = makeTensor(3, {1, 3, 1, 1});
             in.epsilon = makeScalar(4);
             in.y = makeTensor(5, {2, 3, 4, 4});
             in.mean = makeTensor(6, {1, 3, 1, 1});
             in.invVariance = makeTensor(7, {1, 3, 1, 1});
             return makeNormForward(CUDNN_BATCH_NORM, CUDNN_NORM_FWD_TRAINING, in);
         },
         HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT,
         HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, 2},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, 3},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, 4},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, 5},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, 6},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT, 7},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT, -1}}},
        {"batch training with running statistics",
         [] {
             v8::NormForwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.scale = makeTensor(2, {1, 3, 1, 1});
             in.bias = makeTensor(3, {1, 3, 1, 1});
             in.epsilon = makeScalar(4);
             in.y = makeTensor(5, {2, 3, 4, 4});
             in.inputRunningMean = makeTensor(6, {1, 3, 1, 1});
             in.inputRunningVariance = makeTensor(7, {1, 3, 1, 1});
             in.expAvgFactor = makeScalar(8);
             in.outputRunningMean = makeTensor(9, {1, 3, 1, 1});
             in.outputRunningVariance = makeTensor(10, {1, 3, 1, 1});
             return makeNormForward(CUDNN_BATCH_NORM, CUDNN_NORM_FWD_TRAINING, in);
         },
         HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT,
         HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, 4},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT, 6},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT, 7},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT, 8},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT, 9},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT, 10}}},
        {"batch inference",
         [] {
             v8::NormForwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.mean = makeTensor(2, {1, 3, 1, 1});
             in.invVariance = makeTensor(3, {1, 3, 1, 1});
             in.scale = makeTensor(4, {1, 3, 1, 1});
             in.bias = makeTensor(5, {1, 3, 1, 1});
             in.y = makeTensor(6, {2, 3, 4, 4});
             return makeNormForward(CUDNN_BATCH_NORM, CUDNN_NORM_FWD_INFERENCE, in);
         },
         HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT,
         HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, 2},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, 3},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, 4},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, 5},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, 6}}},
        {"layer training",
         [] {
             v8::NormForwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.scale = makeTensor(2, {1, 3, 4, 4});
             in.bias = makeTensor(3, {1, 3, 4, 4});
             in.epsilon = makeScalar(4);
             in.y = makeTensor(5, {2, 3, 4, 4});
             in.mean = makeTensor(6, {2, 1, 1, 1});
             in.invVariance = makeTensor(7, {2, 1, 1, 1});
             return makeNormForward(CUDNN_LAYER_NORM, CUDNN_NORM_FWD_TRAINING, in);
         },
         HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT,
         HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, 2},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, 3},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, 4},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, 5},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, 6},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT, 7}}},
        {"rms training",
         [] {
             v8::NormForwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.scale = makeTensor(2, {1, 3, 4, 4});
             in.epsilon = makeScalar(3);
             in.y = makeTensor(4, {2, 3, 4, 4});
             in.invVariance = makeTensor(5, {2, 1, 1, 1});
             in.bias = makeTensor(6, {1, 3, 4, 4});
             return makeNormForward(CUDNN_RMS_NORM, CUDNN_NORM_FWD_TRAINING, in);
         },
         HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT,
         HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, 2},
          {HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT, 3},
          {HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, 4},
          {HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT, 5},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, 6}}},
    };
    for(const LoweredCase& item : cases)
    {
        expectLowered(item);
    }
}

TEST_F(TestCudnnShimV8CoreLowering, LayerNormCarriesPhaseAndNormalizedDimCount)
{
    v8::NormForwardOperationInputs in;
    in.x = makeTensor(1, {2, 3, 4, 4});
    in.scale = makeTensor(2, {1, 1, 4, 4});
    in.bias = makeTensor(3, {1, 1, 4, 4});
    in.epsilon = makeScalar(4);
    in.y = makeTensor(5, {2, 3, 4, 4});
    lower({makeNormForward(CUDNN_LAYER_NORM, CUDNN_NORM_FWD_INFERENCE, in)});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT)
                  .value<std::int64_t>(),
              2);
    EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT)
                  .value<hipdnnNormFwdPhase_t>(),
              HIPDNN_NORM_FWD_INFERENCE);
    // The epsilon scalar lowers as a runtime pass-by-value tensor.
    EXPECT_TRUE(callOn(slotTensor(op, HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT),
                       HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT));
}

TEST_F(TestCudnnShimV8CoreLowering, NormBackwardSlotsPerMode)
{
    const std::vector<LoweredCase> cases = {
        {"batch backward",
         [] {
             v8::NormBackwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.dy = makeTensor(2, {2, 3, 4, 4});
             in.scale = makeTensor(3, {1, 3, 1, 1});
             in.mean = makeTensor(4, {1, 3, 1, 1});
             in.invVariance = makeTensor(5, {1, 3, 1, 1});
             in.dx = makeTensor(6, {2, 3, 4, 4});
             in.dscale = makeTensor(7, {1, 3, 1, 1});
             in.dbias = makeTensor(8, {1, 3, 1, 1});
             return makeNormBackward(CUDNN_BATCH_NORM, in);
         },
         HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT,
         HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, 2},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, 3},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, 4},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, 5},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, 6},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, 7},
          {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, 8}}},
        {"layer backward",
         [] {
             v8::NormBackwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.dy = makeTensor(2, {2, 3, 4, 4});
             in.scale = makeTensor(3, {1, 3, 4, 4});
             in.mean = makeTensor(4, {2, 1, 1, 1});
             in.invVariance = makeTensor(5, {2, 1, 1, 1});
             in.dx = makeTensor(6, {2, 3, 4, 4});
             in.dscale = makeTensor(7, {1, 3, 4, 4});
             in.dbias = makeTensor(8, {1, 3, 4, 4});
             return makeNormBackward(CUDNN_LAYER_NORM, in);
         },
         HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR_EXT,
         HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY_EXT, 2},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE_EXT, 3},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN_EXT, 4},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE_EXT, 5},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX_EXT, 6},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE_EXT, 7},
          {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS_EXT, 8}}},
        {"rms backward",
         [] {
             v8::NormBackwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.dy = makeTensor(2, {2, 3, 4, 4});
             in.scale = makeTensor(3, {1, 3, 4, 4});
             in.invVariance = makeTensor(4, {2, 1, 1, 1});
             in.dx = makeTensor(5, {2, 3, 4, 4});
             in.dscale = makeTensor(6, {1, 3, 4, 4});
             in.dbias = makeTensor(7, {1, 3, 4, 4});
             return makeNormBackward(CUDNN_RMS_NORM, in);
         },
         HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT,
         HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, 1},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, 2},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT, 3},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT, 5},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, 6},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, 7}}},
        {"rms backward without dbias",
         [] {
             v8::NormBackwardOperationInputs in;
             in.x = makeTensor(1, {2, 3, 4, 4});
             in.dy = makeTensor(2, {2, 3, 4, 4});
             in.scale = makeTensor(3, {1, 3, 4, 4});
             in.invVariance = makeTensor(4, {2, 1, 1, 1});
             in.dx = makeTensor(5, {2, 3, 4, 4});
             in.dscale = makeTensor(6, {1, 3, 4, 4});
             return makeNormBackward(CUDNN_RMS_NORM, in);
         },
         HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT,
         HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
         {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
          {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, -1}}},
    };
    for(const LoweredCase& item : cases)
    {
        expectLowered(item);
    }
}

} // namespace
