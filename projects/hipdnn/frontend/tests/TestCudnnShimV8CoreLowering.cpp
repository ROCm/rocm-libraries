// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v8 core's graph lowering and engine index space, driven with records
// directly (no cuDNN backend descriptors) against the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/lowering.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
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

const char* operationName(cudnnBackendDescriptorType_t /*kind*/)
{
    return "OP";
}

const v8::LoweringLabels LABELS{"OPS", "CONSTANT", &operationName};

std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

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

template <typename Fn>
std::pair<cudnnStatus_t, std::string> shimError(Fn&& fn)
{
    try
    {
        fn();
    }
    catch(const v8::ShimError& error)
    {
        return {error.status(), error.what()};
    }
    return {CUDNN_STATUS_SUCCESS, ""};
}

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
                uids.push_back(uidOf(slot(op, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT)));
            }
        }
        return uids;
    }

    hipdnnBackendDescriptor_t slot(hipdnnBackendDescriptor_t op,
                                   hipdnnBackendAttributeName_t name) const
    {
        for(const auto& call : setAttributesOf(op))
        {
            if(call.name == name)
            {
                return call.value<hipdnnBackendDescriptor_t>();
            }
        }
        return nullptr;
    }

    std::int64_t uidOf(hipdnnBackendDescriptor_t hipdnnTensor) const
    {
        for(const auto& call : setAttributesOf(hipdnnTensor))
        {
            if(call.name == HIPDNN_ATTR_TENSOR_UNIQUE_ID)
            {
                return call.value<std::int64_t>();
            }
        }
        return -1;
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

TEST(TestCudnnShimV8CoreLoweringOrder, RejectsDoubleWriter)
{
    const std::vector<v8::OperationPorts> graph
        = {ports({1}, {3}), ports({10}, {11}), ports({2}, {3})};
    const auto [status, message] = shimError([&] { v8::stableTopologicalOrder(graph, LABELS); });
    EXPECT_EQ(status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(message, "OPS: UID 3 is written by operations 0 and 2");
}

TEST(TestCudnnShimV8CoreLoweringOrder, RejectsCycle)
{
    const std::vector<v8::OperationPorts> graph
        = {ports({1}, {2}), ports({10}, {11}), ports({2}, {1})};
    const auto [status, message] = shimError([&] { v8::stableTopologicalOrder(graph, LABELS); });
    EXPECT_EQ(status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(message, "OPS: the operations form a cycle");
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
        const auto [status, message] = shimError([&] { tensors.add(second); });
        EXPECT_EQ(status, CUDNN_STATUS_BAD_PARAM);
        EXPECT_EQ(message, std::string("OPS: two tensors with UID 7 differ in ") + c.difference);
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

TEST(TestCudnnShimV8CoreLoweringTensors, ConstantOfUncarriableTypeNotSupported)
{
    v8::CanonicalTensors tensors(LABELS);
    const auto [status, message]
        = shimError([&] { tensors.add(constant(5, DataType_t::INT8, std::int8_t{3})); });
    EXPECT_EQ(status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(message, "CONSTANT of UID 5: hipDNN cannot carry a constant of this data type");
}

TEST_F(TestCudnnShimV8CoreLowering, LowersInStableTopologicalOrder)
{
    // relu(2 -> 3) consumes relu(1 -> 2); relu(10 -> 11) is disconnected.
    const std::shared_ptr<const v8::GraphCore> core
        = v8::lowerGraph({relu(2, 3), relu(10, 11), relu(1, 2)}, _handle, LABELS, "tag");

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
    v8::lowerGraph({relu(1, 2), relu(2, 3)}, _handle, LABELS, {});
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
    const auto [cycle, cycleMessage]
        = shimError([&] { v8::lowerGraph({relu(1, 2), relu(2, 1)}, _handle, LABELS, {}); });
    EXPECT_EQ(cycle, CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cycleMessage, "OPS: the operations form a cycle");
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());

    // A norm record without a lowerable target names the operation through the
    // caller's callback.
    v8::NormForwardOperationRecord::Fields norm;
    norm.x = tensor(1);
    norm.y = tensor(2);
    const auto [target, targetMessage] = shimError([&] {
        v8::lowerGraph({std::make_shared<const v8::NormForwardOperationRecord>(
                           v8::OperationTarget::NOT_SET, std::move(norm))},
                       _handle,
                       LABELS,
                       {});
    });
    EXPECT_EQ(target, CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(targetMessage, "OP has no lowering target");
}

TEST_F(TestCudnnShimV8CoreLowering, DenseIndexFollowsSignedId)
{
    installHeuristicResults({5, -2, 300, -40});
    const std::shared_ptr<const v8::GraphCore> core
        = v8::lowerGraph({relu(1, 2)}, _handle, LABELS, {});

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
        = v8::lowerGraph({relu(1, 2)}, _handle, LABELS, {});

    // -1 is dropped and the duplicate 7 is kept once, at its first rank.
    EXPECT_EQ(core->rankedEngineIds(), (std::vector<std::int64_t>{7, 4}));
    EXPECT_EQ(core->engineCount(), 2);
    EXPECT_EQ(core->engineIdAt(0), 4);
    EXPECT_EQ(core->engineIdAt(1), 7);
    EXPECT_EQ(core->engineIndexOf(-1), std::nullopt);
    EXPECT_EQ(core->defaultEngineConfig(-1), nullptr);
}

} // namespace
