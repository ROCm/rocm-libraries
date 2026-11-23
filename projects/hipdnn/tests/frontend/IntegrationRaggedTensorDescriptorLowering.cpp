// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;

namespace
{

namespace fbdata = hipdnn_flatbuffers_sdk::data_objects;

// UIDs for the auxiliary ragged-offset tensors; distinct from the primary
// pointwise tensor UIDs.
constexpr int64_t K_RAGGED_OFFSET_UID = 1399;
constexpr int64_t K_RAGGED_OFFSET_UID_1 = 1398;

// Expected shape/type of every ragged-offset aux tensor built by the helpers.
const std::vector<int64_t> K_RAGGED_OFFSET_DIMS{2, 1, 1, 1};
const std::vector<int64_t> K_RAGGED_OFFSET_STRIDES{1, 1, 1, 1};

// End-to-end lowering of ragged tensors (RFC 0014): builds a frontend graph,
// lowers it through the REAL backend via
// build_operation_graph_via_descriptors(), retrieves the serialized binary
// graph, and deserializes the flatbuffer schema to assert the per-tensor
// ragged-offset link was actually written into the wire image.
class IntegrationRaggedTensorDescriptorLowering : public IntegrationTestFixture
{
protected:
    // Builds a minimal unary pointwise (RELU) graph. When @p withRaggedOffset is
    // true, the input tensor is given a ragged-offset aux tensor.
    static std::shared_ptr<TestableGraphLowering> makePointwiseGraph(bool withRaggedOffset)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("RaggedLoweringGraph")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto in0 = std::make_shared<TensorAttributes>();
        in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
        in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

        if(withRaggedOffset)
        {
            auto raggedOffset = std::make_shared<TensorAttributes>();
            raggedOffset->set_uid(K_RAGGED_OFFSET_UID)
                .set_name("RaggedOffset")
                .set_data_type(DataType::INT64)
                .set_dim({2, 1, 1, 1})
                .set_stride({1, 1, 1, 1});
            in0->set_ragged_offset(raggedOffset);
        }

        PointwiseAttributes pwAttrs;
        pwAttrs.set_name("relu_op");
        pwAttrs.set_mode(PointwiseMode::RELU_FWD);

        auto out0 = graph->pointwise(in0, pwAttrs);
        out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

        return graph;
    }

    // How the two inputs of the binary graph are wired to ragged-offset auxes.
    enum class AuxWiring
    {
        SHARED, // both inputs point at the same aux tensor
        SEPARATE, // each input owns a distinct aux tensor
        AUX_IS_INPUT, // in0's aux is in1, which is also the second ADD operand
        NESTED, // in0's aux itself carries a ragged offset (illegal)
        VIRTUAL, // in0's aux is marked virtual (illegal: no backing storage)
    };

    // Builds a minimal binary pointwise (ADD) graph over in0/in1, wiring the
    // ragged-offset aux tensors per @p wiring.
    static std::shared_ptr<TestableGraphLowering> makeBinaryPointwiseGraph(AuxWiring wiring)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("RaggedBinaryLoweringGraph")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto in0 = std::make_shared<TensorAttributes>();
        in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
        in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

        auto in1 = std::make_shared<TensorAttributes>();
        in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
        in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

        auto makeAux = [](int64_t uid) {
            auto aux = std::make_shared<TensorAttributes>();
            aux->set_uid(uid)
                .set_name("RaggedOffset" + std::to_string(uid))
                .set_data_type(DataType::INT64)
                .set_dim({2, 1, 1, 1})
                .set_stride({1, 1, 1, 1});
            return aux;
        };

        switch(wiring)
        {
        case AuxWiring::SHARED:
        {
            auto aux = makeAux(K_RAGGED_OFFSET_UID);
            in0->set_ragged_offset(aux);
            in1->set_ragged_offset(aux);
            break;
        }
        case AuxWiring::SEPARATE:
            in0->set_ragged_offset(makeAux(K_RAGGED_OFFSET_UID));
            in1->set_ragged_offset(makeAux(K_RAGGED_OFFSET_UID_1));
            break;
        case AuxWiring::AUX_IS_INPUT:
            in0->set_ragged_offset(in1);
            break;
        case AuxWiring::NESTED:
        {
            auto aux = makeAux(K_RAGGED_OFFSET_UID);
            aux->set_ragged_offset(in1);
            in0->set_ragged_offset(aux);
            break;
        }
        case AuxWiring::VIRTUAL:
        {
            auto aux = makeAux(K_RAGGED_OFFSET_UID);
            aux->set_is_virtual(true);
            in0->set_ragged_offset(aux);
            break;
        }
        default:
            break;
        }

        PointwiseAttributes pwAttrs;
        pwAttrs.set_name("add_op");
        pwAttrs.set_mode(PointwiseMode::ADD);

        auto out0 = graph->pointwise(in0, in1, pwAttrs);
        out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

        return graph;
    }

    // Two-call serialization of a backend graph descriptor into its wire bytes.
    static std::vector<uint8_t> serializeDescriptor(hipdnnBackendDescriptor_t desc)
    {
        size_t size = 0;
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &size, nullptr),
                  HIPDNN_STATUS_SUCCESS);
        std::vector<uint8_t> bytes(size);
        if(size != 0)
        {
            EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, size, &size, bytes.data()),
                      HIPDNN_STATUS_SUCCESS);
        }
        return bytes;
    }

    // Lowers a frontend graph through the real backend and returns its serialized
    // wire bytes (the same path lowerAndDeserialize() consumes internally).
    static std::vector<uint8_t> lowerToSerializedBytes(TestableGraphLowering& graph,
                                                       hipdnnHandle_t handle)
    {
        auto validated = graph.validate();
        EXPECT_EQ(validated.code, ErrorCode::OK) << validated.err_msg;

        auto built = graph.build_operation_graph_via_descriptors(handle);
        EXPECT_EQ(built.code, ErrorCode::OK) << built.err_msg;

        auto rawDesc = graph.get_raw_graph_descriptor();
        EXPECT_NE(rawDesc, nullptr); // NOLINT(readability-implicit-bool-conversion)
        if(rawDesc == nullptr)
        {
            return {};
        }
        return serializeDescriptor(rawDesc);
    }

    // Lowers a frontend graph expected to carry a malformed ragged aux and
    // asserts the descriptor lowering path rejects it with INVALID_VALUE.
    // validate() has no ragged-specific checks, so lowering is the enforcement
    // point (mirroring the backend deserialize rejections).
    static void expectLoweringRejected(TestableGraphLowering& graph, hipdnnHandle_t handle)
    {
        auto validated = graph.validate();
        EXPECT_EQ(validated.code, ErrorCode::OK) << validated.err_msg;

        auto built = graph.build_operation_graph_via_descriptors(handle);
        EXPECT_EQ(built.code, ErrorCode::INVALID_VALUE) << built.err_msg;
    }

    // Maps each tensor UID to its ragged-offset link (-1 when absent), the
    // invariant that must survive a serialize -> deserialize -> re-serialize.
    static std::map<int64_t, int64_t> raggedSignature(const fbdata::GraphT& graphT)
    {
        std::map<int64_t, int64_t> signature;
        for(const auto& t : graphT.tensors)
        {
            signature[t->uid] = t->ragged_offset_tensor_uid.has_value()
                                    ? t->ragged_offset_tensor_uid.value()
                                    : -1;
        }
        return signature;
    }
};

} // namespace

// A graph whose input carries a ragged offset must serialize with the
// ragged-offset link recorded on that tensor in the flatbuffer schema.
TEST_F(IntegrationRaggedTensorDescriptorLowering, RaggedOffsetSetsTensorLinkInSchema)
{
    auto graph = makePointwiseGraph(/*withRaggedOffset=*/true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // Sanity: lowering produced a well-formed graph (in0, in0Offset, out0).
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto findTensorIt = [&](int64_t uid) {
        return std::find_if(graphT.tensors.begin(), graphT.tensors.end(), [uid](const auto& t) {
            return t->uid == uid;
        });
    };

    // The primary input tensor must carry the ragged-offset link (by UID) in the
    // serialized schema; this is the sole source of truth for ragged-ness.
    const auto in0It = findTensorIt(K_PW_TENSOR_IN0_UID);
    ASSERT_NE(in0It, graphT.tensors.end()) << "input tensor IN0 missing from serialized graph";
    ASSERT_TRUE((*in0It)->ragged_offset_tensor_uid.has_value())
        << "IN0 must carry ragged_offset_tensor_uid in the serialized schema.";
    EXPECT_EQ((*in0It)->ragged_offset_tensor_uid.value(), K_RAGGED_OFFSET_UID)
        << "IN0's ragged_offset_tensor_uid must point at the ragged-offset aux tensor.";

    auto in0OffsetIt = findTensorIt(K_RAGGED_OFFSET_UID);
    ASSERT_NE(in0OffsetIt, graphT.tensors.end())
        << "input tensor IN0's ragged offset missing from serialized graph";

    // The aux must land in the tensor list with its own dims/strides/dtype so
    // engines can reach its shape and device pointer.
    EXPECT_EQ((*in0OffsetIt)->data_type, fbdata::DataType::INT64);
    EXPECT_EQ((*in0OffsetIt)->dims, K_RAGGED_OFFSET_DIMS);
    EXPECT_EQ((*in0OffsetIt)->strides, K_RAGGED_OFFSET_STRIDES);
}

// Sharing one aux shared_ptr across both inputs must emit exactly one aux tensor,
// with both inputs linking to that single UID.
TEST_F(IntegrationRaggedTensorDescriptorLowering, SharedRaggedOffsetEmitsSingleAux)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::SHARED);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // in0, in1, out0, and a single shared aux.
    ASSERT_EQ(graphT.tensors.size(), 4u);

    const auto auxCount
        = std::count_if(graphT.tensors.begin(), graphT.tensors.end(), [](const auto& t) {
              return t->uid == K_RAGGED_OFFSET_UID;
          });
    EXPECT_EQ(auxCount, 1) << "a shared aux must be emitted exactly once.";

    const auto tensorMap = hipdnn_tests::buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);

    const auto* in0 = tensorMap.at(K_PW_TENSOR_IN0_UID);
    const auto* in1 = tensorMap.at(K_PW_TENSOR_IN1_UID);
    ASSERT_TRUE(in0->ragged_offset_tensor_uid.has_value());
    ASSERT_TRUE(in1->ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(in0->ragged_offset_tensor_uid.value(), K_RAGGED_OFFSET_UID);
    EXPECT_EQ(in1->ragged_offset_tensor_uid.value(), K_RAGGED_OFFSET_UID);
}

// Distinct auxes per input must emit distinct aux tensors, producing a larger
// tensor list than the shared case over the same operands.
TEST_F(IntegrationRaggedTensorDescriptorLowering, SeparateRaggedOffsetsEmitDistinctTensorList)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::SEPARATE);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // in0, in1, out0, and two distinct auxes.
    ASSERT_EQ(graphT.tensors.size(), 5u);

    const auto tensorMap = hipdnn_tests::buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);

    const auto* in0 = tensorMap.at(K_PW_TENSOR_IN0_UID);
    const auto* in1 = tensorMap.at(K_PW_TENSOR_IN1_UID);
    ASSERT_TRUE(in0->ragged_offset_tensor_uid.has_value());
    ASSERT_TRUE(in1->ragged_offset_tensor_uid.has_value());
    EXPECT_NE(in0->ragged_offset_tensor_uid.value(), in1->ragged_offset_tensor_uid.value())
        << "separate auxes must serialize to distinct UIDs.";
    EXPECT_NE(tensorMap.count(K_RAGGED_OFFSET_UID), 0u);
    EXPECT_NE(tensorMap.count(K_RAGGED_OFFSET_UID_1), 0u);
}

// An aux tensor that is also a graph input must dedup by UID: lowering must not
// throw and the shared tensor must appear exactly once.
TEST_F(IntegrationRaggedTensorDescriptorLowering, AuxThatIsAlsoGraphInputEmittedOnce)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::AUX_IS_INPUT);

    fbdata::GraphT graphT;
    ASSERT_NO_THROW(graphT = lowerAndDeserialize(*graph, _handle));

    // in0, in1 (== aux), out0 — the aux is not a separate tensor.
    ASSERT_EQ(graphT.tensors.size(), 3u);

    const auto in1Count
        = std::count_if(graphT.tensors.begin(), graphT.tensors.end(), [](const auto& t) {
              return t->uid == K_PW_TENSOR_IN1_UID;
          });
    EXPECT_EQ(in1Count, 1) << "an aux that is also an input must be emitted exactly once.";

    const auto tensorMap = hipdnn_tests::buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    const auto* in0 = tensorMap.at(K_PW_TENSOR_IN0_UID);
    ASSERT_TRUE(in0->ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(in0->ragged_offset_tensor_uid.value(), K_PW_TENSOR_IN1_UID);
}

// The ragged tensor list (UIDs plus per-tensor ragged-offset links) must be
// identical after a serialize -> deserialize -> re-serialize round trip.
TEST_F(IntegrationRaggedTensorDescriptorLowering, RaggedTensorListStableAcrossReserialize)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::SHARED);

    auto firstBytes = lowerToSerializedBytes(*graph, _handle);
    ASSERT_FALSE(firstBytes.empty());

    hipdnnBackendDescriptor_t revived = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&revived, firstBytes.data(), firstBytes.size()),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(revived, nullptr); // NOLINT(readability-implicit-bool-conversion)

    auto secondBytes = serializeDescriptor(revived);
    EXPECT_EQ(hipdnnBackendDestroyDescriptor(revived), HIPDNN_STATUS_SUCCESS);
    ASSERT_FALSE(secondBytes.empty());

    auto first = fbdata::UnPackGraph(firstBytes.data());
    auto second = fbdata::UnPackGraph(secondBytes.data());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    const auto firstSig = raggedSignature(*first);
    ASSERT_EQ(firstSig.size(), 4u);
    EXPECT_EQ(firstSig.at(K_PW_TENSOR_IN0_UID), K_RAGGED_OFFSET_UID);
    EXPECT_EQ(firstSig.at(K_PW_TENSOR_IN1_UID), K_RAGGED_OFFSET_UID);

    EXPECT_EQ(raggedSignature(*first), raggedSignature(*second))
        << "tensor UID set and ragged-offset links must be stable across reserialize.";
}

// A graph with no ragged offsets must serialize with no ragged-offset link on
// any tensor, so ragged-ness derives as false and engine plugins are not
// spuriously gated on ragged-tensor support.
TEST_F(IntegrationRaggedTensorDescriptorLowering, NonRaggedGraphHasNoTensorLinkInSchema)
{
    auto graph = makePointwiseGraph(/*withRaggedOffset=*/false);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.tensors.size(), 2u);

    // No tensor may carry a ragged-offset link in a non-ragged graph.
    for(const auto& t : graphT.tensors)
    {
        EXPECT_FALSE(t->ragged_offset_tensor_uid.has_value())
            << "tensor uid " << t->uid
            << " unexpectedly carries ragged_offset_tensor_uid in a non-ragged graph.";
    }
}

// A ragged-offset aux that itself carries a ragged offset is illegal: lowering
// must reject it (also breaking the offset cycle) rather than recursing without
// bound while building the tensor descriptors.
TEST_F(IntegrationRaggedTensorDescriptorLowering, NestedRaggedOffsetRejectedInLowering)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::NESTED);
    expectLoweringRejected(*graph, _handle);
}

// A virtual ragged-offset aux has no backing storage and must be rejected at
// lowering, symmetric with the backend deserialize enforcement.
TEST_F(IntegrationRaggedTensorDescriptorLowering, VirtualRaggedOffsetRejectedInLowering)
{
    auto graph = makeBinaryPointwiseGraph(AuxWiring::VIRTUAL);
    expectLoweringRejected(*graph, _handle);
}

#ifdef HIPDNN_ENABLE_SDPA
// The THD SDPA form as a JSON graph file carries it (RFC 0014): Q/O and the LSE packed
// through one offset table, K/V through another, each with its token stride as its
// multiplier, and a 4-byte-aligned LSE. Graph::deserialize() without a handle -- what
// from_json(json) runs -- lifts the backend graph into frontend nodes, and
// build_operation_graph() then re-lowers those nodes; every ragged link, multiplier and
// alignment must survive both, or the engines get a dense graph with no offset tables
// and read every sequence at the padded length.
TEST_F(IntegrationRaggedTensorDescriptorLowering, PackedSdpaSurvivesJsonLiftAndRelower)
{
    constexpr int64_t K_BATCH = 3;
    constexpr int64_t K_HEADS = 2;
    constexpr int64_t K_SEQ = 64;
    constexpr int64_t K_HEAD_DIM = 64;
    constexpr int64_t K_TOKEN_STRIDE = K_HEADS * K_HEAD_DIM;
    constexpr int64_t K_OFFSETS_Q_UID = 20;
    constexpr int64_t K_OFFSETS_KV_UID = 21;
    constexpr int64_t K_LSE_ALIGNMENT = 4;

    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("PackedSdpaLiftGraph")
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto offsets = [](int64_t uid, const std::string& name) {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(uid)
            .set_name(name)
            .set_data_type(DataType::INT32)
            .set_dim({K_BATCH + 1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_alignment(4);
        return t;
    };
    const auto offsetsQ = offsets(K_OFFSETS_Q_UID, "OffsetsQ");
    const auto offsetsKv = offsets(K_OFFSETS_KV_UID, "OffsetsKv");
    const std::vector<int64_t> dims{K_BATCH, K_HEADS, K_SEQ, K_HEAD_DIM};
    const std::vector<int64_t> strides{K_SEQ * K_TOKEN_STRIDE, K_HEAD_DIM, K_TOKEN_STRIDE, 1};
    auto packed = [&](int64_t uid, const std::string& name, const auto& table) {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(uid)
            .set_name(name)
            .set_data_type(DataType::HALF)
            .set_dim(dims)
            .set_stride(strides);
        t->set_ragged_offset(table).set_ragged_offset_multiplier(K_TOKEN_STRIDE);
        return t;
    };
    auto q = packed(1, "Q", offsetsQ);
    auto k = packed(2, "K", offsetsKv);
    auto v = packed(3, "V", offsetsKv);

    SdpaAttributes attributes;
    attributes.set_name("sdpa").set_attn_scale(0.125F).set_generate_stats(true);
    auto outputs = graph->sdpa(q, k, v, attributes);
    outputs[0]->set_uid(4).set_name("O").set_output(true).set_data_type(DataType::HALF);
    outputs[0]->set_dim(dims).set_stride(strides);
    outputs[0]->set_ragged_offset(offsetsQ).set_ragged_offset_multiplier(K_TOKEN_STRIDE);
    outputs[1]->set_uid(5).set_name("LSE").set_output(true).set_data_type(DataType::FLOAT);
    outputs[1]
        ->set_dim({K_BATCH, K_HEADS, K_SEQ, 1})
        .set_stride({K_SEQ * K_HEADS, 1, K_HEADS, 1})
        .set_alignment(K_LSE_ALIGNMENT);
    outputs[1]->set_ragged_offset(offsetsQ).set_ragged_offset_multiplier(K_HEADS);

    ASSERT_EQ(graph->validate().code, ErrorCode::OK);
    std::string originalJson;
    auto result = graph->serialize(originalJson);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // The dnn-benchmarking / Python from_json(json) path: structure-only deserialize, then
    // build_operation_graph(handle), which re-lowers from the lifted nodes.
    auto lifted = std::make_shared<TestableGraphLowering>();
    result = lifted->deserialize(originalJson);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(lifted->validate().code, ErrorCode::OK);
    result = lifted->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::string reloweredJson;
    result = lifted->serialize(reloweredJson);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto byUid = [](const std::string& json) {
        std::map<int64_t, nlohmann::json> tensors;
        const auto parsed = nlohmann::json::parse(json);
        for(const auto& t : parsed.at("tensors"))
        {
            tensors[t.at("uid").get<int64_t>()] = t;
        }
        return tensors;
    };
    const auto before = byUid(originalJson);
    const auto after = byUid(reloweredJson);
    ASSERT_EQ(before.size(), 7u) << "Q, K, V, O, LSE and the two offset tables";
    ASSERT_EQ(after.size(), before.size()) << "an offset table was dropped: " << reloweredJson;
    for(const auto& [uid, tensor] : before)
    {
        ASSERT_EQ(after.count(uid), 1u) << "tensor " << uid << " dropped";
        for(const char* field : {"ragged_offset_tensor_uid",
                                 "ragged_offset_multiplier",
                                 "alignment",
                                 "dims",
                                 "strides",
                                 "data_type"})
        {
            const auto want = tensor.contains(field) ? tensor.at(field) : nlohmann::json();
            const auto got
                = after.at(uid).contains(field) ? after.at(uid).at(field) : nlohmann::json();
            EXPECT_EQ(got, want) << "tensor " << uid << " field " << field;
        }
    }
    EXPECT_EQ(after.at(5).at("ragged_offset_tensor_uid").get<int64_t>(), K_OFFSETS_Q_UID);
    EXPECT_EQ(after.at(2).at("ragged_offset_tensor_uid").get<int64_t>(), K_OFFSETS_KV_UID);
    EXPECT_EQ(after.at(5).at("alignment").get<int64_t>(), K_LSE_ALIGNMENT);
}
#endif // HIPDNN_ENABLE_SDPA
