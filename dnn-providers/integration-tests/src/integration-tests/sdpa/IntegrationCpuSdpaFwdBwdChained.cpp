// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;

// Resolve the UID of a tensor by its name from a GraphWrapper's tensor map.
// Fails the test if no tensor with the given name is found.
int64_t uidByName(const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap,
                  const std::string& name)
{
    for(const auto& [uid, tensor] : tensorMap)
    {
        if(tensor->name() != nullptr && tensor->name()->str() == name)
        {
            return uid;
        }
    }
    ADD_FAILURE() << "Tensor \"" << name << "\" not found in tensor map";
    return -1;
}

// Verify that every element is finite (no NaN/Inf) and that at least one element
// is non-zero. This is a sanity check — the chained test proves that the forward
// LSE output feeds into the backward pass without crashing and produces meaningful
// gradients.
template <typename T>
void expectFiniteAndNonZero(hipdnn_data_sdk::utilities::ITensor& tensor, const std::string& name)
{
    const auto numElements = tensor.elementSpace();
    ASSERT_GT(numElements, static_cast<size_t>(0)) << name << " is empty";

    const auto* data = static_cast<const T*>(tensor.rawHostData());

    bool anyNonZero = false;
    for(size_t i = 0; i < numElements; ++i)
    {
        const auto val = static_cast<float>(data[i]);
        ASSERT_TRUE(std::isfinite(val)) << name << "[" << i << "] is not finite: " << val;
        if(val != 0.0f)
        {
            anyNonZero = true;
        }
    }
    EXPECT_TRUE(anyNonZero) << name << " is all zeros";
}

// ---------------------------------------------------------------------------
// Chained forward → backward CPU-reference test.
//
// Proves that the FP32 LSE tensor produced by the CPU forward reference can
// be consumed by the CPU backward reference, completing the end-to-end
// training path.
// ---------------------------------------------------------------------------
class IntegrationCpuSdpaFwdBwdChained : public ::testing::Test
{
protected:
    // Tensor dimensions — small enough for fast CPU execution.
    static constexpr int64_t BATCH = 2;
    static constexpr int64_t HEADS = 4;
    static constexpr int64_t SEQ_Q = 32;
    static constexpr int64_t SEQ_KV = 32;
    static constexpr int64_t HEAD_DIM = 64;

    const std::vector<int64_t> _qDims = {BATCH, HEADS, SEQ_Q, HEAD_DIM};
    const std::vector<int64_t> _kDims = {BATCH, HEADS, SEQ_KV, HEAD_DIM};
    const std::vector<int64_t> _vDims = {BATCH, HEADS, SEQ_KV, HEAD_DIM};
    const std::vector<int64_t> _oDims = {BATCH, HEADS, SEQ_Q, HEAD_DIM};
};

TEST_F(IntegrationCpuSdpaFwdBwdChained, Bf16NoMaskProducesFiniteGradients)
{
    using hipdnn_data_sdk::utilities::generateStrides;

    const auto qStrides = generateStrides(_qDims);
    const auto kStrides = generateStrides(_kDims);
    const auto vStrides = generateStrides(_vDims);
    const auto oStrides = generateStrides(_oDims);

    // --- Forward graph (with stats / LSE output) ---
    auto fwdBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(_qDims,
                                                                          qStrides,
                                                                          _kDims,
                                                                          kStrides,
                                                                          _vDims,
                                                                          vStrides,
                                                                          _oDims,
                                                                          oStrides,
                                                                          DataType::BFLOAT16,
                                                                          /*withAttnMask=*/false,
                                                                          /*withScale=*/false,
                                                                          /*withStats=*/true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper fwdGraph(
        fwdBuilder.GetBufferPointer(), fwdBuilder.GetSize());

    // Resolve UIDs by tensor name to avoid fragile hardcoded numbers.
    const auto& fwdTensorMap = fwdGraph.getTensorMap();
    const auto fwdQUid = uidByName(fwdTensorMap, "q");
    const auto fwdKUid = uidByName(fwdTensorMap, "k");
    const auto fwdVUid = uidByName(fwdTensorMap, "v");
    const auto fwdOUid = uidByName(fwdTensorMap, "o");
    const auto fwdStatsUid = uidByName(fwdTensorMap, "stats");

    // Allocate tensors and randomize inputs (Q, K, V).
    hipdnn_test_sdk::utilities::GraphTensorBundle fwdBundle(fwdTensorMap);

    constexpr unsigned int SEED = 42;
    fwdBundle.randomizeTensor(fwdQUid, -1.0f, 1.0f, SEED);
    fwdBundle.randomizeTensor(fwdKUid, -1.0f, 1.0f, SEED + 1);
    fwdBundle.randomizeTensor(fwdVUid, -1.0f, 1.0f, SEED + 2);

    // Execute forward CPU reference.
    auto fwdVariantPack = fwdBundle.toHostVariantPack();
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        fwdBuilder.GetBufferPointer(), fwdBuilder.GetSize(), fwdVariantPack);

    // Validate forward outputs are finite.
    using hipdnn_data_sdk::types::bfloat16;
    expectFiniteAndNonZero<bfloat16>(fwdBundle.getTensor(fwdOUid), "O_fwd");
    expectFiniteAndNonZero<float>(fwdBundle.getTensor(fwdStatsUid), "LSE_fwd");

    // --- Backward graph ---
    auto bwdBuilder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        _qDims, qStrides, _kDims, kStrides, _vDims, vStrides, _oDims, oStrides, DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper bwdGraph(
        bwdBuilder.GetBufferPointer(), bwdBuilder.GetSize());

    // Resolve backward UIDs by tensor name.
    const auto& bwdTensorMap = bwdGraph.getTensorMap();
    const auto bwdQUid = uidByName(bwdTensorMap, "q");
    const auto bwdKUid = uidByName(bwdTensorMap, "k");
    const auto bwdVUid = uidByName(bwdTensorMap, "v");
    const auto bwdOUid = uidByName(bwdTensorMap, "o");
    const auto bwdDoUid = uidByName(bwdTensorMap, "do");
    const auto bwdStatsUid = uidByName(bwdTensorMap, "stats");
    const auto bwdDqUid = uidByName(bwdTensorMap, "dq");
    const auto bwdDkUid = uidByName(bwdTensorMap, "dk");
    const auto bwdDvUid = uidByName(bwdTensorMap, "dv");

    hipdnn_test_sdk::utilities::GraphTensorBundle bwdBundle(bwdTensorMap);

    // Copy forward tensors into backward bundle with size safety checks.
    auto copyTensor
        = [](hipdnn_data_sdk::utilities::ITensor& dst, hipdnn_data_sdk::utilities::ITensor& src) {
              const auto srcBytes = src.elementSpace() * src.elementSize();
              const auto dstBytes = dst.elementSpace() * dst.elementSize();
              ASSERT_EQ(srcBytes, dstBytes) << "Tensor byte size mismatch in copy";
              std::memcpy(dst.rawHostData(), src.rawHostData(), srcBytes);
          };

    copyTensor(bwdBundle.getTensor(bwdQUid), fwdBundle.getTensor(fwdQUid));
    copyTensor(bwdBundle.getTensor(bwdKUid), fwdBundle.getTensor(fwdKUid));
    copyTensor(bwdBundle.getTensor(bwdVUid), fwdBundle.getTensor(fwdVUid));
    copyTensor(bwdBundle.getTensor(bwdOUid), fwdBundle.getTensor(fwdOUid));
    copyTensor(bwdBundle.getTensor(bwdStatsUid), fwdBundle.getTensor(fwdStatsUid));

    // Randomize dO (upstream gradient).
    bwdBundle.randomizeTensor(bwdDoUid, -1.0f, 1.0f, SEED + 3);

    // Execute backward CPU reference.
    auto bwdVariantPack = bwdBundle.toHostVariantPack();
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        bwdBuilder.GetBufferPointer(), bwdBuilder.GetSize(), bwdVariantPack);

    // Validate backward outputs are finite and non-zero.
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(bwdDqUid), "dQ");
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(bwdDkUid), "dK");
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(bwdDvUid), "dV");
}

} // namespace
