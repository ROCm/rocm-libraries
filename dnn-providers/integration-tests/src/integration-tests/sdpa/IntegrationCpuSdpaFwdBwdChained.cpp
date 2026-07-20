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

// Verify that every element is finite (no NaN/Inf) and that at least one element
// is non-zero. This is a sanity check — the chained test proves that the forward
// LSE output feeds into the backward pass without crashing and produces meaningful
// gradients.
template <typename T>
void expectFiniteAndNonZero(hipdnn_data_sdk::utilities::ITensor& tensor, const std::string& name)
{
    const auto numElements = tensor.elementCount();
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

    // Allocate tensors and randomize inputs (Q, K, V).
    hipdnn_test_sdk::utilities::GraphTensorBundle fwdBundle(fwdGraph.getTensorMap());

    constexpr unsigned int SEED = 42;
    // UIDs assigned by createValidSdpaFwdGraph: Q=1, K=2, V=3, O=4, Stats=5
    fwdBundle.randomizeTensor(1, -1.0f, 1.0f, SEED); // Q
    fwdBundle.randomizeTensor(2, -1.0f, 1.0f, SEED + 1); // K
    fwdBundle.randomizeTensor(3, -1.0f, 1.0f, SEED + 2); // V

    // Execute forward CPU reference.
    auto fwdVariantPack = fwdBundle.toHostVariantPack();
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        fwdBuilder.GetBufferPointer(), fwdBuilder.GetSize(), fwdVariantPack);

    // Validate forward outputs are finite.
    using hipdnn_data_sdk::types::bfloat16;
    expectFiniteAndNonZero<bfloat16>(fwdBundle.getTensor(4), "O_fwd"); // O (BF16)
    expectFiniteAndNonZero<float>(fwdBundle.getTensor(5), "LSE_fwd"); // Stats/LSE (FP32)

    // --- Backward graph ---
    // createValidSdpaBwdGraph UIDs: Q=1, K=2, V=3, O=4, dO=5, Stats=6, dQ=7, dK=8, dV=9
    auto bwdBuilder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        _qDims, qStrides, _kDims, kStrides, _vDims, vStrides, _oDims, oStrides, DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper bwdGraph(
        bwdBuilder.GetBufferPointer(), bwdBuilder.GetSize());

    hipdnn_test_sdk::utilities::GraphTensorBundle bwdBundle(bwdGraph.getTensorMap());

    // Copy forward inputs (Q, K, V) into backward tensors.
    // Forward UIDs: Q=1,K=2,V=3,O=4,Stats=5
    // Backward UIDs: Q=1,K=2,V=3,O=4,dO=5,Stats=6,dQ=7,dK=8,dV=9
    auto copyTensor = [](hipdnn_data_sdk::utilities::ITensor& dst,
                         hipdnn_data_sdk::utilities::ITensor& src) {
        std::memcpy(dst.rawHostData(), src.rawHostData(), src.elementSpace() * src.elementSize());
    };

    copyTensor(bwdBundle.getTensor(1), fwdBundle.getTensor(1)); // Q
    copyTensor(bwdBundle.getTensor(2), fwdBundle.getTensor(2)); // K
    copyTensor(bwdBundle.getTensor(3), fwdBundle.getTensor(3)); // V
    copyTensor(bwdBundle.getTensor(4), fwdBundle.getTensor(4)); // O (from fwd)
    copyTensor(bwdBundle.getTensor(6), fwdBundle.getTensor(5)); // Stats/LSE (from fwd)

    // Randomize dO (upstream gradient).
    bwdBundle.randomizeTensor(5, -1.0f, 1.0f, SEED + 3); // dO

    // Execute backward CPU reference.
    auto bwdVariantPack = bwdBundle.toHostVariantPack();
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        bwdBuilder.GetBufferPointer(), bwdBuilder.GetSize(), bwdVariantPack);

    // Validate backward outputs are finite and non-zero.
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(7), "dQ"); // dQ (BF16)
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(8), "dK"); // dK (BF16)
    expectFiniteAndNonZero<bfloat16>(bwdBundle.getTensor(9), "dV"); // dV (BF16)
}

} // namespace
