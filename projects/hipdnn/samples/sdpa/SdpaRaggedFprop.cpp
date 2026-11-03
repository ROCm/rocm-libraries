// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

namespace
{

// Ragged SDPA runner: iterates data types but hardcodes the BSHD physical layout
// (the only ragged-legal layout) over BHSD logical dims. Filtering mirrors
// runSdpa() in SdpaFprop.cpp.
template <typename F>
bool runSdpaRagged(F&& f)
{
    bool allPassed = true;

    warnOnUnknownEngineName(f.handle, f.config);

    const std::vector<std::string> dtypes = {"fp32", "fp16", "bf16"};

    for(const auto& dt : dtypes)
    {
        if(!f.config.dtype.empty() && f.config.dtype != dt)
        {
            continue;
        }

        if(dt == "fp32")
        {
            allPassed &= f.template operator()<float, float>(TensorLayout::BSHD);
        }
        else if(dt == "fp16")
        {
            allPassed &= f.template operator()<half, float>(TensorLayout::BSHD);
        }
        else if(dt == "bf16")
        {
            allPassed &= f.template operator()<bfloat16, float>(TensorLayout::BSHD);
        }
    }

    return allPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running ragged SDPA forward graph " << inputType << " [bshd]...\n";

    // ── Dimensions ──────────────────────────────────────────────────────
    // Logical dims: [batch, num_heads, seq_len, head_dim] (BHSD)
    constexpr int64_t B = 3;
    constexpr int64_t H = 4;
    constexpr int64_t S_MAX = 5;
    constexpr int64_t D = 128;
    constexpr int64_t TOKEN_STRIDE = H * D; // elements per token (sequence stride in BSHD)

    // Per-batch sequence lengths (ragged)
    const std::vector<int32_t> qSeqLens = {2, 5, 3};
    const std::vector<int32_t> kvSeqLens = {3, 5, 4};

    // ── Ragged offsets (cumulative sums in token units) ─────────────────
    // Each ragged tensor sets ragged_offset_multiplier = H*D, so the backend scales these
    // token offsets to element offsets. That keeps the offset tables independent of H and D.
    auto computeOffsets = [](const std::vector<int32_t>& seqLens) -> std::vector<int32_t> {
        std::vector<int32_t> offsets;
        offsets.reserve(seqLens.size() + 1);
        offsets.push_back(0);
        for(const auto seqLen : seqLens)
        {
            offsets.push_back(offsets.back() + seqLen);
        }
        return offsets;
    };

    const auto qOffsetsHost = computeOffsets(qSeqLens);
    const auto kvOffsetsHost = computeOffsets(kvSeqLens);

    // ── Graph construction ──────────────────────────────────────────────
    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    setPreferredEngine(graph, config);

    // BHSD logical dims with BSHD physical strides: [S_MAX*H*D, D, H*D, 1].
    // The batch stride is ignored for ragged tensors in favor of the ragged offset.
    const std::vector<int64_t> qDims = {B, H, S_MAX, D};
    const std::vector<int64_t> kvDims = {B, H, S_MAX, D};

    auto bshdStrides = utilities::generateStrides(qDims, layout.strideOrder);

    auto qAttr = std::make_shared<graph::TensorAttributes>();
    qAttr->set_dim(qDims).set_data_type(inputType).set_stride(bshdStrides);

    auto kAttr = std::make_shared<graph::TensorAttributes>();
    kAttr->set_dim(kvDims).set_data_type(inputType).set_stride(bshdStrides);

    auto vAttr = std::make_shared<graph::TensorAttributes>();
    vAttr->set_dim(kvDims).set_data_type(inputType).set_stride(bshdStrides);

    // ── Ragged offset TensorAttributes ({B+1, 1, 1, 1}, INT32) ────────
    const std::vector<int64_t> raggedOffsetDims = {B + 1, 1, 1, 1};
    const auto raggedOffsetStrides = utilities::generateStrides(raggedOffsetDims);

    auto qRaggedAttr = std::make_shared<graph::TensorAttributes>();
    qRaggedAttr->set_dim(raggedOffsetDims)
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_stride(raggedOffsetStrides);

    auto kvRaggedAttr = std::make_shared<graph::TensorAttributes>();
    kvRaggedAttr->set_dim(raggedOffsetDims)
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_stride(raggedOffsetStrides);

    // Wire ragged offsets to primary tensors; the multiplier converts token offsets to
    // elements. K and V share the same offset aux.
    qAttr->set_ragged_offset(qRaggedAttr).set_ragged_offset_multiplier(TOKEN_STRIDE);
    kAttr->set_ragged_offset(kvRaggedAttr).set_ragged_offset_multiplier(TOKEN_STRIDE);
    vAttr->set_ragged_offset(kvRaggedAttr).set_ragged_offset_multiplier(TOKEN_STRIDE);

    // ── Sequence-length TensorAttributes ({B, 1, 1, 1}, INT32) ────────
    const std::vector<int64_t> seqLenDims = {B, 1, 1, 1};
    const auto seqLenStrides = utilities::generateStrides(seqLenDims);

    auto seqLenQAttr = std::make_shared<graph::TensorAttributes>();
    seqLenQAttr->set_dim(seqLenDims)
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_stride(seqLenStrides);

    auto seqLenKvAttr = std::make_shared<graph::TensorAttributes>();
    seqLenKvAttr->set_dim(seqLenDims)
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_stride(seqLenStrides);

    // ── Attention scale (compile-time constant) ────────────────────────
    const float attnScaleVal = 1.0f / std::sqrt(static_cast<float>(D));
    auto attnScale = std::make_shared<graph::TensorAttributes>();
    attnScale->set_dim({1}).set_stride({1}).set_data_type(getDataTypeEnumFromType<float>());
    attnScale->set_compile_time_constant(attnScaleVal);

    // ── SDPA node ──────────────────────────────────────────────────────
    graph::SdpaAttributes sdpaAttributes;
    sdpaAttributes.set_name("sdpa_ragged_fprop_node");
    sdpaAttributes.set_padding_mask(true);
    sdpaAttributes.set_seq_len_q(seqLenQAttr);
    sdpaAttributes.set_seq_len_kv(seqLenKvAttr);
    sdpaAttributes.set_attn_scale(attnScale);

    auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, std::move(sdpaAttributes));
    oAttr->set_output(true);

    // Wire ragged offset to O (uses Q's offsets since O has Q's sequence lengths)
    oAttr->set_ragged_offset(qRaggedAttr).set_ragged_offset_multiplier(TOKEN_STRIDE);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    // ── Runtime tensor allocation ──────────────────────────────────────

    // Ragged offset aux tensors (dense Tensor<int32_t>, held as shared_ptr<ITensor>)
    auto qRaggedOffset
        = std::make_shared<utilities::Tensor<int32_t>>(std::vector<int64_t>{B + 1, 1, 1, 1});
    for(size_t i = 0; i < qOffsetsHost.size(); ++i)
    {
        qRaggedOffset->setHostValue(
            qOffsetsHost[i], static_cast<int64_t>(i), int64_t{0}, int64_t{0}, int64_t{0});
    }

    auto kvRaggedOffset
        = std::make_shared<utilities::Tensor<int32_t>>(std::vector<int64_t>{B + 1, 1, 1, 1});
    for(size_t i = 0; i < kvOffsetsHost.size(); ++i)
    {
        kvRaggedOffset->setHostValue(
            kvOffsetsHost[i], static_cast<int64_t>(i), int64_t{0}, int64_t{0}, int64_t{0});
    }

    // Ragged primaries — demonstrating both construction forms:
    //   Form (a): size-inference (reads ragged_offset[B] * multiplier from aux)
    //   Form (b): explicit physicalElementCount
    // The sequence axis is SDPA_SEQ_AXIS (2) because dims are BHSD logical.

    // Q: Form (a) — size-inference
    utilities::RaggedTensor<InputType> qTensor(
        qDims, bshdStrides, utilities::SDPA_SEQ_AXIS, qRaggedOffset, std::nullopt, TOKEN_STRIDE);

    // K: Form (b) — explicit physicalElementCount (total tokens * H*D elements)
    utilities::RaggedTensor<InputType> kTensor(
        kvDims,
        bshdStrides,
        utilities::SDPA_SEQ_AXIS,
        kvRaggedOffset,
        static_cast<size_t>(static_cast<int64_t>(kvOffsetsHost.back()) * TOKEN_STRIDE),
        TOKEN_STRIDE);

    // V: Form (a) — size-inference, shares kvRaggedOffset with K
    utilities::RaggedTensor<InputType> vTensor(
        kvDims, bshdStrides, utilities::SDPA_SEQ_AXIS, kvRaggedOffset, std::nullopt, TOKEN_STRIDE);

    // O: Form (a) — size-inference, uses Q's ragged offsets
    utilities::RaggedTensor<InputType> oTensor(
        qDims, bshdStrides, utilities::SDPA_SEQ_AXIS, qRaggedOffset, std::nullopt, TOKEN_STRIDE);

    // Sequence-length tensors (dense, non-ragged — constructed as today)
    utilities::Tensor<int32_t> seqLenQTensor(seqLenDims);
    utilities::Tensor<int32_t> seqLenKvTensor(seqLenDims);

    // Fill ragged primaries
    qTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    kTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    vTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    oTensor.fillWithValue(static_cast<InputType>(0.0f));

    // Fill seq_len tensors
    for(int64_t b = 0; b < B; ++b)
    {
        seqLenQTensor.setHostValue(
            qSeqLens[static_cast<size_t>(b)], b, int64_t{0}, int64_t{0}, int64_t{0});
        seqLenKvTensor.setHostValue(
            kvSeqLens[static_cast<size_t>(b)], b, int64_t{0}, int64_t{0}, int64_t{0});
    }

    // ── Variant pack ───────────────────────────────────────────────────
    // Ragged primaries and aux tensors use rawDeviceData().
    // Dense seq_len tensors use .memory().deviceData() (as today).
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[qAttr->get_uid()] = qTensor.rawDeviceData();
    variantPack[kAttr->get_uid()] = kTensor.rawDeviceData();
    variantPack[vAttr->get_uid()] = vTensor.rawDeviceData();
    variantPack[oAttr->get_uid()] = oTensor.rawDeviceData();

    variantPack[qRaggedAttr->get_uid()] = qRaggedOffset->rawDeviceData();
    variantPack[kvRaggedAttr->get_uid()] = kvRaggedOffset->rawDeviceData();

    variantPack[seqLenQAttr->get_uid()] = seqLenQTensor.memory().deviceData();
    variantPack[seqLenKvAttr->get_uid()] = seqLenKvTensor.memory().deviceData();

    // ── Execute ────────────────────────────────────────────────────────
    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    oTensor.memory().markDeviceModified();

    auto oHostPtr = oTensor.memory().hostData();

    std::cout << "First 10 output values: ";
    const auto printCount = std::min(static_cast<size_t>(10), oTensor.elementCount());
    for(size_t i = 0; i < printCount; ++i)
    {
        std::cout << static_cast<float>(oHostPtr[i]) << " ";
    }
    std::cout << '\n';

    // CPU validation: ragged CPU reference is not yet wired.
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation skipped (ragged CPU ref not yet wired).\n";
    }

    std::cout << "Ragged SDPA forward graph execution complete for " << inputType << ".\n\n";
    return true;
}

int main(int argc, char* argv[])
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        auto config = parseCommandLineArgs(argc, argv, SampleType::SDPA);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = runSdpaRagged(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "All ragged SDPA forward runs completed successfully.\n";
            return 0;
        }

        std::cout << "One or more ragged SDPA forward runs failed.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
