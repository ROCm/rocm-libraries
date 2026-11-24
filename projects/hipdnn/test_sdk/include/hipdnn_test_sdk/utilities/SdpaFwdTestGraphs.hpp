// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FloatOperandTestUtils.hpp>

// One-node SDPA forward graphs for the applicability rules of the CPU and GPU plans (fp8 descales
// and scaling, block mask): each field of SdpaGraphSpec changes one thing about an accepted graph.
namespace hipdnn_test_sdk::utilities::sdpa_graph_test
{

using namespace hipdnn_flatbuffers_sdk::data_objects;

enum Uid : int64_t
{
    Q = 1,
    K = 2,
    V = 3,
    O = 4,
    DESCALE_Q = 20,
    DESCALE_K = 21,
    DESCALE_V = 22,
    DESCALE_S = 23,
    SCALE_S = 24,
    SCALE_O = 25,
    AMAX_S = 26,
    AMAX_O = 27,
    SEQ_LEN_Q = 30,
    SEQ_LEN_KV = 31,
    PAGE_TABLE_K = 32,
    DROPOUT_MASK = 40,
    DROPOUT_SCALE = 41,
    BLOCK_MASK = 50,
};

// Q [1, 2, 4, 8] in fp8Type with bf16 (or oType) O and `descales` scalar Q/K/V descales (K's dims
// descaleKDims), stored as descaleStorage says (a baked descale holds bakedDescale). The softmax /
// output scaling fields add one-element tensors of scalingType.
// pagedK pages K through a [1, 1, 2, 1] table over [2, 2, 4, 8] blocks with seq_len tensors (V
// dense at the 8-key extent). dropout adds an explicit dropout mask and scale. blockMaskType adds a
// block mask of that dtype and blockMaskDims (cuDNN's shape for these graphs is [1, 2, 1, 1]).
struct SdpaGraphSpec
{
    DataType fp8Type = DataType::FP8_E4M3;
    DataType oType = DataType::BFLOAT16;
    int descales = 3;
    std::vector<int64_t> descaleKDims = {1};
    OperandStorage descaleStorage = OperandStorage::TENSOR;
    float bakedDescale = 0.5f;
    bool scaleS = false;
    bool descaleS = false;
    bool scaleO = false;
    bool amaxS = false;
    bool amaxO = false;
    DataType scalingType = DataType::FLOAT;
    std::vector<int64_t> scalingDims = {1, 1, 1, 1};
    bool pagedK = false;
    bool dropout = false;
    std::optional<DataType> blockMaskType;
    std::vector<int64_t> blockMaskDims = {1, 2, 1, 1};

    // Every cuDNN sdpa_fp8 scale and amax.
    SdpaGraphSpec& withFullScaling()
    {
        scaleS = descaleS = scaleO = amaxS = amaxO = true;
        return *this;
    }
};

inline flatbuffers::FlatBufferBuilder makeSdpaGraph(const SdpaGraphSpec& spec)
{
    using hipdnn_data_sdk::utilities::generateStrides;
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const auto kDims = spec.pagedK ? std::vector<int64_t>{2, 2, 4, 8} : qDims;
    const auto vDims = spec.pagedK ? std::vector<int64_t>{1, 2, 8, 8} : qDims;
    const auto skv = vDims[2];
    const std::vector<int64_t> scalarDims = {1};
    const std::vector<int64_t> lenDims = {1, 1, 1, 1};
    const std::vector<int64_t> tableDims = {1, 1, 2, 1};
    const std::vector<int64_t> dropoutDims = {1, 2, 4, skv};

    SdpaAttributesT attrs;
    attrs.q_tensor_uid = Q;
    attrs.k_tensor_uid = K;
    attrs.v_tensor_uid = V;
    attrs.o_tensor_uid = O;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto add
        = [&](int64_t uid, const char* name, DataType type, const std::vector<int64_t>& dims) {
              const auto strides = generateStrides(dims);
              tensors.push_back(
                  CreateTensorAttributesDirect(builder, uid, name, type, &strides, &dims));
          };
    add(Q, "Q", spec.fp8Type, qDims);
    add(K, "K", spec.fp8Type, kDims);
    add(V, "V", spec.fp8Type, vDims);
    add(O, "O", spec.oType, qDims);

    const std::vector<std::pair<int64_t, std::optional<int64_t>*>> descaleSlots
        = {{DESCALE_Q, &attrs.descale_q_tensor_uid},
           {DESCALE_K, &attrs.descale_k_tensor_uid},
           {DESCALE_V, &attrs.descale_v_tensor_uid}};
    for(int i = 0; i < spec.descales && i < 3; ++i)
    {
        const auto [uid, slot] = descaleSlots[static_cast<size_t>(i)];
        FloatOperandSpec descale;
        descale.uid = uid;
        descale.storage = spec.descaleStorage;
        descale.bakedValue = spec.bakedDescale;
        descale.dims = uid == DESCALE_K ? spec.descaleKDims : scalarDims;
        addFloatOperandTensor(builder, tensors, descale, "Descale");
        *slot = uid;
    }

    for(const auto& [wanted, uid, slot] :
        {std::tuple{spec.scaleS, SCALE_S, &attrs.scale_s_tensor_uid},
         std::tuple{spec.descaleS, DESCALE_S, &attrs.descale_s_tensor_uid},
         std::tuple{spec.scaleO, SCALE_O, &attrs.scale_o_tensor_uid},
         std::tuple{spec.amaxS, AMAX_S, &attrs.amax_s_tensor_uid},
         std::tuple{spec.amaxO, AMAX_O, &attrs.amax_o_tensor_uid}})
    {
        if(wanted)
        {
            add(uid, "Fp8Scaling", spec.scalingType, spec.scalingDims);
            *slot = uid;
        }
    }

    if(spec.pagedK)
    {
        add(SEQ_LEN_Q, "SeqLenQ", DataType::INT32, lenDims);
        add(SEQ_LEN_KV, "SeqLenKv", DataType::INT32, lenDims);
        add(PAGE_TABLE_K, "PageTableK", DataType::INT32, tableDims);
        attrs.seq_len_q_tensor_uid = SEQ_LEN_Q;
        attrs.seq_len_kv_tensor_uid = SEQ_LEN_KV;
        attrs.page_table_k_tensor_uid = PAGE_TABLE_K;
    }
    if(spec.blockMaskType.has_value())
    {
        add(BLOCK_MASK, "BlockMask", spec.blockMaskType.value(), spec.blockMaskDims);
        attrs.block_mask_tensor_uid = BLOCK_MASK;
    }
    if(spec.dropout)
    {
        add(DROPOUT_MASK, "DropoutMask", DataType::FLOAT, dropoutDims);
        add(DROPOUT_SCALE, "DropoutScale", DataType::FLOAT, scalarDims);
        attrs.dropout_mask_tensor_uid = DROPOUT_MASK;
        attrs.dropout_scale_tensor_uid = DROPOUT_SCALE;
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(
        builder, "SdpaFwdTestGraph", spec.fp8Type, spec.oType, DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace hipdnn_test_sdk::utilities::sdpa_graph_test
