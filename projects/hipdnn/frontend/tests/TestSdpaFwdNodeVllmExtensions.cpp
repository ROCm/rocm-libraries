// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The SDPA forward node's vLLM extensions beyond cuDNN 1.24 (SdpaSemanticsRegister.md SEM-026).
// Every cuDNN-valid graph keeps its result; each extension only accepts a graph cuDNN 1.24
// rejected.

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr int64_t BATCH = 3;
// Q's (and O's) token stride, Hq * D: vLLM's query_start_loc counts tokens.
constexpr int64_t TOKEN_STRIDE = int64_t{8} * 64;

std::shared_ptr<TensorAttributes> tensor(const std::vector<int64_t>& dims,
                                         DataType dataType = DataType::NOT_SET)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims);
    if(dataType != DataType::NOT_SET)
    {
        t->set_data_type(dataType);
    }
    return t;
}

// The token offsets of a ragged tensor: INT32 [B + 1, 1, 1, 1] (vLLM's query_start_loc).
std::shared_ptr<TensorAttributes> offsets()
{
    return tensor({BATCH + 1, 1, 1, 1}, DataType::INT32);
}

std::shared_ptr<TensorAttributes> seqLen()
{
    return tensor({BATCH, 1, 1, 1}, DataType::INT32);
}

// vLLM's paged prefill/decode call: ragged Q and O [B, Hq, max_q, D] packed token by token
// (multiplier Hq * D), K/V containers [num_blocks, Hkv, block_size, D] with page tables
// [B, 1, W, 1], and seq_len_kv (vLLM's seq_lens). No seq_len_q: vLLM has only query_start_loc.
struct VllmGraph
{
    std::shared_ptr<TensorAttributes> q = tensor({BATCH, 8, 16, 64});
    std::shared_ptr<TensorAttributes> o = tensor({BATCH, 8, 16, 64});
    SdpaAttributes attrs;

    VllmGraph()
    {
        q->set_ragged_offset(offsets()).set_ragged_offset_multiplier(TOKEN_STRIDE);
        o->set_ragged_offset(offsets()).set_ragged_offset_multiplier(TOKEN_STRIDE);
        attrs.set_q(q);
        attrs.set_k(tensor({11, 2, 16, 64}));
        attrs.set_v(tensor({11, 2, 16, 64}));
        attrs.set_o(o);
        attrs.set_paged_attention_k_table(tensor({BATCH, 1, 4, 1}, DataType::INT32));
        attrs.set_paged_attention_v_table(tensor({BATCH, 1, 4, 1}, DataType::INT32));
        attrs.set_seq_len_kv(seqLen());
        attrs.set_padding_mask(true);
    }

    error_code_t validate() const
    {
        const GraphAttributes graphAttrs;
        const SdpaFwdNode node(SdpaAttributes(attrs), graphAttrs);
        return node.pre_validate_node().code;
    }
};

} // namespace

// Paged K/V with a ragged Q needs only seq_len_kv; the query lengths are the differences
// of Q's ragged offsets (cuDNN 9.25's cu_seq_len_q + seq_len_kv form).
TEST(TestSdpaFwdNodeVllmExtensions, PagedKvWithRaggedQueryNeedsOnlySeqLenKv)
{
    const VllmGraph graph;
    EXPECT_EQ(graph.validate(), error_code_t::OK);

    auto maxSeqLen = graph;
    maxSeqLen.attrs.set_paged_attention_max_seq_len_kv(64);
    EXPECT_EQ(maxSeqLen.validate(), error_code_t::OK) << "a table exactly ceil(64 / 16) wide";

    auto withoutPaddingMask = graph;
    withoutPaddingMask.attrs.set_padding_mask(false);
    EXPECT_EQ(withoutPaddingMask.validate(), error_code_t::OK) << "lengths pad alone (SEM-008)";

    auto pagedKOnly = graph;
    pagedKOnly.attrs.set_paged_attention_v_table(nullptr);
    pagedKOnly.attrs.set_v(tensor({BATCH, 2, 64, 64}));
    EXPECT_EQ(pagedKOnly.validate(), error_code_t::OK) << "K paged alone";
}

// Ragged Q with seq_len_q too keeps cuDNN's meaning (each offset span a slot, seq_len_q its
// live rows), as before.
TEST(TestSdpaFwdNodeVllmExtensions, PagedKvWithRaggedQueryAndSeqLenQKeepsCudnnForm)
{
    auto graph = VllmGraph();
    graph.attrs.set_seq_len_q(seqLen());
    EXPECT_EQ(graph.validate(), error_code_t::OK);
}

// Each case removes what the relaxed rule still needs, from the accepted vLLM graph.
TEST(TestSdpaFwdNodeVllmExtensions, PagedKvStillNeedsTheLengthsItCannotDerive)
{
    // A dense Q has no offsets to derive the query lengths from (cuDNN 1.24's rule).
    auto denseQuery = VllmGraph();
    denseQuery.q->set_ragged_offset(nullptr);
    denseQuery.q->set_ragged_offset_multiplier(1);
    EXPECT_EQ(denseQuery.validate(), error_code_t::ATTRIBUTE_NOT_SET);

    // Only Q's offsets count: a ragged O beside a dense Q does not stand in for seq_len_q.
    auto raggedOutputOnly = VllmGraph();
    raggedOutputOnly.q->set_ragged_offset(nullptr);
    raggedOutputOnly.q->set_ragged_offset_multiplier(1);
    ASSERT_TRUE(raggedOutputOnly.o->has_ragged_offset());
    EXPECT_EQ(raggedOutputOnly.validate(), error_code_t::ATTRIBUTE_NOT_SET);

    // seq_len_kv is always needed: Q's offsets say nothing about the keys.
    auto noSeqLenKv = VllmGraph();
    noSeqLenKv.attrs.set_seq_len_kv(nullptr);
    EXPECT_EQ(noSeqLenKv.validate(), error_code_t::ATTRIBUTE_NOT_SET);
    noSeqLenKv.attrs.set_seq_len_q(seqLen());
    EXPECT_EQ(noSeqLenKv.validate(), error_code_t::ATTRIBUTE_NOT_SET)
        << "seq_len_q alone is not enough either";
}

// The relaxed rule changes nothing else: seq_len_kv's shape and dtype, the table's width and
// dtype still follow cuDNN.
TEST(TestSdpaFwdNodeVllmExtensions, PagedKvWithRaggedQueryKeepsTheOtherPagedRules)
{
    auto rankOneSeqLen = VllmGraph();
    rankOneSeqLen.attrs.set_seq_len_kv(tensor({BATCH}, DataType::INT32));
    EXPECT_EQ(rankOneSeqLen.validate(), error_code_t::INVALID_VALUE);

    auto int64SeqLen = VllmGraph();
    int64SeqLen.attrs.set_seq_len_kv(tensor({BATCH, 1, 1, 1}, DataType::INT64));
    EXPECT_EQ(int64SeqLen.validate(), error_code_t::INVALID_VALUE);

    auto wideTable = VllmGraph();
    wideTable.attrs.set_paged_attention_max_seq_len_kv(48);
    EXPECT_EQ(wideTable.validate(), error_code_t::INVALID_VALUE)
        << "a table wider than ceil(48 / 16)";

    auto floatTable = VllmGraph();
    floatTable.attrs.set_paged_attention_k_table(tensor({BATCH, 1, 4, 1}, DataType::FLOAT));
    EXPECT_EQ(floatTable.validate(), error_code_t::INVALID_VALUE);
}

// Non-paged K/V under a ragged Q with seq_len_kv alone (SEM-028): PyTorch's varlen seqused_k
// (K/V packed through cu_seqlens_k) and vLLM's non-paged call (dense K/V). The node never
// required both lengths without paging; this pins that it keeps accepting the form.
TEST(TestSdpaFwdNodeVllmExtensions, NonPagedKvWithRaggedQueryNeedsOnlySeqLenKv)
{
    auto packedKv = VllmGraph();
    packedKv.attrs.set_paged_attention_k_table(nullptr);
    packedKv.attrs.set_paged_attention_v_table(nullptr);
    const auto kvOffsets = offsets();
    auto k = tensor({BATCH, 2, 32, 64});
    auto v = tensor({BATCH, 2, 32, 64});
    k->set_ragged_offset(kvOffsets).set_ragged_offset_multiplier(int64_t{2} * 64);
    v->set_ragged_offset(kvOffsets).set_ragged_offset_multiplier(int64_t{2} * 64);
    packedKv.attrs.set_k(k);
    packedKv.attrs.set_v(v);
    EXPECT_EQ(packedKv.validate(), error_code_t::OK) << "seqused_k";
    packedKv.attrs.set_padding_mask(false);
    EXPECT_EQ(packedKv.validate(), error_code_t::OK) << "seqused_k without padding_mask";

    auto denseKv = VllmGraph();
    denseKv.attrs.set_paged_attention_k_table(nullptr);
    denseKv.attrs.set_paged_attention_v_table(nullptr);
    denseKv.attrs.set_k(tensor({BATCH, 2, 32, 64}));
    denseKv.attrs.set_v(tensor({BATCH, 2, 32, 64}));
    EXPECT_EQ(denseKv.validate(), error_code_t::OK) << "dense K/V";

    // seq_len_kv keeps cuDNN's shape and dtype rules.
    denseKv.attrs.set_seq_len_kv(tensor({BATCH, 1, 1, 1}, DataType::INT64));
    EXPECT_EQ(denseKv.validate(), error_code_t::INVALID_VALUE);
}
