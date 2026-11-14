// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// sdpa_fwd: one parameterized hipDNN SDPA forward graph.
//
// A single descriptor covers every layout the attention paths need:
//
//   Q / O   dense  : 4-D [B, Hq, Sq, D] of any strides (encoder-only: strided BSHD views
//                    of vLLM's packed query/output, permuted to BHSD).
//           ragged : 3-D packed [T, Hq, D] plus ragged_offset_q [B+1] INT32 in
//                    token units (vLLM's query_start_loc as is). Described as
//                    [B, Hq, max_seq_len_q, D] with token stride T_s; Q and O
//                    share ONE ragged-offset tensor attribute (one uid), each
//                    with ragged_offset_multiplier = its own token stride
//                    (RFC 0014). seq_len_q [B] INT32 is required.
//   K / V   dense  : 4-D [B, Hkv, Skv, D] of any strides.
//           ragged : 3-D packed [Tk, Hkv, D] plus ragged_offset_kv [B+1] (shared
//                    by K and V) and seq_len_kv [B].
//           paged  : 4-D containers [num_blocks, Hkv, block_size, D] of any
//                    strides (the TRITON_ATTN cache split, interleaved K/V), one
//                    page table [B, M] INT32 (vLLM's block_table_tensor) bound
//                    as BOTH page_table_k and page_table_v (one uid) and
//                    described [B, 1, M', 1] with M' = ceil(max_seq_len_kv /
//                    block_size) <= M, seq_len_kv [B] INT32, and the build-time
//                    bound max_seq_len_kv (cuDNN semantics; INT32 in hipDNN).
//   masks          : diagonal band in hipDNN form: alignment (TOP_LEFT /
//                    BOTTOM_RIGHT) plus left/right bounds; key k is visible to
//                    query q when q+off-left <= k <= q+off+right (-1 = open).
//                    Causal is right_bound 0. vLLM's sliding_window (W-1, 0)
//                    maps to the bounds unchanged (left_bound = W-1).
//   sink           : optional per-head attention sink (cuDNN sink_token, gpt-oss
//                    style): [1, Hq, 1, 1] in FLOAT or the Q dtype, any strides
//                    (vLLM's layer.sinks [Hq] viewed as [1, Hq, 1, 1]). It joins
//                    the softmax max and denominator, unscaled. Whether an engine
//                    takes it (and in which dtype) is the engine's decision.
//   scale          : compile-time attn_scale_value.
//   GQA / MQA      : Hq a multiple of Hkv; engines broadcast K/V heads.
//
// The plan key holds the DESCRIBED layouts (dims, strides, dtypes), never a
// ragged tensor's token count, so a ragged/paged plan built for (B, max bounds)
// serves every step and is capture-invariant. Lengths, offsets and block ids are
// read by the engine from device tensors only.

#include "graphs.h"

#include "../graph_api.h"
#include "../plan_cache.h"
#include "../runtime.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr const char* OP_NAME = "sdpa_fwd";

constexpr int64_t UID_Q = 1;
constexpr int64_t UID_K = 2;
constexpr int64_t UID_V = 3;
constexpr int64_t UID_O = 4;
constexpr int64_t UID_RAGGED_Q = 5; // shared by Q and O
constexpr int64_t UID_RAGGED_KV = 6; // shared by K and V
constexpr int64_t UID_SEQ_LEN_Q = 7;
constexpr int64_t UID_SEQ_LEN_KV = 8;
constexpr int64_t UID_PAGE_TABLE = 9; // shared by page_table_k and page_table_v
constexpr int64_t UID_SINK = 10;
constexpr size_t IO_SLOTS = 4; // Q, K, V, O
constexpr size_t AUX_SLOTS = 6; // ragged Q/KV offsets, seq_len_q/kv, page table, sink
constexpr size_t MAX_SLOTS = IO_SLOTS + AUX_SLOTS;

// One tensor as hipDNN sees it, plus the pointer bound at execute.
struct Described
{
    int64_t uid = 0;
    const char* name = "";
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    fe::DataType type = fe::DataType::FLOAT;
    int64_t raggedUid = 0; // 0: not ragged
    int64_t raggedMultiplier = 1;
    const void* ptr = nullptr;
};

struct SdpaDesc
{
    Described q, k, v, o;
    std::optional<Described> raggedQ, raggedKv, seqLenQ, seqLenKv, pageTable, sink;
    double scale = 1.0;
    bool bottomRight = true;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    int64_t maxSeqLenKv = 0; // paged only
    fe::DataType ioType = fe::DataType::FLOAT;
    const char* layout = "dense";
};

// The optional tensors in slot order (after Q, K, V, O). The plan key lists each
// one, or a placeholder when absent, so graphs with and without a tensor (e.g.
// a sink) never share a plan.
std::array<const std::optional<Described>*, AUX_SLOTS> auxTensors(const SdpaDesc& d)
{
    return {&d.raggedQ, &d.raggedKv, &d.seqLenQ, &d.seqLenKv, &d.pageTable, &d.sink};
}

bool isSdpaType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

// A device INT32 vector [n] described as hipDNN's [n, 1, 1, 1].
Described lengthVector(const at::Tensor& t, int64_t uid, const char* name)
{
    Described d;
    d.uid = uid;
    d.name = name;
    d.dims = {t.size(0), 1, 1, 1};
    d.strides = {t.stride(0), 1, 1, 1};
    d.type = fe::DataType::INT32;
    d.ptr = t.data_ptr();
    return d;
}

Described dense4(const at::Tensor& t, int64_t uid, const char* name)
{
    Described d;
    d.uid = uid;
    d.name = name;
    d.dims = t.sizes().vec();
    d.strides = t.strides().vec();
    d.type = *toHipdnnType(t.scalar_type());
    d.ptr = t.data_ptr();
    return d;
}

// Packed [T, H, D] as the logical [B, H, maxSeq, D] a ragged offset indexes into.
Described ragged3(const at::Tensor& t,
                  int64_t uid,
                  const char* name,
                  int64_t batch,
                  int64_t maxSeq,
                  int64_t raggedUid)
{
    Described d;
    d.uid = uid;
    d.name = name;
    const int64_t tokenStride = t.stride(0);
    d.dims = {batch, t.size(1), maxSeq, t.size(2)};
    d.strides = {maxSeq * tokenStride, t.stride(1), tokenStride, t.stride(2)};
    d.type = *toHipdnnType(t.scalar_type());
    d.raggedUid = raggedUid;
    d.raggedMultiplier = tokenStride;
    d.ptr = t.data_ptr();
    return d;
}

void requireInt32Vector(const at::Tensor& t, const char* name, int64_t length)
{
    TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 1 && t.size(0) == length,
                "sdpa_fwd: ",
                name,
                " must be a 1-D INT32 tensor of length ",
                length,
                ", got ",
                t.scalar_type(),
                " ",
                t.sizes());
}

// Auxiliary tensors are bound as raw device pointers: they must live on Q's
// device, or the engine would dereference a host (or foreign-device) address.
void requireOnDevice(const at::Tensor& t, const at::Tensor& q, const char* name)
{
    TORCH_CHECK(t.device() == q.device(),
                "sdpa_fwd: ",
                name,
                " must be on q's device ",
                q.device(),
                ", got ",
                t.device());
}

struct SdpaInputs
{
    const at::Tensor& q;
    const at::Tensor& k;
    const at::Tensor& v;
    const at::Tensor& out;
    double scale;
    bool causal;
    bool bottomRight;
    int64_t leftBound;
    int64_t rightBound;
    const std::optional<at::Tensor>& raggedOffsetQ;
    const std::optional<at::Tensor>& seqLenQ;
    int64_t maxSeqLenQ;
    const std::optional<at::Tensor>& raggedOffsetKv;
    const std::optional<at::Tensor>& pageTable;
    const std::optional<at::Tensor>& seqLenKv;
    int64_t maxSeqLenKv;
    const std::optional<at::Tensor>& sink;
};

// Validate and describe. Malformed calls are TORCH_CHECK errors (caller bugs);
// valid calls hipDNN cannot express are a Decline.
std::variant<SdpaDesc, Decline> describe(const SdpaInputs& in)
{
    const at::Tensor& q = in.q;
    const at::Tensor& k = in.k;
    const at::Tensor& v = in.v;
    const at::Tensor& o = in.out;
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == o.device(),
                "sdpa_fwd: q, k, v and out must share a device");
    for(const auto& [aux, name] :
        {std::pair{&in.raggedOffsetQ, "ragged_offset_q"},
         std::pair{&in.seqLenQ, "seq_len_q"},
         std::pair{&in.raggedOffsetKv, "ragged_offset_kv"},
         std::pair{&in.pageTable, "page_table"},
         std::pair{&in.seqLenKv, "seq_len_kv"},
         std::pair{&in.sink, "sink"}})
    {
        if(aux->has_value())
        {
            requireOnDevice(**aux, q, name);
        }
    }
    TORCH_CHECK(in.leftBound >= -1 && in.rightBound >= -1,
                "sdpa_fwd: left_bound / right_bound must be >= -1");
    const bool raggedQ = in.raggedOffsetQ.has_value();
    const bool raggedKv = in.raggedOffsetKv.has_value();
    const bool paged = in.pageTable.has_value();
    TORCH_CHECK(!(raggedKv && paged), "sdpa_fwd: K/V are either ragged or paged, not both");

    SdpaDesc d;
    d.scale = in.scale;
    d.bottomRight = in.bottomRight;
    d.leftBound = in.leftBound;
    d.rightBound = in.causal ? 0 : in.rightBound;

    if(!isSdpaType(q.scalar_type()) || k.scalar_type() != q.scalar_type()
       || v.scalar_type() != q.scalar_type() || o.scalar_type() != q.scalar_type())
    {
        return Decline::feature("dtype");
    }
    d.ioType = *toHipdnnType(q.scalar_type());

    // ---- Q / O ----
    int64_t batch = 0;
    if(raggedQ)
    {
        TORCH_CHECK(q.dim() == 3 && o.dim() == 3,
                    "sdpa_fwd: ragged Q/O must be packed [T, H, D], got ",
                    q.sizes(),
                    " and ",
                    o.sizes());
        TORCH_CHECK(q.size(0) == o.size(0) && q.size(1) == o.size(1),
                    "sdpa_fwd: ragged out must be [T, Hq, Dv] matching q");
        const at::Tensor& off = *in.raggedOffsetQ;
        TORCH_CHECK(off.scalar_type() == at::kInt && off.dim() == 1 && off.size(0) >= 2,
                    "sdpa_fwd: ragged_offset_q must be 1-D INT32 [B+1]");
        TORCH_CHECK(in.maxSeqLenQ > 0, "sdpa_fwd: ragged Q needs max_seq_len_q > 0");
        TORCH_CHECK(in.seqLenQ.has_value(), "sdpa_fwd: ragged Q needs seq_len_q");
        batch = off.size(0) - 1;
        d.q = ragged3(q, UID_Q, "q", batch, in.maxSeqLenQ, UID_RAGGED_Q);
        d.o = ragged3(o, UID_O, "o", batch, in.maxSeqLenQ, UID_RAGGED_Q);
        d.raggedQ = lengthVector(off, UID_RAGGED_Q, "ragged_offset_q");
    }
    else
    {
        TORCH_CHECK(q.dim() == 4 && o.dim() == 4,
                    "sdpa_fwd: dense Q/O must be [B, H, S, D], got ",
                    q.sizes(),
                    " and ",
                    o.sizes());
        TORCH_CHECK(o.size(0) == q.size(0) && o.size(1) == q.size(1) && o.size(2) == q.size(2),
                    "sdpa_fwd: out must be [B, Hq, Sq, Dv] matching q");
        batch = q.size(0);
        d.q = dense4(q, UID_Q, "q");
        d.o = dense4(o, UID_O, "o");
    }
    if(in.seqLenQ)
    {
        requireInt32Vector(*in.seqLenQ, "seq_len_q", batch);
        d.seqLenQ = lengthVector(*in.seqLenQ, UID_SEQ_LEN_Q, "seq_len_q");
    }

    // ---- K / V ----
    if(paged)
    {
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                    "sdpa_fwd: paged K/V must be containers [num_blocks, H_kv, block_size, D]");
        TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2),
                    "sdpa_fwd: paged K and V must share num_blocks, H_kv and block_size");
        const at::Tensor& table = *in.pageTable;
        TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 && table.size(0) == batch,
                    "sdpa_fwd: page_table must be INT32 [B, max_blocks], got ",
                    table.scalar_type(),
                    " ",
                    table.sizes(),
                    " for B=",
                    batch);
        TORCH_CHECK(in.seqLenKv.has_value(), "sdpa_fwd: paged K/V needs seq_len_kv");
        TORCH_CHECK(in.maxSeqLenKv > 0, "sdpa_fwd: paged K/V needs max_seq_len_kv > 0");
        TORCH_CHECK(in.maxSeqLenKv <= std::numeric_limits<int32_t>::max(),
                    "sdpa_fwd: max_seq_len_kv ",
                    in.maxSeqLenKv,
                    " exceeds the INT32 bound hipDNN's paged attribute holds");
        const int64_t blockSize = k.size(2);
        if(blockSize == 0)
        {
            // An empty container: nothing to attend to (and no page size to divide by).
            return Decline::feature("empty");
        }
        const int64_t blocks = (in.maxSeqLenKv + blockSize - 1) / blockSize;
        TORCH_CHECK(blocks <= table.size(1),
                    "sdpa_fwd: max_seq_len_kv ",
                    in.maxSeqLenKv,
                    " needs ",
                    blocks,
                    " page-table columns, table has ",
                    table.size(1));
        d.k = dense4(k, UID_K, "k");
        d.v = dense4(v, UID_V, "v");
        Described pt;
        pt.uid = UID_PAGE_TABLE;
        pt.name = "page_table";
        // cuDNN [B, 1, M', 1]; the first M' columns of vLLM's [B, M] block table.
        pt.dims = {batch, 1, blocks, 1};
        pt.strides = {table.stride(0), table.stride(0), table.stride(1), 1};
        pt.type = fe::DataType::INT32;
        pt.ptr = table.data_ptr();
        d.pageTable = pt;
        d.maxSeqLenKv = in.maxSeqLenKv;
        d.layout = raggedQ ? "ragged_q_paged_kv" : "dense_q_paged_kv";
    }
    else if(raggedKv)
    {
        TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "sdpa_fwd: ragged K/V must be packed [T, H, D]");
        const at::Tensor& off = *in.raggedOffsetKv;
        TORCH_CHECK(off.scalar_type() == at::kInt && off.dim() == 1 && off.size(0) == batch + 1,
                    "sdpa_fwd: ragged_offset_kv must be 1-D INT32 [B+1]");
        TORCH_CHECK(in.seqLenKv.has_value(), "sdpa_fwd: ragged K/V needs seq_len_kv");
        TORCH_CHECK(in.maxSeqLenKv > 0, "sdpa_fwd: ragged K/V needs max_seq_len_kv > 0");
        d.k = ragged3(k, UID_K, "k", batch, in.maxSeqLenKv, UID_RAGGED_KV);
        d.v = ragged3(v, UID_V, "v", batch, in.maxSeqLenKv, UID_RAGGED_KV);
        d.raggedKv = lengthVector(off, UID_RAGGED_KV, "ragged_offset_kv");
        d.layout = raggedQ ? "ragged" : "dense_q_ragged_kv";
    }
    else
    {
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "sdpa_fwd: dense K/V must be [B, H_kv, S_kv, D]");
        TORCH_CHECK(k.size(0) == batch && v.size(0) == batch,
                    "sdpa_fwd: dense K/V batch must match Q");
        d.k = dense4(k, UID_K, "k");
        d.v = dense4(v, UID_V, "v");
        d.layout = raggedQ ? "ragged_q_dense_kv" : "dense";
    }
    if(in.seqLenKv)
    {
        requireInt32Vector(*in.seqLenKv, "seq_len_kv", batch);
        d.seqLenKv = lengthVector(*in.seqLenKv, UID_SEQ_LEN_KV, "seq_len_kv");
    }

    const int64_t hq = d.q.dims[1];
    const int64_t hkv = d.k.dims[1];
    TORCH_CHECK(d.v.dims[1] == hkv, "sdpa_fwd: K and V head counts differ");
    TORCH_CHECK(hkv > 0 && hq % hkv == 0,
                "sdpa_fwd: Hq (",
                hq,
                ") must be a multiple of Hkv (",
                hkv,
                ")");
    TORCH_CHECK(d.q.dims[3] == d.k.dims[3], "sdpa_fwd: Q and K head sizes differ");
    TORCH_CHECK(d.o.dims[3] == d.v.dims[3], "sdpa_fwd: out and V head sizes differ");
    if(in.sink)
    {
        const at::Tensor& s = *in.sink;
        TORCH_CHECK(s.dim() == 4 && s.size(0) == 1 && s.size(1) == hq && s.size(2) == 1
                        && s.size(3) == 1,
                    "sdpa_fwd: sink must be [1, Hq, 1, 1] with Hq = ",
                    hq,
                    ", got ",
                    s.sizes());
        if(!isSdpaType(s.scalar_type()))
        {
            return Decline::feature("sink_dtype");
        }
        d.sink = dense4(s, UID_SINK, "sink");
    }

    if(q.numel() == 0 || k.numel() == 0)
    {
        return Decline::feature("empty");
    }
    for(const Described* t : {&d.q, &d.k, &d.v, &d.o})
    {
        if(t->strides[3] != 1)
        {
            return Decline::feature("strided_head_dim");
        }
    }
    return d;
}

// set_ragged_offset_multiplier arrived with hipDNN API 1.4.0 (RFC 0014). Frontends
// without it cannot express a token-unit offset, so such a graph is declined.
template <typename TA>
constexpr bool hasRaggedMultiplier()
{
    return requires(TA& t) { t.set_ragged_offset_multiplier(int64_t{1}); };
}

template <typename TA>
void setRaggedMultiplier(TA& t, int64_t multiplier)
{
    if constexpr(hasRaggedMultiplier<TA>())
    {
        t.set_ragged_offset_multiplier(multiplier);
    }
}

using TensorPtr = std::shared_ptr<fe::graph::TensorAttributes>;

TensorPtr makeTensor(const Described& d)
{
    return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                        .set_name(d.name)
                                        .set_uid(d.uid)
                                        .set_dim(d.dims)
                                        .set_stride(d.strides)
                                        .set_data_type(d.type));
}

void applyLayout(fe::graph::TensorAttributes& t, const Described& d, const TensorPtr& ragged)
{
    t.set_name(d.name).set_uid(d.uid).set_dim(d.dims).set_stride(d.strides).set_data_type(d.type);
    if(d.raggedUid != 0)
    {
        t.set_ragged_offset(ragged);
        if(d.raggedMultiplier != 1)
        {
            setRaggedMultiplier(t, d.raggedMultiplier);
        }
    }
}

std::optional<Decline> unexpressible(const SdpaDesc& d)
{
    if constexpr(!hasRaggedMultiplier<fe::graph::TensorAttributes>())
    {
        for(const Described* t : {&d.q, &d.k, &d.v, &d.o})
        {
            if(t->raggedUid != 0 && t->raggedMultiplier != 1)
            {
                return Decline::feature("ragged_multiplier",
                                        "the hipDNN frontend predates "
                                        "set_ragged_offset_multiplier (API 1.4.0)");
            }
        }
    }
    (void)d;
    return std::nullopt;
}

GraphBuilder builder(const SdpaDesc& d)
{
    return [&d]() -> BuildResult {
        if(auto decline = unexpressible(d))
        {
            return *decline;
        }
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_NAME).set_io_data_type(d.ioType).set_compute_data_type(fe::DataType::FLOAT);

        // Aux tensors first: Q/O and K/V point at one shared ragged attribute each.
        const TensorPtr raggedQ = d.raggedQ ? makeTensor(*d.raggedQ) : nullptr;
        const TensorPtr raggedKv = d.raggedKv ? makeTensor(*d.raggedKv) : nullptr;

        auto q = fe::graph::Graph::tensor(fe::graph::TensorAttributes());
        applyLayout(*q, d.q, raggedQ);
        auto k = fe::graph::Graph::tensor(fe::graph::TensorAttributes());
        applyLayout(*k, d.k, raggedKv);
        auto v = fe::graph::Graph::tensor(fe::graph::TensorAttributes());
        applyLayout(*v, d.v, raggedKv);

        // generate_stats stays unset: the 10.0.0 ASM_SDPA engine treats a present
        // optional as "stats requested" even when false and declines the graph.
        fe::graph::SdpaAttributes attrs;
        attrs.set_name("sdpa_fwd")
            .set_attn_scale(static_cast<float>(d.scale))
            .set_diagonal_alignment(d.bottomRight ? fe::DiagonalAlignment::BOTTOM_RIGHT
                                                  : fe::DiagonalAlignment::TOP_LEFT);
        if(d.leftBound >= 0)
        {
            attrs.set_diagonal_band_left_bound(d.leftBound);
        }
        if(d.rightBound >= 0)
        {
            attrs.set_diagonal_band_right_bound(d.rightBound);
        }
        if(d.seqLenQ)
        {
            attrs.set_seq_len_q(makeTensor(*d.seqLenQ));
        }
        if(d.seqLenKv)
        {
            attrs.set_seq_len_kv(makeTensor(*d.seqLenKv));
        }
        if(d.pageTable)
        {
            // One attribute (one uid) for both tables: vLLM's K and V share blocks.
            const TensorPtr table = makeTensor(*d.pageTable);
            attrs.set_paged_attention_k_table(table)
                .set_paged_attention_v_table(table)
                .set_paged_attention_max_seq_len_kv(static_cast<int32_t>(d.maxSeqLenKv));
        }
        if(d.sink)
        {
            attrs.set_sink_token(makeTensor(*d.sink));
        }

        auto outputs = graph->sdpa(q, k, v, attrs);
        applyLayout(*outputs[0], d.o, raggedQ);
        outputs[0]->set_output(true);

        std::vector<int64_t> slots = {UID_Q, UID_K, UID_V, UID_O};
        for(const auto* aux : auxTensors(d))
        {
            if(aux->has_value())
            {
                slots.push_back((*aux)->uid);
            }
        }
        return GraphDef{.graph = graph, .slotUids = std::move(slots)};
    };
}

// Slot pointers in the order builder() lists slot uids.
std::vector<const void*> slotPointers(const SdpaDesc& d)
{
    std::vector<const void*> ptrs = {d.q.ptr, d.k.ptr, d.v.ptr, d.o.ptr};
    for(const auto* aux : auxTensors(d))
    {
        if(aux->has_value())
        {
            ptrs.push_back((*aux)->ptr);
        }
    }
    return ptrs;
}

void addDescribed(PlanKey& key, const Described& d)
{
    key.add(d.uid).add(static_cast<int64_t>(d.type));
    for(const int64_t v : d.dims)
    {
        key.add(v);
    }
    key.add("s");
    for(const int64_t v : d.strides)
    {
        key.add(v);
    }
    key.add(d.raggedUid).add(d.raggedMultiplier);
}

PlanKey makeKey(const SdpaDesc& d)
{
    PlanKey key(OP_NAME);
    key.add(d.layout).add(d.scale).add(static_cast<int64_t>(d.bottomRight));
    key.add(d.leftBound).add(d.rightBound).add(d.maxSeqLenKv);
    for(const Described* t : {&d.q, &d.k, &d.v, &d.o})
    {
        addDescribed(key, *t);
    }
    for(const auto* aux : auxTensors(d))
    {
        if(aux->has_value())
        {
            addDescribed(key, **aux);
        }
        else
        {
            key.add("-");
        }
    }
    return key;
}

// runGraphOp takes the slot pointers as an initializer list; the slot count is
// 4 plus the auxiliary tensors present.
std::optional<Decline> run(const SdpaDesc& d, const at::Tensor& anchor)
{
    const PlanKey key = makeKey(d);
    const GraphBuilder build = builder(d);
    const std::vector<const void*> p = slotPointers(d);
    switch(p.size())
    {
    case 4:
        return runGraphOp(OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3]});
    case 5:
        return runGraphOp(OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3], p[4]});
    case 6:
        return runGraphOp(OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3], p[4], p[5]});
    case 7:
        return runGraphOp(OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3], p[4], p[5], p[6]});
    case 8:
        return runGraphOp(
            OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]});
    case 9:
        return runGraphOp(
            OP_NAME, key, anchor, build, {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]});
    case MAX_SLOTS:
        return runGraphOp(OP_NAME,
                          key,
                          anchor,
                          build,
                          {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]});
    default:
        break;
    }
    TORCH_CHECK(false, "sdpa_fwd: unexpected slot count ", p.size());
    return std::nullopt;
}

// Build-or-fetch only, reporting the engine that would run. probeEntry keys the
// plan the way runGraphOp does, so a probed plan is the one later calls hit.
std::pair<std::string, std::string>
    probeWithEngine(const PlanKey& key, const at::Tensor& anchor, const GraphBuilder& build)
{
    const std::shared_ptr<PlanEntry> entry = probeEntry(OP_NAME, key, anchor, build);
    if(!entry->plan)
    {
        const Decline& decline = *entry->decline;
        return {decline.reason(), decline.message};
    }
    return {std::string(), entry->plan->engineName};
}

[[noreturn]] void raiseDecline(const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline.reason(), " :: ", decline.message);
}

SdpaInputs inputs(const at::Tensor& q,
                  const at::Tensor& k,
                  const at::Tensor& v,
                  const at::Tensor& out,
                  double scale,
                  bool causal,
                  bool bottomRight,
                  int64_t leftBound,
                  int64_t rightBound,
                  const std::optional<at::Tensor>& raggedOffsetQ,
                  const std::optional<at::Tensor>& seqLenQ,
                  int64_t maxSeqLenQ,
                  const std::optional<at::Tensor>& raggedOffsetKv,
                  const std::optional<at::Tensor>& pageTable,
                  const std::optional<at::Tensor>& seqLenKv,
                  int64_t maxSeqLenKv,
                  const std::optional<at::Tensor>& sink)
{
    return SdpaInputs{.q = q,
                      .k = k,
                      .v = v,
                      .out = out,
                      .scale = scale,
                      .causal = causal,
                      .bottomRight = bottomRight,
                      .leftBound = leftBound,
                      .rightBound = rightBound,
                      .raggedOffsetQ = raggedOffsetQ,
                      .seqLenQ = seqLenQ,
                      .maxSeqLenQ = maxSeqLenQ,
                      .raggedOffsetKv = raggedOffsetKv,
                      .pageTable = pageTable,
                      .seqLenKv = seqLenKv,
                      .maxSeqLenKv = maxSeqLenKv,
                      .sink = sink};
}

py::dict describedDict(const Described& d)
{
    py::dict out;
    out["uid"] = d.uid;
    out["dims"] = d.dims;
    out["strides"] = d.strides;
    out["dtype"] = static_cast<int>(d.type);
    out["ragged_uid"] = d.raggedUid;
    out["ragged_multiplier"] = d.raggedMultiplier;
    return out;
}

} // namespace

void sdpaFwd(const at::Tensor& q,
             const at::Tensor& k,
             const at::Tensor& v,
             const at::Tensor& out,
             double scale,
             bool causal,
             bool bottomRight,
             int64_t leftBound,
             int64_t rightBound,
             const std::optional<at::Tensor>& raggedOffsetQ,
             const std::optional<at::Tensor>& seqLenQ,
             int64_t maxSeqLenQ,
             const std::optional<at::Tensor>& raggedOffsetKv,
             const std::optional<at::Tensor>& pageTable,
             const std::optional<at::Tensor>& seqLenKv,
             int64_t maxSeqLenKv,
             const std::optional<at::Tensor>& sink)
{
    auto described = describe(inputs(q,
                                     k,
                                     v,
                                     out,
                                     scale,
                                     causal,
                                     bottomRight,
                                     leftBound,
                                     rightBound,
                                     raggedOffsetQ,
                                     seqLenQ,
                                     maxSeqLenQ,
                                     raggedOffsetKv,
                                     pageTable,
                                     seqLenKv,
                                     maxSeqLenKv,
                                     sink));
    if(auto* decline = std::get_if<Decline>(&described))
    {
        countDecline(OP_NAME, *decline, q);
        raiseDecline(*decline);
    }
    if(auto decline = run(std::get<SdpaDesc>(described), q))
    {
        raiseDecline(*decline);
    }
}

std::pair<std::string, std::string> sdpaFwdProbe(const at::Tensor& q,
                                                 const at::Tensor& k,
                                                 const at::Tensor& v,
                                                 const at::Tensor& out,
                                                 double scale,
                                                 bool causal,
                                                 bool bottomRight,
                                                 int64_t leftBound,
                                                 int64_t rightBound,
                                                 const std::optional<at::Tensor>& raggedOffsetQ,
                                                 const std::optional<at::Tensor>& seqLenQ,
                                                 int64_t maxSeqLenQ,
                                                 const std::optional<at::Tensor>& raggedOffsetKv,
                                                 const std::optional<at::Tensor>& pageTable,
                                                 const std::optional<at::Tensor>& seqLenKv,
                                                 int64_t maxSeqLenKv,
                                                 const std::optional<at::Tensor>& sink)
{
    auto described = describe(inputs(q,
                                     k,
                                     v,
                                     out,
                                     scale,
                                     causal,
                                     bottomRight,
                                     leftBound,
                                     rightBound,
                                     raggedOffsetQ,
                                     seqLenQ,
                                     maxSeqLenQ,
                                     raggedOffsetKv,
                                     pageTable,
                                     seqLenKv,
                                     maxSeqLenKv,
                                     sink));
    if(auto* decline = std::get_if<Decline>(&described))
    {
        return {decline->reason(), std::string()};
    }
    const SdpaDesc& d = std::get<SdpaDesc>(described);
    return probeWithEngine(makeKey(d), q, builder(d));
}

py::dict sdpaFwdLayout(const at::Tensor& q,
                       const at::Tensor& k,
                       const at::Tensor& v,
                       const at::Tensor& out,
                       double scale,
                       bool causal,
                       bool bottomRight,
                       int64_t leftBound,
                       int64_t rightBound,
                       const std::optional<at::Tensor>& raggedOffsetQ,
                       const std::optional<at::Tensor>& seqLenQ,
                       int64_t maxSeqLenQ,
                       const std::optional<at::Tensor>& raggedOffsetKv,
                       const std::optional<at::Tensor>& pageTable,
                       const std::optional<at::Tensor>& seqLenKv,
                       int64_t maxSeqLenKv,
                       const std::optional<at::Tensor>& sink)
{
    auto described = describe(inputs(q,
                                     k,
                                     v,
                                     out,
                                     scale,
                                     causal,
                                     bottomRight,
                                     leftBound,
                                     rightBound,
                                     raggedOffsetQ,
                                     seqLenQ,
                                     maxSeqLenQ,
                                     raggedOffsetKv,
                                     pageTable,
                                     seqLenKv,
                                     maxSeqLenKv,
                                     sink));
    py::dict result;
    if(auto* decline = std::get_if<Decline>(&described))
    {
        result["decline"] = decline->reason();
        return result;
    }
    const SdpaDesc& d = std::get<SdpaDesc>(described);
    result["decline"] = unexpressible(d) ? unexpressible(d)->reason() : std::string();
    result["layout"] = std::string(d.layout);
    result["scale"] = d.scale;
    result["bottom_right"] = d.bottomRight;
    result["left_bound"] = d.leftBound;
    result["right_bound"] = d.rightBound;
    result["max_seq_len_kv"] = d.maxSeqLenKv;
    result["frontend_ragged_multiplier"] = hasRaggedMultiplier<fe::graph::TensorAttributes>();
    py::dict tensors;
    tensors["q"] = describedDict(d.q);
    tensors["k"] = describedDict(d.k);
    tensors["v"] = describedDict(d.v);
    tensors["o"] = describedDict(d.o);
    const std::array<std::pair<const char*, const std::optional<Described>*>, AUX_SLOTS> aux
        = {{{"ragged_offset_q", &d.raggedQ},
            {"ragged_offset_kv", &d.raggedKv},
            {"seq_len_q", &d.seqLenQ},
            {"seq_len_kv", &d.seqLenKv},
            {"page_table", &d.pageTable},
            {"sink", &d.sink}}};
    for(const auto& [name, t] : aux)
    {
        if(t->has_value())
        {
            tensors[name] = describedDict(**t);
        }
    }
    result["tensors"] = tensors;
    py::list slots;
    for(const int64_t uid : {UID_Q, UID_K, UID_V, UID_O})
    {
        slots.append(uid);
    }
    for(const auto& [name, t] : aux)
    {
        if(t->has_value())
        {
            slots.append((*t)->uid);
        }
    }
    result["slots"] = slots;
    result["key"] = makeKey(d).str();
    return result;
}

} // namespace hipdnn_vllm
