// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// encoder_attention: dense non-causal hipDNN `sdpa` graphs for vLLM's
// MMEncoderAttention (ViT attention).
//
// q [B, S, Hq, D], k [B, S_kv, Hk, D], v [B, S_kv, Hk, Dv] are BSHD tensors
// with arbitrary token/head strides (views of a fused qkv projection pass
// through without a copy). cu_seqlens, when given, splits dim 1 into
// independent images (vLLM's varlen convention, same as torch_sdpa_wrapper):
//
// * B == 1 and every non-empty image has the same length L: one batched graph
//   with batch = number of images. The images are back to back, so image i is
//   the BHSD view at offset i * L tokens.
// * Otherwise: one graph launch per image, with the plan keyed by its length,
//   so all images of one length share a plan.
//
// Every distinct signature is probed (built or fetched) before anything
// launches, so a decline never leaves a half-written output. The image
// lengths are read on the host, which syncs on a device cu_seqlens. That read
// is not capture-invariant, so under HIP-graph capture a cu_seqlens call
// declines with CAPTURE_INVARIANT and the caller falls back.

#include "graphs.h"

#include "../graph_api.h"

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr int64_t UID_Q = 1;
constexpr int64_t UID_K = 2;
constexpr int64_t UID_V = 3;
constexpr int64_t UID_O = 4;
constexpr const char* OP_NAME = "encoder_attention";

bool isAttnType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

// BHSD views of one launch: tokens [start, start + n * len) of every tensor.
struct Launch
{
    at::Tensor q;
    at::Tensor k;
    at::Tensor v;
    at::Tensor o;
};

// [B, S, H, D] -> BHSD view of tokens [start, start + len).
at::Tensor segment(const at::Tensor& t, int64_t start, int64_t len)
{
    return t.narrow(1, start, len).permute({0, 2, 1, 3});
}

// [1, S, H, D] -> BHSD view [n, H, len, D] of n back-to-back images of `len`
// tokens starting at token `start`.
at::Tensor batchedImages(const at::Tensor& t, int64_t start, int64_t n, int64_t len)
{
    const int64_t tokenStride = t.stride(1);
    return t
        .as_strided({n, len, t.size(2), t.size(3)},
                    {len * tokenStride, tokenStride, t.stride(2), t.stride(3)},
                    t.storage_offset() + start * tokenStride)
        .permute({0, 2, 1, 3});
}

void checkShapes(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "encoder_attention: q, k, v must be [B, S, H, D], got ",
                q.sizes(),
                ", ",
                k.sizes(),
                ", ",
                v.sizes());
    TORCH_CHECK(k.size(0) == q.size(0) && v.size(0) == q.size(0), "encoder_attention: batch mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "encoder_attention: k and v sequence lengths differ");
    TORCH_CHECK(k.size(2) == v.size(2), "encoder_attention: k and v head counts differ");
    TORCH_CHECK(k.size(3) == q.size(3), "encoder_attention: q and k head sizes differ");
    TORCH_CHECK(k.size(2) > 0 && q.size(2) % k.size(2) == 0,
                "encoder_attention: num_heads ",
                q.size(2),
                " is not a multiple of num_kv_heads ",
                k.size(2));
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "encoder_attention: q, k, v must share a device");
}

std::optional<Decline> prefilter(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v)
{
    if(!isAttnType(q.scalar_type()))
    {
        return Decline::feature("dtype");
    }
    if(k.scalar_type() != q.scalar_type() || v.scalar_type() != q.scalar_type())
    {
        return Decline::feature("mixed_dtype");
    }
    if(q.stride(3) != 1 || k.stride(3) != 1 || v.stride(3) != 1)
    {
        return Decline::feature("strided_head_dim");
    }
    return std::nullopt;
}

hipStream_t currentStream(const at::Tensor& t)
{
    return at::hip::getCurrentHIPStreamMasqueradingAsCUDA(t.device().index()).stream();
}

bool capturing(const at::Tensor& t)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(currentStream(t), &status) == hipSuccess
           && status == hipStreamCaptureStatusActive;
}

// Image boundaries from cu_seqlens, validated against the token count.
std::vector<int64_t> hostOffsets(const at::Tensor& cuSeqlens, int64_t tokens)
{
    TORCH_CHECK(cuSeqlens.dim() == 1 && cuSeqlens.numel() >= 1,
                "encoder_attention: cu_seqlens must be 1-D with at least one entry");
    TORCH_CHECK(cuSeqlens.scalar_type() == at::kInt || cuSeqlens.scalar_type() == at::kLong,
                "encoder_attention: cu_seqlens must be int32 or int64");
    const at::Tensor host = cuSeqlens.to(at::kCPU, at::kLong).contiguous();
    const int64_t* p = host.data_ptr<int64_t>();
    std::vector<int64_t> offsets(p, p + host.numel());
    TORCH_CHECK(offsets.front() >= 0, "encoder_attention: cu_seqlens[0] must be >= 0");
    for(size_t i = 1; i < offsets.size(); ++i)
    {
        TORCH_CHECK(offsets[i] >= offsets[i - 1], "encoder_attention: cu_seqlens must be non-decreasing");
    }
    TORCH_CHECK(offsets.back() <= tokens,
                "encoder_attention: cu_seqlens[-1]=",
                offsets.back(),
                " exceeds the sequence length ",
                tokens);
    return offsets;
}

std::vector<Launch> plan(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v,
                         const at::Tensor& o,
                         const std::vector<int64_t>& offsets)
{
    std::vector<Launch> launches;
    if(offsets.empty())
    {
        launches.push_back(Launch{.q = segment(q, 0, q.size(1)),
                                  .k = segment(k, 0, k.size(1)),
                                  .v = segment(v, 0, v.size(1)),
                                  .o = segment(o, 0, o.size(1))});
        return launches;
    }

    std::vector<std::pair<int64_t, int64_t>> images; // (start, len), non-empty only
    for(size_t i = 1; i < offsets.size(); ++i)
    {
        if(offsets[i] > offsets[i - 1])
        {
            images.emplace_back(offsets[i - 1], offsets[i] - offsets[i - 1]);
        }
    }
    if(images.empty())
    {
        return launches;
    }
    bool uniform = true;
    for(const auto& image : images)
    {
        uniform = uniform && image.second == images.front().second;
    }
    if(uniform && q.size(0) == 1)
    {
        const int64_t start = images.front().first;
        const auto n = static_cast<int64_t>(images.size());
        const int64_t len = images.front().second;
        launches.push_back(Launch{.q = batchedImages(q, start, n, len),
                                  .k = batchedImages(k, start, n, len),
                                  .v = batchedImages(v, start, n, len),
                                  .o = batchedImages(o, start, n, len)});
        return launches;
    }
    for(const auto& [start, len] : images)
    {
        launches.push_back(Launch{.q = segment(q, start, len),
                                  .k = segment(k, start, len),
                                  .v = segment(v, start, len),
                                  .o = segment(o, start, len)});
    }
    return launches;
}

GraphBuilder builder(const Launch& l, double scale)
{
    return [&l, scale]() -> BuildResult {
        const auto io = *toHipdnnType(l.q.scalar_type());
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_NAME).set_io_data_type(io).set_compute_data_type(fe::DataType::FLOAT);
        auto qT = describeTensor(l.q, UID_Q, "q");
        auto kT = describeTensor(l.k, UID_K, "k");
        auto vT = describeTensor(l.v, UID_V, "v");

        // Non-causal with no stats output: generate_stats and the mask flags
        // stay unset rather than set to false. The TheRock 10.0.0 ASM_SDPA
        // engine tests these optional fields for presence, so an explicit false
        // reads as "stats requested" there and the graph is declined.
        fe::graph::SdpaAttributes attrs;
        attrs.set_name(OP_NAME).set_attn_scale(static_cast<float>(scale));
        auto outputs = graph->sdpa(qT, kT, vT, attrs);
        outputs[0]
            ->set_uid(UID_O)
            .set_dim(l.o.sizes().vec())
            .set_stride(l.o.strides().vec())
            .set_data_type(io)
            .set_output(true);
        return GraphDef{.graph = graph, .slotUids = {UID_Q, UID_K, UID_V, UID_O}};
    };
}

PlanKey makeKey(const Launch& l, double scale)
{
    PlanKey key(OP_NAME);
    key.add(scale).add(l.q).add(l.k).add(l.v).add(l.o);
    return key;
}

[[noreturn]] void raiseDecline(const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline.reason(), " :: ", decline.message);
}

[[noreturn]] void declineAndRaise(const Decline& decline, const at::Tensor& anchor)
{
    countDecline(OP_NAME, decline, anchor);
    raiseDecline(decline);
}

} // namespace

at::Tensor encoderAttention(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            const std::optional<at::Tensor>& cuSeqlens,
                            double scale)
{
    checkShapes(q, k, v);
    if(cuSeqlens.has_value())
    {
        TORCH_CHECK(k.size(1) == q.size(1),
                    "encoder_attention: with cu_seqlens, q and k/v must have the same sequence length");
    }
    if(auto decline = prefilter(q, k, v))
    {
        declineAndRaise(*decline, q);
    }
    at::Tensor o = at::empty({q.size(0), q.size(1), q.size(2), v.size(3)},
                             q.options().memory_format(at::MemoryFormat::Contiguous));
    if(o.numel() == 0 || k.size(1) == 0)
    {
        declineAndRaise(Decline::feature("empty"), q);
    }

    std::vector<int64_t> offsets;
    if(cuSeqlens.has_value())
    {
        if(cuSeqlens->is_cuda() && capturing(q))
        {
            declineAndRaise(Decline{.stage = Stage::CAPTURE_INVARIANT,
                                    .detail = "cu_seqlens",
                                    .message = "image lengths are read on the host"},
                            q);
        }
        offsets = hostOffsets(*cuSeqlens, q.size(1));
    }

    const std::vector<Launch> launches = plan(q, k, v, o, offsets);

    // Probe every distinct signature first: all of them run on hipDNN, or none.
    std::map<std::string, size_t> distinct;
    std::vector<PlanKey> keys;
    keys.reserve(launches.size());
    for(size_t i = 0; i < launches.size(); ++i)
    {
        keys.push_back(makeKey(launches[i], scale));
        if(distinct.emplace(keys.back().str(), i).second)
        {
            if(auto decline = probeGraphOp(OP_NAME, keys.back(), q, builder(launches[i], scale)))
            {
                declineAndRaise(*decline, q);
            }
        }
    }

    for(size_t i = 0; i < launches.size(); ++i)
    {
        const Launch& l = launches[i];
        if(auto decline = runGraphOp(OP_NAME,
                                     keys[i],
                                     q,
                                     builder(l, scale),
                                     {l.q.data_ptr(), l.k.data_ptr(), l.v.data_ptr(), l.o.data_ptr()}))
        {
            raiseDecline(*decline);
        }
    }

    // Tokens past cu_seqlens[-1] belong to no image (padding); keep them defined.
    if(!offsets.empty())
    {
        const int64_t covered = offsets.back();
        if(covered < o.size(1))
        {
            o.narrow(1, covered, o.size(1) - covered).zero_();
        }
        if(offsets.front() > 0)
        {
            o.narrow(1, 0, offsets.front()).zero_();
        }
    }
    return o;
}

} // namespace hipdnn_vllm
