// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// conv_fwd: hipDNN `conv_fprop` (+ bias) graph.
//
// x [N, C, *spatial] and w [K, C/groups, *kernel] (2-D or 3-D spatial) are
// described with their own dims and strides; groups follow from w's channel
// dim. The output is a new contiguous [N, K, *out] tensor. Every dim and stride
// is in the plan key, so each input resolution gets its own plan (vision patch
// embeddings see few distinct resolutions).
//
// Bias. The MIOpen provider serves conv + bias only as its ConvFwdBiasActiv
// fusion, which requires a third, activation node. The fused graph is
//   conv_fprop -> pointwise(ADD, bias[1, K, 1, ...]) -> pointwise(RELU_FWD)
// with relu_lower_clip = -FLT_MAX and relu_upper_clip = FLT_MAX, i.e. MIOpen's
// CLAMP activation over the whole finite range, an identity on finite values.
// The intermediates are virtual fp32. If no engine takes that graph, the plain
// conv graph runs on hipDNN (stats op "conv_fprop") and the bias is added with
// ATen; the call is counted as (conv_fwd, hipdnn, PARTIAL).
//
// A decline never fails the model: the op counts route=fallback and runs
// at::convolution (what vLLM's Conv*Layer._forward_conv calls), unless
// `strict` is set, in which case it raises the decline.

#include "graphs.h"

#include "../graph_api.h"
#include "../stats.h"

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

#include <atomic>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr int64_t UID_X = 1;
constexpr int64_t UID_W = 2;
constexpr int64_t UID_B = 3;
constexpr int64_t UID_Y = 4;
constexpr const char* OP_NAME = "conv_fwd";
// The bias-less conv graph run on behalf of a conv_fwd call with bias.
constexpr const char* OP_CONV_ONLY = "conv_fprop";
constexpr const char* REASON_PARTIAL = "PARTIAL";

struct ConvParams
{
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    int64_t groups;
};

enum class Variant
{
    PLAIN, // conv only
    FUSED_BIAS, // conv -> add bias -> identity clamp
};

bool isConvType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

bool capturing(const at::Tensor& anchor)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(anchor.device().index()).stream();
    return hipStreamIsCapturing(stream, &status) == hipSuccess
           && status == hipStreamCaptureStatusActive;
}

ConvParams checkArgs(const at::Tensor& x,
                     const at::Tensor& weight,
                     const std::optional<at::Tensor>& bias,
                     at::IntArrayRef stride,
                     at::IntArrayRef padding,
                     at::IntArrayRef dilation,
                     int64_t groups)
{
    const int64_t sp = x.dim() - 2;
    TORCH_CHECK(sp == 2 || sp == 3, "conv_fwd: x must be 4-D or 5-D, got ", x.sizes());
    TORCH_CHECK(weight.dim() == x.dim(), "conv_fwd: weight rank must match x, got ", weight.sizes());
    TORCH_CHECK(std::cmp_equal(stride.size(), sp) && std::cmp_equal(padding.size(), sp)
                    && std::cmp_equal(dilation.size(), sp),
                "conv_fwd: stride/padding/dilation must have ",
                sp,
                " entries");
    for(int64_t i = 0; i < sp; ++i)
    {
        const auto u = static_cast<size_t>(i);
        TORCH_CHECK(stride[u] >= 1 && dilation[u] >= 1 && padding[u] >= 0,
                    "conv_fwd: stride and dilation must be >= 1 and padding >= 0, got stride=",
                    stride,
                    " padding=",
                    padding,
                    " dilation=",
                    dilation);
        TORCH_CHECK(x.size(i + 2) + 2 * padding[u] >= dilation[u] * (weight.size(i + 2) - 1) + 1,
                    "conv_fwd: dilated kernel ",
                    weight.sizes(),
                    " is larger than the padded input ",
                    x.sizes());
    }
    TORCH_CHECK(groups >= 1 && weight.size(0) % groups == 0 && weight.size(1) * groups == x.size(1),
                "conv_fwd: channels ",
                x.size(1),
                " / weight ",
                weight.sizes(),
                " do not match groups=",
                groups);
    TORCH_CHECK(!bias || (bias->dim() == 1 && bias->size(0) == weight.size(0)),
                "conv_fwd: bias must be [out_channels]");
    TORCH_CHECK(x.device() == weight.device() && (!bias || bias->device() == x.device()),
                "conv_fwd: tensors must share a device");
    return {.stride = stride.vec(), .padding = padding.vec(), .dilation = dilation.vec(), .groups = groups};
}

std::optional<Decline>
    prefilter(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    if(!isConvType(x.scalar_type()) || weight.scalar_type() != x.scalar_type()
       || (bias && bias->scalar_type() != x.scalar_type()))
    {
        return Decline::feature("dtype");
    }
    if(x.numel() == 0)
    {
        return Decline::feature("empty");
    }
    if(bias && bias->stride(0) != 1)
    {
        return Decline::feature("strided_bias");
    }
    return std::nullopt;
}

std::vector<int64_t> outDims(const at::Tensor& x, const at::Tensor& w, const ConvParams& p)
{
    std::vector<int64_t> dims{x.size(0), w.size(0)};
    for(size_t i = 0; i < p.stride.size(); ++i)
    {
        const auto d = static_cast<int64_t>(i) + 2;
        dims.push_back((x.size(d) + 2 * p.padding[i] - p.dilation[i] * (w.size(d) - 1) - 1) / p.stride[i]
                       + 1);
    }
    return dims;
}

std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::vector<int64_t> biasDims(const at::Tensor& y)
{
    std::vector<int64_t> dims(static_cast<size_t>(y.dim()), 1);
    dims[1] = y.size(1);
    return dims;
}

GraphBuilder
    builder(Variant variant, const at::Tensor& x, const at::Tensor& w, const at::Tensor& y, const ConvParams& p)
{
    return [variant, &x, &w, &y, &p]() -> BuildResult {
        const auto io = *toHipdnnType(x.scalar_type());
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_NAME)
            .set_io_data_type(io)
            .set_intermediate_data_type(fe::DataType::FLOAT)
            .set_compute_data_type(fe::DataType::FLOAT);

        fe::graph::ConvFpropAttributes attrs;
        attrs.set_padding(p.padding).set_stride(p.stride).set_dilation(p.dilation);
        auto out = graph->conv_fprop(describeTensor(x, UID_X, "x"), describeTensor(w, UID_W, "w"), attrs);

        const std::vector<int64_t> yDims(y.sizes().begin(), y.sizes().end());
        const std::vector<int64_t> yStrides(y.strides().begin(), y.strides().end());
        std::vector<int64_t> slots{UID_X, UID_W};
        if(variant == Variant::FUSED_BIAS)
        {
            out->set_dim(yDims).set_stride(contiguousStrides(yDims)).set_data_type(fe::DataType::FLOAT);
            const std::vector<int64_t> bDims = biasDims(y);
            auto bT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                                   .set_name("bias")
                                                   .set_uid(UID_B)
                                                   .set_dim(bDims)
                                                   .set_stride(contiguousStrides(bDims))
                                                   .set_data_type(io));
            out = graph->pointwise(
                out, bT, fe::graph::PointwiseAttributes().set_mode(fe::PointwiseMode::ADD));
            out->set_dim(yDims).set_stride(contiguousStrides(yDims)).set_data_type(fe::DataType::FLOAT);
            out = graph->pointwise(out,
                                   fe::graph::PointwiseAttributes()
                                       .set_mode(fe::PointwiseMode::RELU_FWD)
                                       .set_relu_lower_clip(std::numeric_limits<float>::lowest())
                                       .set_relu_upper_clip(std::numeric_limits<float>::max()));
            slots.push_back(UID_B);
        }
        out->set_uid(UID_Y).set_dim(yDims).set_stride(yStrides).set_data_type(io).set_output(true);
        slots.push_back(UID_Y);
        return GraphDef{.graph = graph, .slotUids = slots};
    };
}

PlanKey makeKey(const char* op,
                Variant variant,
                const at::Tensor& x,
                const at::Tensor& w,
                const at::Tensor& y,
                const ConvParams& p)
{
    PlanKey key(op);
    key.add(static_cast<int64_t>(variant));
    for(const auto* v : {&p.stride, &p.padding, &p.dilation})
    {
        for(const int64_t e : *v)
        {
            key.add(e);
        }
        key.add("|");
    }
    key.add(p.groups).add(x).add(w).add(y);
    return key;
}

[[noreturn]] void raiseDecline(const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline.reason(), " :: ", decline.message);
}

at::Tensor nativeConv(const at::Tensor& x,
                      const at::Tensor& w,
                      const std::optional<at::Tensor>& bias,
                      const ConvParams& p)
{
    return at::convolution(x, w, bias, p.stride, p.padding, p.dilation, false, {0}, p.groups);
}

at::Tensor declined(const Decline& decline,
                    bool strict,
                    const at::Tensor& x,
                    const at::Tensor& w,
                    const std::optional<at::Tensor>& bias,
                    const ConvParams& p)
{
    if(strict)
    {
        raiseDecline(decline);
    }
    return nativeConv(x, w, bias, p);
}

void countPartial(const at::Tensor& anchor)
{
    static std::atomic<uint64_t>* const s_eager
        = Stats::instance().cell(OP_NAME, ROUTE_HIPDNN, REASON_PARTIAL, false);
    static std::atomic<uint64_t>* const s_capture
        = Stats::instance().cell(OP_NAME, ROUTE_HIPDNN, REASON_PARTIAL, true);
    (capturing(anchor) ? s_capture : s_eager)->fetch_add(1, std::memory_order_relaxed);
}

// Decline of the graph the op would run for this signature, without counting.
std::optional<Decline> probe(const at::Tensor& x,
                             const at::Tensor& w,
                             bool hasBias,
                             const at::Tensor& y,
                             const ConvParams& p)
{
    if(!hasBias)
    {
        return probeGraphOp(
            OP_NAME, makeKey(OP_NAME, Variant::PLAIN, x, w, y, p), x, builder(Variant::PLAIN, x, w, y, p));
    }
    auto fused = probeGraphOp(OP_NAME,
                              makeKey(OP_NAME, Variant::FUSED_BIAS, x, w, y, p),
                              x,
                              builder(Variant::FUSED_BIAS, x, w, y, p));
    if(!fused)
    {
        return std::nullopt;
    }
    auto plain = probeGraphOp(OP_CONV_ONLY,
                              makeKey(OP_CONV_ONLY, Variant::PLAIN, x, w, y, p),
                              x,
                              builder(Variant::PLAIN, x, w, y, p));
    return plain ? fused : std::nullopt;
}

} // namespace

at::Tensor convFwd(const at::Tensor& x,
                   const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias,
                   at::IntArrayRef stride,
                   at::IntArrayRef padding,
                   at::IntArrayRef dilation,
                   int64_t groups,
                   bool strict)
{
    const ConvParams p = checkArgs(x, weight, bias, stride, padding, dilation, groups);
    if(auto decline = prefilter(x, weight, bias))
    {
        countDecline(OP_NAME, *decline, x);
        return declined(*decline, strict, x, weight, bias, p);
    }
    at::Tensor y = at::empty(outDims(x, weight, p), x.options().memory_format(at::MemoryFormat::Contiguous));

    if(!bias)
    {
        if(auto decline = runGraphOp(OP_NAME,
                                     makeKey(OP_NAME, Variant::PLAIN, x, weight, y, p),
                                     x,
                                     builder(Variant::PLAIN, x, weight, y, p),
                                     {x.data_ptr(), weight.data_ptr(), y.data_ptr()}))
        {
            return declined(*decline, strict, x, weight, bias, p);
        }
        return y;
    }

    const PlanKey fusedKey = makeKey(OP_NAME, Variant::FUSED_BIAS, x, weight, y, p);
    const GraphBuilder fusedBuilder = builder(Variant::FUSED_BIAS, x, weight, y, p);
    const std::optional<Decline> fusedDecline = probeGraphOp(OP_NAME, fusedKey, x, fusedBuilder);
    if(!fusedDecline)
    {
        if(auto decline = runGraphOp(OP_NAME,
                                     fusedKey,
                                     x,
                                     fusedBuilder,
                                     {x.data_ptr(), weight.data_ptr(), bias->data_ptr(), y.data_ptr()}))
        {
            return declined(*decline, strict, x, weight, bias, p);
        }
        return y;
    }

    // No fused engine: hipDNN conv plus an ATen bias add.
    const PlanKey plainKey = makeKey(OP_CONV_ONLY, Variant::PLAIN, x, weight, y, p);
    const GraphBuilder plainBuilder = builder(Variant::PLAIN, x, weight, y, p);
    if(probeGraphOp(OP_CONV_ONLY, plainKey, x, plainBuilder))
    {
        countDecline(OP_NAME, *fusedDecline, x);
        return declined(*fusedDecline, strict, x, weight, bias, p);
    }
    if(auto decline = runGraphOp(
           OP_CONV_ONLY, plainKey, x, plainBuilder, {x.data_ptr(), weight.data_ptr(), y.data_ptr()}))
    {
        countDecline(OP_NAME, *decline, x);
        return declined(*decline, strict, x, weight, bias, p);
    }
    y.add_(bias->view(biasDims(y)));
    countPartial(x);
    return y;
}

std::string convFwdProbe(const at::Tensor& x,
                         const at::Tensor& weight,
                         const std::optional<at::Tensor>& bias,
                         at::IntArrayRef stride,
                         at::IntArrayRef padding,
                         at::IntArrayRef dilation,
                         int64_t groups)
{
    const ConvParams p = checkArgs(x, weight, bias, stride, padding, dilation, groups);
    if(auto decline = prefilter(x, weight, bias))
    {
        return decline->reason();
    }
    const at::Tensor y = at::empty(outDims(x, weight, p), x.options().device(at::kMeta));
    auto decline = probe(x, weight, bias.has_value(), y, p);
    return decline ? decline->reason() : std::string();
}

} // namespace hipdnn_vllm
