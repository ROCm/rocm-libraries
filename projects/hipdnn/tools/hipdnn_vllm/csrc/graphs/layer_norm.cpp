// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// layer_norm: hipDNN `layernorm` graph, INFERENCE phase.
//
// x [..., N] is flattened to rows [R, N] and each row block is described as the
// 4-D view [r, N, 1, 1] with the row stride kept, so hipDNN normalizes over
// every non-batch dim, i.e. over N. Rows are a no-copy view when the hidden dim
// is packed and the leading dims collapse (a padded row stride is kept). Any
// other layout, e.g. the transposed [B, S, N] output of a patch-embedding conv,
// is first packed with contiguous(), which is what ATen's layer_norm does for
// such inputs too. Scale and bias are [1, N, 1, 1]. The graph needs both; an
// absent weight or bias (elementwise_affine=False, bias=False) is passed as
// ones / zeros.
//
// Row count (until engines take a dynamic row count): when
// an exact plan for R rows is already cached (layer_norm_probe builds one; use
// it to pre-build HIP-graph capture sizes), the call is one launch. Otherwise R
// is split into power-of-two row blocks (its binary decomposition), so the
// plans per hidden size are bounded by log2(max rows) + 1. No plan is ever
// built for an arbitrary R on the hot path, and the choice depends only on the
// shape and the cache, so capture records the same launches as eager warm-up.
//
// A decline never fails the model: the op counts route=fallback and computes
// the vLLM LayerNorm.forward_native formula with ATen (normalize in the wider
// of the input and weight dtypes, cast back to the input dtype), unless
// `strict` is set, in which case it raises the decline.

#include "graphs.h"

#include "../graph_api.h"
#include "../plan_cache.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr int64_t UID_X = 1;
constexpr int64_t UID_SCALE = 2;
constexpr int64_t UID_BIAS = 3;
constexpr int64_t UID_EPS = 4;
constexpr int64_t UID_Y = 5;
constexpr const char* OP_NAME = "layer_norm";

bool isNormType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

// The cached entry for `key`, or null (no build, no counting).
std::shared_ptr<PlanEntry> cachedEntry(const at::Tensor& anchor, const PlanKey& key)
{
    const c10::DeviceGuard guard(anchor.device());
    const Runtime& runtime = Runtime::forDevice(anchor.device().index());
    return planCache().get(runtime.keyPrefix() + key.str());
}

void checkArgs(const at::Tensor& x,
               const std::optional<at::Tensor>& weight,
               const std::optional<at::Tensor>& bias)
{
    TORCH_CHECK(x.dim() >= 1, "layer_norm: x must have at least one dimension");
    const int64_t n = x.size(-1);
    for(const auto* p : {&weight, &bias})
    {
        if(p->has_value())
        {
            const at::Tensor& t = **p;
            TORCH_CHECK(t.dim() == 1 && t.size(0) == n,
                        "layer_norm: weight/bias must be [hidden] matching x's last dim, got ",
                        t.sizes(),
                        " for x ",
                        x.sizes());
            TORCH_CHECK(t.device() == x.device(), "layer_norm: tensors must share a device");
        }
    }
    TORCH_CHECK(!bias.has_value() || weight.has_value(),
                "layer_norm: bias without weight is not a vLLM LayerNorm configuration");
}

// Plugin-side pre-filters on valid inputs hipDNN cannot take.
std::optional<Decline> prefilter(const at::Tensor& x,
                                 const std::optional<at::Tensor>& weight,
                                 const std::optional<at::Tensor>& bias)
{
    if(!isNormType(x.scalar_type()) || (weight && !isNormType(weight->scalar_type()))
       || (bias && !isNormType(bias->scalar_type())))
    {
        return Decline::feature("dtype");
    }
    if(weight && bias && weight->scalar_type() != bias->scalar_type())
    {
        return Decline::feature("mixed_affine_dtype");
    }
    if(x.numel() == 0)
    {
        return Decline::feature("empty");
    }
    if((weight && weight->stride(0) != 1) || (bias && bias->stride(0) != 1))
    {
        return Decline::feature("strided_affine");
    }
    return std::nullopt;
}

// [R, N] rows view of x, or nullopt when the hidden dim is not packed or the
// leading dims do not collapse.
std::optional<at::Tensor> rowsView(const at::Tensor& x)
{
    const int64_t n = x.size(-1);
    if(n > 1 && x.stride(-1) != 1)
    {
        return std::nullopt;
    }
    const int64_t count = x.numel() / n;
    if(count == 1)
    {
        // One row: its stride is irrelevant, so describe it packed.
        return x.as_strided({1, n}, {n, 1});
    }
    // The leading dims collapse into one row dim when, ignoring size-1 dims,
    // each stride equals the next (inner) dim's stride times its size.
    int64_t rowStride = -1;
    int64_t expect = 0;
    for(int64_t i = x.dim() - 2; i >= 0; --i)
    {
        if(x.size(i) == 1)
        {
            continue;
        }
        if(rowStride < 0)
        {
            rowStride = x.stride(i);
        }
        else if(x.stride(i) != expect)
        {
            return std::nullopt;
        }
        expect = x.stride(i) * x.size(i);
    }
    return x.as_strided({count, n}, {rowStride, 1});
}

void rowLayout(const at::Tensor& t, std::vector<int64_t>& dims, std::vector<int64_t>& strides)
{
    dims = {t.size(0), t.size(1), 1, 1};
    strides = {t.stride(0), 1, 1, 1};
}

GraphBuilder builder(const at::Tensor& x,
                     const at::Tensor& scale,
                     const at::Tensor& bias,
                     const at::Tensor& y,
                     double eps)
{
    return [&x, &scale, &bias, &y, eps]() -> BuildResult {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_NAME)
            .set_io_data_type(*toHipdnnType(x.scalar_type()))
            .set_compute_data_type(fe::DataType::FLOAT);

        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        rowLayout(x, dims, strides);
        auto xT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("x")
                                               .set_uid(UID_X)
                                               .set_dim(dims)
                                               .set_stride(strides)
                                               .set_data_type(*toHipdnnType(x.scalar_type())));

        const int64_t n = x.size(1);
        const std::vector<int64_t> pDims{1, n, 1, 1};
        const std::vector<int64_t> pStrides{n, 1, 1, 1};
        auto sT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("scale")
                                               .set_uid(UID_SCALE)
                                               .set_dim(pDims)
                                               .set_stride(pStrides)
                                               .set_data_type(*toHipdnnType(scale.scalar_type())));
        auto bT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("bias")
                                               .set_uid(UID_BIAS)
                                               .set_dim(pDims)
                                               .set_stride(pStrides)
                                               .set_data_type(*toHipdnnType(bias.scalar_type())));
        auto epsT = fe::graph::Graph::tensor(
            fe::graph::TensorAttributes(static_cast<float>(eps)).set_name("eps").set_uid(UID_EPS));

        fe::graph::LayernormAttributes attrs;
        attrs.set_forward_phase(fe::NormFwdPhase::INFERENCE).set_epsilon(epsT);
        auto outputs = graph->layernorm(xT, sT, bT, attrs);

        rowLayout(y, dims, strides);
        outputs[0]
            ->set_uid(UID_Y)
            .set_dim(dims)
            .set_stride(strides)
            .set_data_type(*toHipdnnType(y.scalar_type()))
            .set_output(true);
        return GraphDef{.graph = graph, .slotUids = {UID_X, UID_SCALE, UID_BIAS, UID_Y}};
    };
}

PlanKey makeKey(const at::Tensor& x,
                const at::Tensor& scale,
                const at::Tensor& bias,
                const at::Tensor& y,
                double eps)
{
    PlanKey key(OP_NAME);
    key.add(eps).add(x).add(scale).add(bias).add(y);
    return key;
}

[[noreturn]] void raiseDecline(const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline.reason(), " :: ", decline.message);
}

// vLLM LayerNorm.forward_native.
at::Tensor nativeLayerNorm(const at::Tensor& x,
                           const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& bias,
                           double eps)
{
    const at::ScalarType compute
        = weight ? at::promote_types(x.scalar_type(), weight->scalar_type()) : x.scalar_type();
    const at::Tensor xc = x.scalar_type() == compute ? x : x.to(compute);
    const at::Tensor y = at::layer_norm(xc, {x.size(-1)}, weight, bias, eps);
    return y.scalar_type() == x.scalar_type() ? y : y.to(x.scalar_type());
}

at::Tensor declined(const Decline& decline,
                    bool strict,
                    const at::Tensor& x,
                    const std::optional<at::Tensor>& weight,
                    const std::optional<at::Tensor>& bias,
                    double eps)
{
    if(strict)
    {
        raiseDecline(decline);
    }
    return nativeLayerNorm(x, weight, bias, eps);
}

struct Affine
{
    at::Tensor scale;
    at::Tensor bias;
};

Affine affineFor(const at::Tensor& x,
                 const std::optional<at::Tensor>& weight,
                 const std::optional<at::Tensor>& bias)
{
    const auto paramType = weight ? weight->scalar_type() : x.scalar_type();
    const auto opts = x.options().dtype(paramType);
    return {.scale = weight ? *weight : at::ones({x.size(-1)}, opts),
            .bias = bias ? *bias : at::zeros({x.size(-1)}, opts)};
}

// Signature-level probe for one row block. Shared by the op and the probe.
// `anchor` is the caller's device tensor: `rows` may be a meta view.
std::optional<Decline> probeRows(const at::Tensor& rows, const Affine& a, double eps, const at::Tensor& anchor)
{
    const at::Tensor y = at::empty(rows.sizes(), rows.options().device(at::kMeta));
    return probeGraphOp(
        OP_NAME, makeKey(rows, a.scale, a.bias, y, eps), anchor, builder(rows, a.scale, a.bias, y, eps));
}

} // namespace

at::Tensor layerNorm(const at::Tensor& x,
                     const std::optional<at::Tensor>& weight,
                     const std::optional<at::Tensor>& bias,
                     double eps,
                     bool strict)
{
    checkArgs(x, weight, bias);
    if(auto decline = prefilter(x, weight, bias))
    {
        countDecline(OP_NAME, *decline, x);
        return declined(*decline, strict, x, weight, bias, eps);
    }
    std::optional<at::Tensor> rows = rowsView(x);
    if(!rows)
    {
        rows = rowsView(x.contiguous());
    }
    const Affine a = affineFor(x, weight, bias);
    const at::Tensor y = at::empty(rows->sizes(), x.options().memory_format(at::MemoryFormat::Contiguous));

    const int64_t total = rows->size(0);
    // One launch when the exact plan exists; otherwise power-of-two blocks,
    // largest first. A cached exact decline (from the probe) falls back without
    // building block plans for a signature hipDNN already rejected.
    const std::shared_ptr<PlanEntry> exactEntry = cachedEntry(x, makeKey(*rows, a.scale, a.bias, y, eps));
    if(exactEntry && !exactEntry->plan)
    {
        countDecline(OP_NAME, *exactEntry->decline, x);
        return declined(*exactEntry->decline, strict, x, weight, bias, eps);
    }
    const bool exact = exactEntry != nullptr;
    int64_t off = 0;
    while(off < total)
    {
        const int64_t left = total - off;
        int64_t block = left;
        if(!exact)
        {
            block = static_cast<int64_t>(std::bit_floor(static_cast<uint64_t>(left)));
        }
        const at::Tensor xb = rows->narrow(0, off, block);
        const at::Tensor yb = y.narrow(0, off, block);
        const PlanKey key = makeKey(xb, a.scale, a.bias, yb, eps);
        if(auto decline = runGraphOp(OP_NAME,
                                     key,
                                     x,
                                     builder(xb, a.scale, a.bias, yb, eps),
                                     {xb.data_ptr(), a.scale.data_ptr(), a.bias.data_ptr(), yb.data_ptr()}))
        {
            // runGraphOp counted this call as a fallback. Blocks before `off`
            // ran on hipDNN; the native result replaces the whole output.
            return declined(*decline, strict, x, weight, bias, eps);
        }
        off += block;
    }
    return y.view(x.sizes());
}

std::string layerNormProbe(const at::Tensor& x,
                           const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& bias,
                           double eps)
{
    checkArgs(x, weight, bias);
    if(auto decline = prefilter(x, weight, bias))
    {
        return decline->reason();
    }
    std::optional<at::Tensor> rows = rowsView(x);
    if(!rows)
    {
        // The op packs this layout first; probe the packed signature.
        rows = rowsView(at::empty(x.sizes(), x.options().device(at::kMeta)));
    }
    // The probe checks the exact signature (one block of R rows).
    auto decline = probeRows(*rows, affineFor(x, weight, bias), eps, x);
    return decline ? decline->reason() : std::string();
}

} // namespace hipdnn_vllm
