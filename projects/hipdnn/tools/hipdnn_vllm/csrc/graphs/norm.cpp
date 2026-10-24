// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// rms_norm: hipDNN `rmsnorm` graph, INFERENCE phase.
//
// hipDNN's rmsnorm normalizes over the maximal trailing suffix of dims where the
// scale's dims equal x's (frontend validateScaleNormalizedShape; the scale's
// leading dims must be 1), and the shipped engines take 4-D or 5-D tensors. A
// row-major x [d0, ..., dk, N] therefore has its leading dims collapsed, without
// copying, into one row dim R and is described as [R, N, 1, 1] with the scale
// [1, N, 1, 1]: the normalized suffix is [N, 1, 1]. The row stride is kept, so a
// padded row stride (a view of a fused projection) is passed through; leading
// dims that do not collapse into one stride decline. eps is baked into the graph
// as a compile-time scalar, so it is part of the key.

#include "graphs.h"

#include "../graph_api.h"

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr int64_t UID_X = 1;
constexpr int64_t UID_W = 2;
constexpr int64_t UID_EPS = 3;
constexpr int64_t UID_Y = 4;
constexpr size_t RANK = 4; // the smallest rank the engines take
constexpr const char* OP_NAME = "rms_norm";

bool isNormType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

// Plugin-side pre-filters. Shapes are checked with TORCH_CHECK (caller error);
// valid inputs hipDNN cannot take are a Decline.
std::optional<Decline> prefilter(const at::Tensor& x, const at::Tensor& weight)
{
    TORCH_CHECK(x.dim() >= 1, "rms_norm: x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "rms_norm: weight must be [hidden] matching x's last dim, got ",
                weight.sizes(),
                " for x ",
                x.sizes());
    TORCH_CHECK(x.device() == weight.device(), "rms_norm: x and weight must share a device");
    if(!isNormType(x.scalar_type()) || !isNormType(weight.scalar_type()))
    {
        return Decline::feature("dtype");
    }
    if(x.numel() == 0)
    {
        return Decline::feature("empty");
    }
    if(x.stride(-1) != 1 || weight.stride(0) != 1)
    {
        // Elements within a row are strided: not a row-contiguous norm input.
        return Decline::feature("strided_rows");
    }
    return std::nullopt;
}

// x [d0, ..., dk, N] as the no-copy view [R, N], or nullopt when the leading
// dims cannot be described by one row stride.
std::optional<at::Tensor> asRows(const at::Tensor& x)
{
    const int64_t hidden = x.size(-1);
    const std::array<int64_t, 2> shape{x.numel() / hidden, hidden};
    const auto strides = at::detail::computeStride(x.sizes(), x.strides(), at::IntArrayRef(shape));
    if(!strides)
    {
        return std::nullopt;
    }
    return x.as_strided(shape, *strides, x.storage_offset());
}

// [R, N] -> dims/strides of [R, N, 1, 1].
void rowLayout(const at::Tensor& t, std::vector<int64_t>& dims, std::vector<int64_t>& strides)
{
    dims.assign(RANK, 1);
    strides.assign(RANK, 1);
    dims[0] = t.size(0);
    strides[0] = t.stride(0);
    dims[1] = t.size(1);
    strides[1] = t.stride(1);
}

GraphBuilder builder(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps)
{
    return [&x, &weight, &y, eps]() -> BuildResult {
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

        std::vector<int64_t> wDims(dims.size(), 1);
        std::vector<int64_t> wStrides(dims.size(), 0);
        wDims[1] = weight.size(0);
        wStrides[1] = weight.stride(0);
        auto wT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("weight")
                                               .set_uid(UID_W)
                                               .set_dim(wDims)
                                               .set_stride(wStrides)
                                               .set_data_type(*toHipdnnType(weight.scalar_type())));

        auto epsT = fe::graph::Graph::tensor(
            fe::graph::TensorAttributes(static_cast<float>(eps)).set_name("eps").set_uid(UID_EPS));

        fe::graph::RMSNormAttributes attrs;
        attrs.set_forward_phase(fe::NormFwdPhase::INFERENCE).set_epsilon(epsT);
        auto outputs = graph->rmsnorm(xT, wT, attrs);

        rowLayout(y, dims, strides);
        outputs[0]
            ->set_uid(UID_Y)
            .set_dim(dims)
            .set_stride(strides)
            .set_data_type(*toHipdnnType(y.scalar_type()))
            .set_output(true);
        return GraphDef{.graph = graph, .slotUids = {UID_X, UID_W, UID_Y}};
    };
}

PlanKey makeKey(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps)
{
    PlanKey key(OP_NAME);
    key.add(eps).add(x).add(weight).add(y);
    return key;
}

[[noreturn]] void raiseDecline(const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline.reason(), " :: ", decline.message);
}

} // namespace

std::string declineErrorPrefix(const char* op)
{
    return std::string("hipdnn_vllm decline [") + op + "]: ";
}

std::optional<Decline> rmsNormPrefilter(const at::Tensor& x, const at::Tensor& weight)
{
    return prefilter(x, weight);
}

std::optional<at::Tensor> rmsNormAsRows(const at::Tensor& x)
{
    return asRows(x);
}

void rmsNormRowLayout(const at::Tensor& t, std::vector<int64_t>& dims, std::vector<int64_t>& strides)
{
    rowLayout(t, dims, strides);
}

std::optional<Decline>
    rmsNormInto(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps)
{
    const PlanKey key = makeKey(x, weight, y, eps);
    return runGraphOp(
        OP_NAME, key, x, builder(x, weight, y, eps), {x.data_ptr(), weight.data_ptr(), y.data_ptr()});
}

std::optional<Decline> rmsNormPrepare(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps, const at::Tensor& anchor)
{
    const PlanKey key = makeKey(x, weight, y, eps);
    return probeGraphOp(OP_NAME, key, anchor, builder(x, weight, y, eps));
}

at::Tensor rmsNorm(const at::Tensor& xIn, const at::Tensor& weight, double eps)
{
    std::optional<Decline> decline = prefilter(xIn, weight);
    const std::optional<at::Tensor> rows = decline ? std::nullopt : asRows(xIn);
    if(!decline && !rows)
    {
        decline = Decline::feature("noncollapsible_rows");
    }
    if(decline)
    {
        countDecline(OP_NAME, *decline, xIn);
        raiseDecline(*decline);
    }
    const at::Tensor& x = *rows;
    const at::Tensor y = at::empty(x.sizes(), x.options().memory_format(at::MemoryFormat::Contiguous));
    const PlanKey key = makeKey(x, weight, y, eps);
    if(auto failed = runGraphOp(
           OP_NAME, key, x, builder(x, weight, y, eps), {x.data_ptr(), weight.data_ptr(), y.data_ptr()}))
    {
        raiseDecline(*failed);
    }
    return y.view(xIn.sizes());
}

std::string rmsNormProbe(const at::Tensor& xIn, const at::Tensor& weight, double eps)
{
    if(auto decline = prefilter(xIn, weight))
    {
        return decline->reason();
    }
    const std::optional<at::Tensor> rows = asRows(xIn);
    if(!rows)
    {
        return Decline::feature("noncollapsible_rows").reason();
    }
    // The output is always contiguous; a meta tensor gives its strides without
    // allocating device memory.
    const at::Tensor y = at::empty(rows->sizes(), rows->options().device(at::kMeta));
    const PlanKey key = makeKey(*rows, weight, y, eps);
    const auto decline = probeGraphOp(OP_NAME, key, *rows, builder(*rows, weight, y, eps));
    return decline ? decline->reason() : std::string();
}

} // namespace hipdnn_vllm
