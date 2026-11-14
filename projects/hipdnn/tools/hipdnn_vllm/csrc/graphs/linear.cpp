// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// linear: hipDNN `matmul` (+ pointwise ADD bias) graph for vLLM's unquantized
// linear layers. y[M, N] = x[M, K] · Wᵀ + bias, with W the
// [N, K] weight vLLM stores.
//
// * x is flattened to [M, K] (a view when the leading dims allow it) and
//   described with its own row stride, so views of fused projections pass
//   through without a copy.
// * W is described as the transposed view Wᵀ [K, N] with strides
//   [1, W.stride(0)]: column-major with leading dimension = W's row stride.
//   vLLM pads weight rows on gfx11 to avoid cache aliasing
//   (maybe_pad_weight_avoid_cache_aliasing), so the row stride may exceed K.
// * bias is the hipBLASLt bias epilogue: pointwise ADD of a [1, N] tensor to
//   the virtual matmul output.
//
// M (the token count) is dynamic and no matmul engine takes an override shape
// yet, so every M has its own exact plan in the process plan cache (an LRU).
// Under HIP-graph capture the plan must already exist: vLLM runs each capture
// size eagerly before capturing it (cudagraph_num_of_warmups), which builds
// it. The Runtime's capture rules apply (runtime.h): a miss under capture
// declines with CAPTURE_INVARIANT instead of building inside the capture, and
// a plan a capture uses is pinned.

#include "graphs.h"

#include "../graph_api.h"
#include "../plan_cache.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr int64_t UID_X = 1;
constexpr int64_t UID_W = 2;
constexpr int64_t UID_BIAS = 3;
constexpr int64_t UID_Y = 4;
constexpr const char* OP_NAME = "linear";
// hipBLASLt kernels assume 16-byte aligned operand pointers.
constexpr uintptr_t ALIGN_BYTES = 16;

bool isGemmType(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

bool aligned(const at::Tensor& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % ALIGN_BYTES == 0;
}

// The operands as hipDNN sees them: x2 [M, K], Wᵀ [K, N], y [M, N], bias [1, N].
struct Problem
{
    at::Tensor x2;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    // Row strides with size-1 dims normalized (torch allows any stride there),
    // so the key and the hipBLASLt leading dimensions depend on the layout only.
    int64_t xRowStride = 0;
    int64_t wRowStride = 0;
    bool hasBias = false;
};

void checkArgs(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    TORCH_CHECK(x.dim() >= 1, "linear: x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(-1),
                "linear: weight must be [out, in] with in == x's last dim, got ",
                weight.sizes(),
                " for x ",
                x.sizes());
    TORCH_CHECK(x.device() == weight.device(), "linear: x and weight must share a device");
    if(bias.has_value())
    {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0),
                    "linear: bias must be [out] matching weight's first dim, got ",
                    bias->sizes());
        TORCH_CHECK(bias->device() == x.device(), "linear: bias must be on x's device");
    }
}

// Plugin-side pre-filters: valid inputs hipDNN cannot take are a Decline.
std::optional<Decline> prefilter(const at::Tensor& x,
                                 const at::Tensor& weight,
                                 const std::optional<at::Tensor>& bias,
                                 Problem& p)
{
    checkArgs(x, weight, bias);
    if(!x.is_cuda())
    {
        return Decline::feature("device");
    }
    const at::ScalarType t = x.scalar_type();
    if(!isGemmType(t) || weight.scalar_type() != t
       || (bias.has_value() && bias->scalar_type() != t))
    {
        return Decline::feature("dtype");
    }
    p.k = x.size(-1);
    p.n = weight.size(0);
    p.m = p.k == 0 ? 0 : x.numel() / p.k;
    if(x.numel() == 0 || p.n == 0 || p.k == 0)
    {
        return Decline::feature("empty");
    }

    // Flatten leading dims; reshape copies only when no view exists.
    p.x2 = x.dim() == 2 ? x : x.reshape({p.m, p.k});
    if(p.k != 1 && p.x2.stride(1) != 1)
    {
        // Column-major x (a transposed view) is a valid hipBLASLt layout, but
        // vLLM never passes one; copy it rather than add a second plan family.
        p.x2 = p.x2.contiguous();
    }
    p.xRowStride = p.m == 1 ? p.k : p.x2.stride(0);
    if(p.xRowStride < p.k)
    {
        return Decline::feature("x_layout");
    }
    p.wRowStride = p.n == 1 ? p.k : weight.stride(0);
    if((p.k != 1 && weight.stride(1) != 1) || p.wRowStride < p.k)
    {
        return Decline::feature("weight_layout");
    }
    p.hasBias = bias.has_value();
    if(p.hasBias && p.n != 1 && bias->stride(0) != 1)
    {
        return Decline::feature("bias_layout");
    }
    if(!aligned(p.x2) || !aligned(weight) || (p.hasBias && !aligned(*bias)))
    {
        return Decline::feature("align");
    }
    return std::nullopt;
}

GraphBuilder builder(const Problem& p, at::ScalarType type)
{
    return [&p, type]() -> BuildResult {
        const fe::DataType dt = *toHipdnnType(type);
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_NAME)
            .set_io_data_type(dt)
            .set_intermediate_data_type(fe::DataType::FLOAT)
            .set_compute_data_type(fe::DataType::FLOAT);

        auto xT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("x")
                                               .set_uid(UID_X)
                                               .set_dim({p.m, p.k})
                                               .set_stride({p.xRowStride, 1})
                                               .set_data_type(dt));
        auto wT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("weight_t")
                                               .set_uid(UID_W)
                                               .set_dim({p.k, p.n})
                                               .set_stride({1, p.wRowStride})
                                               .set_data_type(dt));

        fe::graph::MatmulAttributes mm;
        mm.set_name("linear_matmul");
        auto out = graph->matmul(xT, wT, mm);
        std::vector<int64_t> slots{UID_X, UID_W};
        if(p.hasBias)
        {
            out->set_data_type(fe::DataType::FLOAT);
            auto bT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                                   .set_name("bias")
                                                   .set_uid(UID_BIAS)
                                                   .set_dim({1, p.n})
                                                   .set_stride({p.n, 1})
                                                   .set_data_type(dt));
            fe::graph::PointwiseAttributes add;
            add.set_name("linear_bias").set_mode(fe::PointwiseMode::ADD);
            out = graph->pointwise(out, bT, add);
            slots.push_back(UID_BIAS);
        }
        out->set_uid(UID_Y)
            .set_dim({p.m, p.n})
            .set_stride({p.n, 1})
            .set_data_type(dt)
            .set_output(true);
        slots.push_back(UID_Y);
        return GraphDef{.graph = graph, .slotUids = std::move(slots)};
    };
}

PlanKey makeKey(const Problem& p, at::ScalarType type)
{
    PlanKey key(OP_NAME);
    key.add(static_cast<int64_t>(type))
        .add(p.m)
        .add(p.n)
        .add(p.k)
        .add(p.xRowStride)
        .add(p.wRowStride)
        .add(static_cast<int64_t>(p.hasBias));
    return key;
}

struct LinearResult
{
    at::Tensor y;
    std::optional<Decline> decline;
};

LinearResult declined(Decline decline)
{
    return LinearResult{.y = at::Tensor(), .decline = std::move(decline)};
}

LinearResult
    run(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    Problem p;
    if(auto decline = prefilter(x, weight, bias, p))
    {
        countDecline(OP_NAME, *decline, x);
        return declined(std::move(*decline));
    }
    const at::ScalarType type = x.scalar_type();
    const PlanKey key = makeKey(p, type);

    const at::Tensor y2
        = at::empty({p.m, p.n}, x.options().memory_format(at::MemoryFormat::Contiguous));
    std::optional<Decline> decline;
    if(p.hasBias)
    {
        decline = runGraphOp(OP_NAME,
                             key,
                             x,
                             builder(p, type),
                             {p.x2.data_ptr(), weight.data_ptr(), bias->data_ptr(), y2.data_ptr()});
    }
    else
    {
        decline = runGraphOp(
            OP_NAME, key, x, builder(p, type), {p.x2.data_ptr(), weight.data_ptr(), y2.data_ptr()});
    }
    if(decline)
    {
        return declined(std::move(*decline));
    }
    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end() - 1);
    outShape.push_back(p.n);
    return LinearResult{.y = x.dim() == 2 ? y2 : y2.view(outShape), .decline = std::nullopt};
}

} // namespace

at::Tensor
    linear(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    LinearResult r = run(x, weight, bias);
    if(r.decline)
    {
        TORCH_CHECK(
            false, declineErrorPrefix(OP_NAME), r.decline->reason(), " :: ", r.decline->message);
    }
    return r.y;
}

std::tuple<std::optional<at::Tensor>, std::string, std::string>
    linearTry(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    LinearResult r = run(x, weight, bias);
    if(r.decline)
    {
        return {std::nullopt, r.decline->reason(), r.decline->message};
    }
    return {std::move(r.y), std::string(), std::string()};
}

std::pair<std::string, std::string> linearProbe(const at::Tensor& x,
                                                const at::Tensor& weight,
                                                const std::optional<at::Tensor>& bias)
{
    Problem p;
    if(auto decline = prefilter(x, weight, bias, p))
    {
        return {decline->reason(), std::string()};
    }
    const at::ScalarType type = x.scalar_type();
    const std::shared_ptr<PlanEntry> entry
        = probeEntry(OP_NAME, makeKey(p, type), x, builder(p, type));
    if(!entry->plan)
    {
        return {entry->decline ? entry->decline->reason() : std::string("NO_BUILD"), std::string()};
    }
    return {std::string(), entry->plan->engineName};
}

} // namespace hipdnn_vllm
