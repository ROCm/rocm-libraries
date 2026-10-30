// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// custom_op: one hipDNN CUSTOM_OP node, the opaque op an engine claims by op id
// and payload (e.g. megakernel.<model> with its JSON key, docs in the megakernel
// engine). The plugin does not interpret either: the graph is the input tensors
// (any count, uids 1..n in order, no outputs) plus `op_id` and the payload bytes.
//
// The plan key is the op id, the payload bytes and every input's dims, strides
// and dtype, never the addresses: a call with new tensors of the same shape is a
// cache hit and rebinds the pointers.
//
// custom_op_prepare builds (or fetches) the plan without running it and returns
// the decline reason, "" when an engine took it. It never raises on a decline
// and builds nothing under capture (the decline is CAPTURE_INVARIANT).
// custom_op_exec runs the plan. On a decline it counts route=fallback and
// returns (the caller owns the fallback; the op has no outputs to fill), or
// raises when `strict`; a plan miss under capture is such a decline.
//
// Opt-in: until the bootstrap enables it (`custom_op` in HIPDNN_VLLM_OPS) both
// calls decline DISABLED without building a graph, so no engine ever sees a
// CUSTOM_OP node from a process that did not ask for one.

#include "graphs.h"

#include "../graph_api.h"

#include <ATen/ATen.h>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

constexpr const char* OP_NAME = "custom_op";

std::atomic<bool> g_enabled{false};

void checkArgs(const std::string& opId, const std::vector<at::Tensor>& inputs)
{
    TORCH_CHECK(!opId.empty(), "custom_op: op_id must not be empty");
    TORCH_CHECK(!inputs.empty(), "custom_op: at least one input tensor is required");
    for(const auto& t : inputs)
    {
        TORCH_CHECK(t.is_cuda(), "custom_op: inputs must be device tensors");
        TORCH_CHECK(t.device() == inputs[0].device(), "custom_op: inputs must share a device");
    }
}

std::optional<Decline> prefilter(const std::vector<at::Tensor>& inputs)
{
    if(!g_enabled.load(std::memory_order_relaxed))
    {
        return Decline{.stage = Stage::DISABLED, .detail = {}, .message = {}};
    }
    for(const auto& t : inputs)
    {
        if(!toHipdnnType(t.scalar_type()))
        {
            return Decline::feature("dtype");
        }
    }
    return std::nullopt;
}

PlanKey planKey(const std::string& opId, const std::string& payload, const std::vector<at::Tensor>& inputs)
{
    PlanKey key(OP_NAME);
    // Lengths first, so no op id / payload byte sequence can alias another.
    key.add(static_cast<int64_t>(opId.size())).add(opId.c_str());
    key.add(static_cast<int64_t>(payload.size())).add(payload.c_str());
    for(const auto& t : inputs)
    {
        key.add(t);
    }
    return key;
}

GraphBuilder builder(const std::string& opId, const std::string& payload, const std::vector<at::Tensor>& inputs)
{
    return [&]() -> BuildResult {
        auto graph = std::make_shared<fe::graph::Graph>();
        // The engine ignores the compute type; the frontend requires one.
        graph->set_compute_data_type(fe::DataType::FLOAT);
        std::vector<std::shared_ptr<fe::graph::TensorAttributes>> tensors;
        std::vector<int64_t> uids;
        for(size_t i = 0; i < inputs.size(); ++i)
        {
            const auto uid = static_cast<int64_t>(i + 1);
            tensors.push_back(describeTensor(inputs[i], uid, "in" + std::to_string(i)));
            uids.push_back(uid);
        }
        fe::graph::CustomOpAttributes attrs;
        attrs.set_custom_op_id(opId).set_compute_data_type(fe::DataType::FLOAT);
        attrs.set_data(std::vector<uint8_t>(payload.begin(), payload.end()));
        graph->custom_op(std::move(tensors), 0, std::move(attrs));
        return GraphDef{.graph = graph, .slotUids = std::move(uids)};
    };
}

} // namespace

void customOpSetEnabled(bool enabled)
{
    g_enabled.store(enabled, std::memory_order_relaxed);
}

std::string customOpPrepare(const std::string& opId,
                            const std::string& payload,
                            const std::vector<at::Tensor>& inputs)
{
    checkArgs(opId, inputs);
    if(auto d = prefilter(inputs))
    {
        return d->reason();
    }
    const auto d = probeGraphOp(OP_NAME, planKey(opId, payload, inputs), inputs[0], builder(opId, payload, inputs));
    return d ? d->reason() : std::string();
}

void customOpExec(const std::string& opId,
                  const std::string& payload,
                  const std::vector<at::Tensor>& inputs,
                  bool strict)
{
    checkArgs(opId, inputs);
    std::optional<Decline> decline = prefilter(inputs);
    if(decline)
    {
        countDecline(OP_NAME, *decline, inputs[0]);
    }
    else
    {
        std::vector<const void*> ptrs;
        ptrs.reserve(inputs.size());
        for(const auto& t : inputs)
        {
            ptrs.push_back(t.data_ptr());
        }
        decline = runGraphOp(OP_NAME,
                             planKey(opId, payload, inputs),
                             inputs[0],
                             builder(opId, payload, inputs),
                             ptrs.data(),
                             ptrs.size());
    }
    if(decline && strict)
    {
        TORCH_CHECK(false, declineErrorPrefix(OP_NAME), decline->reason(), " :: ", decline->message);
    }
}

} // namespace hipdnn_vllm
