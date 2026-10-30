// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "graph_api.h"

#include "stats.h"

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/core/DeviceGuard.h>

#include <array>
#include <cstdio>
#include <vector>

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

hipStream_t currentStream(const at::Tensor& anchor)
{
    return at::hip::getCurrentHIPStreamMasqueradingAsCUDA(anchor.device().index()).stream();
}

bool capturing(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &status) == hipSuccess
           && status == hipStreamCaptureStatusActive;
}

std::string fullKey(const Runtime& runtime, const PlanKey& key)
{
    return runtime.keyPrefix() + key.str();
}

} // namespace

std::optional<fe::DataType> toHipdnnType(at::ScalarType type)
{
    switch(type)
    {
    case at::kFloat:
        return fe::DataType::FLOAT;
    case at::kHalf:
        return fe::DataType::HALF;
    case at::kBFloat16:
        return fe::DataType::BFLOAT16;
    case at::kDouble:
        return fe::DataType::DOUBLE;
    case at::kInt:
        return fe::DataType::INT32;
    case at::kLong:
        return fe::DataType::INT64;
    case at::kChar:
        return fe::DataType::INT8;
    case at::kByte:
        return fe::DataType::UINT8;
    case at::kBool:
        return fe::DataType::BOOLEAN;
    default:
        return std::nullopt;
    }
}

std::shared_ptr<fe::graph::TensorAttributes>
    describeTensor(const at::Tensor& t, int64_t uid, const std::string& name)
{
    const auto type = toHipdnnType(t.scalar_type());
    TORCH_CHECK(type.has_value(), "hipdnn_vllm: dtype ", t.scalar_type(), " has no hipDNN type");
    return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                        .set_name(name)
                                        .set_uid(uid)
                                        .set_dim(t.sizes().vec())
                                        .set_stride(t.strides().vec())
                                        .set_data_type(*type));
}

PlanKey::PlanKey(const char* op)
    : _key(op)
{
    _key.reserve(160);
}

PlanKey& PlanKey::add(int64_t v)
{
    _key.push_back('|');
    _key.append(std::to_string(v));
    return *this;
}

PlanKey& PlanKey::add(double v)
{
    // Exact bits: two attributes that differ in the last ulp are distinct graphs.
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "|%a", v);
    _key.append(buffer.data());
    return *this;
}

PlanKey& PlanKey::add(const char* s)
{
    _key.push_back('|');
    _key.append(s);
    return *this;
}

PlanKey& PlanKey::add(const at::Tensor& t)
{
    _key.append("|t");
    _key.append(std::to_string(static_cast<int>(t.scalar_type())));
    _key.push_back('[');
    for(const int64_t d : t.sizes())
    {
        _key.append(std::to_string(d));
        _key.push_back(',');
    }
    _key.push_back(':');
    for(const int64_t s : t.strides())
    {
        _key.append(std::to_string(s));
        _key.push_back(',');
    }
    _key.push_back(']');
    return *this;
}

std::optional<Decline> runGraphOp(const char* op,
                                  const PlanKey& key,
                                  const at::Tensor& anchor,
                                  const GraphBuilder& builder,
                                  const void* const* slotPtrs,
                                  size_t numSlots)
{
    const c10::DeviceGuard guard(anchor.device());
    Runtime& runtime = Runtime::forDevice(anchor.device().index());
    hipStream_t stream = currentStream(anchor);
    const std::shared_ptr<PlanEntry> entry
        = runtime.getOrBuild(op, fullKey(runtime, key), builder, stream);
    if(!entry->plan)
    {
        (capturing(stream) ? entry->captureCell : entry->eagerCell)
            ->fetch_add(1, std::memory_order_relaxed);
        return entry->decline;
    }

    at::Tensor workspace;
    void* workspacePtr = nullptr;
    if(entry->plan->workspaceBytes > 0)
    {
        // Caching allocator: no hipMalloc on the hot path, and graph-pool backed
        // during capture.
        workspace = at::empty({entry->plan->workspaceBytes},
                              at::TensorOptions().dtype(at::kByte).device(anchor.device()));
        workspacePtr = workspace.data_ptr();
    }
    return runtime.execute(*entry, slotPtrs, numSlots, workspacePtr, stream);
}

std::optional<Decline> runGraphOp(const char* op,
                                  const PlanKey& key,
                                  const at::Tensor& anchor,
                                  const GraphBuilder& builder,
                                  std::initializer_list<const void*> slotPtrs)
{
    return runGraphOp(op, key, anchor, builder, slotPtrs.begin(), slotPtrs.size());
}

std::shared_ptr<PlanEntry>
    probeEntry(const char* op, const PlanKey& key, const at::Tensor& anchor, const GraphBuilder& builder)
{
    const c10::DeviceGuard guard(anchor.device());
    Runtime& runtime = Runtime::forDevice(anchor.device().index());
    return runtime.getOrBuild(op, fullKey(runtime, key), builder, currentStream(anchor));
}

std::optional<Decline>
    probeGraphOp(const char* op, const PlanKey& key, const at::Tensor& anchor, const GraphBuilder& builder)
{
    const std::shared_ptr<PlanEntry> entry = probeEntry(op, key, anchor, builder);
    return entry->plan ? std::nullopt : entry->decline;
}

void countDecline(const char* op, const Decline& decline, const at::Tensor& anchor)
{
    const bool capture = anchor.is_cuda() && capturing(currentStream(anchor));
    Stats::instance().count(op, ROUTE_FALLBACK, decline.reason(), capture);
}

} // namespace hipdnn_vllm
