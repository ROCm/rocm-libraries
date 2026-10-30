// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The API graph builders (csrc/graphs/*.cpp) use to define an op:
//
//   PlanKey key("rms_norm");              // op kind
//   key.add(eps).add(x).add(weight);      // every graph-affecting attribute
//   auto d = runGraphOp("rms_norm", key, x, [&]() -> BuildResult {
//       auto g = std::make_shared<Graph>();
//       ... describeTensor(x, kX, "x") ...
//       return GraphDef{g, {kX, kW, kY}};  // uid bound to each execute slot
//   }, {x.data_ptr(), w.data_ptr(), y.data_ptr()});
//   if (d) { /* declined: d->reason() */ }
//
// runGraphOp resolves the current torch stream of `anchor`'s device, finds or
// builds the plan, counts the call, allocates the workspace from torch's
// caching allocator and executes with the slot pointers. It never syncs.

#pragma once

#include "decline.h"
#include "runtime.h"

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>

namespace hipdnn_vllm
{

// torch dtype -> hipDNN DataType, for formats both sides state exactly.
std::optional<hipdnn_frontend::DataType> toHipdnnType(at::ScalarType type);

// A tensor attribute with the tensor's dims, strides and dtype. The caller must
// have checked toHipdnnType(t.scalar_type()).
std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    describeTensor(const at::Tensor& t, int64_t uid, const std::string& name);

// Builder for plan-cache keys (op kind, graph-affecting attributes,
// dims, strides, dtypes; the Runtime prefixes device ordinal and engine order).
class PlanKey
{
public:
    explicit PlanKey(const char* op);
    PlanKey& add(int64_t v);
    PlanKey& add(double v);
    PlanKey& add(const char* s);
    PlanKey& add(const at::Tensor& t); // dims, strides, dtype (not data)
    const std::string& str() const
    {
        return _key;
    }

private:
    std::string _key;
};

// Build-or-fetch + execute. Returns std::nullopt when hipDNN ran the op, or the
// Decline when it did not (already counted as a fallback for `op`).
std::optional<Decline> runGraphOp(const char* op,
                                  const PlanKey& key,
                                  const at::Tensor& anchor,
                                  const GraphBuilder& builder,
                                  std::initializer_list<const void*> slotPtrs);
// Same, for a slot count known only at run time.
std::optional<Decline> runGraphOp(const char* op,
                                  const PlanKey& key,
                                  const at::Tensor& anchor,
                                  const GraphBuilder& builder,
                                  const void* const* slotPtrs,
                                  size_t numSlots);

// Build-or-fetch only: whether hipDNN accepts this signature, without counting a
// call. Used by supports_args-style probes. Under capture it never builds (a
// miss is CAPTURE_INVARIANT), like runGraphOp.
std::optional<Decline>
    probeGraphOp(const char* op, const PlanKey& key, const at::Tensor& anchor, const GraphBuilder& builder);
// probeGraphOp's entry itself (plan with engine name, or the decline).
std::shared_ptr<PlanEntry>
    probeEntry(const char* op, const PlanKey& key, const at::Tensor& anchor, const GraphBuilder& builder);

// Count a call the op declined before reaching hipDNN (pre-filters).
void countDecline(const char* op, const Decline& decline, const at::Tensor& anchor);

} // namespace hipdnn_vllm
