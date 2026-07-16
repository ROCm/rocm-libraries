// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "dispatcher/AotInstance.hpp"
#include "dispatcher/SdpaProblem.hpp"

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{
class IGraph;
}

namespace rocke_client::dispatcher
{

// Translates an SDPA op-graph into the normalized SdpaProblem ("graph ->
// normalized form"). Pure graph decode: NO HIP calls, so it is unit-testable
// without a device. The problem's `arch` is left empty here and filled by the
// dispatcher (which needs the HIP stream).
//
// Acts as an allowlist for the rocKE FMHA-fwd-MFMA family: it accepts only the
// SDPA forward graphs the family can realistically serve today and returns
// std::nullopt for everything else, so the engine never accepts a graph it has
// no instance for. Declined cases:
//   - not exactly one node, or the node is not SdpaAttributes;
//   - any of Q/K/V/O missing from the tensor map, or not rank-4;
//   - inconsistent Q/K/V/O element type, or a type outside the fp16/bf16 family;
//   - K/V/O head dim != Q head dim (family serves a single head_size);
//   - a physical layout that is neither BSHD- nor BHSD-contiguous;
//   - a contradictory mask configuration;
//   - unsupported capabilities: dropout != 0, alibi/padding masks, an explicit
//     scale, generate_stats/max_seq_len_kv, or any optional feature/output tensor
//     (additive mask, paged KV, varlen, dropout machinery, FP8, stats) that has
//     no representation in the #8866 selection contract.
//
// Shape, dtype, layout and mask_mode remain selection keys captured into the
// problem; the dispatcher's AOT catalog makes the final accept/reject decision.
std::optional<SdpaProblem>
    translate(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

// The op-agnostic launch bindings a captured SDPA node yields, plus the runtime
// batch dimension. The bindings carry the Q/K/V/O tensor uids and the SDPA launch
// scalars (log2-domain scale, seqlens, per-tensor token/head strides), each keyed
// by the kernel ABI's argument name so launch::bindArgs consumes them with no SDPA
// knowledge. `batch` is a grid dimension, not a kernel argument, so it lives here
// rather than in the bindings; sdpaGridSymbols() feeds it to launch::evalGrid.
// Dims are [B, H, S, D]; the token axis is dim 2 and the head axis dim 1, so
// strideToken = strides[2] and strideHead = strides[1]. Pure POD: no HIP handles.
struct SdpaLaunchInputs
{
    LaunchBindings bindings;
    std::int64_t batch = 0;
};

// Launch-oriented read of the single SDPA node translate() accepts: captures the
// concrete Q/K/V/O uids, per-tensor strides, seqlens and derived log2 scale into a
// LaunchBindings keyed by the FMHA ABI argument names. Returns std::nullopt for
// any graph translate() rejects, so a caller can rely on selection having accepted
// the graph before inputs are produced. Pure graph decode: NO HIP calls.
std::optional<SdpaLaunchInputs>
    buildSdpaLaunchInputs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

// Grid symbol table (symbol name -> value) for an SDPA kernel, from its compile
// spec and the runtime batch, ready to feed launch::evalGrid().
std::unordered_map<std::string, std::int64_t> sdpaGridSymbols(const CompileSpec& spec,
                                                              std::int64_t batch);

} // namespace rocke_client::dispatcher
