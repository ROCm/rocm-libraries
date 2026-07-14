// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

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

} // namespace rocke_client::dispatcher
