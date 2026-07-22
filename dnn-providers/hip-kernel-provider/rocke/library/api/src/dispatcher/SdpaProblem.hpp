// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "dispatcher/AotInstance.hpp" // AttrValue

namespace rocke_client::dispatcher
{

// Physical tensor layout inferred from a graph tensor's dims + strides.
// The rocKE SDPA FMHA MFMA family (PR #8866) is canonical-layout "BSHD".
enum class TensorLayout
{
    BSHD,
    BHSD,
    OTHER
};

const char* toString(TensorLayout layout);

// The normalized form of an SDPA graph: the arch/shape/attribute inputs the
// dispatcher matches against the AOT catalog. This is the "graph -> normalized
// form" step; it is produced by SdpaGraphAdapter::translate() and carries no
// device pointers and does nothing HIP-specific (arch is filled separately).
struct SdpaProblem
{
    std::string op = "sdpa_fwd";
    std::string arch; // filled by the dispatcher; empty when undetectable
    std::string dtype; // provider spelling, e.g. "fp16"
    TensorLayout layout = TensorLayout::OTHER;
    std::int64_t batch = 0;
    std::int64_t seqlenQ = 0;
    std::int64_t seqlenK = 0;
    std::int64_t numQueryHeads = 0;
    std::int64_t numKvHeads = 0;
    std::int64_t headSize = 0;
    std::string maskMode = "none";
    double dropoutProbability = 0.0;
    bool paddingMask = false;
    bool alibiMask = false;
    std::string scalePolicy = "default_1_over_sqrt_d";

    // The runtime attribute view matched against
    // AotInstance::attributeConstraints. Keys and value kinds mirror the
    // selection.attribute_constraints contract in PR #8866:
    // mask_mode(string), dropout_probability(double), scale_policy(string),
    // padding_mask(bool), alibi_mask(bool).
    std::map<std::string, AttrValue> attributes() const;
};

} // namespace rocke_client::dispatcher
