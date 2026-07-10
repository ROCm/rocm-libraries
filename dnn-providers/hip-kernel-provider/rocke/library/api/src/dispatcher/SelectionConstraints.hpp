// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <string>

#include "dispatcher/AotInstance.hpp"
#include "dispatcher/SdpaProblem.hpp"

namespace rocke_client::dispatcher
{

// Returns whether a runtime attribute set satisfies a set of attribute
// constraints. This is the C++ mirror of the authoritative Python contract in
// rocke_client_aot.instance_schema.attributes_match_constraints (PR #8866):
//   - a constrained attribute that is absent from `attributes` => no match;
//   - `equals`     : value must equal the operand;
//   - `not_equals` : value must not equal the operand;
//   - `one_of`     : value must be one of the operands;
//   - all present operators for an attribute must hold.
bool attributesMatchConstraints(const std::map<std::string, AttrValue>& attributes,
                                const AttributeConstraints& constraints);

// Returns whether an AOT instance can serve a normalized problem: exact shape
// match (dtype, layout, seqlen_q/k, head counts, head_size, mask_mode), the
// batch range, and the attribute constraints. Mirrors the PR #8866 selection
// contract; `compile_spec.block_size_{q,k}` are intentionally not compared.
bool satisfies(const AotInstance& instance, const SdpaProblem& problem);

// As above, but reuses a pre-built problem attribute view so a caller filtering
// many candidates against one problem builds it once (see SdpaProblem::attributes).
bool satisfies(const AotInstance& instance,
               const SdpaProblem& problem,
               const std::map<std::string, AttrValue>& problemAttributes);

} // namespace rocke_client::dispatcher
