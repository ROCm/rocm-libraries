// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace rocke_client::dispatcher
{

// A single runtime attribute value, matched against an AotInstance's
// selection constraints. The alternatives mirror the JSON value kinds that the
// rocKE AOT sidecar/instance schema (PR #8866) uses for
// selection.attribute_constraints: booleans (padding_mask, alibi_mask),
// numbers (dropout_probability), and strings (mask_mode, scale_policy).
//
// Two construction pitfalls the kpack catalog parser (and tests) MUST avoid,
// because std::variant matching is alternative-exact:
//   * String literals: AttrValue{"none"} selects the `bool` alternative (a
//     const char* converts to bool, not to std::string), silently yielding
//     `true`. Always wrap string values: AttrValue{std::string("none")}.
//   * Numbers: equality only holds within the same alternative, so a
//     `double` problem value never compares equal to an `int64_t` operand.
//     Widen numeric constraint operands to `double` at parse time to match the
//     values SdpaProblem::attributes() produces (e.g. dropout_probability).
using AttrValue = std::variant<bool, std::int64_t, double, std::string>;

// One selection.attribute_constraints[name] rule. Mirrors the Python contract in
// rocke_client_aot.instance_schema.attributes_match_constraints: any subset of
// {equals, not_equals, one_of} may be present and ALL present operators must hold.
struct AttributeRule
{
    std::optional<AttrValue> equals;
    std::optional<AttrValue> notEquals;
    std::optional<std::vector<AttrValue>> oneOf;

    // A rule with no operator set is malformed: the Python producer rejects it at
    // catalog-parse time (instance_schema.normalize_attribute_constraints raises
    // InstanceError). Selection treats it as unsatisfiable so a malformed instance
    // can never be dispatched; the future catalog parser MUST also reject it.
    bool empty() const
    {
        return !equals.has_value() && !notEquals.has_value() && !oneOf.has_value();
    }
};

// selection.attribute_constraints: attribute name -> rule.
using AttributeConstraints = std::map<std::string, AttributeRule>;

// selection.batch: the inclusive [min, max] batch sizes this prebuilt instance
// serves. Unlike the exact-matched shape keys (seqlen/heads/head_size), batch is
// a runtime launch dimension, so one instance advertises a range and selection
// accepts any problem.batch within it.
struct BatchRange
{
    std::int64_t min = 1;
    std::int64_t max = 1;
};

// The exact kernel build parameters (aot_list.json "compile_spec"). The shape
// fields are matched exactly against a runtime problem; block_size_{q,k} are
// kernel-internal tiling and are NOT part of selection (kept for completeness).
struct CompileSpec
{
    std::string dtype; // I/O element type (Q/K/V/O), e.g. "fp16". Compute/
        // accumulation is fp32 (enforced by the adapter's compute_data_type
        // gate), so it is not a separate key.
    std::string canonicalLayout; // e.g. "BSHD"
    std::int64_t seqlenQ = 0;
    std::int64_t seqlenK = 0;
    std::int64_t numQueryHeads = 0;
    std::int64_t numKvHeads = 0;
    std::int64_t headSize = 0;
    std::int64_t blockSizeQ = 0; // kernel tiling; unused for selection
    std::int64_t blockSizeK = 0; // kernel tiling; unused for selection
    std::string maskMode; // e.g. "none"
};

// One checked-in, AOT-built kernel instance (one element of a per-arch
// aot_list.json array in PR #8866).
//
// Kept a cheap-to-copy value type: selection returns the winner by value, so
// avoid move-only or expensive-to-copy members (runtime state belongs on the
// execution-time plan, not here).
//
// NOTE: the instance list does NOT carry the sidecar `cache_key` (the kernel_id).
// Until the sidecar index lands (kpack fast-follow), `name` is the unique, stable
// selection handle. The kernel_id + launch metadata are resolved from the sidecar
// at plan-construction time, which is out of scope for this ticket.
struct AotInstance
{
    std::string name; // unique within a catalog
    std::string op; // "sdpa_fwd"
    std::string arch; // "gfx942"
    CompileSpec compileSpec;
    BatchRange batch;
    AttributeConstraints attributeConstraints;
};

} // namespace rocke_client::dispatcher
