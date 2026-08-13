// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "JsonLogicEvaluator.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Context for feature extraction containing all bound variables.
///
/// Provides bindings for:
/// - $device.*: Device properties (cu_count, arch, total_global_mem, etc.)
/// - $kernel.*: Kernel metadata from KMD fields (tile_m, split_k, dtype, etc.)
/// - $q.*: Problem/query properties from graph match (batch, seqlen, heads, etc.)
class FeatureExtractionContext
{
public:
    using ValueMap = std::unordered_map<std::string, VariableContext::ValueType>;

    /// Bind device properties ($device.*).
    void bindDeviceVars(const ValueMap& props);

    /// Bind kernel metadata ($kernel.*).
    void bindKernelVars(const ValueMap& metadata);

    /// Drop all $kernel.* bindings, leaving $device.* and $q.* intact.
    ///
    /// Call this between candidates when reusing a context, so a candidate whose
    /// metadata omits a field does not inherit the previous candidate's value.
    void clearKernelVars();

    /// Bind problem/query properties ($q.*).
    void bindQueryVars(const ValueMap& queryProps);

    /// Bind a single variable by full name.
    void bind(const std::string& fullName, VariableContext::ValueType value);

    /// Get the underlying VariableContext for evaluation.
    const VariableContext& getContext() const
    {
        return _ctx;
    }

    /// Clear all bindings.
    void clear();

    /// Check if all required variables are bound.
    bool hasAllVars(const std::unordered_set<std::string>& required) const;

    /// Get list of missing variables from a required set.
    std::vector<std::string> getMissingVars(const std::unordered_set<std::string>& required) const;

private:
    VariableContext _ctx;
};

/// @brief Feature extractor for UHD heuristic evaluation.
///
/// Extracts feature vectors from a features_signature (ordered list of JsonLogic
/// expressions) using a FeatureExtractionContext. Supports RFC 0019 §6.4 derived
/// values which are evaluated in order and bound to $derived.* namespace before
/// evaluating the main signature. Also computes signature hashes for contract validation.
class FeatureExtractor
{
public:
    /// Construct with a features signature (list of JsonLogic expression strings).
    /// @param signature Ordered list of feature expressions.
    /// @param derived Optional derived values (RFC 0019 §6.4) as (name, expression) pairs.
    explicit FeatureExtractor(const std::vector<std::string>& signature,
                               const std::vector<std::pair<std::string, std::string>>& derived
                               = {});

    /// Extract feature vector from context.
    /// @returns Ordered vector of feature values matching the signature.
    /// @throws JsonLogicError if any expression fails to evaluate.
    std::vector<double> extract(const FeatureExtractionContext& ctx) const;

    /// Evaluate only the entries that do not reference $kernel.* (RFC 0019 §6 step 2).
    ///
    /// Problem ($q.*) and device ($device.*) features are shared across every
    /// candidate, so they are evaluated once per selection rather than once per
    /// candidate. Kernel-dependent slots are left at zero for extractKernelInto().
    ///
    /// @returns A full-width row with shared slots populated.
    /// @throws JsonLogicError if any shared expression fails to evaluate.
    std::vector<double> extractSharedRow(const FeatureExtractionContext& ctx) const;

    /// Fill the $kernel.*-dependent slots of a row from extractSharedRow().
    ///
    /// @param ctx Context with this candidate's kernel metadata bound.
    /// @param row Row to update in place; must be featureCount() wide.
    /// @throws JsonLogicError if any kernel-dependent expression fails to evaluate.
    void extractKernelInto(const FeatureExtractionContext& ctx, std::vector<double>& row) const;

    /// Get the number of features in the signature.
    size_t featureCount() const
    {
        return _parsedExprs.size();
    }

    /// Number of signature entries that reference $kernel.* (re-evaluated per candidate).
    size_t kernelDependentCount() const
    {
        return _kernelIndices.size();
    }

    /// Get all variable references in the signature.
    const std::unordered_set<std::string>& getVariableRefs() const
    {
        return _varRefs;
    }

    /// Validate that all referenced variables in the signature are present in context.
    bool validateContext(const FeatureExtractionContext& ctx) const;

    /// Get missing variables that are referenced but not bound in context.
    std::vector<std::string> getMissingVariables(const FeatureExtractionContext& ctx) const;

    /// Parse one features_signature entry.
    ///
    /// RFC 0019 §7.2 allows two forms: a bare field reference (`$q.seqlen_q`) or a
    /// derived JsonLogic expression (`{"log2": ["$q.seqlen_q"]}`). A bare reference is
    /// not valid JSON on its own, so it is lifted to a JSON string rather than parsed.
    /// Pre-quoted entries (`"\"$q.seqlen_q\""`) still parse to the same node.
    ///
    /// @throws JsonLogicError if a non-reference entry is not valid JSON.
    static nlohmann::json parseSignatureEntry(const std::string& entry);

    /// Compute SHA-256 hash of the features signature.
    /// This hash is embedded in trained models for contract validation.
    ///
    /// Entries are canonicalized through parseSignatureEntry() before hashing, so the
    /// bare and pre-quoted spellings of a reference agree. Order is significant —
    /// RFC 0019 §7.2 requires the signature to match training exactly, so a permuted
    /// signature must not produce a matching hash.
    static std::string computeHash(const std::vector<std::string>& signature);

    /// Get the hash of this extractor's signature.
    const std::string& getSignatureHash() const
    {
        return _signatureHash;
    }

    /// Validate that a set of KMD field names covers all $kernel.* references.
    /// Returns true if every $kernel.<field> in the signature has a matching field name.
    bool validateAgainstKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const;

    /// Get $kernel.* field names referenced but not in KMD.
    std::vector<std::string>
        getMissingKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const;

private:
    /// Evaluate and bind derived values to $derived.* namespace (RFC 0019 §6.4).
    /// @param ctx Context to bind derived values into (mutable because binding is lazy).
    void evaluateDerived(FeatureExtractionContext& ctx) const;

    // Derived values (RFC 0019 §6.4): ordered (name, parsed-expression) pairs
    std::vector<std::pair<std::string, nlohmann::json>> _parsedDerived;
    /// Derived value indices that depend on $kernel.* (must be re-evaluated per candidate)
    std::unordered_set<size_t> _kernelDependentDerivedIndices;

    std::vector<nlohmann::json> _parsedExprs;
    std::unordered_set<std::string> _varRefs;
    /// Signature positions with no $kernel.* reference — evaluated once per selection.
    std::vector<size_t> _sharedIndices;
    /// Signature positions referencing $kernel.* — evaluated once per candidate.
    std::vector<size_t> _kernelIndices;
    std::string _signatureHash;
    JsonLogicEvaluator _evaluator;
};

} // namespace hipdnn_backend::heuristics::uhd
