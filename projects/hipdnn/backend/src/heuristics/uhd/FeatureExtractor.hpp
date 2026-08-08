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

    /// Bind problem/query properties ($q.*).
    void bindQueryVars(const ValueMap& queryProps);

    /// Bind a single variable by full name.
    void bind(const std::string& fullName, VariableContext::ValueType value);

    /// Get the underlying VariableContext for evaluation.
    const VariableContext& getContext() const { return _ctx; }

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
/// expressions) using a FeatureExtractionContext. Also computes signature hashes
/// for contract validation.
class FeatureExtractor
{
public:
    /// Construct with a features signature (list of JsonLogic expression strings).
    explicit FeatureExtractor(const std::vector<std::string>& signature);

    /// Extract feature vector from context.
    /// @returns Ordered vector of feature values matching the signature.
    /// @throws JsonLogicError if any expression fails to evaluate.
    std::vector<double> extract(const FeatureExtractionContext& ctx) const;

    /// Get the number of features in the signature.
    size_t featureCount() const { return _parsedExprs.size(); }

    /// Get all variable references in the signature.
    const std::unordered_set<std::string>& getVariableRefs() const { return _varRefs; }

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
    const std::string& getSignatureHash() const { return _signatureHash; }

    /// Validate that a set of KMD field names covers all $kernel.* references.
    /// Returns true if every $kernel.<field> in the signature has a matching field name.
    bool validateAgainstKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const;

    /// Get $kernel.* field names referenced but not in KMD.
    std::vector<std::string>
        getMissingKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const;

private:
    std::vector<nlohmann::json> _parsedExprs;
    std::unordered_set<std::string> _varRefs;
    std::string _signatureHash;
    JsonLogicEvaluator _evaluator;
};

} // namespace hipdnn_backend::heuristics::uhd
