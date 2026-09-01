// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/JsonLogicEvaluator.hpp>

#include <cstdint>
#include <optional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/Sha256.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace hipdnn_plugin_sdk::ingestor::uhd
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
    static std::string computeHash(
        const std::vector<std::string>& signature,
        const std::map<std::string, std::map<std::string, int32_t>>& categoricalEncoding = {});

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


// ============================================================================
// FeatureExtractionContext
// ============================================================================

inline void FeatureExtractionContext::bindDeviceVars(const ValueMap& props)
{
    _ctx.bindNamespace("device", props);
}

inline void FeatureExtractionContext::bindKernelVars(const ValueMap& metadata)
{
    _ctx.bindNamespace("kernel", metadata);
}

inline void FeatureExtractionContext::clearKernelVars()
{
    _ctx.clearNamespace("kernel");
}

inline void FeatureExtractionContext::bindQueryVars(const ValueMap& queryProps)
{
    _ctx.bindNamespace("q", queryProps);
}

inline void FeatureExtractionContext::bind(const std::string& fullName, VariableContext::ValueType value)
{
    _ctx.bind(fullName, std::move(value));
}

inline void FeatureExtractionContext::clear()
{
    _ctx.clear();
}

inline bool FeatureExtractionContext::hasAllVars(const std::unordered_set<std::string>& required) const
{
    for(const auto& var : required)
    {
        if(!_ctx.has(var))
        {
            return false;
        }
    }
    return true;
}

inline std::vector<std::string>
    FeatureExtractionContext::getMissingVars(const std::unordered_set<std::string>& required) const
{
    std::vector<std::string> missing;
    for(const auto& var : required)
    {
        if(!_ctx.has(var))
        {
            missing.push_back(var);
        }
    }
    return missing;
}

// ============================================================================
// FeatureExtractor
// ============================================================================

namespace detail
{

/// Largest numeric literal magnitude the cross-language canonical form is safe for.
///
/// Above this the two JSON writers stop agreeing, in three independent ways:
///   - nlohmann switches a double to scientific notation at 1e15, Python's repr at
///     1e16, so the whole decade renders differently ("1e+15" vs "1000000000000000.0");
///   - an integer outside int64/uint64 degrades to double in nlohmann while Python
///     keeps arbitrary precision -- and the lossy conversion makes distinct values
///     collide, so C++ would accept a hash computed over a *different* signature;
///   - non-finite values are a Python json extension that nlohmann rejects outright.
///
/// Feature literals are tile sizes, dimensions and thresholds, so this bound is many
/// orders of magnitude clear of anything real. Rejecting is the honest option: a
/// signature we cannot canonicalize identically on both sides has no usable hash.
inline constexpr double MAX_SAFE_NUMERIC_LITERAL = 1e15;

inline void validateNumericLiterals(const nlohmann::json& node)
{
    if(node.is_number())
    {
        const double value = node.get<double>();
        if(!std::isfinite(value))
        {
            throw JsonLogicError("features_signature contains a non-finite numeric literal");
        }
        if(std::abs(value) >= MAX_SAFE_NUMERIC_LITERAL)
        {
            throw JsonLogicError(
                "features_signature contains the numeric literal " + node.dump()
                + ", whose magnitude is at or above 1e15. The generator and the runtime "
                  "render such values differently, so the features_hash would not match. "
                  "Rescale the feature instead.");
        }
        return;
    }

    if(node.is_array())
    {
        for(const auto& element : node)
        {
            validateNumericLiterals(element);
        }
        return;
    }

    if(node.is_object())
    {
        for(const auto& item : node.items())
        {
            validateNumericLiterals(item.value());
        }
    }
}

} // namespace detail

inline nlohmann::json FeatureExtractor::parseSignatureEntry(const std::string& entry)
{
    // A bare reference such as `$q.seqlen_q` is the RFC 0019 §7.2 canonical spelling
    // and what tools/uhd_gen emits, but it is not valid JSON. Lift it to a JSON string
    // so it reaches the evaluator's variable branch.
    if(!entry.empty() && entry.front() == '$')
    {
        // Not a braced init list: `nlohmann::json{entry}` builds a one-element
        // *array*, not a string, which would reach the evaluator as the wrong node
        // type. The explicit constructor call is required for correctness here.
        // NOLINTNEXTLINE(modernize-return-braced-init-list)
        return nlohmann::json(entry);
    }

    return JsonLogicEvaluator::parse(entry);
}

inline FeatureExtractor::FeatureExtractor(const std::vector<std::string>& signature,
                                     const std::vector<std::pair<std::string, std::string>>& derived)
{
    // Parse and analyze derived values (RFC 0019 §6.4)
    _parsedDerived.reserve(derived.size());
    const std::string kernelPrefix = "$kernel.";
    const std::string derivedPrefix = "$derived.";

    // First pass: identify directly kernel-dependent derived values
    for(size_t i = 0; i < derived.size(); ++i)
    {
        const auto& [name, exprStr] = derived[i];
        auto parsed = parseSignatureEntry(exprStr);
        _parsedDerived.emplace_back(name, std::move(parsed));

        // Check if this derived value depends on $kernel.*
        auto vars = JsonLogicEvaluator::extractVariables(_parsedDerived.back().second);
        const bool kernelDependent
            = std::any_of(vars.begin(), vars.end(), [&kernelPrefix](const std::string& v) {
                  return v == "$kernel" || v.rfind(kernelPrefix, 0) == 0;
              });

        if(kernelDependent)
        {
            _kernelDependentDerivedIndices.insert(i);
        }
    }

    // Second pass: propagate kernel-dependency transitively through $derived.* references
    // A derived value is kernel-dependent if it references another kernel-dependent derived.
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(size_t i = 0; i < _parsedDerived.size(); ++i)
        {
            if(_kernelDependentDerivedIndices.count(i) > 0)
            {
                continue; // Already marked
            }

            auto vars = JsonLogicEvaluator::extractVariables(_parsedDerived[i].second);
            for(const auto& var : vars)
            {
                if(var.rfind(derivedPrefix, 0) == 0)
                {
                    const std::string refName = var.substr(derivedPrefix.length());
                    // Find the referenced derived value
                    for(size_t j = 0; j < _parsedDerived.size(); ++j)
                    {
                        if(_parsedDerived[j].first == refName
                           && _kernelDependentDerivedIndices.count(j) > 0)
                        {
                            _kernelDependentDerivedIndices.insert(i);
                            changed = true;
                            break;
                        }
                    }
                    if(changed)
                    {
                        break;
                    }
                }
            }
            if(changed)
            {
                break;
            }
        }
    }

    _parsedExprs.reserve(signature.size());

    for(const auto& exprStr : signature)
    {
        auto parsed = parseSignatureEntry(exprStr);
        _parsedExprs.push_back(std::move(parsed));

        // Extract variable references from this expression
        auto vars = JsonLogicEvaluator::extractVariables(_parsedExprs.back());
        _varRefs.insert(vars.begin(), vars.end());

        // Partition by whether this entry varies per candidate (RFC 0019 §6 step 2).
        //
        // The bare "$kernel" case matters: extractVariables reports the *syntactic*
        // reference, but the shape/rank operators resolve a *synthesized* name --
        // {"shape": ["$kernel", 0]} reads $kernel.shape_0. Testing only for the
        // "$kernel." prefix would file that entry as shared, so it would be evaluated
        // in the shared pass before any kernel metadata is bound.
        //
        // Additionally, check if this references any kernel-dependent derived values.
        // A derived value is kernel-dependent if it transitively depends on $kernel.*.
        bool kernelDependent
            = std::any_of(vars.begin(), vars.end(), [&kernelPrefix](const std::string& v) {
                  return v == "$kernel" || v.rfind(kernelPrefix, 0) == 0;
              });

        // Check for derived value references (RFC 0019 §6.4)
        if(!kernelDependent)
        {
            for(const auto& var : vars)
            {
                if(var.rfind(derivedPrefix, 0) == 0)
                {
                    // Extract derived value name
                    const std::string derivedName = var.substr(derivedPrefix.length());
                    // Check if this derived value is kernel-dependent
                    for(size_t i = 0; i < _parsedDerived.size(); ++i)
                    {
                        if(_parsedDerived[i].first == derivedName
                           && _kernelDependentDerivedIndices.count(i) > 0)
                        {
                            kernelDependent = true;
                            break;
                        }
                    }
                    if(kernelDependent)
                    {
                        break;
                    }
                }
            }
        }

        const size_t index = _parsedExprs.size() - 1;
        if(kernelDependent)
        {
            _kernelIndices.push_back(index);
        }
        else
        {
            _sharedIndices.push_back(index);
        }
    }

    _signatureHash = computeHash(signature);
}

inline std::vector<double> FeatureExtractor::extract(const FeatureExtractionContext& ctx) const
{
    // Evaluate and bind derived values (RFC 0019 §6.4)
    // Note: This requires a mutable context - derived values are lazily computed
    auto& mutableCtx = const_cast<FeatureExtractionContext&>(ctx);
    evaluateDerived(mutableCtx);

    std::vector<double> features;
    features.reserve(_parsedExprs.size());

    for(const auto& expr : _parsedExprs)
    {
        features.push_back(_evaluator.evaluateDouble(expr, ctx.getContext()));
    }

    return features;
}

inline std::vector<double> FeatureExtractor::extractSharedRow(const FeatureExtractionContext& ctx) const
{
    // Evaluate and bind kernel-independent derived values (RFC 0019 §6.4)
    auto& mutableCtx = const_cast<FeatureExtractionContext&>(ctx);
    for(size_t i = 0; i < _parsedDerived.size(); ++i)
    {
        // Skip kernel-dependent derived values in the shared pass
        if(_kernelDependentDerivedIndices.count(i) > 0)
        {
            continue;
        }
        const auto& [name, expr] = _parsedDerived[i];
        const double value = _evaluator.evaluateDouble(expr, ctx.getContext());
        mutableCtx.bind("$derived." + name, value);
    }

    std::vector<double> row(_parsedExprs.size(), 0.0);

    for(const size_t i : _sharedIndices)
    {
        row[i] = _evaluator.evaluateDouble(_parsedExprs[i], ctx.getContext());
    }

    return row;
}

inline void FeatureExtractor::extractKernelInto(const FeatureExtractionContext& ctx,
                                         std::vector<double>& row) const
{
    if(row.size() != _parsedExprs.size())
    {
        throw JsonLogicError("extractKernelInto: row width " + std::to_string(row.size())
                             + " does not match signature width "
                             + std::to_string(_parsedExprs.size()));
    }

    // Evaluate kernel-dependent derived values (RFC 0019 §6.4)
    auto& mutableCtx = const_cast<FeatureExtractionContext&>(ctx);
    for(size_t i = 0; i < _parsedDerived.size(); ++i)
    {
        // Only evaluate kernel-dependent derived values here
        if(_kernelDependentDerivedIndices.count(i) == 0)
        {
            continue;
        }
        const auto& [name, expr] = _parsedDerived[i];
        const double value = _evaluator.evaluateDouble(expr, ctx.getContext());
        mutableCtx.bind("$derived." + name, value);
    }

    for(const size_t i : _kernelIndices)
    {
        row[i] = _evaluator.evaluateDouble(_parsedExprs[i], ctx.getContext());
    }
}

inline bool FeatureExtractor::validateContext(const FeatureExtractionContext& ctx) const
{
    return ctx.hasAllVars(_varRefs);
}

inline std::vector<std::string>
    FeatureExtractor::getMissingVariables(const FeatureExtractionContext& ctx) const
{
    return ctx.getMissingVars(_varRefs);
}

inline std::string FeatureExtractor::computeHash(
    const std::vector<std::string>& signature,
    const std::map<std::string, std::map<std::string, int32_t>>& categoricalEncoding)
{
    // Canonical form is the parsed signature dumped as compact JSON, matching Python's
    // json.dumps(signature, separators=(",", ":")) in tools/uhd_gen. Parsing first means
    // the bare and pre-quoted spellings of a reference collapse to the same node, and
    // nlohmann handles escaping. Order is preserved deliberately: RFC 0019 §7.2 requires
    // the signature to match training exactly, so a permuted signature must hash
    // differently.
    nlohmann::json canonical = nlohmann::json::array();
    for(const auto& entry : signature)
    {
        canonical.push_back(parseSignatureEntry(entry));
    }

    // Reject literals the two languages would render differently before hashing them.
    detail::validateNumericLiterals(canonical);

    std::string serialized;
    try
    {
        serialized = canonical.dump();
    }
    catch(const nlohmann::json::exception& e)
    {
        // dump() rejects invalid UTF-8 (type_error.316). That path never went through
        // parse() -- a bare "$ref" is lifted straight to a JSON string -- so it has to
        // be converted here or it escapes as a raw nlohmann type.
        throw JsonLogicError("features_signature cannot be serialized: " + std::string(e.what()));
    }

    // RFC 0019 §6.3: the hash fingerprints the *resolved* feature contract -- the
    // canonicalized signature and the categorical encoding -- so editing the encoding
    // invalidates the contract instead of passing silently. §6.5 makes the point directly:
    // "features_hash does not catch it because the signature text is unchanged."
    //
    // Appended only when there is an encoding, so a UHD that reads no string field hashes
    // exactly as it did before this field existed. Every shipped model is such a UHD, and
    // changing their hashes would invalidate contracts that are in fact intact.
    if(!categoricalEncoding.empty())
    {
        nlohmann::json encoding = nlohmann::json::object();
        for(const auto& [field, codes] : categoricalEncoding)
        {
            for(const auto& [value, code] : codes)
            {
                encoding[field][value] = code;
            }
        }
        // std::map is key-sorted and nlohmann's default object is too, so both sides render
        // the same bytes without an explicit sort.
        serialized += "|" + encoding.dump();
    }

    const std::string fullHash = sha256(serialized);

    // Return first 16 chars with sha256: prefix to match Python format
    return "sha256:" + fullHash.substr(0, 16);
}

inline bool FeatureExtractor::validateAgainstKmdFields(
    const std::unordered_set<std::string>& kmdFieldNames) const
{
    const std::string kernelPrefix = "$kernel.";

    for(const auto& var : _varRefs)
    {
        if(var.rfind(kernelPrefix, 0) == 0)
        {
            // Extract field name after "$kernel."
            const std::string fieldName = var.substr(kernelPrefix.length());
            if(kmdFieldNames.find(fieldName) == kmdFieldNames.end())
            {
                return false;
            }
        }
    }
    return true;
}

inline std::vector<std::string> FeatureExtractor::getMissingKmdFields(
    const std::unordered_set<std::string>& kmdFieldNames) const
{
    const std::string kernelPrefix = "$kernel.";
    std::vector<std::string> missing;

    for(const auto& var : _varRefs)
    {
        if(var.rfind(kernelPrefix, 0) == 0)
        {
            const std::string fieldName = var.substr(kernelPrefix.length());
            if(kmdFieldNames.find(fieldName) == kmdFieldNames.end())
            {
                missing.push_back(fieldName);
            }
        }
    }
    return missing;
}

inline void FeatureExtractor::evaluateDerived(FeatureExtractionContext& ctx) const
{
    // RFC 0019 §6.4: Evaluate derived values in order and bind to $derived.* namespace.
    // Each expression can reference $device.*, $kernel.*, $q.*, and earlier $derived.* entries.
    // NOLINTNEXTLINE(modernize-loop-convert) - index reserved for future error reporting
    for(size_t i = 0; i < _parsedDerived.size(); ++i)
    {
        const auto& [name, expr] = _parsedDerived[i];
        try
        {
            const double value = _evaluator.evaluateDouble(expr, ctx.getContext());
            ctx.bind("$derived." + name, value);
        }
        catch(const JsonLogicError& e)
        {
            std::ostringstream oss;
            oss << "Failed to evaluate derived value '$derived." << name
                << "': " << e.what();
            throw JsonLogicError(oss.str());
        }
    }
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
