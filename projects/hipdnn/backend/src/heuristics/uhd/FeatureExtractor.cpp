// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FeatureExtractor.hpp"
#include "Sha256.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace
{
// SHA-256 implementation now lives in Sha256.cpp
} // anonymous namespace

namespace hipdnn_backend::heuristics::uhd
{

// ============================================================================
// FeatureExtractionContext
// ============================================================================

void FeatureExtractionContext::bindDeviceVars(const ValueMap& props)
{
    _ctx.bindNamespace("device", props);
}

void FeatureExtractionContext::bindKernelVars(const ValueMap& metadata)
{
    _ctx.bindNamespace("kernel", metadata);
}

void FeatureExtractionContext::clearKernelVars()
{
    _ctx.clearNamespace("kernel");
}

void FeatureExtractionContext::bindQueryVars(const ValueMap& queryProps)
{
    _ctx.bindNamespace("q", queryProps);
}

void FeatureExtractionContext::bind(const std::string& fullName, VariableContext::ValueType value)
{
    _ctx.bind(fullName, std::move(value));
}

void FeatureExtractionContext::clear()
{
    _ctx.clear();
}

bool FeatureExtractionContext::hasAllVars(const std::unordered_set<std::string>& required) const
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

std::vector<std::string>
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

namespace
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
constexpr double MAX_SAFE_NUMERIC_LITERAL = 1e15;

void validateNumericLiterals(const nlohmann::json& node)
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

} // namespace

nlohmann::json FeatureExtractor::parseSignatureEntry(const std::string& entry)
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

FeatureExtractor::FeatureExtractor(const std::vector<std::string>& signature,
                                     const std::vector<std::pair<std::string, std::string>>& derived)
{
    // Parse and analyze derived values (RFC 0019 §6.4)
    _parsedDerived.reserve(derived.size());
    const std::string kernelPrefix = "$kernel.";

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
        const bool kernelDependent
            = std::any_of(vars.begin(), vars.end(), [&kernelPrefix](const std::string& v) {
                  return v == "$kernel" || v.rfind(kernelPrefix, 0) == 0;
              });

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

std::vector<double> FeatureExtractor::extract(const FeatureExtractionContext& ctx) const
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

std::vector<double> FeatureExtractor::extractSharedRow(const FeatureExtractionContext& ctx) const
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

void FeatureExtractor::extractKernelInto(const FeatureExtractionContext& ctx,
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

bool FeatureExtractor::validateContext(const FeatureExtractionContext& ctx) const
{
    return ctx.hasAllVars(_varRefs);
}

std::vector<std::string>
    FeatureExtractor::getMissingVariables(const FeatureExtractionContext& ctx) const
{
    return ctx.getMissingVars(_varRefs);
}

std::string FeatureExtractor::computeHash(const std::vector<std::string>& signature)
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
    validateNumericLiterals(canonical);

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

    const std::string fullHash = sha256(serialized);

    // Return first 16 chars with sha256: prefix to match Python format
    return "sha256:" + fullHash.substr(0, 16);
}

bool FeatureExtractor::validateAgainstKmdFields(
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

std::vector<std::string> FeatureExtractor::getMissingKmdFields(
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

void FeatureExtractor::evaluateDerived(FeatureExtractionContext& ctx) const
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

} // namespace hipdnn_backend::heuristics::uhd
