// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FeatureExtractor.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

// For SHA-256 we use a minimal implementation or rely on OpenSSL if available.
// For now, use a simple hash that can be replaced with proper SHA-256 later.
#include <functional>

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

FeatureExtractor::FeatureExtractor(const std::vector<std::string>& signature)
{
    _parsedExprs.reserve(signature.size());

    for(const auto& exprStr : signature)
    {
        auto parsed = JsonLogicEvaluator::parse(exprStr);
        _parsedExprs.push_back(std::move(parsed));

        // Extract variable references from this expression
        auto vars = JsonLogicEvaluator::extractVariables(_parsedExprs.back());
        _varRefs.insert(vars.begin(), vars.end());
    }

    _signatureHash = computeHash(signature);
}

std::vector<double> FeatureExtractor::extract(const FeatureExtractionContext& ctx) const
{
    std::vector<double> features;
    features.reserve(_parsedExprs.size());

    for(const auto& expr : _parsedExprs)
    {
        features.push_back(_evaluator.evaluateDouble(expr, ctx.getContext()));
    }

    return features;
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
    // Concatenate all signature strings with a separator
    std::ostringstream oss;
    for(size_t i = 0; i < signature.size(); ++i)
    {
        if(i > 0)
        {
            oss << '\0';
        }
        oss << signature[i];
    }
    const std::string combined = oss.str();

    // Use std::hash for now (replace with proper SHA-256 in production)
    // This is a placeholder - the actual implementation should use OpenSSL
    // or a header-only SHA-256 implementation.
    const std::size_t h = std::hash<std::string>{}(combined);

    // Format as hex string
    std::ostringstream hexStream;
    hexStream << std::hex << std::setfill('0') << std::setw(16) << h;

    // Pad to 64 chars (SHA-256 length) for interface consistency
    std::string result = hexStream.str();
    while(result.length() < 64)
    {
        result.insert(0, 1, '0');
    }

    return result;
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

std::vector<std::string>
    FeatureExtractor::getMissingKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const
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

} // namespace hipdnn_backend::heuristics::uhd
