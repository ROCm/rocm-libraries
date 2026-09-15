// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/heuristics/uhd/JsonLogicEvaluator.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>

namespace hipdnn_plugin_sdk::uhd
{

/// Device and kernel metadata have reserved namespaces. Problem bindings already
/// carry the names published by the engine; no synthetic query namespace is added.
class FeatureExtractionContext
{
public:
    using ValueMap = std::unordered_map<std::string, VariableContext::ValueType>;

    void bindDeviceVars(const ValueMap& props)
    {
        _ctx.bindNamespace("device", props);
    }
    void bindKernelVars(const ValueMap& props)
    {
        _ctx.bindNamespace("kernel", props);
    }
    void clearKernelVars()
    {
        _ctx.clearNamespace("kernel");
    }
    void bindQueryVars(const ValueMap& props)
    {
        for(const auto& [name, value] : props)
        {
            bind(name, value);
        }
    }
    void bind(const std::string& name, VariableContext::ValueType value)
    {
        if(name.empty())
        {
            throw JsonLogicError("Empty published symbol name");
        }
        _ctx.bind(name.front() == '$' ? name : "$" + name, std::move(value));
    }
    const VariableContext& getContext() const
    {
        return _ctx;
    }

    /// @brief Export published feature names, without the expression reference prefix.
    nlohmann::json toJson() const
    {
        auto result = nlohmann::json::object();
        for(const auto& [name, value] : _ctx.bindings())
        {
            const auto key = !name.empty() && name.front() == '$' ? name.substr(1) : name;
            std::visit([&](const auto& held) { result[key] = held; }, value);
        }
        return result;
    }
    void clear()
    {
        _ctx.clear();
    }
    bool hasAllVars(const std::unordered_set<std::string>& required) const
    {
        return std::all_of(
            required.begin(), required.end(), [&](const auto& name) { return _ctx.has(name); });
    }
    std::vector<std::string> getMissingVars(const std::unordered_set<std::string>& required) const
    {
        std::vector<std::string> missing;
        for(const auto& name : required)
        {
            if(!_ctx.has(name))
            {
                missing.push_back(name);
            }
        }
        return missing;
    }

private:
    VariableContext _ctx;
};

/// Compiles the complete inline signature into one shared descriptor DAG. The
/// workspace belongs to a selection, not the extractor, so cached engines are safe
/// to use concurrently. All output rows and node caches are allocated once per
/// selection, never once per candidate.
class FeatureExtractor
{
public:
    struct Workspace
    {
        expression::Program::Workspace expressions;
        std::vector<double> values;
    };

    explicit FeatureExtractor(const std::vector<nlohmann::json>& signature,
                              const expression::CategoricalEncoding& encoding = {})
        : _program(signature, encoding)
        , _signatureHash(computeHash(signature, encoding))
    {
        for(size_t i = 0; i < _program.size(); ++i)
        {
            (_program.kernelDependent(i) ? _kernelIndices : _sharedIndices).push_back(i);
        }
    }

    std::vector<double> extract(const FeatureExtractionContext& ctx) const
    {
        auto work = prepare(ctx);
        extractKernelInto(ctx, work);
        return std::move(work.values);
    }

    Workspace prepare(const FeatureExtractionContext& ctx) const
    {
        Workspace work{_program.workspace(), std::vector<double>(featureCount(), 0.0)};
        _program.prepare(ctx.getContext(), work.expressions);
        for(const auto i : _sharedIndices)
        {
            work.values[i] = expression::Program::number(
                _program.evaluate(i, ctx.getContext(), work.expressions));
        }
        return work;
    }

    void extractKernelInto(const FeatureExtractionContext& ctx, Workspace& work) const
    {
        if(work.values.size() != featureCount())
        {
            throw JsonLogicError("Feature workspace width mismatch");
        }
        _program.resetCandidate(work.expressions);
        for(const auto i : _kernelIndices)
        {
            work.values[i] = expression::Program::number(
                _program.evaluate(i, ctx.getContext(), work.expressions));
        }
    }

    size_t featureCount() const
    {
        return _program.size();
    }
    size_t kernelDependentCount() const
    {
        return _kernelIndices.size();
    }
    size_t compiledNodeCount() const
    {
        return _program.nodeCount();
    }
    size_t sharedNodeCount() const
    {
        return _program.sharedNodeCount();
    }
    size_t candidateNodeCount() const
    {
        return _program.candidateNodeCount();
    }
    const std::unordered_set<std::string>& getVariableRefs() const
    {
        return _program.variables();
    }
    const std::string& getSignatureHash() const
    {
        return _signatureHash;
    }
    bool validateContext(const FeatureExtractionContext& ctx) const
    {
        return ctx.hasAllVars(getVariableRefs());
    }
    std::vector<std::string> getMissingVariables(const FeatureExtractionContext& ctx) const
    {
        return ctx.getMissingVars(getVariableRefs());
    }
    bool validateAgainstKmdFields(const std::unordered_set<std::string>& fields) const
    {
        return getMissingKmdFields(fields).empty();
    }
    std::vector<std::string>
        getMissingKmdFields(const std::unordered_set<std::string>& fields) const
    {
        std::vector<std::string> missing;
        for(const auto& reference : getVariableRefs())
        {
            constexpr std::string_view PREFIX = "$kernel.";
            if(reference.rfind(PREFIX, 0) == 0
               && fields.count(reference.substr(PREFIX.size())) == 0)
            {
                missing.push_back(reference.substr(PREFIX.size()));
            }
        }
        return missing;
    }

    /// Compact, sorted-key JSON AST plus the optional sorted categorical vocabulary.
    /// This preserves hashes of existing canonical raw-reference signatures.
    static std::string computeHash(const std::vector<nlohmann::json>& signature,
                                   const expression::CategoricalEncoding& encoding = {})
    {
        validateSignature(signature);
        try
        {
            std::string serialized = nlohmann::json(signature).dump();
            if(!encoding.empty())
            {
                nlohmann::json canonicalEncoding = nlohmann::json::object();
                for(const auto& [field, codes] : encoding)
                {
                    if(field.empty() || field.front() != '$' || codes.empty())
                    {
                        throw JsonLogicError("categorical_encoding requires full references and "
                                             "nonempty vocabularies");
                    }
                    canonicalEncoding[field] = codes;
                }
                serialized += "|" + canonicalEncoding.dump();
            }
            return "sha256:" + sha256(serialized).substr(0, 16);
        }
        catch(const nlohmann::json::exception& error)
        {
            throw JsonLogicError("features_signature cannot be serialized: "
                                 + std::string(error.what()));
        }
    }

private:
    static void validateLiterals(const nlohmann::json& node, size_t depth, size_t& visited)
    {
        if(depth > 2 * expression::Program::MAX_EXPRESSION_DEPTH + 2
           || ++visited > expression::Program::MAX_INPUT_NODES)
        {
            throw JsonLogicError("features_signature exceeds depth or size bound");
        }
        if(node.is_number())
        {
            const double value = node.get<double>();
            if(!std::isfinite(value) || std::abs(value) >= 1e15)
            {
                throw JsonLogicError(
                    "features_signature numeric literal must be finite with magnitude below 1e15");
            }
        }
        else if(node.is_array() || node.is_object())
        {
            for(const auto& child : node)
            {
                validateLiterals(child, depth + 1, visited);
            }
        }
    }

    static const std::vector<nlohmann::json>&
        validateSignature(const std::vector<nlohmann::json>& signature)
    {
        size_t visited = 0;
        for(const auto& entry : signature)
        {
            if(!entry.is_object()
               && !(entry.is_string() && !entry.get_ref<const std::string&>().empty()
                    && entry.get_ref<const std::string&>().front() == '$'))
            {
                throw JsonLogicError(
                    "Feature entry must be a bare reference or an inline expression object");
            }
            validateLiterals(entry, 0, visited);
        }
        return signature;
    }

    expression::Program _program;
    std::vector<size_t> _sharedIndices;
    std::vector<size_t> _kernelIndices;
    std::string _signatureHash;
};

} // namespace hipdnn_plugin_sdk::uhd
