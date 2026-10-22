// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/heuristics/uhd/Expressions.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>

namespace hipdnn_plugin_sdk::uhd
{

/// Variables for feature extraction. Kernel metadata lives under `kernel.`; problem
/// bindings use the engine's published names unprefixed.
class FeatureExtractionContext
{
public:
    using ValueMap = std::unordered_map<std::string, VariableContext::ValueType>;

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
        if(name.empty() || name == "$")
        {
            throw JsonLogicError("Empty published symbol name");
        }
        _ctx.bind(name, std::move(value));
    }
    const VariableContext& getContext() const
    {
        return _ctx;
    }

    /// @brief Export published feature names, without the expression reference prefix.
    nlohmann::json toJson() const
    {
        auto result = nlohmann::json::object();
        for(const auto& binding : _ctx.bindings())
        {
            const auto& name = binding.first;
            std::visit([&](const auto& held) { result[name] = held; }, binding.second);
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

/// Compiles a `features_signature` once and evaluates it per selection.
/// prepare() evaluates non-kernel entries once; extractKernelInto() re-evaluates only
/// `$kernel` entries per candidate (RFC 0019 §9.3). The Workspace is per selection, so one
/// extractor may be used concurrently.
class FeatureExtractor
{
public:
    struct Workspace
    {
        std::vector<double> values;
        const FeatureExtractor* owner = nullptr;
    };

    explicit FeatureExtractor(const std::vector<nlohmann::json>& signature,
                              const CategoricalEncoding& encoding = {})
        : _signatureHash(computeHash(signature, encoding))
        , _expressions(signature)
    {
        _vocabularies.resize(_expressions.size());
        for(size_t i = 0; i < _expressions.size(); ++i)
        {
            (_expressions.referencesKernel(i) ? _kernelIndices : _sharedIndices).push_back(i);
            const auto& reference = _expressions.reference(i);
            if(const auto found = encoding.find(reference);
               !reference.empty() && found != encoding.end())
            {
                _vocabularies[i] = found->second;
            }
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
        Workspace work{std::vector<double>(featureCount(), 0.0), this};
        for(const auto i : _sharedIndices)
        {
            work.values[i] = evaluate(i, ctx.getContext());
        }
        return work;
    }

    void extractKernelInto(const FeatureExtractionContext& ctx, Workspace& work) const
    {
        if(work.owner != this || work.values.size() != featureCount())
        {
            throw JsonLogicError("Feature workspace belongs to a different signature");
        }
        for(const auto i : _kernelIndices)
        {
            work.values[i] = evaluate(i, ctx.getContext());
        }
    }

    size_t featureCount() const
    {
        return _expressions.size();
    }
    size_t kernelDependentCount() const
    {
        return _kernelIndices.size();
    }
    const std::unordered_set<std::string>& getVariableRefs() const
    {
        return _expressions.variables();
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
    /// The `$kernel.*` fields the signature reads that @p fields does not declare, each once.
    /// `$kernel.tile[0]` is covered by `tile` or by `tile[0]`.
    std::vector<std::string>
        getMissingKmdFields(const std::unordered_set<std::string>& fields) const
    {
        constexpr std::string_view PREFIX = "$kernel.";
        std::vector<std::string> missing;
        for(const auto& reference : getVariableRefs())
        {
            auto field = kernelFieldOf(reference);
            if(!field || fields.count(*field) != 0
               || fields.count(reference.substr(PREFIX.size())) != 0)
            {
                continue;
            }
            if(std::find(missing.begin(), missing.end(), *field) == missing.end())
            {
                missing.push_back(std::move(*field));
            }
        }
        return missing;
    }

    /// The KMD field a `$kernel.*` reference reads (index stripped: `tile[0]` -> `tile`),
    /// or nullopt for any other reference. `$kernel.priority` is the kernel's declared
    /// priority, bound for every candidate, not a KMD field or knob (RFC 0019 §6.1).
    static std::optional<std::string> kernelFieldOf(const std::string& reference)
    {
        constexpr std::string_view PREFIX = "$kernel.";
        if(reference.rfind(PREFIX, 0) != 0 || reference == "$kernel.priority")
        {
            return std::nullopt;
        }
        const auto field = std::string_view(reference).substr(PREFIX.size());
        return std::string(field.substr(0, field.find('[')));
    }

    /// Hash of the compact sorted-key JSON signature plus the sorted categorical vocabulary.
    /// The format must stay stable: deployed models record this hash.
    static std::string computeHash(const std::vector<nlohmann::json>& signature,
                                   const CategoricalEncoding& encoding = {})
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
    /// Feature @p i as a number. Encoded categorical strings read as their code; any other
    /// string or unresolved value throws (the row is unscorable; no value is substituted).
    double evaluate(size_t i, const VariableContext& ctx) const
    {
        const auto& reference = _expressions.reference(i);
        if(reference.empty())
        {
            return ExpressionSet::number(_expressions.resolve(i, ctx));
        }

        const auto* bound = ctx.find(reference);
        if(bound == nullptr)
        {
            throw JsonLogicError("Undefined variable: " + reference);
        }
        const double result = std::visit(
            [&](const auto& held) -> double {
                using T = std::decay_t<decltype(held)>;
                if constexpr(std::is_same_v<T, std::string>)
                {
                    const auto& vocabulary = _vocabularies[i];
                    if(vocabulary.empty())
                    {
                        throw JsonLogicError("Type error: string used where a number is required");
                    }
                    const auto code = vocabulary.find(held);
                    if(code == vocabulary.end())
                    {
                        throw JsonLogicError(
                            "Categorical value has no code in categorical_encoding");
                    }
                    return static_cast<double>(code->second);
                }
                else
                {
                    return static_cast<double>(held);
                }
            },
            *bound);
        if(!std::isfinite(result))
        {
            throw JsonLogicError("Non-finite expression value");
        }
        return result;
    }

    /// @p node has passed ExpressionSet::checkBounds, which bounds this recursion.
    static void validateLiterals(const nlohmann::json& node)
    {
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
                validateLiterals(child);
            }
        }
    }

    static void validateSignature(const std::vector<nlohmann::json>& signature)
    {
        size_t visited = 0;
        for(const auto& entry : signature)
        {
            if(!entry.is_object()
               && (!entry.is_string() || entry.get_ref<const std::string&>().empty()
                   || entry.get_ref<const std::string&>().front() != '$'))
            {
                throw JsonLogicError(
                    "Feature entry must be a bare reference or an inline expression object");
            }
            ExpressionSet::checkBounds(entry, visited);
            validateLiterals(entry);
        }
    }

    std::string _signatureHash;
    ExpressionSet _expressions;
    /// Per entry, the vocabulary of a bare reference the descriptor declares categorical.
    std::vector<std::map<std::string, int32_t>> _vocabularies;
    std::vector<size_t> _sharedIndices;
    std::vector<size_t> _kernelIndices;
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
