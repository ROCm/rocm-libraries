// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_backend::heuristics::config
{
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_version = hipdnn_data_sdk::detail::autotune_config::version;

/// Dimension value meaning "match any value in this slot".
inline constexpr int64_t WILDCARD_DIM = -1;

/// View into one logical tensor: its schema tensor field name plus pointers to
/// live dim and stride vectors. The matcher does not own this data; callers
/// must keep the underlying vectors alive for the duration of the match call.
struct TensorView
{
    std::string_view tensorId;
    const std::vector<int64_t>* dim;
    const std::vector<int64_t>* stride;
};

/// Pattern for a single tensor: an optional schema tensor field name, a list of
/// expected dimensions, and optional strides, with -1 as a per-slot wildcard.
/// When `stride` is empty no stride matching is performed.
struct TensorPattern
{
    std::optional<std::string> tensorId;
    std::vector<int64_t> dim;
    std::vector<int64_t> stride;
    bool matches(const TensorView& tensor) const
    {
        const auto& tdim = *tensor.dim;
        if(dim.size() != tdim.size())
        {
            return false;
        }
        for(size_t i = 0; i < dim.size(); ++i)
        {
            if(dim[i] != WILDCARD_DIM && dim[i] != tdim[i])
            {
                return false;
            }
        }
        if(!stride.empty())
        {
            const auto& tstride = *tensor.stride;
            if(stride.size() != tstride.size())
            {
                return false;
            }
            for(size_t i = 0; i < stride.size(); ++i)
            {
                if(stride[i] != WILDCARD_DIM && stride[i] != tstride[i])
                {
                    return false;
                }
            }
        }
        return true;
    }
};
struct Criterion
{
    std::string key;
    int64_t value = 0;
};

/// A single engine-override rule (one operation, one criteria set, one engine,
/// ordered tensor patterns).
struct OperationRule
{
    std::string op;
    std::string engineName;
    std::vector<Criterion> criteria;
    std::vector<TensorPattern> tensors;
    bool useNamedTensorIds = false;

    bool matches(const std::vector<Criterion>& actualCriteria,
                 const std::vector<TensorView>& inputs) const
    {
        if(criteria.size() != actualCriteria.size() || tensors.size() != inputs.size())
        {
            return false;
        }
        for(size_t i = 0; i < criteria.size(); ++i)
        {
            if(criteria[i].key != actualCriteria[i].key
               || criteria[i].value != actualCriteria[i].value)
            {
                return false;
            }
        }
        return matchesTensors(inputs);
    }

private:
    bool matchesTensors(const std::vector<TensorView>& inputs) const
    {
        if(!useNamedTensorIds)
        {
            return matchesLegacyPositional(inputs);
        }

        std::vector<uint8_t> used(inputs.size(), 0);
        for(const auto& pattern : tensors)
        {
            if(!pattern.tensorId.has_value() || !matchesNamed(pattern, inputs, used))
            {
                return false;
            }
        }
        return true;
    }

    bool matchesLegacyPositional(const std::vector<TensorView>& inputs) const
    {
        for(size_t i = 0; i < tensors.size(); ++i)
        {
            if(!tensors[i].matches(inputs[i]))
            {
                return false;
            }
        }
        return true;
    }

    static bool matchesNamed(const TensorPattern& pattern,
                             const std::vector<TensorView>& inputs,
                             std::vector<uint8_t>& used)
    {
        for(size_t i = 0; i < inputs.size(); ++i)
        {
            if(used[i] == 0 && inputs[i].tensorId == *pattern.tensorId
               && pattern.matches(inputs[i]))
            {
                used[i] = 1;
                return true;
            }
        }
        return false;
    }
};

namespace detail
{

/// FNV-1a hash over a flat vector<int64_t> key.
struct DimKeyHash
{
    size_t operator()(const std::vector<int64_t>& key) const noexcept
    {
        size_t h = 14695981039346656037ULL;
        for(int64_t v : key)
        {
            const auto* p = reinterpret_cast<const unsigned char*>(&v);
            for(size_t b = 0; b < sizeof(int64_t); ++b)
            {
                h ^= static_cast<size_t>(p[b]);
                h *= 1099511628211ULL;
            }
        }
        return h;
    }
};

} // namespace detail

/// Loaded set of engine-override rules (process-lifetime cache around
/// HIPDNN_HEUR_CONFIG_PATH). Rules are evaluated in declaration order;
/// first match wins.
class EngineOverrideConfig
{
public:
    EngineOverrideConfig() = default;

    explicit EngineOverrideConfig(std::vector<OperationRule> rules)
    {
        for(size_t i = 0; i < rules.size(); ++i)
        {
            indexRule(std::move(rules[i]), i);
        }
    }

    static std::optional<EngineOverrideConfig> load(const std::string& filepath)
    {
        std::ifstream file(filepath);
        if(!file.is_open())
        {
            return std::nullopt;
        }
        try
        {
            return parseJson(nlohmann::json::parse(file));
        }
        catch(const nlohmann::json::exception&)
        {
            return std::nullopt;
        }
    }

    static std::optional<EngineOverrideConfig> loadFromContent(const std::string& content)
    {
        try
        {
            return parseJson(nlohmann::json::parse(content));
        }
        catch(const nlohmann::json::exception&)
        {
            return std::nullopt;
        }
    }

    /// Read HIPDNN_HEUR_CONFIG_PATH and load the referenced config.
    /// Returns nullopt when the variable is unset / empty / the file cannot
    /// be opened or parsed. Called once per heuristic finalize so env changes
    /// take effect without process restart and the path stays testable.
    static std::optional<EngineOverrideConfig> loadFromEnv()
    {
        static constexpr const char* ENV_VAR = "HIPDNN_HEUR_CONFIG_PATH";
        const std::string path
            = hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv(ENV_VAR, ""));
        if(path.empty())
        {
            return std::nullopt;
        }
        return load(path);
    }

    /// Scan rules in declaration order; return the first matching engine ID or nullopt.
    std::optional<int64_t> matchOperation(const std::string& op,
                                          const std::vector<TensorView>& tensors) const
    {
        return matchOperation(op, {}, tensors);
    }

    /// Scan rules in declaration order with operation-specific criteria.
    std::optional<int64_t> matchOperation(const std::string& op,
                                          const std::vector<Criterion>& criteria,
                                          const std::vector<TensorView>& tensors) const
    {
        const auto normalizedCriteria = normalizeCriteria(criteria);
        for(const auto& entry : _rules)
        {
            if(entry.rule.op != op)
            {
                continue;
            }
            if(entry.rule.matches(normalizedCriteria, tensors))
            {
                return entry.engineId;
            }
        }
        return std::nullopt;
    }

    size_t ruleCount() const
    {
        return _rules.size();
    }

private:
    struct IndexedRule
    {
        OperationRule rule;
        int64_t engineId;
    };

    std::vector<IndexedRule> _rules;

    static std::vector<Criterion> normalizeCriteria(std::vector<Criterion> criteria)
    {
        std::sort(criteria.begin(), criteria.end(), [](const Criterion& lhs, const Criterion& rhs) {
            return lhs.key < rhs.key;
        });
        return criteria;
    }

    static void normalizeRule(OperationRule& rule)
    {
        rule.criteria = normalizeCriteria(std::move(rule.criteria));
    }

    static int64_t getConfigVersion(const nlohmann::json& j)
    {
        if(j.contains(config_json::VERSION))
        {
            return j.at(config_json::VERSION).get<int64_t>();
        }
        return config_version::DEFAULT;
    }

    static bool usesNamedTensorIds(int64_t configVersion)
    {
        return configVersion >= config_version::NAMED_TENSOR_IDS;
    }

    static EngineOverrideConfig parseJson(const nlohmann::json& j)
    {
        const bool useNamedTensorIds = usesNamedTensorIds(getConfigVersion(j));
        std::vector<OperationRule> rules;
        for(const auto& entry : j.at(config_json::ENGINE_OVERRIDES))
        {
            OperationRule rule;
            rule.useNamedTensorIds = useNamedTensorIds;
            rule.op = entry.at(config_json::OP).get<std::string>();
            rule.engineName = entry.at(config_json::ENGINE_NAME).get<std::string>();
            if(entry.contains(config_json::CRITERIA))
            {
                const auto& criteria = entry.at(config_json::CRITERIA);
                if(!criteria.is_object())
                {
                    throw nlohmann::json::type_error::create(
                        302, "criteria must be an object", &criteria);
                }
                for(const auto& item : criteria.items())
                {
                    rule.criteria.push_back(Criterion{item.key(), item.value().get<int64_t>()});
                }
            }
            for(const auto& t : entry.at(config_json::TENSORS))
            {
                TensorPattern pat;
                if(t.contains(config_json::TENSOR_ID))
                {
                    pat.tensorId = t.at(config_json::TENSOR_ID).get<std::string>();
                }
                pat.dim = t.at(config_json::DIM).get<std::vector<int64_t>>();
                if(t.contains(config_json::STRIDE))
                {
                    pat.stride = t.at(config_json::STRIDE).get<std::vector<int64_t>>();
                }
                rule.tensors.push_back(std::move(pat));
            }
            rules.push_back(std::move(rule));
        }
        return EngineOverrideConfig(std::move(rules));
    }

    void indexRule(OperationRule rule, size_t /*order*/)
    {
        normalizeRule(rule);
        const int64_t resolvedId = hipdnn_data_sdk::utilities::engineNameOrIdToId(rule.engineName);
        _rules.push_back(IndexedRule{std::move(rule), resolvedId});
    }
};

} // namespace hipdnn_backend::heuristics::config
