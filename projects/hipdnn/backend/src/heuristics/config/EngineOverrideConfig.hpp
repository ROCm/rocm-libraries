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
#include <iterator>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hipdnn_backend::heuristics::config
{
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_version = hipdnn_data_sdk::detail::autotune_config::version;

/// Dimension value meaning "match any value in this slot".
inline constexpr int64_t WILDCARD_DIM = -1;

inline constexpr int NLOHMANN_TYPE_MISMATCH_ERROR_ID = 302;

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

        const auto matchesElement = [](int64_t expected, int64_t actual) {
            return expected == WILDCARD_DIM || expected == actual;
        };
        if(!std::equal(dim.begin(), dim.end(), tdim.begin(), matchesElement))
        {
            return false;
        }

        const auto& tstride = *tensor.stride;
        return stride.empty()
               || (stride.size() == tstride.size()
                   && std::equal(stride.begin(), stride.end(), tstride.begin(), matchesElement));
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
        return std::equal(criteria.begin(),
                          criteria.end(),
                          actualCriteria.begin(),
                          [](const Criterion& lhs, const Criterion& rhs) {
                              return lhs.key == rhs.key && lhs.value == rhs.value;
                          })
               && matchesTensors(inputs);
    }

private:
    bool matchesTensors(const std::vector<TensorView>& inputs) const
    {
        if(!useNamedTensorIds)
        {
            return matchesLegacyPositional(inputs);
        }

        return std::all_of(tensors.begin(), tensors.end(), [&](const TensorPattern& pattern) {
            return pattern.tensorId.has_value() && matchesNamed(pattern, inputs);
        });
    }

    bool matchesLegacyPositional(const std::vector<TensorView>& inputs) const
    {
        return std::equal(tensors.begin(),
                          tensors.end(),
                          inputs.begin(),
                          [](const TensorPattern& pattern, const TensorView& input) {
                              return pattern.matches(input);
                          });
    }

    static bool matchesNamed(const TensorPattern& pattern, const std::vector<TensorView>& inputs)
    {
        const auto it = std::find_if(inputs.begin(), inputs.end(), [&](const TensorView& input) {
            return input.tensorId == *pattern.tensorId && pattern.matches(input);
        });
        return it != inputs.end();
    }
};

/// Loaded set of engine-override rules (process-lifetime cache around
/// HIPDNN_HEUR_CONFIG_PATH). Rules are evaluated in declaration order;
/// first match wins.
class EngineOverrideConfig
{
public:
    EngineOverrideConfig() = default;

    explicit EngineOverrideConfig(std::vector<OperationRule> rules)
    {
        std::transform(
            rules.begin(), rules.end(), std::back_inserter(_rules), [](OperationRule& rule) {
                normalizeRule(rule);
                const int64_t resolvedId
                    = hipdnn_data_sdk::utilities::engineNameOrIdToId(rule.engineName);
                return IndexedRule{std::move(rule), resolvedId};
            });
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
                        NLOHMANN_TYPE_MISMATCH_ERROR_ID, "criteria must be an object", &criteria);
                }
                for(const auto& item : criteria.items())
                {
                    rule.criteria.push_back(Criterion{item.key(), item.value().get<int64_t>()});
                }
            }
            const auto& tensors = entry.at(config_json::TENSORS);
            std::unordered_set<std::string_view> tensorIds;
            tensorIds.reserve(tensors.size());
            for(const auto& t : tensors)
            {
                TensorPattern pat;
                if(useNamedTensorIds)
                {
                    if(!t.contains(config_json::TENSOR_ID))
                    {
                        throw nlohmann::json::type_error::create(
                            NLOHMANN_TYPE_MISMATCH_ERROR_ID,
                            "versioned tensor entry must contain tensor_id",
                            &t);
                    }
                    const auto& tensorId
                        = t.at(config_json::TENSOR_ID).get_ref<const std::string&>();
                    if(!tensorIds.emplace(tensorId.data(), tensorId.size()).second)
                    {
                        throw nlohmann::json::type_error::create(
                            NLOHMANN_TYPE_MISMATCH_ERROR_ID,
                            "versioned tensor_id entries must be unique",
                            &t);
                    }
                    pat.tensorId = tensorId;
                }
                else if(t.contains(config_json::TENSOR_ID))
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
};

} // namespace hipdnn_backend::heuristics::config
