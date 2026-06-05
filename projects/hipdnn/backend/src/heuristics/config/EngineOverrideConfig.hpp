// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_backend::heuristics::config
{

/// Dimension value meaning "match any value in this slot".
inline constexpr int64_t WILDCARD_DIM = -1;

/// View into one tensor: pointers to the live dim and stride vectors.
/// The matcher does not own this data; callers must keep the underlying
/// vectors alive for the duration of the match call.
struct TensorView
{
    const std::vector<int64_t>* dim;
    const std::vector<int64_t>* stride;
};

/// Pattern for a single tensor: a list of expected dimensions and optional strides,
/// with -1 as a per-slot wildcard. When `stride` is empty no stride matching is
/// performed.
struct TensorPattern
{
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

/// A single engine-override rule (one operation, one engine, ordered tensor patterns).
struct OperationRule
{
    std::string op;
    std::string engineName;
    std::vector<TensorPattern> tensors;

    bool matches(const std::vector<TensorView>& inputs) const
    {
        if(tensors.size() != inputs.size())
        {
            return false;
        }
        for(size_t i = 0; i < tensors.size(); ++i)
        {
            if(!tensors[i].matches(inputs[i]))
            {
                return false;
            }
        }
        return true;
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
/// first match wins. Internally split per-op into an exact hash bucket and
/// an order-preserving wildcard list, reconciled by declaration index.
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
        const auto opIt = _index.find(op);
        if(opIt == _index.end())
        {
            return std::nullopt;
        }
        const OpBucket& bucket = opIt->second;

        std::optional<ExactEntry> exactHit;
        {
            const auto key = buildDimKey(tensors);
            const auto eit = bucket.exact.find(key);
            if(eit != bucket.exact.end())
            {
                exactHit = eit->second;
            }
        }

        for(const auto& entry : bucket.wildcards)
        {
            if(exactHit && entry.order > exactHit->order)
            {
                break;
            }
            if(entry.rule.matches(tensors))
            {
                return entry.engineId;
            }
        }

        if(exactHit)
        {
            return exactHit->engineId;
        }
        return std::nullopt;
    }

    size_t ruleCount() const
    {
        size_t n = 0;
        for(const auto& [op, bucket] : _index)
        {
            n += bucket.exact.size() + bucket.wildcards.size();
        }
        return n;
    }

private:
    struct ExactEntry
    {
        int64_t engineId;
        size_t order;
    };

    struct WildcardEntry
    {
        OperationRule rule;
        int64_t engineId;
        size_t order;
    };

    struct OpBucket
    {
        std::unordered_map<std::vector<int64_t>, ExactEntry, detail::DimKeyHash> exact;
        std::vector<WildcardEntry> wildcards;
    };

    std::unordered_map<std::string, OpBucket> _index;

    static EngineOverrideConfig parseJson(const nlohmann::json& j)
    {
        std::vector<OperationRule> rules;
        for(const auto& entry : j.at("engine_overrides"))
        {
            OperationRule rule;
            rule.op = entry.at("op").get<std::string>();
            rule.engineName = entry.at("engine_name").get<std::string>();
            for(const auto& t : entry.at("tensors"))
            {
                TensorPattern pat;
                pat.dim = t.at("dim").get<std::vector<int64_t>>();
                if(t.contains("stride"))
                {
                    pat.stride = t.at("stride").get<std::vector<int64_t>>();
                }
                rule.tensors.push_back(std::move(pat));
            }
            rules.push_back(std::move(rule));
        }
        return EngineOverrideConfig(std::move(rules));
    }

    static bool hasWildcard(const std::vector<TensorPattern>& patterns)
    {
        for(const auto& p : patterns)
        {
            for(const int64_t d : p.dim)
            {
                if(d == WILDCARD_DIM)
                {
                    return true;
                }
            }
            if(!p.stride.empty())
            {
                return true;
            }
        }
        return false;
    }

    static std::vector<int64_t> buildDimKey(const std::vector<TensorPattern>& patterns)
    {
        std::vector<int64_t> key;
        for(const auto& p : patterns)
        {
            key.push_back(static_cast<int64_t>(p.dim.size()));
            key.insert(key.end(), p.dim.begin(), p.dim.end());
        }
        return key;
    }

    static std::vector<int64_t> buildDimKey(const std::vector<TensorView>& tensors)
    {
        std::vector<int64_t> key;
        for(const auto& t : tensors)
        {
            const auto& d = *t.dim;
            key.push_back(static_cast<int64_t>(d.size()));
            key.insert(key.end(), d.begin(), d.end());
        }
        return key;
    }

    void indexRule(OperationRule rule, size_t order)
    {
        const int64_t resolvedId = hipdnn_data_sdk::utilities::engineNameToId(rule.engineName);
        OpBucket& bucket = _index[rule.op];
        if(hasWildcard(rule.tensors))
        {
            bucket.wildcards.push_back(WildcardEntry{std::move(rule), resolvedId, order});
        }
        else
        {
            const auto key = buildDimKey(rule.tensors);
            bucket.exact.try_emplace(key, ExactEntry{resolvedId, order});
        }
    }
};

} // namespace hipdnn_backend::heuristics::config
