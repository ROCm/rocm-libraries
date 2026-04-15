// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <fstream>
#include <nlohmann/json.hpp>
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB

namespace hipdnn_frontend::engine_override
{

/// Dimension value meaning "match any value in this slot".
inline constexpr int64_t WILDCARD_DIM = -1;

/// Pattern for a single tensor: a list of expected dimensions and optional strides,
/// with -1 as a per-slot wildcard.
/// When `stride` is empty no stride matching is performed.
struct TensorPattern
{
    std::vector<int64_t> dim;
    std::vector<int64_t> stride; ///< Empty = do not match on stride.

    /// Returns true iff tensor.get_dim() matches this pattern element-by-element,
    /// and (when stride is non-empty) tensor.get_stride() matches stride element-by-element.
    /// Rejects immediately when rank differs; skips per-element check for WILDCARD_DIM slots.
    bool matches(const graph::TensorAttributes& tensor) const
    {
        const auto& tdim = tensor.get_dim();
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
            const auto& tstride = tensor.get_stride();
            // Nonmatching strides assume wildcard semantics.
            const size_t rank = std::min(stride.size(), tstride.size());
            for(size_t i = 0; i < rank; ++i)
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

/// A single engine-override rule.
struct OperationRule
{
    std::string op; ///< "conv_fprop" / "conv_dgrad" / "conv_wgrad"
    std::string engineName; ///< Engine name resolved to an ID via engineNameToId()
    std::vector<TensorPattern> tensors; ///< Ordered patterns for operation inputs

    /// Returns true iff every tensor in `tensors` matches the corresponding pattern.
    /// Rejects immediately when the tensor count differs.
    bool matches(const std::vector<std::shared_ptr<graph::TensorAttributes>>& inputs) const
    {
        if(tensors.size() != inputs.size())
        {
            return false;
        }
        for(size_t i = 0; i < tensors.size(); ++i)
        {
            if(!tensors[i].matches(*inputs[i]))
            {
                return false;
            }
        }
        return true;
    }
};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// FNV-1a hash over a flat vector<int64_t> key.
struct DimKeyHash
{
    size_t operator()(const std::vector<int64_t>& key) const noexcept
    {
        size_t h = 14695981039346656037ULL; // FNV-1a offset basis (64-bit)
        for(int64_t v : key)
        {
            const auto* p = reinterpret_cast<const unsigned char*>(&v);
            for(size_t b = 0; b < sizeof(int64_t); ++b)
            {
                h ^= static_cast<size_t>(p[b]);
                h *= 1099511628211ULL; // FNV-1a prime
            }
        }
        return h;
    }
};

// ── EngineOverrideConfig ──────────────────────────────────────────────────────

/// Loaded set of engine-override rules.
/// Rules are evaluated in declaration order; first match wins.
///
/// Internally rules are split per op name (strategy 1) and further divided
/// into exact rules — where every dimension is concrete and no stride constraint
/// is present — stored in a hash map for O(1) lookup, and wildcard rules kept
/// in a declaration-order vector (strategy 2).  The two structures are reconciled
/// via declaration index so that first-match semantics are preserved across the
/// partition.
class EngineOverrideConfig
{
public:
    /// Default-construct an empty config (no rules).
    EngineOverrideConfig() = default;

    /// Construct directly from a vector of rules (useful for tests).
    explicit EngineOverrideConfig(std::vector<OperationRule> rules)
    {
        for(size_t i = 0; i < rules.size(); ++i)
        {
            indexRule(std::move(rules[i]), i);
        }
    }

    /// Load from an explicit file path.
    /// Returns nullopt on missing file, parse error, or when JSON support is
    /// compiled out (HIPDNN_FRONTEND_SKIP_JSON_LIB defined).
    static std::optional<EngineOverrideConfig> load(const std::string& filepath)
    {
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
        std::ifstream file(filepath);
        if(!file.is_open())
        {
            HIPDNN_FE_LOG_WARN("EngineOverrideConfig: cannot open file: " << filepath);
            return std::nullopt;
        }

        try
        {
            auto config = parseJson(nlohmann::json::parse(file));
            HIPDNN_FE_LOG_INFO("EngineOverrideConfig: loaded " << config.ruleCount()
                                                               << " rule(s) from " << filepath);
            return config;
        }
        catch(const nlohmann::json::exception& e)
        {
            HIPDNN_FE_LOG_WARN("EngineOverrideConfig: JSON parse error in " << filepath << ": "
                                                                            << e.what());
            return std::nullopt;
        }
#else
        (void)filepath;
        HIPDNN_FE_LOG_WARN(
            "EngineOverrideConfig: JSON support not compiled in; engine override file ignored.");
        return std::nullopt;
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
    }

    /// Load from a JSON string in memory.
    /// Returns nullopt on parse error or when JSON support is compiled out.
    static std::optional<EngineOverrideConfig> loadFromContent(const std::string& content)
    {
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
        try
        {
            auto config = parseJson(nlohmann::json::parse(content));
            HIPDNN_FE_LOG_INFO("EngineOverrideConfig: loaded " << config.ruleCount()
                                                               << " rule(s) from inline content");
            return config;
        }
        catch(const nlohmann::json::exception& e)
        {
            HIPDNN_FE_LOG_WARN(
                "EngineOverrideConfig: JSON parse error in inline content: " << e.what());
            return std::nullopt;
        }
#else
        (void)content;
        HIPDNN_FE_LOG_WARN(
            "EngineOverrideConfig: JSON support not compiled in; engine override content ignored.");
        return std::nullopt;
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
    }

    /// Return a pointer to the process-lifetime config loaded from
    /// HIPDNN_ENGINE_OVERRIDE_FILE (read and cached on the first call,
    /// thread-safe per C++11).  Returns nullptr when the variable is unset,
    /// empty, the file cannot be opened, or JSON support is compiled out.
    /// Leading and trailing whitespace in the path value is ignored.
    static const EngineOverrideConfig* loadFromEnv()
    {
        static constexpr const char* ENV_VAR = "HIPDNN_ENGINE_OVERRIDE_FILE";
        static const std::optional<EngineOverrideConfig> s_cached = []() {
            std::string path = hipdnn_data_sdk::utilities::getEnv(ENV_VAR, "");
            const auto first = path.find_first_not_of(" \t\r\n");
            if(first == std::string::npos)
            {
                return std::optional<EngineOverrideConfig>{};
            }
            path = path.substr(first, path.find_last_not_of(" \t\r\n") - first + 1);
            return load(path);
        }();
        return s_cached ? &*s_cached : nullptr;
    }

    /// Scan rules in declaration order; return the first matching enginedId or nullopt.
    ///
    /// Strategy 1: only the bucket for `op` is examined (hash map lookup).
    /// Strategy 2: within the bucket, exact rules are probed in O(1) via hash map;
    ///             wildcard rules are scanned linearly but the scan terminates as
    ///             soon as a lower-order exact match is known to exist.
    std::optional<int64_t>
        matchOperation(const std::string& op,
                       const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors) const
    {
        // Strategy 1: find the op bucket
        const auto opIt = _index.find(op);
        if(opIt == _index.end())
        {
            return std::nullopt;
        }
        const OpBucket& bucket = opIt->second;

        // Strategy 2a: O(1) probe of the exact map
        std::optional<ExactEntry> exactHit;
        {
            const auto key = buildDimKey(tensors);
            const auto eit = bucket.exact.find(key);
            if(eit != bucket.exact.end())
            {
                exactHit = eit->second;
            }
        }

        // Strategy 2b: linear scan of wildcard rules in declaration order.
        // Wildcards are stored in ascending order, so once the current entry's
        // order exceeds the exact hit's order, no further wildcard can win.
        for(const auto& entry : bucket.wildcards)
        {
            if(exactHit && entry.order > exactHit->order)
            {
                break; // exact match has earlier declaration; no wildcard can beat it
            }
            if(entry.rule.matches(tensors))
            {
                // This wildcard has lower or equal order to any exact hit (loop
                // would have broken otherwise), so it is the first-match winner.
                HIPDNN_FE_LOG_INFO("EngineOverrideConfig: matched op="
                                   << op << " enginedId=" << entry.enginedId << " (wildcard rule)");
                return entry.enginedId;
            }
        }

        if(exactHit)
        {
            HIPDNN_FE_LOG_INFO("EngineOverrideConfig: matched op="
                               << op << " enginedId=" << exactHit->enginedId << " (exact rule)");
            return exactHit->enginedId;
        }
        return std::nullopt;
    }

private:
    /// enginedId and declaration index for an exact-match rule.
    struct ExactEntry
    {
        int64_t enginedId;
        size_t order; ///< position in the original rule list (0 = first)
    };

    /// Wildcard rule paired with its declaration index and resolved engine ID.
    struct WildcardEntry
    {
        OperationRule rule;
        int64_t enginedId; ///< resolved from rule.engineName at index time
        size_t order;
    };

    /// Per-op rule storage partitioned into exact and wildcard buckets.
    struct OpBucket
    {
        /// Exact rules: no WILDCARD_DIM anywhere and no stride constraints.
        /// Key = rank-prefixed flattened dims of all input tensors.
        /// When two rules share a key, only the first (lowest order) is kept.
        std::unordered_map<std::vector<int64_t>, ExactEntry, DimKeyHash> exact;

        /// Wildcard rules in ascending declaration order.
        std::vector<WildcardEntry> wildcards;
    };

    std::unordered_map<std::string, OpBucket> _index;

    // ── helpers ───────────────────────────────────────────────────────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    /// Parse a nlohmann::json object into an EngineOverrideConfig.
    /// Throws nlohmann::json::exception on malformed input; callers handle it.
    ///
    /// Expected JSON format:
    /// @code{.json}
    /// {
    ///   "engine_overrides": [
    ///     {
    ///       "op": "conv_fprop",
    ///       "engine_name": "MIOPEN_ENGINE",
    ///       "tensors": [
    ///         { "dim": [1, 3, 224, 224], "stride": [150528, 50176, 224, 1] },
    ///         { "dim": [64, 3, 7, 7] }
    ///       ]
    ///     },
    ///     {
    ///       "op": "conv_fprop",
    ///       "engine_name": "HIPBLASLT_ENGINE",
    ///       "tensors": [
    ///         { "dim": [-1, -1, -1, -1] },
    ///         { "dim": [-1, -1, -1, -1] }
    ///       ]
    ///     }
    ///   ]
    /// }
    /// @endcode
    ///
    /// Notes:
    /// - `engine_name` must be a registered engine name (e.g. "MIOPEN_ENGINE").
    /// - `-1` in `dim` or `stride` is a wildcard matching any value.
    /// - `stride` is optional per tensor; omitting it skips stride matching.
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
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB

    /// Returns true if any dim slot in any pattern is WILDCARD_DIM, or if any
    /// pattern carries a stride constraint (stride-constrained rules use the
    /// linear wildcard scan so that TensorPattern::matches() is always called).
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

    /// Build a dim key from a rule's tensor patterns.
    /// Format: [rank₀, dim₀₀, dim₀₁, …, rank₁, dim₁₀, …]
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

    /// Build a dim key from live tensor attributes.
    static std::vector<int64_t>
        buildDimKey(const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors)
    {
        std::vector<int64_t> key;
        for(const auto& t : tensors)
        {
            const auto& d = t->get_dim();
            key.push_back(static_cast<int64_t>(d.size()));
            key.insert(key.end(), d.begin(), d.end());
        }
        return key;
    }

    /// Insert one rule into the appropriate bucket of _index.
    /// Resolves engineName to an int64_t ID via engineNameToId().
    void indexRule(OperationRule rule, size_t order)
    {
        const int64_t resolvedId = hipdnn_data_sdk::utilities::engineNameToId(rule.engineName);
        OpBucket& bucket = _index[rule.op]; // keyed by op (strategy 1)
        if(hasWildcard(rule.tensors))
        {
            bucket.wildcards.push_back(WildcardEntry{std::move(rule), resolvedId, order});
        }
        else
        {
            const auto key = buildDimKey(rule.tensors);
            // try_emplace keeps the first (lowest-order) entry for duplicate keys.
            bucket.exact.try_emplace(key, ExactEntry{resolvedId, order});
        }
    }

    /// Total rule count across all buckets (exact + wildcard).
    size_t ruleCount() const
    {
        size_t n = 0;
        for(const auto& [op, bucket] : _index)
        {
            n += bucket.exact.size() + bucket.wildcards.size();
        }
        return n;
    }
};

/// Match op/tensors against a config and return the first matching enginedId.
/// When `config` is null the process-lifetime config loaded from
/// HIPDNN_ENGINE_OVERRIDE_FILE is used (read once on first call, thread-safe
/// per C++11).  Passing an explicit config bypasses the env-var lookup entirely,
/// which is useful for testing or when the caller manages the config lifetime.
/// Returns nullopt when no rule matches or JSON support is compiled out.
inline std::optional<int64_t>
    checkEngineOverride(const std::string& op,
                        const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors,
                        const EngineOverrideConfig* config = nullptr)
{
    if(config == nullptr)
    {
        config = EngineOverrideConfig::loadFromEnv();
    }
    if(config == nullptr)
    {
        return std::nullopt;
    }
    return config->matchOperation(op, tensors);
}

} // namespace hipdnn_frontend::engine_override
