// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>

#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace miopen {
namespace solver {
namespace conv {

/// The default (non-AI, untuned) configuration a CK grouped solver derives from its valid-kernel
/// list: the kernel_id it runs and whether the list was taken with TF32.
struct CachedDefaultConfig
{
    std::string kernel_id;
    bool use_tf32;
};

/// Identifies what the default depends on: the solver, the device (the CK plugin and the ranked
/// list are chosen by its name), the problem as the perf-db keys it (lengths, pads, strides,
/// dilations, layouts, types, TF32, non-packed strides), the alpha/beta case (part of CK's
/// arguments, not of the perf-db key), and whether the ranked list is used at all.
inline std::string DefaultConfigKey(const char* solver,
                                    const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
                                    bool ranked_defaults)
{
    std::ostringstream key;
    key << solver << ';' << ctx.GetStream().GetDeviceName() << ';';
    problem.Serialize(key);
    key << ';' << static_cast<int>(problem.GetAlphaBetaCase()) << ';' << ranked_defaults;
    return key.str();
}

/// Process-wide memo of default configurations, never evicted: the default depends only on the
/// key, and computing it means asking CK about every instance for the full valid-kernel list on
/// every perf-db miss. Tuning and the AI heuristics build their own lists and do not use it.
/// kernel_ids are stored once each; entries point at them.
class DefaultConfigCache
{
public:
    static DefaultConfigCache& Get()
    {
        static DefaultConfigCache cache;
        return cache;
    }

    std::optional<CachedDefaultConfig> Load(const std::string& key) const
    {
        std::shared_lock lock{mutex};
        const auto it = entries.find(key);
        if(it == entries.end())
            return std::nullopt;
        return CachedDefaultConfig{*it->second.kernel_id, it->second.use_tf32};
    }

    void Store(const std::string& key, const CachedDefaultConfig& config)
    {
        std::unique_lock lock{mutex};
        const auto& kernel_id = *kernel_ids.insert(config.kernel_id).first;
        entries.try_emplace(key, Entry{&kernel_id, config.use_tf32});
    }

private:
    struct Entry
    {
        const std::string* kernel_id;
        bool use_tf32;
    };

    mutable std::shared_mutex mutex;
    std::set<std::string> kernel_ids;
    std::unordered_map<std::string, Entry> entries;
};

} // namespace conv
} // namespace solver
} // namespace miopen
