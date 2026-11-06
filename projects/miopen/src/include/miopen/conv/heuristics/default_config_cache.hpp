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
#include <string_view>
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
inline std::string DefaultConfigKey(std::string_view solver,
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
/// key, and computing it asks CK about every instance for the full valid-kernel list, which would
/// otherwise happen on every perf-db miss. Tuning and the AI heuristics build their own lists and
/// do not use it. A config set from a cached default has an empty valid_kernels (index 0), so its
/// IsValidValue() is false; only tuning iterates the list, and SetNextValue() fills it.
/// kernel_ids are stored once each; entries point at them.
class DefaultConfigCache
{
public:
    static DefaultConfigCache& Get()
    {
        static DefaultConfigCache cache;
        return cache;
    }

    /// The default for `key`: the cached one, or the one `compute()` returns, which is then kept
    /// when it exists. `compute()` runs outside the lock; threads racing on a new key compute the
    /// same default, and the first one stored is kept.
    template <class Compute>
    std::optional<CachedDefaultConfig> GetOrCompute(const std::string& key, Compute&& compute)
    {
        {
            std::shared_lock lock{mutex};
            if(const auto it = entries.find(key); it != entries.end())
                return CachedDefaultConfig{*it->second.kernel_id, it->second.use_tf32};
        }
        std::optional<CachedDefaultConfig> computed = compute();
        if(computed)
        {
            std::unique_lock lock{mutex};
            const auto& kernel_id = *kernel_ids.insert(computed->kernel_id).first;
            entries.try_emplace(key, Entry{&kernel_id, computed->use_tf32});
        }
        return computed;
    }

private:
    struct Entry
    {
        const std::string* kernel_id;
        bool use_tf32;
    };

    std::shared_mutex mutex;
    std::set<std::string> kernel_ids;
    std::unordered_map<std::string, Entry> entries;
};

} // namespace conv
} // namespace solver
} // namespace miopen
