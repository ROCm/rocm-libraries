// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/ck_problem_cache.hpp>

#include <mutex>
#include <optional>
#include <set>
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

/// Identifies everything the default depends on: the solver; the stream's device
/// (ctx.GetStream().GetDeviceName()); the problem as the perf-db keys it (lengths, pads, strides,
/// dilations, layouts, types, TF32, non-packed strides); the alpha/beta case (part of CK's
/// arguments, not of the perf-db key); and whether the ranked list is used at all.
///
/// The ranked list is chosen by the stream's device, but InitValidKernels() asks the CK plugin of
/// the current HIP device. The key assumes both are the same arch, as they are whenever the CK
/// solvers work at all: with different archs they would launch one arch's kernels on the other.
inline std::string DefaultConfigKey(std::string_view solver,
                                    std::string_view device,
                                    const miopen::conv::ProblemDescription& problem,
                                    bool ranked_defaults)
{
    std::ostringstream key;
    key << solver << ';' << device << ';';
    problem.Serialize(key);
    key << ';' << static_cast<int>(problem.GetAlphaBetaCase()) << ';' << ranked_defaults;
    return key.str();
}

inline std::string DefaultConfigKey(std::string_view solver,
                                    const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
                                    bool ranked_defaults)
{
    return DefaultConfigKey(solver, ctx.GetStream().GetDeviceName(), problem, ranked_defaults);
}

/// Process-wide memo of default configurations (bounded and switchable, see CKProblemCache): the
/// default depends only on the key, and computing it asks CK about every instance for the full
/// valid-kernel list, which would otherwise happen on every perf-db miss. Tuning and the AI
/// heuristics build their own lists and do not use it. A config set from a cached default has an
/// empty valid_kernels (index 0), so its IsValidValue() is false; only tuning iterates the list,
/// and SetNextValue() fills it. kernel_ids are stored once each and kept when the cache starts
/// over: there are at most as many as the plugin has instances.
class DefaultConfigCache
{
public:
    static DefaultConfigCache& Get()
    {
        static DefaultConfigCache cache;
        return cache;
    }

    /// The default for `key`: the cached one, or the one `compute()` returns, which is then kept
    /// when it exists.
    template <class Compute>
    std::optional<CachedDefaultConfig> GetOrCompute(const std::string& key, Compute&& compute)
    {
        const auto entry = entries.GetOrCompute(key, [&]() -> std::optional<Entry> {
            std::optional<CachedDefaultConfig> computed = compute();
            if(!computed)
                return std::nullopt;
            return Entry{&Intern(computed->kernel_id), computed->use_tf32};
        });
        if(!entry)
            return std::nullopt;
        return CachedDefaultConfig{*entry->kernel_id, entry->use_tf32};
    }

private:
    struct Entry
    {
        const std::string* kernel_id;
        bool use_tf32;
    };

    const std::string& Intern(const std::string& kernel_id)
    {
        std::lock_guard lock{kernel_ids_mutex};
        return *kernel_ids.insert(kernel_id).first;
    }

    std::mutex kernel_ids_mutex;
    std::set<std::string> kernel_ids;
    CKProblemCache<std::unordered_map<std::string, Entry>> entries;
};

} // namespace conv
} // namespace solver
} // namespace miopen
