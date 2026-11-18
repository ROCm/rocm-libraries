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

/// The untuned configuration HeuristicInit of a CK grouped backward-weights solver chooses, by
/// the AI heuristics or from the default valid-kernel list: the kernel_id it runs (with its
/// "+<split_k>" suffix), whether it uses TF32, and its split_k.
struct CachedDefaultConfig
{
    std::string kernel_id;
    bool use_tf32;
    int split_k;
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

/// The key of a HeuristicInit result: DefaultConfigKey, and whether the AI heuristics run. When
/// they do, it also holds the device's compute-unit count, the one input of their features that
/// is neither in the problem nor in the device name; their models are fixed for the process.
inline std::string HeuristicConfigKey(std::string_view solver,
                                      std::string_view device,
                                      std::size_t compute_units,
                                      const miopen::conv::ProblemDescription& problem,
                                      bool ranked_defaults,
                                      bool ai_heuristics)
{
    auto key = DefaultConfigKey(solver, device, problem, ranked_defaults);
    key += ai_heuristics ? ";ai;" + std::to_string(compute_units) : std::string{";noai"};
    return key;
}

inline std::string HeuristicConfigKey(std::string_view solver,
                                      const ExecutionContext& ctx,
                                      const miopen::conv::ProblemDescription& problem,
                                      bool ranked_defaults,
                                      bool ai_heuristics)
{
    return HeuristicConfigKey(solver,
                              ctx.GetStream().GetDeviceName(),
                              ctx.GetStream().GetMaxComputeUnits(),
                              problem,
                              ranked_defaults,
                              ai_heuristics);
}

/// Process-wide memo of HeuristicInit results (bounded and switchable, see CKProblemCache): the
/// result depends only on the key, and computing it asks CK about every instance for the full
/// valid-kernel list (and, with the AI heuristics, runs their model), which would otherwise
/// happen on every perf-db miss. Tuning builds its own list and does not use it. A cache hit
/// sets only kernel_id, use_tf32 and split_k; valid_kernels and index keep what HeuristicInit
/// left (empty), so they need not match kernel_id and IsValidValue() can be false. Nothing reads
/// them before tuning, which starts from a default-constructed config. kernel_ids are stored
/// once each and kept when the cache starts over: there are at most as many as the plugin has
/// instances and split_k values.
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
            return Entry{&Intern(computed->kernel_id), computed->use_tf32, computed->split_k};
        });
        if(!entry)
            return std::nullopt;
        return CachedDefaultConfig{*entry->kernel_id, entry->use_tf32, entry->split_k};
    }

private:
    struct Entry
    {
        const std::string* kernel_id;
        bool use_tf32;
        int split_k;
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

/// HeuristicInit of the CK grouped backward-weights solvers, through DefaultConfigCache. On a miss
/// it runs the AI heuristics when `ai_heuristics` is set (`run_ai()` returns whether they chose a
/// configuration), and otherwise, or when they choose none, builds cfg's valid-kernel list and
/// picks the default from it (ranked, or the first entry, with split_k = 1). Either way cfg ends
/// with the cached kernel_id, use_tf32 and split_k. The perf configs befriend it to reach
/// InitValidKernels() and use_tf32.
template <class Config, class State, class RunAI>
void InitHeuristicConfigFromCache(Config& cfg,
                                  State& state,
                                  const ExecutionContext& ctx,
                                  const miopen::conv::ProblemDescription& problem,
                                  std::string_view solver,
                                  bool ranked_defaults,
                                  bool uses_split_k,
                                  bool ai_heuristics,
                                  RunAI&& run_ai)
{
    const auto chosen = DefaultConfigCache::Get().GetOrCompute(
        HeuristicConfigKey(solver, ctx, problem, ranked_defaults, ai_heuristics),
        [&]() -> std::optional<CachedDefaultConfig> {
            if(ai_heuristics && run_ai())
                return CachedDefaultConfig{cfg.kernel_id, cfg.use_tf32, cfg.split_k};
            cfg.InitValidKernels(problem);
            if(cfg.valid_kernels.empty())
                return std::nullopt;
            if(ranked_defaults)
                cfg.DefaultKernelFromList(ctx);
            state.SetResult(cfg.index, cfg.split_k, uses_split_k);
            return CachedDefaultConfig{cfg.kernel_id, cfg.use_tf32, 1};
        });
    if(chosen)
    {
        cfg.kernel_id = chosen->kernel_id;
        cfg.use_tf32  = chosen->use_tf32;
        cfg.split_k   = chosen->split_k;
    }
}

} // namespace conv
} // namespace solver
} // namespace miopen
