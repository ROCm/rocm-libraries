// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
{

/// What rocKE's runtime selection reads of one implicit-GEMM instance that already serves
/// the problem: its kind and its baked load and store widths.
struct AotSelectionKey
{
    /// The kind. rocKE's kind is (direction, epilogue, split_k, two_stage, group_merge);
    /// every packaged instance is forward with the default split-K, two-stage and group
    /// merge (make_conv_aot_config.py refuses any other), so only the epilogue varies.
    std::string epilogue;
    int64_t vectorA;
    int64_t vectorB;
    int64_t vectorC;
};

/// Which of @p serving to run: rocKE's _widest_vector_sizes in
/// rocke/library/benchmarks/common/benchmark_implicit_gemm_conv.py, the default policy of
/// its --run-from-cache benchmark (no --all-vector-sizes), applied to the implicit-GEMM
/// instances that serve one problem (KernelCache.compatible). One flag per entry, in
/// order.
///
/// - Default-epilogue instances store scalars. They are dropped when any serving
///   instance stores wider than 1, since then a cshuffle instance of the same
///   configuration stores wider.
/// - Per kind, only the widest width combination present is kept, widest being the
///   greatest (a + b + c, a, b, c). Every width an instance bakes divides the problem's
///   channel run, so that combination is the problem's widest; a configuration that
///   cannot reach it (its tile admits only narrower loads, or its grid does not fit) is
///   dropped.
///
/// The policy depends on which instances serve, not on one instance, so it runs over the
/// whole candidate set (hipkernel:Gfx950ConvFwdAot registers it as a CandidateFilterFn).
inline std::vector<bool> widestVectorSelection(const std::vector<AotSelectionKey>& serving)
{
    using Rank = std::tuple<int64_t, int64_t, int64_t, int64_t>;
    const auto rank = [](const AotSelectionKey& key) {
        return Rank{key.vectorA + key.vectorB + key.vectorC, key.vectorA, key.vectorB, key.vectorC};
    };

    // rocKE groups the store widths by (direction, split_k, two_stage, group_merge),
    // which is one group here.
    int64_t widestStore = 0;
    for(const auto& key : serving)
    {
        widestStore = std::max(widestStore, key.vectorC);
    }
    std::vector<bool> keep(serving.size());
    for(size_t i = 0; i < serving.size(); ++i)
    {
        keep[i] = !(serving[i].epilogue == "default" && widestStore > 1);
    }

    std::map<std::string, Rank> best;
    for(size_t i = 0; i < serving.size(); ++i)
    {
        if(!keep[i])
        {
            continue;
        }
        const auto [it, inserted] = best.emplace(serving[i].epilogue, rank(serving[i]));
        if(!inserted)
        {
            it->second = std::max(it->second, rank(serving[i]));
        }
    }
    for(size_t i = 0; i < serving.size(); ++i)
    {
        keep[i] = keep[i] && rank(serving[i]) == best.at(serving[i].epilogue);
    }
    return keep;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
