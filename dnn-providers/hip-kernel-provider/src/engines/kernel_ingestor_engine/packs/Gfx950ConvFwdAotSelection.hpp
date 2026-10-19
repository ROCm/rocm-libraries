// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
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
    /// Group-merge degree, part of the kind for widestPerKindSelection; 1 when unmerged.
    int64_t groupMerge = 1;
};

/// The widest vector width a channel run of @p channels admits: the largest of 8, 4, 2
/// and 1 that divides it.
inline int64_t widestVectorWidth(int64_t channels)
{
    for(const int64_t width : {int64_t{8}, int64_t{4}, int64_t{2}})
    {
        if(channels % width == 0)
        {
            return width;
        }
    }
    return 1;
}

/// Which of @p serving to run, for a problem of @p cpg input and @p kpg output channels
/// per group. One flag per entry, in order.
///
/// The problem's widest combination is the widest load width C/groups admits and the
/// widest store width K/groups admits (widestVectorWidth). When some serving instance
/// bakes exactly that combination, only those run, of either epilogue: in a catalog that
/// builds every width this is rocKE's _widest_vector_sizes (the default policy of
/// --run-from-cache in rocke/library/benchmarks/common/benchmark_implicit_gemm_conv.py).
/// The shipped catalog builds only some combinations, so when none serving bakes the
/// problem's widest, every serving instance runs: the selection falls back to whatever
/// widths the catalog supports for the problem rather than to a narrower guess.
///
/// The policy depends on which instances serve, not on one instance, so it runs over the
/// whole candidate set (hipkernel:Gfx950ConvFwdAot registers it as a CandidateFilterFn).
inline std::vector<bool>
    widestVectorSelection(const std::vector<AotSelectionKey>& serving, int64_t cpg, int64_t kpg)
{
    const int64_t load = widestVectorWidth(cpg);
    const int64_t store = widestVectorWidth(kpg);
    std::vector<bool> keep(serving.size());
    bool anyWidest = false;
    for(size_t i = 0; i < serving.size(); ++i)
    {
        keep[i] = serving[i].vectorA == load && serving[i].vectorB == load
                  && serving[i].vectorC == store;
        anyWidest = anyWidest || keep[i];
    }
    if(!anyWidest)
    {
        std::fill(keep.begin(), keep.end(), true);
    }
    return keep;
}

/// Which of @p serving to run: rocKE's current --run-from-cache policy,
/// _widest_vector_sizes in rocke/library/benchmarks/common/benchmark_implicit_gemm_conv.py,
/// which hipkernel:Gfx942ConvFwdAot follows. One flag per entry, in order.
///
/// "Widest" is taken per kernel kind, the epilogue and the merge degree (no packaged
/// instance is split-K, two-stage or stream-K): of each kind, only the instances whose
/// (a + b + c, a, b, c) is the greatest serving one run. Before that, default-epilogue
/// instances drop out where a serving instance of the same merge degree stores wider
/// than 1. A merged instance records its derived widths as 0, so every serving merged
/// instance of its kind runs. gfx942_conv_fwd_aot_runtime_candidates in
/// convolution_forward.py is the same policy.
inline std::vector<bool> widestPerKindSelection(const std::vector<AotSelectionKey>& serving)
{
    using Rank = std::array<int64_t, 4>;
    const auto rank = [](const AotSelectionKey& key) {
        return Rank{key.vectorA + key.vectorB + key.vectorC, key.vectorA, key.vectorB, key.vectorC};
    };
    std::map<int64_t, int64_t> widestStore;
    for(const auto& key : serving)
    {
        auto& widest = widestStore[key.groupMerge];
        widest = std::max(widest, key.vectorC);
    }
    std::vector<bool> keep(serving.size());
    std::map<std::pair<std::string, int64_t>, Rank> best;
    for(size_t i = 0; i < serving.size(); ++i)
    {
        keep[i] = !(serving[i].epilogue == "default" && widestStore[serving[i].groupMerge] > 1);
        if(keep[i])
        {
            const auto kind = std::make_pair(serving[i].epilogue, serving[i].groupMerge);
            const auto found = best.find(kind);
            if(found == best.end() || found->second < rank(serving[i]))
            {
                best[kind] = rank(serving[i]);
            }
        }
    }
    for(size_t i = 0; i < serving.size(); ++i)
    {
        keep[i] = keep[i]
                  && best.at(std::make_pair(serving[i].epilogue, serving[i].groupMerge))
                         == rank(serving[i]);
    }
    return keep;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
