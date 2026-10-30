// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace hip_kernel_provider::kernel_ingestor_engine::gfx942_conv_fwd
{

/// What rocKE's runtime selection reads of one implicit-GEMM instance that already serves
/// the problem: its kind and its baked load and store widths.
struct AotSelectionKey
{
    /// The kind. rocKE's kind is (direction, epilogue, split_k, two_stage, group_merge);
    /// every packaged instance is forward with the default split-K and two-stage
    /// (make_conv_aot_config.py refuses any other), and a group-merged instance records
    /// derived widths (0) and is never compared (aotCandidateFilter keeps it), so only
    /// the epilogue varies.
    std::string epilogue;
    int64_t vectorA;
    int64_t vectorB;
    int64_t vectorC;
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
/// whole candidate set (hipkernel:Gfx942ConvFwdAot registers it as a CandidateFilterFn).
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx942_conv_fwd
