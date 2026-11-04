// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

#include <algorithm>
#include <cstddef>

// A TensileLite kernel's MX scale layout, as serialized in
// problemType.mxScaleFormat. Only HostPreSwizzle reads scales the host has
// already swizzled. The others read the scale tensor as the caller packs it.
enum class rocblaslt_mx_scale_layout : int
{
    NoSwizzle       = 0,
    HostPreSwizzle  = 1,
    InMemorySwizzle = 2,
};

// What a problem's A and B scale modes require of a kernel's MX scale layout.
enum class rocblaslt_mx_scale_need
{
    Any, // no MX scales
    Natural, // packed MX scales: any layout but HostPreSwizzle
    PreSwizzled, // Block_32_UE8M0_32_8_EXT: HostPreSwizzle only
    Conflict, // packed on one side, pre-swizzled on the other: no layout
};

inline rocblaslt_mx_scale_need
    rocblaslt_mx_scale_need_for(RocblasltContractionProblem::ScalingFormat scale)
{
    using Scale = RocblasltContractionProblem::ScalingFormat;
    switch(scale)
    {
    case Scale::Block_32_UE8M0:
    case Scale::Block_16_UE8M0:
    case Scale::Block_32_UE4M3:
    case Scale::Block_16_UE4M3:
    case Scale::Block_32_UE5M3:
    case Scale::Block_16_UE5M3:
        return rocblaslt_mx_scale_need::Natural;
    case Scale::Block_32_UE8M0_32_8_EXT:
        return rocblaslt_mx_scale_need::PreSwizzled;
    default:
        return rocblaslt_mx_scale_need::Any;
    }
}

inline rocblaslt_mx_scale_need rocblaslt_mx_scale_need_combine(rocblaslt_mx_scale_need a,
                                                               rocblaslt_mx_scale_need b)
{
    if(a == rocblaslt_mx_scale_need::Any)
        return b;
    if(b == rocblaslt_mx_scale_need::Any || a == b)
        return a;
    return rocblaslt_mx_scale_need::Conflict;
}

inline rocblaslt_mx_scale_need
    rocblaslt_mx_scale_need_for(RocblasltContractionProblem::ScalingFormat scaleA,
                                RocblasltContractionProblem::ScalingFormat scaleB)
{
    return rocblaslt_mx_scale_need_combine(rocblaslt_mx_scale_need_for(scaleA),
                                           rocblaslt_mx_scale_need_for(scaleB));
}

inline bool rocblaslt_mx_scale_layout_ok(rocblaslt_mx_scale_need need, int kernelLayout)
{
    const bool preSwizzled
        = kernelLayout == static_cast<int>(rocblaslt_mx_scale_layout::HostPreSwizzle);
    switch(need)
    {
    case rocblaslt_mx_scale_need::Any:
        return true;
    case rocblaslt_mx_scale_need::Natural:
        return !preSwizzled;
    case rocblaslt_mx_scale_need::PreSwizzled:
        return preSwizzled;
    case rocblaslt_mx_scale_need::Conflict:
        return false;
    }
    return false;
}

inline const char* rocblaslt_mx_scale_mismatch_message(rocblaslt_mx_scale_need need)
{
    switch(need)
    {
    case rocblaslt_mx_scale_need::Natural:
        return "packed MX scales cannot use a HostPreSwizzle kernel";
    case rocblaslt_mx_scale_need::PreSwizzled:
        return "Block_32_UE8M0_32_8_EXT scales need a HostPreSwizzle kernel";
    case rocblaslt_mx_scale_need::Conflict:
        return "scale A and scale B need different MX scale layouts";
    default:
        return "kernel MX scale layout does not match the scale mode";
    }
}

// search(n) returns the top n candidates in rank order. Candidates that drop()
// refuses are removed, and the search is widened by that many until requested
// candidates remain or the search returns fewer than it was asked for.
template <typename Search, typename Drop>
auto rocblaslt_find_top_kept(Search&& search, int requested, Drop&& drop)
{
    auto kept = search(requested);
    for(int request = requested;;)
    {
        size_t const found = kept.size();
        kept.erase(std::remove_if(kept.begin(), kept.end(), drop), kept.end());
        size_t const dropped = found - kept.size();
        if(dropped == 0 || kept.size() >= static_cast<size_t>(requested)
           || found < static_cast<size_t>(request))
            break;
        request += static_cast<int>(dropped);
        kept = search(request);
    }
    return kept;
}
