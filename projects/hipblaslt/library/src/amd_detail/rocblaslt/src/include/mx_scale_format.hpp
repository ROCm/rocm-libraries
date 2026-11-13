// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

#include <Tensile/MXScaleFormat.hpp>

#include <algorithm>
#include <cstddef>

// What a problem's A and B scale modes require of a kernel's MX scale layout.
enum class rocblaslt_mx_scale_need
{
    Any, // no MX scales
    Natural, // packed MX scales: NoSwizzle or InMemorySwizzle
    PreSwizzled, // Block_32_UE8M0_32_8_EXT: HostPreSwizzle
    Conflict, // packed on one side, pre-swizzled on the other: no layout
    PreSwizzled128x4, // Block_32_UE8M0_128_4_EXT: HostPreSwizzle128x4
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
    case Scale::Block_32_UE8M0_128_4_EXT:
        return rocblaslt_mx_scale_need::PreSwizzled128x4;
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
    using Layout = TensileLite::MXScaleFormat;
    switch(need)
    {
    case rocblaslt_mx_scale_need::Any:
        return true;
    case rocblaslt_mx_scale_need::Natural:
        return kernelLayout == static_cast<int>(Layout::NoSwizzle)
               || kernelLayout == static_cast<int>(Layout::InMemorySwizzle);
    case rocblaslt_mx_scale_need::PreSwizzled:
        return kernelLayout == static_cast<int>(Layout::HostPreSwizzle);
    case rocblaslt_mx_scale_need::Conflict:
        return false;
    case rocblaslt_mx_scale_need::PreSwizzled128x4:
        return kernelLayout == static_cast<int>(Layout::HostPreSwizzle128x4);
    }
    return false;
}

inline const char* rocblaslt_mx_scale_mismatch_message(rocblaslt_mx_scale_need need)
{
    switch(need)
    {
    case rocblaslt_mx_scale_need::Natural:
        return "packed MX scales need a NoSwizzle or InMemorySwizzle kernel";
    case rocblaslt_mx_scale_need::PreSwizzled:
        return "Block_32_UE8M0_32_8_EXT scales need a HostPreSwizzle kernel";
    case rocblaslt_mx_scale_need::PreSwizzled128x4:
        return "Block_32_UE8M0_128_4_EXT scales need a HostPreSwizzle128x4 kernel";
    case rocblaslt_mx_scale_need::Conflict:
        return "scale A and scale B need different MX scale layouts";
    default:
        return "kernel MX scale layout does not match the scale mode";
    }
}

// search(n) returns the top n candidates in rank order, and drop() refuses
// candidates. While fewer than requested survive and the search returned all
// it was asked for, the request grows by the number dropped from that result,
// so it doubles when every candidate is dropped: a search that keeps nothing
// costs about log2(library size) calls, all for the problem's own type.
// Returns at most requested candidates, in rank order.
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
    if(kept.size() > static_cast<size_t>(requested))
        kept.resize(requested);
    return kept;
}
