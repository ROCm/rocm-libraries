// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Shared by the MX scale relayout kernel (MxScaleRelayout.cpp, compiled at run time
// with HIPRTC) and the host code that launches it, so both use one parameter layout.
// HIPRTC offers only a few standard headers; <cstdint> is one of them.
#include <cstdint>

namespace hipblaslt_plugin::mx_scale_relayout
{

// Embedded source name and entry points of the kernel (see cmake/KernelEmbedding.cmake).
constexpr const char* SOURCE_NAME = "MxScaleRelayout.cpp";
// mxDataGenerator preSwizzleScalesGFX950: [mn][kb] padded to (32, 8) and viewed as
// (mn/32, 2, 16, kb/8, 2, 4), permuted (0, 3, 5, 2, 4, 1).
constexpr const char* GFX950_KERNEL = "MxScaleRelayoutGfx950";
// mxDataGenerator preSwizzleScalesGFX1250 (32-element blocks): [mn][kb] padded to
// kb % 4 == 0 and permuted [mn][kb/4][4] -> [kb/4][mn][4].
constexpr const char* GFX1250_KERNEL = "MxScaleRelayoutGfx1250";

// Padding of the swizzled layouts: the gfx950 one pads the scale to whole tiles of 32
// MN rows by 8 K blocks (one 256-byte granule each), the gfx1250 one pads the K blocks
// to whole groups of 4.
constexpr uint32_t GFX950_TILE_MN = 32;
constexpr uint32_t GFX950_TILE_KB = 8;
constexpr uint32_t GFX1250_GROUP_KB = 4;

// One workgroup of THREADS lanes rewrites a TILE x TILE block of (mn, kb) scale bytes.
constexpr uint32_t TILE = 64;
constexpr uint32_t THREADS = 256;

// One scale rewrite: a packed scale tensor (`srcKContiguous`: [mn][kBlocks], otherwise
// [kBlocks][mn]) into a swizzled layout of outMn x outKb (padding included) at `dst`.
// Padding bytes are written as zero. `src` and `dst` must not overlap.
struct Params
{
    const uint8_t* src;
    uint8_t* dst;
    uint32_t mn;
    uint32_t kBlocks;
    uint32_t outMn;
    uint32_t outKb;
    bool srcKContiguous;
    bool wordLoads; // source rows are whole, 4-byte aligned words
    bool wordStores; // `dst` is 4-byte aligned
};

// Up to two rewrites with the same layout in one launch (the A and B scales of one
// GEMM): blocks [0, firstBlockOfSecond) tile `first`, the rest tile `second`.
struct Batch
{
    Params first;
    Params second;
    uint32_t tilesKFirst; // tiles along first.outKb
    uint32_t tilesKSecond; // tiles along second.outKb
    uint32_t firstBlockOfSecond;
};

} // namespace hipblaslt_plugin::mx_scale_relayout
