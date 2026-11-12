// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// The cluster workgroups a TDM pipeline's GetTDMWorkgroupMask multicasts a tile to: those in the
// same M tile (A), the same N tile (B), or either.
enum class MultiCastDirection
{
    kM,
    kN,
    kMN
};

} // namespace ck_tile
