#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The eight tiles the degree model is scored at, and where each comes from.

The shipped dispatch tile (64x64x64) is a placeholder -- grouped_convolution.py
says so in as many words ("Hard-coded tile parameters (to be replaced by
sweep-derived tuning tables)") and it is used for *every* implicit-GEMM conv
request, not just depthwise. It is therefore a control, not a reference point.

The other seven are taken from Composable Kernel's tuned instance lists rather
than invented, on the grounds that a tile somebody already tuned for this exact
arrangement is better evidence than a tile chosen to span a grid. Two sources:

  * ``device_grouped_conv_fwd_xdl_merged_groups_instance.hpp`` -- CK's merged
    groups instances, the direct analogue of this feature. Six distinct tiles,
    every one of them with NPerBlock <= 64 and KPerBlock <= 32.
  * ``device_grouped_conv_fwd_xdl_{,comp_}instance.hpp`` -- CK's general
    grouped conv forward set, for breadth. 64x64x64 does not appear anywhere in
    it; exactly one of its ~290 instances uses KPerBlock = 64.

Format is ``MxNxK/warpMxwarpN/warpTileMN``. warp counts are chosen so the
block size and the per-wave Xdl counts match the CK instance being copied.
"""

TILES = (
    # spec                      source
    ("64x64x64/2x2/32", "rocke dispatch (placeholder, control)"),
    ("64x16x16/1x1/16", "CK merged-groups, most instances (9)"),
    ("64x16x32/1x1/16", "CK merged-groups (6)"),
    ("32x64x32/1x1/32", "CK merged-groups (3)"),
    ("128x32x32/4x1/32", "CK merged-groups V3 (4)"),
    ("64x64x32/2x2/16", "CK merged-groups (2) + CK general (33)"),
    ("128x64x32/2x2/32", "CK merged-groups (4) + CK general (20)"),
    ("256x32x64/4x1/32", "rocke 35-shape tile-swept winner"),
)


def specs():
    return [t for t, _ in TILES]


if __name__ == "__main__":
    for spec, src in TILES:
        print(f"{spec:<20} {src}")
