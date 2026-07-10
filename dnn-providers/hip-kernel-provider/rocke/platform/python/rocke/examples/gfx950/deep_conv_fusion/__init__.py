# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 deep-conv-fusion case study: conv0 -> ReLU -> 1x1 conv1 -> ReLU ->
2x2 stride-2 maxpool fused into a single kernel with no HBM intermediates.

Scripts:
  deep_fused_conv_pool_verify -- build/verify harness + numpy reference
  compare_pool_tile_configs   -- A/B timing of pool-tile / config variants
  profile_best_config         -- single blocking dispatch for rocprofv3 capture

The accompanying ``*.md`` notes in this directory are the experiment log /
analysis for the optimization levers (LDS handoff elimination, barrier merge,
m-decode bypass, async/direct probes). These pin ``arch="gfx950"`` and drive
``rocke.instances.gfx950.deep_fused_conv_pool``.
"""
