# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Automatic selection of LoopModel parameters (`LoopModelAutoSelect`).

  space      the values each free parameter may take
  evaluate   tier 1 (θ, budget) and tier 2 (GIR timeline) judgement of one candidate
  resources  the GPRs a candidate needs, counted without alignment
  hw         gfx1250 costs, mirrored from StinkyTofu
  perf       the nest walk (tier 0), the analytic cost (tier 1), the steady-body timeline (tier 2)
  asmsim     tier 3: the emitted steady loop timed in order (Expand CLI only)
  search     the split pre-pass and the order + register search
  report     what was chosen, why, and what came next
"""
