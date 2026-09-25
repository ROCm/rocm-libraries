# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side drivers for the DSA lightning indexer (gfx942).

Two lanes, matching the KDA builder package layout:

* ``hostpack`` -- torch-free numpy input generation, bf16 packing, and the score
  oracle (``ref_indexer_scores``). This is the correctness reference the numeric
  test grades the kernel against; it runs anywhere, no GPU required.
* ``indexer`` -- the on-GPU lane: build the kernel, launch it, and compare its
  scores against the oracle. Needs a gfx942 device.
* ``manifest`` -- the run_manifest adapter that exposes the kernel to the
  cluster benchmark CLI.

Nothing is exported at package level; import the submodule you need.
"""
