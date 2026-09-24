# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side drivers for the DSA lightning indexer.

Arch-neutral (numpy) host lanes shared by the gfx942 and gfx950 kernels:

* ``hostpack`` -- torch-free numpy input generation, bf16 packing, and the score
  oracle (``ref_indexer_scores``). This is the correctness reference the numeric
  tests grade the kernel against; it runs anywhere, no GPU required.
* ``manifest`` -- the ``run_manifest`` adapter (the on-GPU lane): it allocates
  device buffers, launches the kernel, and grades the scores against the oracle,
  and also exposes the kernel to the cluster benchmark CLI. Needs a matching
  device to actually run.

Nothing is exported at package level; import the submodule you need.
"""
