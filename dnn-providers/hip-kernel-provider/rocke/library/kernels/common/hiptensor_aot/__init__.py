# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic kernels of hipTensor's rocKE backend (built ahead of time into its kernel pack).

* :mod:`.ew` -- strided elementwise kernel (permute, elementwise binary / trinary, contraction
  operand packing and epilogue);
* :mod:`.reduce` -- strided reduction kernels (row, col, split finalize);
* :mod:`.host` -- the launch-argument rules hipTensor's C++ runtime implements;
* :mod:`.jit_ew` -- the problem-specialized form of :mod:`.ew` for hipTensor's JIT (recipes rolled
  at build time, replayed by the C++ recipe VM), and its vector classes with the geometry as
  kernel arguments;
* :mod:`.ewt` -- tiled elementwise kernel for transposing layouts;
* :mod:`.dgett` -- direct GETT: any contraction read and written in place, no workspace;
* :mod:`.rgett` -- rocKE's ring GETT body with runtime index maps (f16 / bf16, gfx950);
* :mod:`.gett_host` -- the planner rules of the two GETT kernels.

Every kernel takes its problem as kernel arguments (rank bucket R8), so one compiled kernel
serves every problem of its dtype row. Python engine only, like the other library kernels.
"""
