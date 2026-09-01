# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The kernels this engine's packs compile at plan-build time, named by stem.
#
# Two targets embed them, because embedding is per-target: the provider, for the packs it
# registers, and the test binary, which links those same packs and so needs its own copy.
# One list rather than two, since a pack whose kernel reached only one of them fails at
# plan build with a missing embedded source -- a runtime error, from a set of files CMake
# had in hand all along.
#
# Resolved against this file's own directory, so an includer's location does not matter.
set(HIPDNN_INGESTOR_PACK_KERNEL_DIR "${CMAKE_CURRENT_LIST_DIR}/kernels")
set(HIPDNN_INGESTOR_PACK_KERNELS PointwiseAdd PointwiseMul PointwiseSub ConvFwd)
