# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The kernels this engine's packs compile at plan-build time, named by stem.
#
# Embedding is per-target. One target embeds this set: the unit-test binary, which links
# these packs statically.
#
# Resolved against this file's own directory, so an includer's location does not matter.
set(HIPDNN_INGESTOR_PACK_KERNEL_DIR "${CMAKE_CURRENT_LIST_DIR}/test_descriptors/unit/pointwise/kernels")
set(HIPDNN_INGESTOR_PACK_KERNELS PointwiseAdd PointwiseMul PointwiseSub)
