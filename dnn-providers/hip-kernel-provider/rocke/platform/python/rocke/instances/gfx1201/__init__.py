# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 (RDNA4, wave32, WMMA) arch-specific instance builders (hybrid layout).

Put a kernel here only when its *algorithm* genuinely exploits a gfx1201-only
capability in a way that changes the kernel structure versus the shared
``instances/common/`` version. Shared, arch-polymorphic kernels (e.g. the
deep-fused conv/pool body, driven by the resolved ``MmaOp``) live in
``instances/common/``; this module only pins the WMMA geometry.

The gfx1201 deep-fused conv/pool builder moved to the rocke *library*
convolution vertical: :mod:`kernels.gfx1201.deep_fused_conv_pool`.
"""
