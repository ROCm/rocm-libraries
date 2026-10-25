# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor-network monokernels (gfx950, EXPERIMENTAL: not reached by any dispatcher).

:mod:`.circuit_kernel` runs a whole contraction path -- a quantum-circuit
network, every gate one extent-2-mode contraction -- in one persistent launch
(phases of gates applied as MFMA blocks on LDS tiles, a grid barrier between
phases). :mod:`.circuit_plan` holds the host planners and an f64 reference,
:mod:`.circuit_emulate` a lane-exact NumPy emulation of the kernel's index
math. Python engine builder only; the emitted IR uses mirrored ops plus empty
``inline_asm`` copies.
"""
