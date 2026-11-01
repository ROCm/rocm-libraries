# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor-contraction (hipTensor-style) kernel families.

* :mod:`.gett` -- native GETT: one launch reads both operands in place
  through per-mode strides and writes D through a permuted alpha / beta
  epilogue (gfx950, f16 / bf16; the ring and MN-layout bodies are mirrored by the C++
  engine, ``rocke/instance_tensor_gett.h``).
* :mod:`.gett_plan` -- the host-side planner (mode orders, K split, mixed-radix
  row / column digits).
"""

from .gett import (  # noqa: F401
    TENSOR_GETT_ARCHS,
    TENSOR_GETT_DTYPES,
    GettHooks,
    GettKnobs,
    TensorGettSpec,
    build_tensor_gett,
    is_valid_spec,
    kc_body_errors,
    kc_body_trait,
    kc_operand_modes,
    launch_operands,
    mn_body_errors,
    mn_body_trait,
    tensor_gett_block,
    tensor_gett_grid,
    tensor_gett_signature,
)
from .gett_plan import GettPlan, plan_gett  # noqa: F401
