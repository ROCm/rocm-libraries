# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Arch-polymorphic instance builders (hybrid layout).

Kernels here have a single algorithm that works across every supported gfx
target, parametrizing only over target-selected values (the MMA ``op_id`` from
``ArchTarget.mma``, waitcnt encoding, datalayout, wave primitives). A builder in
``common/`` must contain **no** ``if arch == ...`` around structural control
flow — only target lookups.

Kernels whose *algorithm* genuinely diverges per arch live in
``instances/gfx950/`` / ``instances/gfx942/`` instead. Per-family validity rules
live in ``instances/<family>/policy.py``.

See ``dsl_docs/architecture/multi_arch_data_layout.md``.
"""
