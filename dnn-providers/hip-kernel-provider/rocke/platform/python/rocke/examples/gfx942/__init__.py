# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 (CDNA3 / MI300) example showcases (hybrid layout).

These pin ``arch="gfx942"`` and reuse the arch-neutral harnesses in
``examples/common/`` so they add only gfx942-specific specs/configs without
duplicating build+verify plumbing. See
``dsl_docs/architecture/multi_arch_data_layout.md``.
"""
