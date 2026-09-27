# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Arch-neutral examples and harnesses (hybrid layout).

Examples here take ``--arch`` / ``--isa`` and run on any supported target. They
are imported by the per-arch showcases under ``examples/gfx942/`` and
``examples/gfx950/`` so arch-specific demos add only the spec/config they need,
without duplicating the build+verify plumbing.

See ``dsl_docs/architecture/multi_arch_data_layout.md`` → "Code Organization".
"""
