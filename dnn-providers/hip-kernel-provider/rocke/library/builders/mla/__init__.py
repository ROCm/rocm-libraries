# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Architecture-neutral MLA (Multi-head Latent Attention) references.

This is the first *family*-scoped rather than arch-scoped directory under
``library/builders/`` -- deliberate, because the numerical reference is
arch-neutral (see ``DESIGN.md`` section 8.1 layout note). Arch-specific parity
and bench entry points still belong under ``builders/gfx942/attention/`` and
``builders/gfx950/attention/`` next to their siblings, not here.
"""
