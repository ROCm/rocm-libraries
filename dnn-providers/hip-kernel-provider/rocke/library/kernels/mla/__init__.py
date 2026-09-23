# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MLA (Multi-head Latent Attention) kernel definitions.

Family-scoped rather than arch-scoped, mirroring ``library/builders/mla/`` --
the arch lives in the module name (``mla_prefill_gfx942``) so one family
directory can hold several targets without a per-arch package each.
"""
