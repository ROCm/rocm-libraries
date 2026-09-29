# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Built-in architecture profiles for LDS conflict prediction."""

from .gfx90a import GFX90A_PROFILE, Gfx90aProfile


__all__ = ["BUILTIN_PROFILES", "GFX90A_PROFILE", "Gfx90aProfile"]


BUILTIN_PROFILES = (GFX90A_PROFILE,)
