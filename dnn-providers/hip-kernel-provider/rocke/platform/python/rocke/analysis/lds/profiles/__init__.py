# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Built-in architecture profiles for LDS conflict prediction."""

from .gfx90a import GFX90A_PROFILE, Gfx90aProfile
from .gfx942 import GFX942_PROFILE, Gfx942Profile


__all__ = [
    "BUILTIN_PROFILES",
    "GFX90A_PROFILE",
    "GFX942_PROFILE",
    "Gfx90aProfile",
    "Gfx942Profile",
]


BUILTIN_PROFILES = (GFX90A_PROFILE, GFX942_PROFILE)
