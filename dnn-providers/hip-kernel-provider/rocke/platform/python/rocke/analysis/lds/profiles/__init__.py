# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Built-in architecture profiles for LDS conflict prediction."""

from .gfx90a import GFX90A_PROFILE, Gfx90aProfile
from .gfx950 import GFX950_PROFILE, Gfx950Profile

__all__ = [
    "BUILTIN_PROFILES",
    "GFX90A_PROFILE",
    "GFX950_PROFILE",
    "Gfx90aProfile",
    "Gfx950Profile",
]


BUILTIN_PROFILES = (GFX90A_PROFILE, GFX950_PROFILE)
