# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit gfx942 LDS conflict profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ProfileIdentity
from .gfx90a import Gfx90aProfile


__all__ = ["GFX942_PROFILE", "Gfx942Profile"]


@dataclass(frozen=True)
class Gfx942Profile(Gfx90aProfile):
    """Public gfx942 identity using the reviewed shared prediction rules."""

    identity: ProfileIdentity = field(
        default_factory=lambda: ProfileIdentity(target="gfx942", profile_version=1)
    )


GFX942_PROFILE = Gfx942Profile()
