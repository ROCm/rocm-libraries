# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exact target selection for LDS conflict profiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .model import ProfileIdentity


__all__ = [
    "LdsProfile",
    "UnsupportedLdsTargetError",
    "registered_targets",
    "resolve_profile",
]


class UnsupportedLdsTargetError(ValueError):
    """Raised when no explicit LDS conflict profile exists for a target."""


@runtime_checkable
class LdsProfile(Protocol):
    """Architecture-specific rules consumed by the common predictor.

    Collision keys and lane phases are empirical prediction rules. They do not
    assert a unique physical implementation.
    """

    identity: ProfileIdentity
    supported_wave_sizes: frozenset[int]
    supported_opcodes: frozenset[str]

    def phase_key(self, opcode: str, lane: int) -> tuple[int, ...]:
        """Return the independent execution phase for one lane."""

    def collision_key(self, opcode: str, lds_byte_address: int) -> int:
        """Return the empirical address-equivalence key for one access."""


def _profiles() -> Sequence[LdsProfile]:
    # Local import keeps profile modules independent of registry construction.
    from .profiles import BUILTIN_PROFILES

    return BUILTIN_PROFILES


def _profile_map() -> dict[str, LdsProfile]:
    profiles: dict[str, LdsProfile] = {}
    for profile in _profiles():
        target = profile.identity.target
        if target in profiles:
            raise RuntimeError(f"duplicate LDS profile target: {target}")
        profiles[target] = profile
    return profiles


def registered_targets() -> tuple[str, ...]:
    """Return explicitly registered profile targets in stable order."""

    return tuple(sorted(_profile_map()))


def resolve_profile(target: str) -> LdsProfile:
    """Select an exact target profile without aliases or fallback behavior."""

    if not isinstance(target, str):
        raise TypeError("target must be a string")
    profiles = _profile_map()
    try:
        return profiles[target]
    except KeyError as exc:
        choices = ", ".join(sorted(profiles))
        raise UnsupportedLdsTargetError(
            f"unsupported LDS target {target!r}; registered targets: {choices}"
        ) from exc
