# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Profile registry tests for the public LDS conflict expert."""

import pytest

from rocke.analysis.lds.registry import (
    UnsupportedLdsTargetError,
    registered_targets,
    resolve_profile,
)


def test_registry_selects_explicit_gfx90a_profile():
    profile = resolve_profile("gfx90a")

    assert profile.identity.target == "gfx90a"
    assert profile.identity.profile_version == 1
    assert registered_targets() == ("gfx90a", "gfx942")


def test_registry_selects_explicit_gfx942_profile():
    profile = resolve_profile("gfx942")

    assert profile.identity.target == "gfx942"
    assert profile.identity.profile_version == 1


@pytest.mark.parametrize("target", ["gfx950", "GFX942", " gfx942 ", ""])
def test_registry_rejects_unknown_targets_without_fallback(target):
    with pytest.raises(UnsupportedLdsTargetError, match="unsupported LDS target"):
        resolve_profile(target)


def test_registry_rejects_non_string_target():
    with pytest.raises(TypeError, match="target must be a string"):
        resolve_profile(90)
