# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Profile registry tests for the public LDS conflict expert."""

import pytest
from rocke.analysis.lds.registry import (
    UnsupportedLdsTargetError,
    registered_targets,
    resolve_profile,
)


@pytest.mark.parametrize("target", ["gfx90a", "gfx950"])
def test_registry_selects_explicit_profile(target):
    profile = resolve_profile(target)

    assert profile.identity.target == target
    assert profile.identity.profile_version == 1
    assert registered_targets() == ("gfx90a", "gfx950")


@pytest.mark.parametrize("target", ["gfx942", "GFX90A", " gfx90a ", ""])
def test_registry_rejects_unknown_targets_without_fallback(target):
    with pytest.raises(UnsupportedLdsTargetError, match="unsupported LDS target"):
        resolve_profile(target)


def test_registry_rejects_non_string_target():
    with pytest.raises(TypeError, match="target must be a string"):
        resolve_profile(90)
