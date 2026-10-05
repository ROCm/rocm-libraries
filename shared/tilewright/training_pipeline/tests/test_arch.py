# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from lib.arch import canonical_arch


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gfx950", "gfx950"),
        ("gfx942", "gfx942"),
        ("gfx1201", "gfx1201"),
        ("gfx90a", "gfx90a"),
        ("gfx1250", "gfx1250"),
        ("gfx1250v0", "gfx1250"),
        ("gfx1250v12", "gfx1250"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx1250v0:xnack+", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx942-xnack+", "gfx942"),
        ("gfx942-sramecc+-xnack-", "gfx942"),
        ("  gfx950 ", "gfx950"),
    ],
)
def test_canonical_arch(name, expected):
    assert canonical_arch(name) == expected


@pytest.mark.parametrize("empty", ["", None])
def test_canonical_arch_passes_empty_through(empty):
    assert canonical_arch(empty) == empty
