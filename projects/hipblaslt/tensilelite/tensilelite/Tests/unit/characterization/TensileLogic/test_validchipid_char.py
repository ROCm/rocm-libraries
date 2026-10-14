################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the pure helpers of ``TensileLogic.ValidChipId``.

The end-to-end ``_validateChipId`` flow is already exercised by the existing
``test_ValidChipId.py``; the baseline left two helper details uncovered:

* ``_chipIdDirFromPath`` line 114 — the fall-through when *no* path component
  matches the ``gfx`` / ``gfx_id<chip>`` / ``gfx_<chip>`` patterns (a logic
  file whose placement path has no arch directory at all).
* ``_fallbackFamily`` branch 72->71 — the loop arm where a candidate source is
  *not* in the target arch (so it is skipped). For ``gfx950`` every source
  shares the same fallback root, so the skip arm is never taken with a
  realistic ``gfx``; we drive it with an intentionally cross-arch pair
  (``id=75a3`` against ``gfx942``) to pin the skip behaviour.

Two placement branches in ``_validateChipIdPlacement`` (lines 129 and 155) are
**unreachable** given the production registries — see ``resistance.md``.

These helpers are pure string/set logic over the chip-ID registries, so the
snapshots pin the registry-derived behaviour: a registry edit that changes a
family or chip-ID set fails loudly here. ``_ChipIdDir`` results are snapshotted
via ``._asdict()``; set-returning helpers are sorted first.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import (
    _archChipIds,
    _chipIdDirFromPath,
    _chipIdKey,
    _defaultChipIds,
    _fallbackFamily,
    _sourceChipIds,
)

pytestmark = pytest.mark.unit


# --- _chipIdDirFromPath -----------------------------------------------------

@pytest.mark.parametrize("name,parts", [
    ("variant_id_dir", "gfx950/gfx950_id75a3/Equality/logic.yaml"),
    ("malformed_id_dir", "gfx950/gfx950_75a3/Equality/logic.yaml"),
    ("base_arch_dir", "gfx950/gfx950/Equality/logic.yaml"),
    # No gfx950 component anywhere in the path -> end-of-function fallback (L114).
    ("no_arch_component", "some/random/place/logic.yaml"),
    # Nearest chip-ID dir wins over an enclosing plain 'gfx950' ancestor.
    ("nearest_wins", "gfx950/checkout/gfx950/gfx950_id75a3/Equality/logic.yaml"),
])
def test_chip_id_dir_from_path(name, parts, snapshot):
    result = _chipIdDirFromPath("gfx950", Path(parts))
    assert result._asdict() == snapshot(name=name)


# --- _fallbackFamily --------------------------------------------------------

def test_fallback_family_default_is_singleton(snapshot):
    # A default chip ID has no fallback entry -> family is just itself.
    assert sorted(_fallbackFamily("id=75a0", "gfx950")) == snapshot


def test_fallback_family_source_includes_siblings(snapshot):
    # A source chip ID expands to itself + its fallback + all siblings that
    # share that fallback root (every gfx950 source shares id=75a0).
    assert sorted(_fallbackFamily("id=75a3", "gfx950")) == snapshot


def test_fallback_family_cross_arch_skip_arm(snapshot):
    # Intentionally cross-arch: id=75a3's siblings are gfx950 chip IDs, none of
    # which are in gfx942's arch set, so the expansion loop skips every
    # candidate (branch 72->71). Family stays {chip, direct fallback}.
    assert sorted(_fallbackFamily("id=75a3", "gfx942")) == snapshot


# --- registry-derived set helpers -------------------------------------------

def test_chip_id_key(snapshot):
    # Lower-cases and prefixes; pins the key format other helpers depend on.
    assert {
        "lower": _chipIdKey("75a3"),
        "upper": _chipIdKey("75A3"),
    } == snapshot


@pytest.mark.parametrize("gfx", ["gfx950", "gfx942", "gfx900"])
def test_arch_source_default_partition(gfx, snapshot):
    # arch = source ∪ default for each arch (gfx900 has no registry entries ->
    # all three empty). Pins the partition derived from the registries.
    assert {
        "arch": sorted(_archChipIds(gfx)),
        "source": sorted(_sourceChipIds(gfx)),
        "default": sorted(_defaultChipIds(gfx)),
    } == snapshot(name=gfx)
