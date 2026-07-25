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

"""Mutation-killing characterization test for ``_validateChipIdPlacement``.

Pins the current behavior of the placement validator for ``gfx950`` (the only
chip-ID-aware architecture today). The test data is anchored on the live
``Tensile.Common.Architectures`` tables:

    arch    = id=75a0 id=75a2 id=75a3 id=75a8 id=75b0 id=75b2 id=75b3 id=75b8
    source  =         id=75a2 id=75a3 id=75a8 id=75b0 id=75b2 id=75b3 id=75b8
    default = id=75a0
    family(id=75a3) = all eight arch IDs

Each test drives one branch/return site of the function and asserts the exact
string (or ``None``) the function returns today. Every listed survivor mutant
(assignment-to-``None``, argument-to-``None``, dropped-argument, negated
condition, and ``sorted(None)`` literal mutations) changes at least one of these
observable return values or raises where the original returns cleanly, so each
assertion below distinguishes original from mutant. Behavior is pinned exactly
as it runs now, including the surprising cases (a base-logic path with no
chip-ID directory reports against the full arch list; a family violation lists
the whole fallback family). A later bugfix would have to update these tests.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import _validateChipIdPlacement

pytestmark = pytest.mark.unit


GFX = "gfx950"

ARCH_SORTED = [
    "id=75a0",
    "id=75a2",
    "id=75a3",
    "id=75a8",
    "id=75b0",
    "id=75b2",
    "id=75b3",
    "id=75b8",
]


def test_base_logic_disallowed_chip_id_reports_arch_list():
    """No chip-ID directory + a chip ID outside the arch set returns the base
    error listing the full sorted arch set and the found IDs.

    Reaches the ``not hasChipIdDir`` -> ``not issubset(arch_ids)`` return. Kills
    the ``arch_ids`` assignment mutants (mutant 1 ``None`` -> ``issubset(None)``
    raises; mutant 2 ``_archChipIds(None)`` -> empty arch list in the message),
    the ``chip_id_dir`` construction mutants (mutants 7-11 raise on line 125 or
    on ``.hasChipIdDir``), the negated outer guard (mutant 12, which falls
    through to the non-source branch with ``chipId=None``), the negated
    ``issubset`` guard (mutant 13, which returns ``None``), ``issubset(None)``
    (mutant 14 raises), and both ``sorted(None)`` literals in this message
    (mutants 15, 16 raise).
    """
    result = _validateChipIdPlacement(GFX, {"id=9999"}, Path("logic.yaml"))
    assert result == (
        f"base gfx950 logic may only declare chip IDs available for gfx950 "
        f"{ARCH_SORTED}; found ['id=9999']"
    )


def test_base_logic_allowed_chip_id_returns_none():
    """No chip-ID directory + only arch-valid chip IDs returns ``None``.

    Reaches the ``not hasChipIdDir`` -> ``issubset`` -> ``return None`` site.
    Distinguishes the ``arch_ids`` mutants (mutant 1 raises; mutant 2 empties the
    arch set so ``issubset`` fails and the base message is returned) and the
    negated ``issubset`` guard (mutant 13, which returns the base message here).
    """
    result = _validateChipIdPlacement(GFX, {"id=75a0"}, Path("logic.yaml"))
    assert result is None


def test_malformed_chip_id_directory_reports_format():
    """A ``gfx950_<chip>`` directory (missing the ``id`` token) is malformed and
    returns the format-requirement message naming the offending directory.

    Reaches the ``not isValidFormat`` return. Kills the negated
    ``isValidFormat`` guard (mutant 17), which would skip this branch and fall
    through to a clean ``None`` for this input.
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75a3"}, Path("gfx950_75a3/logic.yaml")
    )
    assert result == "chip-ID directory 'gfx950_75a3' must use gfx950_id<chip> format"


def test_non_source_chip_id_directory_reports_non_source():
    """A well-formed chip-ID directory whose chip ID is a default (non-source)
    fallback returns the non-source error.

    ``id=75a0`` is in the arch set but not a source fallback. Reaches the
    ``chipId not in source_ids`` return. Kills ``source_ids=None`` (mutant 3
    raises on ``in None``) and the negated membership guard (mutant 18, which
    skips to the default-fallback branch and returns that message instead).
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75a0"}, Path("gfx950_id75a0/logic.yaml")
    )
    assert result == "gfx950_id directory uses non-source chip ID id=75a0"


def test_chip_id_directory_missing_own_id_reports_must_contain():
    """A source chip-ID directory whose YAML omits that chip ID returns the
    'must contain' error.

    Directory ``gfx950_id75a3`` with device list ``{id=75b0}`` (a source, so it
    passes the non-source check) but missing ``id=75a3``. Reaches the
    ``chipId not in device_ids`` return. Kills ``source_ids=_sourceChipIds(None)``
    (mutant 4, empty -> non-source message), the negated non-source guard
    (mutant 18, returns non-source message), and the negated ``chipId in
    device_ids`` guard (mutant 19, which falls through to a clean ``None``).
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75b0"}, Path("gfx950_id75a3/logic.yaml")
    )
    assert result == "id=75a3 directory must contain id=75a3 in the YAML Device list"


def test_chip_id_directory_declares_default_fallback():
    """A source chip-ID directory that also declares a default fallback chip ID
    returns the default-fallback error listing those IDs.

    Directory ``gfx950_id75a3`` with device list ``{id=75a3, id=75a0}``; the
    default ``id=75a0`` is disallowed. Reaches the ``declared_default_ids``
    return. Kills ``default_ids=None`` / ``intersection(None)`` (mutants 5, 21
    raise), ``default_ids=_defaultChipIds(None)`` (mutant 6, empty -> returns
    ``None``), ``declared_default_ids=None`` (mutant 20, skips the branch ->
    ``None``), the negated ``chipId in device_ids`` guard (mutant 19, returns the
    must-contain message), and the ``sorted(None)`` literal in this message
    (mutant 22 raises).
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75a3", "id=75a0"}, Path("gfx950_id75a3/logic.yaml")
    )
    assert result == (
        "id=75a3 directory may not declare default fallback chip IDs ['id=75a0']"
    )


def test_chip_id_directory_out_of_fallback_family():
    """A source chip-ID directory declaring a chip ID outside its fallback
    family returns the family-violation error listing the sorted family.

    Directory ``gfx950_id75a3`` with device list ``{id=75a3, id=9999}``;
    ``id=9999`` is outside the family (all eight arch IDs). Reaches the
    ``not issubset(family)`` return. Kills ``family=None`` / ``issubset(None)``
    (mutants 23, 29 raise), ``_fallbackFamily(None, ...)`` (mutant 24 raises),
    ``_fallbackFamily(chipId, None)`` (mutant 25, shrinks the family so the
    reported list differs), the dropped-argument variants (mutants 26, 27
    raise), the negated ``issubset(family)`` guard (mutant 28, returns ``None``),
    and both ``sorted(None)`` literals in this message (mutants 30, 31 raise).
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75a3", "id=9999"}, Path("gfx950_id75a3/logic.yaml")
    )
    assert result == (
        f"id=75a3 directory may only declare chip IDs in fallback family "
        f"{ARCH_SORTED}; found ['id=75a3', 'id=9999']"
    )


def test_chip_id_directory_valid_family_member_returns_none():
    """A source chip-ID directory declaring its own ID plus another in-family
    source ID passes and returns ``None``.

    Directory ``gfx950_id75a3`` with device list ``{id=75a3, id=75b0}``. Both
    are in the family; ``id=75b0`` is a source (not a default), so no branch
    fires. Reaches the final ``return None``. Kills ``_fallbackFamily(chipId,
    None)`` (mutant 25), which shrinks the family to ``{id=75a0, id=75a3}`` so
    ``id=75b0`` becomes out-of-family and the function returns the family error
    instead of ``None``; also kills the negated ``issubset(family)`` guard
    (mutant 28), which returns the family error here.
    """
    result = _validateChipIdPlacement(
        GFX, {"id=75a3", "id=75b0"}, Path("gfx950_id75a3/logic.yaml")
    )
    assert result is None
