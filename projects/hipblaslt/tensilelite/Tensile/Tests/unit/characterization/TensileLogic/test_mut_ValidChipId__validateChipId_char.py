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

"""Mutation-killing characterization test for ``_validateChipId``.

Pins the current end-to-end behavior of the top-level chip-ID validator across
its five reachable outcomes. The module is imported by dotted path
(``from Tensile.TensileLogic.ValidChipId import _validateChipId``) so that each
call is routed through the mutmut trampoline; the sibling ``test_ValidChipId.py``
loads the module with ``importlib.spec_from_file_location`` against the source
path, which bypasses the trampoline and therefore never exercises the mutants.

Each test drives one branch/return site and asserts the exact return value and
the exact ``stderr`` text the function emits today (via ``_reportChipIdFailure``).
Behavior is pinned as it runs now, including two non-obvious facts a later
refactor would have to revisit:

  * ``_extractArchInfo`` rejects an empty Device list before ``_validateChipId``
    ever reaches its own ``not arch_info.DeviceIds`` guard, so that guard's
    reporting branch is unreachable dead code today.
  * an invalid device predicate surfaces through the *inner* ``except
    (LogicFileError, ValueError)`` as ``ValidChipId failed (ValueError): ...``,
    not through the extract-time ``except LogicFileError``.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import _validateChipId

pytestmark = pytest.mark.unit


def _writeLogic(path: Path, *, gfx="gfx950", name="gfx950", devices="Device 75a0", lines=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines is None:
        lines = ["- MinimumRequiredVersion: 4.33.0", f"- {name}", f"- {gfx}", f"- [{devices}]"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_valid_base_logic_returns_true_silently(tmp_path, capsys):
    """A well-formed gfx950 base-directory logic file with a valid default chip
    ID passes every gate and returns ``True`` with no diagnostics.

    Reaches the final ``return True``. Distinguishes the mutants that break the
    happy path: ``placement_path``/``report_path`` set to ``None`` (raise at
    placement), ``arch_info = None`` (attribute error), the dropped/``None``
    ``_extractArchInfo`` arguments, the negated ``not arch_info.DeviceIds`` guard
    (mutant enters the empty-list branch and returns ``False``), the broken
    ``_verifyPredicate`` and ``set(...)`` calls, every broken
    ``_validateChipIdPlacement`` argument, ``_validateChipIdPlacement`` with a
    ``None`` gfx (yields a placement error), and the final ``return True`` flip.
    """
    logic = _writeLogic(tmp_path / "gfx950" / "gfx950" / "Equality" / "logic.yaml",
                        devices="Device 75a0")
    result = _validateChipId(logic)
    assert result is True
    assert capsys.readouterr().err == ""


def test_non_gated_arch_returns_true_before_validation(tmp_path, capsys):
    """A non-chip-ID-gated architecture (gfx942) short-circuits to ``True`` at the
    ``supportsChipIdPredicate`` guard without inspecting device IDs or placement.

    Reaches ``return True`` at the ``not supportsChipIdPredicate`` guard. Kills
    the flipped early ``return True`` -> ``return False`` (mutant 18) and the
    negated ``supportsChipIdPredicate`` guard (mutant 16, which then falls through
    to a gfx942 placement error and returns ``False``).
    """
    logic = _writeLogic(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "logic.yaml",
                        gfx="gfx942", name="aquavanjaram", devices="Device 20cu")
    result = _validateChipId(logic)
    assert result is True
    assert capsys.readouterr().err == ""


def test_variant_directory_missing_own_chip_id_reports_placement(tmp_path, capsys):
    """A gfx950_id75a3 variant directory whose YAML omits id=75a3 fails placement
    and returns ``False`` with the 'must contain' diagnostic naming the file.

    Reaches the ``if placement_error`` -> report -> ``return False`` path. Kills
    ``placement_error = None`` (mutant 31, skips the branch -> ``True``), the
    negated ``supportsChipIdPredicate`` guard passing ``None`` (mutant 17,
    short-circuits to ``True``), the ``_reportChipIdFailure`` argument mutants in
    this branch (``None`` path, ``None`` detail, dropped args), and the flipped
    ``return False`` -> ``return True`` (mutant 42).
    """
    logic = _writeLogic(tmp_path / "gfx950" / "gfx950_id75a3" / "Equality" / "logic.yaml",
                        devices="Device 75b0")
    result = _validateChipId(logic)
    assert result is False
    assert capsys.readouterr().err == (
        f"Error: id=75a3 directory must contain id=75a3 in the YAML Device list "
        f"(file: {logic})\n"
    )


def test_extract_failure_reports_via_outer_except(tmp_path, capsys):
    """A logic file with only three list-format header lines fails
    ``_extractArchInfo``'s list-length check and returns ``False`` through the
    extract-time ``except LogicFileError``.

    Reaches the first ``except`` -> report -> ``return False``. Kills the
    ``report_path``/``placement_path`` ``None``/``and`` mutants (which drop the
    real file path from the message), the extract-except
    ``_reportChipIdFailure`` argument mutants (``None`` path, ``None`` detail,
    dropped args), and the flipped ``return False`` -> ``return True`` (mutant
    15). The inner parser message is owned by ``_extractArchInfo`` and pinned by
    prefix only.
    """
    logic = _writeLogic(tmp_path / "gfx950" / "gfx950" / "Equality" / "logic.yaml",
                        lines=["- MinimumRequiredVersion: 4.33.0", "- gfx950", "- gfx950"])
    result = _validateChipId(logic)
    assert result is False
    err = capsys.readouterr().err
    assert err.startswith(
        "Error: Chip ID validation failed: Expected at least 4 list-format header lines"
    )
    assert f"(file: {logic})\n" == err[err.index(" (file: ") + 1:]


def test_invalid_predicate_reports_via_inner_except(tmp_path, capsys):
    """A device ID not associated with the architecture (74a0 on gfx950) passes
    extraction (``validateDeviceIds=False``) but fails ``_verifyPredicate``,
    surfacing through the inner ``except (LogicFileError, ValueError)``.

    Reaches the inner ``except`` -> report -> ``return False``. Kills
    ``validateDeviceIds=True`` (mutant 10, which would raise at extract time and
    report the extract-except message instead), the inner-except
    ``_reportChipIdFailure`` argument mutants (``None`` path, ``None`` detail,
    dropped args), ``type(None).__name__`` -> ``NoneType`` (mutant 48), and the
    flipped final ``return False`` -> ``return True`` (mutant 49). The exact
    ``ValidChipId failed (ValueError): ...`` wrapper distinguishes all of them.
    """
    logic = _writeLogic(tmp_path / "gfx950" / "gfx950" / "Equality" / "logic.yaml",
                        devices="Device 74a0")
    result = _validateChipId(logic)
    assert result is False
    assert capsys.readouterr().err == (
        f"Error: ValidChipId failed (ValueError): Invalid predicate: id=74a0: "
        f"device ID is not associated with gfx950 (file: {logic})\n"
    )
