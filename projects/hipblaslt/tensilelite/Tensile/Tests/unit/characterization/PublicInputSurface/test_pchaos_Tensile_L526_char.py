################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: Tensile/Tensile.py lines 526 and 529.

Branch d8f43265b665a4b721072cb052d2fbd64d526645 (L526):
  Predicate: ``altFormat and len(configPaths) > 2``
  TRUE  -> printExit fires ("Only 1 or 2 config_files are accepted...")
  FALSE -> falls through to elif (L529) or continues

Branch covering L529:
  Predicate: ``not altFormat and len(configPaths) != 1``
  TRUE  -> printExit fires ("Only 1 config_file is accepted...")
  FALSE -> continues into executeStepsInConfig

Tests are CPU-only, deterministic, and pin ACTUAL observed behavior.
They do NOT exercise GPU dispatch or deep YAML parsing.
"""

import importlib
import os
import tempfile

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.Tensile")


# ---------------------------------------------------------------------------
# (1) Pure-helper: extracted predicate from L526
# ---------------------------------------------------------------------------

def alt_format_rejected(alt_format: bool, n_config_files: int) -> bool:
    """Mirror of Tensile/Tensile.py:526 predicate: ``altFormat and len(configPaths) > 2``.

    Returns True when the alternate-format branch fires printExit
    (alt format selected AND more than 2 config files supplied).
    """
    return alt_format and n_config_files > 2


def test_pure_helper_true_witness():
    # TRUE: altFormat=True, 3 config files -> predicate True -> printExit fires
    assert alt_format_rejected(True, 3) is True


def test_pure_helper_false_alt_format_false():
    # FALSE: altFormat=False, configPaths=3 -> short-circuit on altFormat
    assert alt_format_rejected(False, 3) is False


def test_pure_helper_false_too_few_configs():
    # FALSE: altFormat=True, configPaths=1 -> comparison >2 fails
    assert alt_format_rejected(True, 1) is False


def test_pure_helper_false_boundary_exactly_two():
    # FALSE: altFormat=True, configPaths=2 -> 2 > 2 is False (boundary)
    assert alt_format_rejected(True, 2) is False


def test_pure_helper_full_domain_enumeration():
    # Enumerate the seeded domain [False,True] x [0,3]
    expected = {
        (False, 0): False,
        (False, 1): False,
        (False, 2): False,
        (False, 3): False,
        (True, 0): False,
        (True, 1): False,
        (True, 2): False,
        (True, 3): True,   # unique TRUE point in the seeded domain
    }
    for (af, n), want in expected.items():
        got = alt_format_rejected(af, n)
        assert got is want, f"alt_format_rejected({af!r}, {n}) should be {want}, got {got}"


# ---------------------------------------------------------------------------
# (2) Real-entry pin: call Tensile.Tensile(userArgs) and observe SystemExit
# ---------------------------------------------------------------------------

def _make_fake_configs(tmpdir, n):
    """Create n zero-byte yaml files in tmpdir and return their paths."""
    paths = []
    for i in range(n):
        p = os.path.join(tmpdir, f"c{i}.yaml")
        open(p, "w").close()
        paths.append(p)
    return paths


def test_real_entry_l526_true_branch_triggers_system_exit(capsys):
    """altFormat=True + 3 config files -> L526 fires printExit -> SystemExit(-1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = _make_fake_configs(tmpdir, 3)
        outdir = os.path.join(tmpdir, "out")
        userArgs = ["--alternate-format"] + configs + [outdir]

        with pytest.raises(SystemExit) as exc_info:
            M.Tensile(userArgs)

        assert exc_info.value.code == -1
        captured = capsys.readouterr()
        assert "Only 1 or 2 config_files are accepted" in captured.out


def test_real_entry_l526_false_two_configs_passes_guard(capsys):
    """altFormat=True + exactly 2 config files -> L526 predicate False -> no SystemExit from L526.

    Execution continues past both guards and may raise later (deep YAML parse),
    but it must NOT raise SystemExit with the L526 message.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = _make_fake_configs(tmpdir, 2)
        outdir = os.path.join(tmpdir, "out")
        userArgs = ["--alternate-format"] + configs + [outdir]

        try:
            M.Tensile(userArgs)
        except SystemExit as e:
            captured = capsys.readouterr()
            # Must NOT be the L526 message
            assert "Only 1 or 2 config_files are accepted" not in captured.out, (
                f"L526 guard fired unexpectedly; exit code={e.code}, stdout={captured.out!r}"
            )
        except Exception:
            # Any other exception (deep YAML parse, NoneType, etc.) is fine.
            pass


def test_real_entry_l529_true_branch_triggers_system_exit(capsys):
    """altFormat=False + 2 config files -> L529 fires printExit -> SystemExit(-1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = _make_fake_configs(tmpdir, 2)
        outdir = os.path.join(tmpdir, "out")
        # No --alternate-format: altFormat=False, 2 configs -> != 1 -> L529 fires
        userArgs = configs + [outdir]

        with pytest.raises(SystemExit) as exc_info:
            M.Tensile(userArgs)

        assert exc_info.value.code == -1
        captured = capsys.readouterr()
        assert "Only 1 config_file is accepted" in captured.out


def test_real_entry_l529_false_one_config_passes_guard(capsys):
    """altFormat=False + exactly 1 config file -> L529 predicate False -> no SystemExit from L529.

    Execution continues past both guards; may fail deeper (YAML parse), but must
    NOT fire the L529 guard message.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = _make_fake_configs(tmpdir, 1)
        outdir = os.path.join(tmpdir, "out")
        userArgs = configs + [outdir]

        try:
            M.Tensile(userArgs)
        except SystemExit as e:
            captured = capsys.readouterr()
            assert "Only 1 config_file is accepted" not in captured.out, (
                f"L529 guard fired unexpectedly; exit code={e.code}, stdout={captured.out!r}"
            )
        except Exception:
            # Deep parse failures are expected and acceptable here.
            pass
