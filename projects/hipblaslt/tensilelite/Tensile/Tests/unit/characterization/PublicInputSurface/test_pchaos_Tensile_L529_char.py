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
################################################################################

"""Characterization tests for Tensile.py L526 (if) and L529 (elif) config-file guards.

Branch ID: 01e8ac7f371264a85285039d91ea87c8514acc62
Source:    Tensile/Tensile.py:529 (elif)
Predicate: not altFormat and len(configPaths) != 1

Inputs:
  altFormat   — bool  — args.AlternateFormat (--alternate-format, store_true, default False)
  configPaths — list  — args.ConfigFile (nargs='+', len inspected by predicate)

Run-1 canonical contract (two test groups):
  (1) TestAltFormatRejectedHelper — pure extracted predicates, no argparse/filesystem.
  (2) TestTensileArgparseGuards   — real Tensile(userArgs) entry calls; asserts that
      L526/L529 printExit (sys.exit(-1)) fires or does NOT fire for each witness.

CPU-only.  Deterministic.  Add-only.
"""

import sys
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# (1) Pure-helper tests — extracted predicates, fully static
# ---------------------------------------------------------------------------

def alt_format_rejected(alt_format: bool, n_config_files: int) -> bool:
    """L526 guard predicate: rejects when altFormat and more than 2 config files."""
    return bool(alt_format and n_config_files > 2)


def default_format_rejected(alt_format: bool, n_config_files: int) -> bool:
    """L529 guard predicate (branch under test): rejects when not altFormat and n != 1."""
    return bool((not alt_format) and n_config_files != 1)


class TestAltFormatRejectedHelper:
    """Pure predicate assertions — no argparse, no filesystem, no GPU.

    Witness grid (from verified fragment 01e8ac7f...):
      alt_format x n_config_files grid (F/T x 1/2/3):

      L526 TRUE  cells: (True, 3)
      L526 FALSE cells: (False,1), (False,2), (True,1), (True,2)

      L529 TRUE  cells: (False,2), (False,3)
      L529 FALSE cells: (False,1), (True,1),  (True,2)
    """

    # ---- L526: alt_format_rejected ----

    def test_l526_true_alt_true_n3(self):
        """(True, 3) → L526 fires (only 1 or 2 alt-format configs allowed)."""
        assert alt_format_rejected(True, 3) is True

    def test_l526_false_alt_false_n1(self):
        """(False, 1) → L526 does not fire."""
        assert alt_format_rejected(False, 1) is False

    def test_l526_false_alt_false_n2(self):
        """(False, 2) → L526 does not fire (alt_format is False)."""
        assert alt_format_rejected(False, 2) is False

    def test_l526_false_alt_true_n1(self):
        """(True, 1) → L526 does not fire (n<=2 is fine for alt format)."""
        assert alt_format_rejected(True, 1) is False

    def test_l526_false_alt_true_n2(self):
        """(True, 2) → L526 does not fire (exactly 2 is the alt-format limit)."""
        assert alt_format_rejected(True, 2) is False

    # ---- L529: default_format_rejected ----

    def test_l529_true_alt_false_n2(self):
        """(False, 2) → L529 fires (default format accepts only 1 config)."""
        assert default_format_rejected(False, 2) is True

    def test_l529_true_alt_false_n3(self):
        """(False, 3) → L529 fires (still > 1 config, not alt format)."""
        assert default_format_rejected(False, 3) is True

    def test_l529_false_alt_false_n1(self):
        """(False, 1) → L529 does not fire (happy path: exactly 1 config)."""
        assert default_format_rejected(False, 1) is False

    def test_l529_false_alt_true_n1(self):
        """(True, 1) → L529 does not fire (alt_format=True skips this elif)."""
        assert default_format_rejected(True, 1) is False

    def test_l529_false_alt_true_n2(self):
        """(True, 2) → L529 does not fire (alt_format=True skips this elif)."""
        assert default_format_rejected(True, 2) is False


# ---------------------------------------------------------------------------
# (2) Real-entry pin tests — call Tensile.Tensile(userArgs) and pin actual behavior
# ---------------------------------------------------------------------------

class TestTensileArgparseGuards:
    """Real Tensile() entry-point tests — assert L526/L529 guards fire or do not fire.

    Execution stops at the guard (printExit → sys.exit(-1)) for TRUE witnesses,
    or proceeds past the guard into deeper (file-based) parsing for FALSE witnesses.
    No monkeypatching of the guard itself; we observe raw SystemExit(code).
    """

    @staticmethod
    def _userArgs(config_files: list, output_path: str, alternate_format: bool = False) -> list:
        """Build a userArgs list accepted by argparse.parse_args() inside Tensile()."""
        args = list(config_files) + [output_path]
        if alternate_format:
            args.append("--alternate-format")
        return args

    # ---- TRUE witnesses: guard fires → sys.exit(-1) ----

    def test_l529_default_format_two_configs_exits(self, tmp_path):
        """L529 TRUE (False,2): altFormat=False, n=2 → SystemExit(-1) from printExit."""
        import Tensile.Tensile as TM

        cfg1 = str(tmp_path / "c1.yaml")
        cfg2 = str(tmp_path / "c2.yaml")
        userArgs = self._userArgs([cfg1, cfg2], str(tmp_path / "out"))

        with pytest.raises(SystemExit) as exc_info:
            TM.Tensile(userArgs)

        assert exc_info.value.code == -1

    def test_l529_default_format_three_configs_exits(self, tmp_path):
        """L529 TRUE (False,3): altFormat=False, n=3 → SystemExit(-1) from printExit."""
        import Tensile.Tensile as TM

        cfg1 = str(tmp_path / "c1.yaml")
        cfg2 = str(tmp_path / "c2.yaml")
        cfg3 = str(tmp_path / "c3.yaml")
        userArgs = self._userArgs([cfg1, cfg2, cfg3], str(tmp_path / "out"))

        with pytest.raises(SystemExit) as exc_info:
            TM.Tensile(userArgs)

        assert exc_info.value.code == -1

    def test_l526_alt_format_three_configs_exits(self, tmp_path):
        """L526 TRUE (True,3): altFormat=True, n=3 → SystemExit(-1) from printExit (sibling if)."""
        import Tensile.Tensile as TM

        cfg1 = str(tmp_path / "c1.yaml")
        cfg2 = str(tmp_path / "c2.yaml")
        cfg3 = str(tmp_path / "c3.yaml")
        userArgs = self._userArgs([cfg1, cfg2, cfg3], str(tmp_path / "out"),
                                  alternate_format=True)

        with pytest.raises(SystemExit) as exc_info:
            TM.Tensile(userArgs)

        assert exc_info.value.code == -1

    # ---- FALSE witnesses: guard does NOT fire → deeper parsing reached ----

    def test_l526_alt_format_two_configs_guard_does_not_fire(self, tmp_path):
        """L526 FALSE (True,2): altFormat=True, n=2 → guard does NOT fire.

        Execution proceeds past L526/L529 and hits a deeper parsing error
        (e.g. FileNotFoundError on the config YAML).  We assert exit code is
        NOT -1 (the guard exit code).
        """
        import Tensile.Tensile as TM

        cfg1 = str(tmp_path / "c1.yaml")
        cfg2 = str(tmp_path / "c2.yaml")
        userArgs = self._userArgs([cfg1, cfg2], str(tmp_path / "out"),
                                  alternate_format=True)

        try:
            TM.Tensile(userArgs)
            # If we somehow return cleanly, guard definitely did not fire.
        except SystemExit as e:
            assert e.code != -1, (
                "L526 guard fired for (altFormat=True, n=2) but should not: "
                "alternate format allows exactly 1 or 2 config files"
            )
        except Exception:
            # Any non-SystemExit means we cleared the guard — pass.
            pass

    def test_l529_default_format_one_config_guard_does_not_fire(self, tmp_path):
        """L529 FALSE (False,1): altFormat=False, n=1 → guard does NOT fire (happy path).

        Execution proceeds past L526/L529 and hits a deeper parsing error.
        We assert exit code is NOT -1.
        """
        import Tensile.Tensile as TM

        cfg1 = str(tmp_path / "c1.yaml")
        userArgs = self._userArgs([cfg1], str(tmp_path / "out"))

        try:
            TM.Tensile(userArgs)
        except SystemExit as e:
            assert e.code != -1, (
                "L529 guard fired for (altFormat=False, n=1) but should not: "
                "exactly 1 config file is the accepted default-format case"
            )
        except Exception:
            pass
