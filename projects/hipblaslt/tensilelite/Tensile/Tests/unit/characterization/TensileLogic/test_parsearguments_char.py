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

"""Characterization tests for ``TensileLogic.ParseArguments.parseArguments``.

``parseArguments`` builds an ``ArgumentParser`` and parses ``sys.argv``. Each
test sets ``sys.argv`` (via monkeypatch) and snapshots the parsed namespace as
a normalised dict (``Path`` values stringified so the snapshot is
platform-stable). Both the ``--check-all`` and ``--check-only-custom-kernels``
arms of the mutually exclusive group are pinned, plus the two ``SystemExit``
paths argparse raises (mutually-exclusive conflict and missing required
``LogicPath``).
"""

import sys

import pytest

from Tensile.TensileLogic.ParseArguments import parseArguments

pytestmark = pytest.mark.unit


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["TensileLogic", *argv])
    args = parseArguments()
    # Normalise: stringify any Path-valued options for stable snapshots.
    return {k: (str(v) if v is not None and not isinstance(v, (int, bool, str)) else v)
            for k, v in sorted(vars(args).items())}


def test_minimal_check_all(monkeypatch, snapshot):
    assert _parse(monkeypatch, ["logic/dir", "--check-all"]) == snapshot


def test_check_only_custom_kernels_with_options(monkeypatch, snapshot):
    assert _parse(monkeypatch, [
        "logic/dir", "--check-only-custom-kernels",
        "-v", "2", "--jobs", "8", "--cxx-compiler", "amdclang++",
        "--known-bugs", "kb.yaml",
    ]) == snapshot


def test_defaults_only_logicpath(monkeypatch, snapshot):
    # No check flag: parses fine (the "no checks" decision is made later in
    # _setup, not by the parser).
    assert _parse(monkeypatch, ["logic/dir"]) == snapshot


def test_mutually_exclusive_flags_exit(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["TensileLogic", "logic/dir", "--check-all", "--check-only-custom-kernels"],
    )
    with pytest.raises(SystemExit):
        parseArguments()


def test_missing_logicpath_exit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["TensileLogic", "--check-all"])
    with pytest.raises(SystemExit):
        parseArguments()
