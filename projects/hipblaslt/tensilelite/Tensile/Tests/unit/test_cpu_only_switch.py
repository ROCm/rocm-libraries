################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Unit tests for the GPU-less ``--cpu-only`` switch (P0.5 prerequisite).

This file accumulates the T1-T12 rigor-gate suite from GPU-MOCK-PR.md. This commit
covers the flag-plumbing tier:

* T1 ``test_flag_default_off`` - the CLI flag parses correctly (absent->False,
  present->True), the internal ``globalParameters["CpuOnly"]`` plumbing key resets to
  ``False`` via ``restoreDefaultGlobalParameters()``, and the flag is NOT exposed on the
  documented ``--global-parameters`` surface.
* T2 ``test_arg_validation`` - pins the behavior commit-2 establishes at the
  common-arguments parser layer: ``--cpu-only`` parses without requiring an arch at parse
  time (no premature SystemExit), yielding ``cpuOnly=True`` with ``gpuTargets`` still
  ``None``. The ``--cpu-only`` *requires an arch* contract is enforced/pinned in the ISA
  commit (its own test), not here.

GPU-less safety: every test monkeypatches ``builtins.input`` to raise so any accidental
stdin read (e.g. ``get_user_max_frequency``) fails loudly instead of hanging unattended.
"""

import argparse

import pytest

pytestmark = pytest.mark.unit

from Tensile import Tensile
from Tensile.Common.GlobalParameters import (
    globalParameters,
    restoreDefaultGlobalParameters,
    defaultGlobalParameters,
)


@pytest.fixture(autouse=True)
def _no_stdin(monkeypatch):
    """Fail loudly on any unattended stdin read instead of hanging on a GPU-less host."""

    def _boom(*args, **kwargs):
        raise AssertionError("builtins.input() called on the --cpu-only path")

    monkeypatch.setattr("builtins.input", _boom)


def _parse(argv):
    """Parse ``argv`` through the shared addCommonArguments parser used by the script."""
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    return argParser.parse_args(argv)


def test_flag_default_off(monkeypatch):
    """T1: flag absent->False, present->True; internal plumbing key resets to False;
    flag is not on the --global-parameters surface."""
    # Absent -> default False.
    args = _parse([])
    assert args.cpuOnly is False

    # Present -> True.
    args = _parse(["--cpu-only"])
    assert args.cpuOnly is True

    # The undocumented internal plumbing key exists and defaults False, and
    # restoreDefaultGlobalParameters() resets it to False.
    assert defaultGlobalParameters["CpuOnly"] is False
    globalParameters["CpuOnly"] = True  # simulate a prior run flipping it on
    restoreDefaultGlobalParameters()
    try:
        assert globalParameters["CpuOnly"] is False
    finally:
        restoreDefaultGlobalParameters()

    # The flag must NOT be advertised on the documented --global-parameters help surface.
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    help_text = argParser.format_help()
    # --cpu-only is its own flag, present in help...
    assert "--cpu-only" in help_text
    # ...but it is not threaded through the --global-parameters key=value mechanism.
    gp_action = next(
        a for a in argParser._actions if "--global-parameters" in a.option_strings
    )
    assert "CpuOnly" not in (gp_action.help or "")
    # And eval-style --global-parameters parsing never references CpuOnly.
    assert "cpuOnly" not in (gp_action.help or "")


def test_arg_validation():
    """T2: --cpu-only without an arch parses cleanly at the common-arguments layer
    (no premature SystemExit), yielding cpuOnly=True and gpuTargets unset.

    The --cpu-only-requires-arch contract is enforced and pinned in the ISA commit; this
    test pins only what flag plumbing (commit 2) establishes: the flag is orthogonal to
    --gpu-targets at parse time.
    """
    args = _parse(["--cpu-only"])
    assert args.cpuOnly is True
    # gpuTargets lives on the Tensile() main parser, not addCommonArguments; the common
    # parser must not synthesize or require it, so the attribute is simply absent here.
    assert not hasattr(args, "gpuTargets")

    # Off by default and independent of other common args.
    args = _parse(["--device", "0"])
    assert args.cpuOnly is False
