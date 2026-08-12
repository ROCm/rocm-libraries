################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: ``if args.platform:`` in
``Tensile/Tensile.py`` at line 239, inside ``argUpdatedGlobalParameters``.

Branch 765305e2fbcf1ee08927bffdac198278bded30ee.

The predicate is a bare Python truthiness test on the CLI integer flag
``-p / --platform`` (type=int, no default => None when omitted):

  * TRUE branch  -> ``args.platform`` is a nonzero int (e.g. 1); the block
                    prints a Command-line override message and stores
                    ``rv["Platform"] = args.platform``.
  * FALSE branch -> ``args.platform`` is 0 (explicit) or None (omitted);
                    the block is skipped and ``rv`` does NOT get a
                    ``"Platform"`` key.

Witnesses confirmed via z3 (sat on both sides) and in-container argparse replay:
  TRUE:  platform=1  -> rv contains "Platform"
  FALSE: platform=0  -> rv does NOT contain "Platform"
  FALSE: platform=None (omitted) -> rv does NOT contain "Platform"

These tests pin ACTUAL observed behavior; they do not assert anything
aspirational.
"""

import argparse

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: build a minimal args namespace that satisfies argUpdatedGlobalParameters
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Return a Namespace matching the fields consumed by argUpdatedGlobalParameters."""
    defaults = dict(
        platform=None,
        RuntimeLanguage=None,
        CodeObjectVersion=None,
        debug=False,
        client_lock=None,
        prebuilt_client=None,
        MXScaleFormat=0,
        global_parameters=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Unit: pure predicate truthiness (no Tensile import; no side-effects)
# ---------------------------------------------------------------------------

def test_platform_predicate_true_nonzero_int():
    """bool(1) is True -> the TRUE branch is taken."""
    assert bool(1) is True


def test_platform_predicate_false_zero():
    """bool(0) is False -> the FALSE branch is taken."""
    assert bool(0) is False


def test_platform_predicate_false_none():
    """bool(None) is False -> the FALSE branch is taken when -p is omitted."""
    assert bool(None) is False


# ---------------------------------------------------------------------------
# Integration: call argUpdatedGlobalParameters directly, pin rv["Platform"]
# ---------------------------------------------------------------------------

def test_platform_true_branch_sets_rv_key(capsys):
    """TRUE branch (platform=1): rv contains 'Platform' == 1."""
    from Tensile.Tensile import argUpdatedGlobalParameters

    args = _make_args(platform=1)
    rv = argUpdatedGlobalParameters(args)
    assert "Platform" in rv, "Expected 'Platform' key in rv for truthy platform"
    assert rv["Platform"] == 1


def test_platform_false_branch_zero_no_rv_key():
    """FALSE branch (platform=0): rv does NOT contain 'Platform'."""
    from Tensile.Tensile import argUpdatedGlobalParameters

    args = _make_args(platform=0)
    rv = argUpdatedGlobalParameters(args)
    assert "Platform" not in rv, (
        "Expected no 'Platform' key in rv for falsy platform=0; got rv={}".format(rv)
    )


def test_platform_false_branch_none_no_rv_key():
    """FALSE branch (platform=None, omitted): rv does NOT contain 'Platform'."""
    from Tensile.Tensile import argUpdatedGlobalParameters

    args = _make_args(platform=None)
    rv = argUpdatedGlobalParameters(args)
    assert "Platform" not in rv, (
        "Expected no 'Platform' key in rv for falsy platform=None; got rv={}".format(rv)
    )
