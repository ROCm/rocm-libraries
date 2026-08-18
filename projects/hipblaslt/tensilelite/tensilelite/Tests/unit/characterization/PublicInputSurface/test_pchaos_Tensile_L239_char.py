################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: OpenCL platform selection in
``Tensile/Tensile.py``'s ``argUpdatedGlobalParameters``.

This file used to pin the ``if args.platform:`` branch, which stored
``rv["Platform"] = args.platform`` from the ``-p / --platform`` CLI flag.  That
flag and that branch are gone: OpenCL is no longer a supported runtime, so the
option was dropped from ``addCommonArguments`` and the assignment was replaced
by an explicit refusal.

What is pinned now:

  * ``-p / --platform`` is not a recognised option any more.
  * ``args.platform`` is never read, so a namespace still carrying it produces
    no ``"Platform"`` key rather than an error.
  * ``Platform`` can still reach ``rv`` through ``--global-parameters``, and
    that path exits via ``printExit`` instead of silently configuring OpenCL.

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
        # Retained deliberately: the point of the first test below is that this
        # field is now inert, so the namespace still offers it.
        platform=None,
        RuntimeLanguage=None,
        CodeObjectVersion=None,
        debug=False,
        # --validate-metadata: action="store_true", so a real argparse Namespace
        # always carries this (default False) -- see
        # TensileMain/test_tensile_helpers_char.py for dedicated ValidateMetadata
        # coverage.
        ValidateMetadata=False,
        client_lock=None,
        prebuilt_client=None,
        MXScaleFormat=0,
        global_parameters=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# The flag is gone from the public CLI surface
# ---------------------------------------------------------------------------

def test_platform_flag_is_no_longer_a_recognised_option():
    """``--platform`` is left over as an unknown argument rather than parsed."""
    from Tensile.Tensile import addCommonArguments

    parser = argparse.ArgumentParser()
    addCommonArguments(parser)

    _, unknown = parser.parse_known_args(["--platform", "1"])
    assert "--platform" in unknown

    assert "platform" not in {action.dest for action in parser._actions}


# ---------------------------------------------------------------------------
# argUpdatedGlobalParameters no longer reads args.platform at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", [1, 0, None], ids=["nonzero", "zero", "omitted"])
def test_args_platform_is_ignored(platform):
    """Whatever the namespace carries, no 'Platform' key is produced."""
    from Tensile.Tensile import argUpdatedGlobalParameters

    rv = argUpdatedGlobalParameters(_make_args(platform=platform))
    assert "Platform" not in rv, (
        "args.platform is no longer consulted; got rv={}".format(rv)
    )


# ---------------------------------------------------------------------------
# The one surviving route to a Platform key is refused outright
# ---------------------------------------------------------------------------

def test_platform_via_global_parameters_exits(capsys):
    """--global-parameters Platform=1 reaches rv, and is rejected there."""
    from Tensile.Tensile import argUpdatedGlobalParameters

    args = _make_args(global_parameters=[("Platform", 1)])
    with pytest.raises(SystemExit) as excinfo:
        argUpdatedGlobalParameters(args)

    assert excinfo.value.code == -1
    assert "OpenCL platform selection is no longer supported" in capsys.readouterr().out
