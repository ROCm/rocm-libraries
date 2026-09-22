# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for ROCM-26842 (SEC-00394).

Tensile's ``--global-parameters`` option accepts ``key=value`` overrides. Historically
the value was passed through ``eval()``, letting any CLI/CI-supplied argument execute
arbitrary Python. These tests lock in the fix: values are parsed as Python literals via
``ast.literal_eval``, and any expression that would execute code is rejected.

These live in the top-level unit tree (not ``characterization/``) on purpose: they assert
intended behavior (a security invariant), not a pinned golden, and the coverage lane runs
this directory while excluding ``characterization/``.
"""

import argparse

import pytest

import Tensile.Tensile as T

pytestmark = pytest.mark.unit


def test_global_parameters_parses_literals():
    p = argparse.ArgumentParser()
    T.addCommonArguments(p)
    args = p.parse_args(["--global-parameters", "I=5", "B=True", "L=[1, 2, 3]", "S='a=b'"])
    gp = dict(args.global_parameters)
    assert gp["I"] == 5
    assert gp["B"] is True
    assert gp["L"] == [1, 2, 3]
    # A quoted string containing '=' survives split("=", 1).
    assert gp["S"] == "a=b"


def test_global_parameters_rejects_code_execution(tmp_path):
    """A ``--global-parameters`` value that would execute code on ``eval()`` must be
    rejected by the parser and never run. Under the old ``eval()`` this payload created
    the marker file and parsing succeeded; literal parsing rejects it (SystemExit) and
    leaves no side effect."""
    p = argparse.ArgumentParser()
    T.addCommonArguments(p)
    marker = tmp_path / "pwned"
    payload = f"X=open({str(marker)!r}, 'w').close()"
    with pytest.raises(SystemExit):
        p.parse_args(["--global-parameters", payload])
    assert not marker.exists(), "eval() executed attacker-controlled code"
