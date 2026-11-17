# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for ROCM-26842 (SEC-00394).

Tensile's CLI parsers accept ``key=value`` overrides on ``--global-parameters`` /
``--benchmark-parameters``. Historically the value was passed through ``eval()``,
which let any CLI/CI-supplied argument execute arbitrary Python. These tests lock in
the fix: values are parsed as Python literals via ``ast.literal_eval`` and any
expression that would execute code is rejected (never run).
"""

import argparse
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# A payload that, under the old eval(), would create a marker file as a side effect.
# Literal parsing must reject it and leave no file behind.
def _code_payload(marker):
    return f"X=open({str(marker)!r}, 'w').close()"


# --- TensileCreateLib.ParseArguments.splitExtraParameters (module-level) ---------
from Tensile.TensileCreateLib.ParseArguments import splitExtraParameters as _createlib_split


def test_createlib_split_parses_literals():
    assert _createlib_split("Foo=5") == ("Foo", 5)
    assert _createlib_split("Bar='baz'") == ("Bar", "baz")
    assert _createlib_split("L=[1, 2, 3]") == ("L", [1, 2, 3])
    # A quoted string containing '=' survives split("=", 1).
    assert _createlib_split("S='a=b'") == ("S", "a=b")


def test_createlib_split_rejects_code_execution(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(argparse.ArgumentTypeError):
        _createlib_split(_code_payload(marker))
    assert not marker.exists(), "eval() executed attacker-controlled code"


# --- Tensile.Tensile.addCommonArguments / --global-parameters --------------------
def test_global_parameters_parses_literals():
    import Tensile.Tensile as T

    p = argparse.ArgumentParser()
    T.addCommonArguments(p)
    args = p.parse_args(["--global-parameters", "I=5", "B=True", "L=[1, 2, 3]", "S='a=b'"])
    gp = dict(args.global_parameters)
    assert gp["I"] == 5
    assert gp["B"] is True
    assert gp["L"] == [1, 2, 3]
    assert gp["S"] == "a=b"


def test_global_parameters_rejects_code_execution(tmp_path):
    import Tensile.Tensile as T

    p = argparse.ArgumentParser()
    T.addCommonArguments(p)
    marker = tmp_path / "pwned"
    with pytest.raises(SystemExit):
        p.parse_args(["--global-parameters", _code_payload(marker)])
    assert not marker.exists(), "eval() executed attacker-controlled code"


# --- TensileBenchmarkCluster / --benchmark-parameters ----------------------------
def _cluster_parse(argv):
    from Tensile.TensileBenchmarkCluster import TensileBenchmarkCluster

    parse = TensileBenchmarkCluster._TensileBenchmarkCluster__parseArgs
    with patch("sys.argv", argv):
        return parse(None)


def test_benchmark_parameters_parses_literals():
    ns = _cluster_parse(["s", "logic", "deploy", "--benchmark-parameters", "Foo=5", "Bar='baz'"])
    assert ("Foo", 5) in ns.benchmark_parameters
    assert ("Bar", "baz") in ns.benchmark_parameters


def test_benchmark_parameters_rejects_code_execution(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(SystemExit):
        _cluster_parse(["s", "logic", "deploy", "--benchmark-parameters", _code_payload(marker)])
    assert not marker.exists(), "eval() executed attacker-controlled code"
