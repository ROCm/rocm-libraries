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

"""Characterization tests for the pure support-functions slice of
``Tensile.SolutionStructs.Solution`` (L165-439): the type-mismatch collector
machinery, the benchmark-arg dataclasses, and the index helpers. The
cap-coupled ``Solution`` class (L444+) is out of this slice.
"""

import importlib

import pytest

from Tensile.Activation import ActivationType

# NOTE: SolutionStructs/__init__.py re-exports the `Solution` *class*, which
# shadows the submodule attribute, so `import Tensile.SolutionStructs.Solution
# as S` would bind S to the class. Use import_module to get the real module.
S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit


def _render_collector(collector):
    """Object-free, sorted view of the collector for snapshotting."""
    return {
        "|".join(k): {
            "count": v["count"],
            "values": sorted(v["values"]),
            "files": sorted(v["files"]),
        }
        for k, v in sorted(collector.items(), key=lambda kv: kv[0])
    }


# ===========================================================================
# _getExpectedTypes + _expectedParamTypes + _skipTypeCheck
# ===========================================================================

def test_get_expected_types_synthetic(snapshot):
    # -1 sentinel skipped; empty list skipped; concrete lists -> set of types.
    validParams = {
        "AnyValue": -1,
        "EmptyList": [],
        "IntParam": [0, 1, 2],
        "BoolParam": [True, False],
        "StrParam": ["a", "b"],
        "MixedParam": [0, "x"],
    }
    result = {k: sorted(t.__name__ for t in v) for k, v in S._getExpectedTypes(validParams).items()}
    assert result == snapshot


def test_expected_param_types_is_populated():
    # The import-time map is built from validParameters and is non-empty.
    assert len(S._expectedParamTypes) > 0
    assert all(isinstance(v, set) for v in S._expectedParamTypes.values())


def test_skip_type_check_roster(snapshot):
    assert sorted(S._skipTypeCheck) == snapshot


# ===========================================================================
# type-mismatch collector: reset / get / merge
# ===========================================================================

def test_collector_reset(isolated_collector):
    with isolated_collector() as collector:
        collector[("X", "int", "bool")] = {"count": 1, "values": {"1"}, "files": set()}
        S.resetTypeMismatchCollector()
        assert collector == {}


def test_collector_get_returns_copy(isolated_collector, snapshot):
    with isolated_collector() as collector:
        collector[("UseBeta", "int", "bool")] = {"count": 2, "values": {"0", "1"}, "files": {"a.yaml"}}
        got = S.getTypeMismatchCollector()
        # Mutating the returned copy must not touch the live collector.
        got[("UseBeta", "int", "bool")]["count"] = 999
        assert collector[("UseBeta", "int", "bool")]["count"] == 2
        assert _render_collector(got) == snapshot


def test_collector_merge_new_and_existing(isolated_collector, snapshot):
    with isolated_collector() as collector:
        collector[("A", "int", "bool")] = {"count": 1, "values": {"1"}, "files": {"x.yaml"}}
        S.mergeTypeMismatchCollector({
            ("A", "int", "bool"): {"count": 2, "values": {"0"}, "files": {"y.yaml"}},  # existing
            ("B", "float", "int"): {"count": 5, "values": {"1.5"}, "files": {"z.yaml"}},  # new
        })
        assert _render_collector(collector) == snapshot


# ===========================================================================
# validateParameterTypes
# ===========================================================================

def test_validate_clean(isolated_collector, snapshot):
    with isolated_collector() as collector:
        # BufferLoad is a real bool Solution param; a bool -> no mismatch.
        S.mergeMismatchRecords(S.validateParameterTypes({"BufferLoad": True}, srcFile="f.yaml"))
        assert _render_collector(collector) == snapshot


def test_validate_mismatch_with_srcfile(isolated_collector, snapshot):
    with isolated_collector() as collector:
        # bool param given an int -> mismatch recorded with the file.
        S.mergeMismatchRecords(S.validateParameterTypes({"BufferLoad": 1}, srcFile="f.yaml"))
        assert _render_collector(collector) == snapshot


def test_validate_mismatch_no_srcfile(isolated_collector, snapshot):
    with isolated_collector() as collector:
        S.mergeMismatchRecords(S.validateParameterTypes({"BufferLoad": 1}, srcFile=""))
        assert _render_collector(collector) == snapshot


def test_validate_mismatch_accumulates(isolated_collector, snapshot):
    with isolated_collector() as collector:
        # Same (param, actual, expected) twice -> the existing-key branch;
        # count accumulates and both values are recorded.
        S.mergeMismatchRecords(S.validateParameterTypes({"BufferLoad": 1}, srcFile="a.yaml"))
        S.mergeMismatchRecords(S.validateParameterTypes({"BufferLoad": 2}, srcFile="b.yaml"))
        assert _render_collector(collector) == snapshot


def test_validate_skips_unknown_and_skiplist(isolated_collector, snapshot):
    with isolated_collector() as collector:
        # Unknown key + a _skipTypeCheck key (DataType) -> both ignored.
        S.mergeMismatchRecords(S.validateParameterTypes({"NotARealParam": 1, "DataType": "wrong-type-but-skipped"}))
        assert _render_collector(collector) == snapshot


# ===========================================================================
# printTypeMismatchSummary
# ===========================================================================

def test_print_summary_empty_returns_zero(isolated_collector, capsys, snapshot):
    with isolated_collector():
        rv = S.printTypeMismatchSummary(numFiles=0)
        out = capsys.readouterr().out
        assert {"return": rv, "stdout": out} == snapshot


def test_print_summary_empty_with_files(isolated_collector, capsys, snapshot):
    with isolated_collector():
        rv = S.printTypeMismatchSummary(numFiles=7)
        out = capsys.readouterr().out
        assert {"return": rv, "stdout": out} == snapshot


def test_print_summary_populated(isolated_collector, capsys, snapshot):
    with isolated_collector() as collector:
        collector[("UseBeta", "int", "bool")] = {"count": 3, "values": {"0", "1"}, "files": {"a.yaml", "b.yaml"}}
        collector[("DepthU", "float", "int")] = {"count": 1, "values": {"8.0"}, "files": {"a.yaml"}}
        rv = S.printTypeMismatchSummary(numFiles=2)
        out = capsys.readouterr().out
        assert {"return": rv, "stdout": out} == snapshot


# ===========================================================================
# Fbs enum
# ===========================================================================

def test_fbs_enum(snapshot):
    assert {m.name: m.value for m in S.Fbs} == snapshot


# ===========================================================================
# FactorDimArgs
# ===========================================================================

def test_factor_dim_args_disabled(make_pt, snapshot):
    # Neither UseScaleAlphaVec nor UseBias -> empty.
    args = S.FactorDimArgs(make_pt(), [0, 1])
    assert {"factorDims": args.factorDims, "total": args.totalProblemSizes, "str": str(args)} == snapshot


def test_factor_dim_args_enabled(make_pt, snapshot):
    args = S.FactorDimArgs(make_pt(UseScaleAlphaVec=1), [0, 1])
    assert {"factorDims": args.factorDims, "total": args.totalProblemSizes} == snapshot


def test_factor_dim_args_bad_dim_warns(make_pt, snapshot):
    # dim not in [0,1] -> printWarning (still appended).
    args = S.FactorDimArgs(make_pt(UseBias=1), [0, 5])
    assert {"factorDims": args.factorDims, "total": args.totalProblemSizes} == snapshot


# ===========================================================================
# BiasTypeArgs
# ===========================================================================

def test_bias_type_args_disabled(make_pt, snapshot):
    args = S.BiasTypeArgs(make_pt(), ["s"])
    assert {"biasTypes": [str(b) for b in args.biasTypes], "total": args.totalProblemSizes, "str": str(args)} == snapshot


def test_bias_type_args_enabled(make_pt, snapshot):
    # UseBias -> default BiasDataTypeList is [Float]; "s" (Float) is in it.
    args = S.BiasTypeArgs(make_pt(UseBias=1), ["s"])
    assert {"biasTypes": [b.toName() for b in args.biasTypes], "total": args.totalProblemSizes} == snapshot


def test_bias_type_args_unsupported_dtype_warns(make_pt, snapshot):
    # "d" (Double) not in the default [Float] list -> printWarning, still appended.
    args = S.BiasTypeArgs(make_pt(UseBias=1), ["d"])
    assert {"biasTypes": [b.toName() for b in args.biasTypes], "total": args.totalProblemSizes} == snapshot


def test_bias_type_args_empty_exits(make_pt):
    # UseBias set but no bias types provided -> printExit (sys.exit).
    with pytest.raises(SystemExit):
        S.BiasTypeArgs(make_pt(UseBias=1), [])


# ===========================================================================
# ActivationArgs
# ===========================================================================

def test_activation_args_none_early_return(make_pt, snapshot):
    args = S.ActivationArgs(make_pt(), [])  # default ActivationType == none
    assert {"settings": len(args.settingList), "total": args.totalProblemSizes, "str": str(args)} == snapshot


def test_activation_args_all_with_enum(make_pt, snapshot):
    pt = make_pt(Activation=True, ActivationType="all")
    args = S.ActivationArgs(pt, [[{"Enum": "relu"}]])
    assert {"settings": [str(s.activationEnum) for s in args.settingList], "total": args.totalProblemSizes} == snapshot


def test_activation_args_all_missing_enum_exits(make_pt):
    pt = make_pt(Activation=True, ActivationType="all")
    # config entry with no "Enum" -> actSetting.activationEnum stays "" -> printExit.
    with pytest.raises(SystemExit):
        S.ActivationArgs(pt, [[{"NotEnum": "x"}]])


def test_activation_args_all_empty_config_exits(make_pt):
    pt = make_pt(Activation=True, ActivationType="all")
    # ActivationType == 'all' but no settings collected -> printExit.
    with pytest.raises(SystemExit):
        S.ActivationArgs(pt, [])


def test_activation_args_specific_type_branch(snapshot):
    # A problemType whose ActivationType is neither none nor all/hipblaslt_all
    # exercises the else branch (enum := the problem's ActivationType). Real
    # ProblemType only yields none/all/hipblaslt_all, so use a dict stand-in.
    fake_pt = {"ActivationType": ActivationType("relu")}
    args = S.ActivationArgs(fake_pt, [[{"Enum": "gelu"}]])
    assert {"settings": [str(s.activationEnum) for s in args.settingList], "total": args.totalProblemSizes} == snapshot


# ===========================================================================
# isPackedIndex / isExtractableIndex
# ===========================================================================

def test_is_packed_index(make_pt, snapshot):
    pt = make_pt()
    ks = {"ProblemType": pt}
    free = pt["IndicesFree"]
    result = {
        "indices_free": list(free),
        "free0_packed": S.isPackedIndex(ks, free[0]),
        "nonfree_packed": S.isPackedIndex(ks, 999),
    }
    assert result == snapshot


@pytest.mark.parametrize("tc", ["A", "B", "x"], ids=["A", "B", "x"])
def test_is_extractable_index(tc, snapshot):
    ks = {"PackedC0IndicesX": [0, 1, 2], "PackedC1IndicesX": [3, 4, 5]}
    # index 0 in C0[:-1]=[0,1]; index 3 in C1[:-1]=[3,4]; index 9 in neither.
    result = {idx: S.isExtractableIndex(ks, idx, tc) for idx in (0, 3, 9)}
    assert result == snapshot
