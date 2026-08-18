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

"""Characterization tests for the solution-file writer:
``_writeSolutionsHeader`` (version/problem-sizes/bias/activation header),
``_findBodyOffset`` (header/body boundary), and ``writeSolutions`` (non-cache
full write + cache same-size / shifted-header in-place rewrites).

The ``writeSolutions`` non-cache path calls ``solution.getAttributes()``,
``ProblemType.state`` and ``problemTypeToEnum`` on each solution. Rather than
build a live ``Solution`` (heavy; see ``resistance.md``), the contract is
driven with a synthetic solution that exposes exactly those surfaces — this
pins the *writer's* behaviour (header + body layout, ISA truncation, dtype
enum-to-int conversion) deterministically. The version token is normalised to
``<VERSION>`` in every snapshot.
"""

from types import SimpleNamespace

import pytest

from Tensile import __version__
import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit


def _norm(text):
    """Normalise the embedded Tensile version to a stable token."""
    return text.replace(__version__, "<VERSION>")


# ---------------------------------------------------------------------------
# Synthetic solution / problem-size collaborators
# ---------------------------------------------------------------------------

class _Enum:
    """Stand-in for a dtype enum member: exposes ``.value``."""
    def __init__(self, value):
        self.value = value


def _problem_type_state():
    """A minimal ProblemType.state dict shaped for ``problemTypeToEnum``."""
    return {
        "OperationType": "GEMM",
        "DataType": _Enum(0),
        "MacDataTypeA": _Enum(0),
        "MacDataTypeB": _Enum(0),
        "DataTypeA": _Enum(0),
        "DataTypeB": _Enum(0),
        "DataTypeE": _Enum(0),
        "DataTypeAmaxD": _Enum(0),
        "DestDataType": _Enum(0),
        "ComputeDataType": _Enum(0),
        "BiasDataTypeList": [_Enum(0)],
        "ActivationComputeDataType": _Enum(0),
        "ActivationType": _Enum("none"),
        "F32XdlMathOp": _Enum(0),
    }


class _FakeProblemType:
    @property
    def state(self):
        return _problem_type_state()


class _FakeSolution:
    def __init__(self, index):
        self._index = index

    def getAttributes(self):
        return {
            "SolutionIndex": self._index,
            "SolutionNameMin": f"sol{self._index}",
            "ISA": (9, 4, 2, "extra-truncated"),
            "ProblemType": _FakeProblemType(),
        }


def _problem_sizes(ranges, exacts):
    return SimpleNamespace(ranges=ranges, exacts=exacts)


def _bias_args(values):
    return SimpleNamespace(biasTypes=[_Enum(v) for v in values])


def _activation_args(enums):
    return SimpleNamespace(settingList=[SimpleNamespace(activationEnum=e) for e in enums])


# ===========================================================================
# _writeSolutionsHeader
# ===========================================================================

def _header(problemSizes, biasArgs, actArgs):
    import io
    buf = io.StringIO()
    L._writeSolutionsHeader(buf, problemSizes, biasArgs, actArgs)
    return _norm(buf.getvalue())


def test_header_minimal(snapshot):
    # All optionals None -> version + empty ProblemSizes only.
    assert _header(None, None, None) == snapshot


def test_header_with_problem_sizes(snapshot):
    ps = _problem_sizes(ranges=[[16, 16, 1, 16]], exacts=[[128, 64, 1, 256]])
    assert _header(ps, None, None) == snapshot


def test_header_with_bias_and_activation(snapshot):
    ps = _problem_sizes(ranges=[], exacts=[])
    assert _header(ps, _bias_args([0, 4]), _activation_args(["relu", "gelu"])) == snapshot


# ===========================================================================
# _findBodyOffset
# ===========================================================================

def test_find_body_offset(tmp_path, snapshot):
    p = tmp_path / "sol.yaml"
    p.write_text(
        "- MinimumRequiredVersion: 5.0.0\n"
        "- ProblemSizes:\n"
        "  - Range: [16, 16, 1, 16]\n"
        "- SolutionIndex: 0\n"
        "  Name: k0\n"
    )
    headerKeys = {"MinimumRequiredVersion", "ProblemSizes"}
    offset = L._findBodyOffset(str(p), headerKeys)
    body = p.read_text()[offset:]
    assert {"offset": offset, "body": body} == snapshot


def test_find_body_offset_no_body(tmp_path, snapshot):
    # Header-only file: offset lands at EOF (the `if not line` return).
    p = tmp_path / "sol.yaml"
    p.write_text("- MinimumRequiredVersion: 5.0.0\n- ProblemSizes:\n")
    headerKeys = {"MinimumRequiredVersion", "ProblemSizes"}
    offset = L._findBodyOffset(str(p), headerKeys)
    assert {"offset": offset, "at_eof": offset == len(p.read_text())} == snapshot


# ===========================================================================
# writeSolutions — non-cache full write
# ===========================================================================

def test_write_solutions_full(tmp_path, snapshot):
    p = tmp_path / "sol.yaml"
    ps = _problem_sizes(ranges=[[16, 16, 1, 16]], exacts=[])
    L.writeSolutions(str(p), ps, None, None, [_FakeSolution(0), _FakeSolution(1)])
    assert _norm(p.read_text()) == snapshot


# ===========================================================================
# writeSolutions — cache=True (rewrite header in place)
# ===========================================================================

def _seed_file(p):
    """Write a file whose header matches what _writeSolutionsHeader(None,...)
    produces, followed by a solution body."""
    import io
    buf = io.StringIO()
    L._writeSolutionsHeader(buf, None, None, None)
    p.write_text(buf.getvalue() + "- SolutionIndex: 0\n  Name: k0\n")
    return buf.getvalue()


def test_write_solutions_cache_same_size(tmp_path, snapshot):
    # New header identical in size to the old -> overwrite in place, body intact.
    p = tmp_path / "sol.yaml"
    _seed_file(p)
    L.writeSolutions(str(p), None, None, None, [], cache=True)
    assert _norm(p.read_text()) == snapshot


def test_write_solutions_cache_shifted(tmp_path, snapshot):
    # New header larger than old -> body must be shifted down.
    p = tmp_path / "sol.yaml"
    _seed_file(p)
    ps = _problem_sizes(ranges=[[16, 16, 1, 16], [32, 32, 1, 32]], exacts=[[1, 2, 3, 4]])
    L.writeSolutions(str(p), ps, None, None, [], cache=True)
    assert _norm(p.read_text()) == snapshot
