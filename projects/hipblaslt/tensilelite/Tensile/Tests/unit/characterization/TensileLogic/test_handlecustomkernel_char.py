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

"""Characterization tests for ``TensileLogic.HandleCustomKernel``.

* ``prepareCustomKernelConfig`` — pure string formatting of an MI-param dict
  into a developer-facing diff. Snapshot the produced text. Inputs are sized so
  ``pformat`` wraps one key per line, exercising the ``ISA`` skip, the quote
  removal, and both the trailing-comma and no-comma arms.
* ``hasCustomKernel`` — line scan of a logic file; True/False/empty-name cases
  via ``tmp_path`` fixtures.
* ``handleCustomKernel`` — branch logic. The non-custom early return is real;
  the custom-kernel paths inject the config via a monkeypatched
  ``getCustomKernelConfig`` (the "fixture/injection" the goal allows) so the
  test does not depend on a specific shipped ``.s`` file. The MI-length-4 path
  returns ``(sol, True)`` directly; the not-4/0 path additionally runs the
  ``matrixInstructionToMIParameters`` + hint branch (its stdout is incidental,
  not snapshotted).
"""

import copy

import pytest

import Tensile.TensileLogic.HandleCustomKernel as hck_mod
from Tensile.TensileLogic.HandleCustomKernel import (
    handleCustomKernel,
    hasCustomKernel,
    prepareCustomKernelConfig,
)

pytestmark = pytest.mark.unit


# --- prepareCustomKernelConfig (pure) ---------------------------------------

def test_prepare_custom_kernel_config(snapshot):
    miParams = {
        "ISA": (9, 0, 10), "MatrixInstM": 16, "MatrixInstN": 16,
        "MatrixInstK": 16, "MatrixInstB": 1, "MIWaveGroup": [2, 2],
        "MIWaveTile": [4, 4], "MIInputPerThread": 8,
    }
    assert prepareCustomKernelConfig(miParams, [16, 16, 16, 1]) == snapshot


# --- hasCustomKernel --------------------------------------------------------

def test_has_custom_kernel_true(tmp_path, snapshot):
    f = tmp_path / "withck.yaml"
    f.write_text("- foo\n  CustomKernelName: MyKernel\n- bar\n", encoding="utf-8")
    assert hasCustomKernel(f) == snapshot


def test_has_custom_kernel_false_no_line(tmp_path, snapshot):
    f = tmp_path / "nock.yaml"
    f.write_text("- foo\n  SomeOtherKey: 1\n", encoding="utf-8")
    assert hasCustomKernel(f) == snapshot


def test_has_custom_kernel_false_empty_name(tmp_path, snapshot):
    # Empty-string name (ends with '') is treated as "no custom kernel".
    f = tmp_path / "emptyck.yaml"
    f.write_text("- foo\n  CustomKernelName: ''\n", encoding="utf-8")
    assert hasCustomKernel(f) == snapshot


# --- handleCustomKernel -----------------------------------------------------

def test_handle_non_custom_returns_false(snapshot):
    # No CustomKernelName -> early (sol, False), sol unchanged.
    sol = {"SolutionIndex": 0}
    out, is_custom = handleCustomKernel(sol, {})
    assert {"is_custom": is_custom, "sol": out} == snapshot


def test_handle_custom_mi_length_4(monkeypatch, snapshot):
    # MI length 4 -> normal path, no hint, returns (sol, True).
    config = {"MatrixInstruction": [16, 16, 16, 1], "WorkGroup": [16, 16, 1]}
    monkeypatch.setattr(hck_mod, "getCustomKernelConfig", lambda name, d, dir: dict(config))
    sol = {"CustomKernelName": "DummyKernel", "SolutionIndex": 0}
    out, is_custom = handleCustomKernel(sol, {})
    assert {"is_custom": is_custom, "sol": out} == snapshot


def test_handle_custom_mi_wrong_length_runs_hint(monkeypatch, isa_info_map, snapshot):
    # MI length 9 (not 4 or 0) -> hint branch: matrixInstructionToMIParameters
    # + prepareCustomKernelConfig run (stdout incidental). Returns (sol, True).
    config = {"MatrixInstruction": [16, 16, 16, 1, 1, 2, 2, 2, 2]}
    monkeypatch.setattr(hck_mod, "getCustomKernelConfig", lambda name, d, dir: dict(config))
    sol = {
        "CustomKernelName": "DummyKernel", "SolutionIndex": 0,
        "WavefrontSize": 64, "ProblemType": {"DataType": "h"},
    }
    out, is_custom = handleCustomKernel(sol, isa_info_map)
    assert {
        "is_custom": is_custom,
        "matrix_instruction": out["MatrixInstruction"],
    } == snapshot
