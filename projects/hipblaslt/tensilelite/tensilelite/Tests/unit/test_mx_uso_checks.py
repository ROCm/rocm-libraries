# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The hardware regression must reject vacuous and numerically-close row checks."""

import numpy as np
import pytest

from tensilelite.Tests.common.mx_uso_helpers import assert_row_uniform, assert_static_partials

pytestmark = pytest.mark.unit


@pytest.fixture
def tensor(tmp_path):
    path = tmp_path / "tensor_D.bin"
    # Two column-major 3x2 matrices, each with identical rows and distinct columns.
    np.array([1, 1, 1, -2, -2, -2, 3, 3, 3, 4, 4, 4], dtype="<f4").tofile(path)
    return path


def test_identical_rows_in_each_batch(tensor):
    assert_row_uniform(tensor, [3, 2, 2, 128])


def test_one_bit_row_difference_is_not_a_tolerance_check(tensor):
    bits = np.fromfile(tensor, dtype="<u4")
    bits[-1] += 1  # One ULP in a non-first batch and non-first column.
    bits.tofile(tensor)
    with pytest.raises(AssertionError, match="Rows differ bitwise"):
        assert_row_uniform(tensor, [3, 2, 2, 128])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_uniform_but_vacuous_output_fails(tensor, value):
    np.full(12, value, dtype="<f4").tofile(tensor)
    with pytest.raises(AssertionError, match="Nonfinite|Constant"):
        assert_row_uniform(tensor, [3, 2, 2, 128])


def test_truncated_dump_fails(tensor):
    with pytest.raises(AssertionError, match="Wrong tensor size"):
        assert_row_uniform(tensor, [3, 3, 2, 128])


_LAUNCH = """TensileLite::StreamK LAUNCH SUMMARY  kernel='mx_kernel'
  TileProcessingStrategy=StreamK WorkAssignment=StaticGrid EffectiveWorkAssignment=StaticGrid
    reduction = tree
    isDynamic = no
    partials = yes
    allocated = yes
    dpOnly = no
    final = 64
    tiles = 8
"""


def test_static_partial_launch_is_required():
    assert_static_partials(_LAUNCH, "mx_kernel")


@pytest.mark.parametrize(
    "old,new",
    [
        ("partials = yes", "partials = no"),
        ("allocated = yes", "allocated = no"),
        ("reduction = tree", "reduction = parallel(DP)"),
        ("isDynamic = no", "isDynamic = yes"),
        ("dpOnly = no", "dpOnly = yes"),
        ("final = 64", "final = 8"),
        ("kernel='mx_kernel'", "kernel='different_kernel'"),
    ],
)
def test_wrong_launch_cannot_hide_behind_a_passing_launch(old, new):
    with pytest.raises(AssertionError):
        assert_static_partials(_LAUNCH + _LAUNCH.replace(old, new), "mx_kernel")


def test_missing_launch_summary_fails():
    with pytest.raises(AssertionError, match="No StreamK launch summary"):
        assert_static_partials("PASSED", "mx_kernel")


def test_replay_visits_each_solution_and_problem_in_compressed_library(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tensilelite.Common.GlobalParameters import globalParameters
    from tensilelite.Tests.common.mx_uso_helpers import run_mx_uso_checks
    from tensilelite.Utilities.BinaryLibraryIO import writeMessagePack

    source = tmp_path / "1_BenchmarkProblems/group/00_Final/caches/hash/source"
    source.mkdir(parents=True)
    library = source / "TensileLibrary.dat"
    writeMessagePack(
        library,
        {
            "library": {},
            "solutions": [
                {"index": 0, "name": "solution0", "kernelName": "mx0"},
                {"index": 1, "name": "solution1", "kernelName": "mx1"},
            ],
        },
        indexed=True,
    )
    (source / "ClientParameters.ini").write_text(
        f"library-file={library}\n"
        "d-type=Float\nc-equal-d=False\n"
        "problem-identifier=Contraction_l_Alik_Bljk_Cijk_Dijk\n"
        "problem-size=3,2,1,128\nd-strides=1,3,6\n"
        "problem-size=4,2,1,256\nd-strides=1,4,8\n"
    )
    seen = set()

    def client_run(command, *, cwd, env, stdout, stderr, timeout):
        from pathlib import Path

        values = dict(line.split("=", 1) for line in Path(command[-1]).read_text().splitlines())
        solution, problem = int(values["solution-start-idx"]), int(values["problem-start-idx"])
        assert values["num-solutions"] == values["num-problems"] == "1"
        assert values["uniform-summation-order"] == "1"
        assert values["num-elements-to-validate"] == "-1"
        assert env["TENSILE_PERSISTENT_FIXED_GRID"] == "64"
        assert env["TENSILE_PERSISTENT_DYNAMIC_GRID"] == "0"
        seen.add((solution, problem, cwd))
        tensor = np.repeat([1, 2], [3 + problem, 3 + problem]).astype("<f4")
        for name in ("D", "Ref"):
            tensor.tofile(cwd / f"tensor_{name}.bin")
        stdout.write(_LAUNCH.replace("mx_kernel", f"mx{solution}"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setitem(globalParameters, "ClientExecutionLockPath", None)
    monkeypatch.setattr("tensilelite.Tests.common.mx_uso_helpers.subprocess.run", client_run)
    run_mx_uso_checks(tmp_path, "/client", {"grid": 64, "expected_solutions": 2})
    assert {(solution, problem) for solution, problem, _ in seen} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert len({cwd for _, _, cwd in seen}) == 4
