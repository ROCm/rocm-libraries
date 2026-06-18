################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

from __future__ import annotations

import builtins
import filecmp
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from Tensile import TensileLibLogicToYaml as M
from Tensile.Common.GlobalParameters import globalParameters

_TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
_LIBLOGIC_FIXTURE_PATH = _TEST_DATA_DIR / "TensileLibLogicToYaml_liblogic.yaml"
_EXPECTED_CONFIG_FIXTURE_PATH = _TEST_DATA_DIR / "TensileLibLogicToYaml_expected_config.yaml"


def _read_liblogic_fixture() -> str:
    """Return ``test_data/TensileLibLogicToYaml_liblogic.yaml`` for regression tests.

    Returns:
        Raw UTF-8 text of the sample library logic.

    Raises:
        FileNotFoundError: If the fixture file is missing from ``test_data/``.
        OSError: If the file cannot be read.
    """
    return _LIBLOGIC_FIXTURE_PATH.read_text(encoding="utf-8")


def _read_expected_config_fixture() -> str:
    """Return ``test_data/TensileLibLogicToYaml_expected_config.yaml`` (golden config).

    Returns:
        Raw UTF-8 text of the expected generator output.

    Raises:
        FileNotFoundError: If the fixture file is missing from ``test_data/``.
        OSError: If the file cannot be read.
    """
    return _EXPECTED_CONFIG_FIXTURE_PATH.read_text(encoding="utf-8")


def test_format_compact_range_non_list() -> None:
    """Non-list *rng* is stringified.

    Returns:
        None.

    Raises:
        None.
    """
    assert M._formatCompactRange((1, 2, 3)) == "(1, 2, 3)"


def test_format_compact_range_short_list() -> None:
    """Short lists are emitted in full.

    Returns:
        None.

    Raises:
        None.
    """
    assert M._formatCompactRange([1, 2, 3]) == "[1, 2, 3]"


def test_format_compact_range_long_list() -> None:
    """Long lists use leading and trailing segments with ellipsis.

    Returns:
        None.

    Raises:
        None.
    """
    long = list(range(20))
    out = M._formatCompactRange(long, start_elements=2, end_elements=2)
    assert out == "[0, 1, ..., 18, 19]"


def test_build_fork_parameter_comment_metadata_stable() -> None:
    """Metadata keys are a non-empty subset of merged defaults.

    Returns:
        None.

    Raises:
        None.
    """
    a = M.buildForkParameterCommentMetadata()
    b = M.buildForkParameterCommentMetadata()
    assert a == b
    assert a
    assert all(v.startswith(" # Default Value:") for v in a.values())


@pytest.mark.parametrize(
    "rest,expected",
    [
        ("", None),
        ("foo", None),
        (": notalist", None),
        (": [1, 2", None),
        (": [a]", None),
        (": []", None),
        (": [1.0, 2]", None),
        (": [1, 2, 3]", [1, 2, 3]),
    ],
)
def test_parse_matrix_instruction_list_from_colon_rest(
    rest: str, expected: Any
) -> None:
    """Parser handles invalid tails and a valid int list.

    Args:
        rest: Colon-rest fixture fragment.
        expected: Expected parse result.

    Returns:
        None.

    Raises:
        None.
    """
    assert M.parseMatrixInstructionListFromColonRest(rest) == expected


def test_parse_matrix_instruction_nested_brackets() -> None:
    """A closing ``]`` that does not end the outer list continues scanning.

    Returns:
        None.

    Raises:
        None.
    """
    assert M.parseMatrixInstructionListFromColonRest(": [[1, 2, 3]]") is None


def test_parse_matrix_instruction_literal_eval_syntax_error() -> None:
    """``ast.literal_eval`` failure on a closed bracket segment returns ``None``.

    Returns:
        None.

    Raises:
        None.
    """
    assert M.parseMatrixInstructionListFromColonRest(": [1,,2]") is None


def test_format_matrix_instruction_cms_comment_short() -> None:
    """Fewer than nine MI components yields no CMS suffix.

    Returns:
        None.

    Raises:
        None.
    """
    assert M.formatMatrixInstructionCmsComment([1] * 8) is None


def test_format_matrix_instruction_cms_comment_full() -> None:
    """Nine MI components produce a CMS comment string.

    Returns:
        None.

    Raises:
        None.
    """
    mi = [16, 16, 32, 1, 1, 7, 6, 1, 4]
    s = M.formatMatrixInstructionCmsComment(mi)
    assert s is not None
    assert "#CMS — MT" in s


def test_yaml_custom_representers() -> None:
    """Quoted, FlowList, and None serialize with registered representers.

    Returns:
        None.

    Raises:
        None.
    """
    buf = io.StringIO()
    yaml.dump(
        {"q": M.Quoted("x"), "f": M.FlowList([1, 2]), "n": None},
        buf,
        default_flow_style=False,
        sort_keys=False,
        Dumper=yaml.Dumper,
    )
    text = buf.getvalue()
    assert '"x"' in text
    assert "[1, 2]" in text.replace("\n", "")


def test_inject_fork_exits_on_benchmark_join() -> None:
    """Fork block ends when ``BenchmarkJoinParameters`` begins.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Foo: [1]\n"
        "    BenchmarkJoinParameters:\n"
        "    other:\n"
    )
    meta = {"Foo": " # meta"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "Foo: [1] # meta" in out
    assert "BenchmarkJoinParameters" in out


def test_inject_fork_exits_on_benchmark_final() -> None:
    """Fork block ends when ``BenchmarkFinalParameters`` begins.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Bar: [2]\n"
        "    BenchmarkFinalParameters:\n"
    )
    meta = {"Bar": " # mb"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "Bar: [2] # mb" in out


def test_inject_groups_matrix_instruction_cms() -> None:
    """Nine-element MI receives a CMS suffix when metadata is absent for MI.

    Returns:
        None.

    Raises:
        None.
    """
    mi = [16, 16, 32, 1, 1, 7, 6, 1, 4]
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        f"      - - MatrixInstruction: {mi}\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {})
    assert "#CMS — MT" in out


def test_inject_groups_matrix_instruction_metadata_fallback() -> None:
    """Short MI list falls back to MatrixInstruction metadata when present.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "      - - MatrixInstruction: [1, 2, 3]\n"
        "    BenchmarkFinalParameters:\n"
    )
    meta = {"MatrixInstruction": " #DEF"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "MatrixInstruction: [1, 2, 3] #DEF" in out.replace("\n", " ")


def test_inject_groups_workgroup_and_miarch() -> None:
    """WorkGroup and MIArchVgpr lines receive metadata when keys exist.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "          WorkGroup: [1, 2, 3]\n"
        "          MIArchVgpr: false\n"
        "    BenchmarkFinalParameters:\n"
    )
    meta = {"WorkGroup": " #WG", "MIArchVgpr": " #MV"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "WorkGroup: [1, 2, 3] #WG" in out.replace("\n", " ")
    assert "MIArchVgpr: false #MV" in out.replace("\n", " ")


def test_inject_groups_idempotent_default_value() -> None:
    """Lines that already contain ``Default Value:`` are not modified.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "      - - MatrixInstruction: [1, 2, 3] # Default Value: x\n"
        "    BenchmarkFinalParameters:\n"
    )
    meta = {"MatrixInstruction": " # would-append"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "# would-append" not in out


def test_inject_groups_mi_no_matrix_instruction_metadata() -> None:
    """MI line with no CMS and no ``MatrixInstruction`` metadata leaves the line.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "      - - MatrixInstruction: [1, 2, 3]\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {})
    assert "MatrixInstruction: [1, 2, 3]" in out
    assert "#DEF" not in out


def test_inject_groups_unrecognized_line_under_groups() -> None:
    """A ``Groups`` child line that matches no regex is passed through unchanged.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "          OtherKey: 1\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(
        src, {"WorkGroup": " #W", "MIArchVgpr": " #M"}
    )
    assert "OtherKey: 1" in out


def test_inject_groups_workgroup_without_metadata() -> None:
    """``WorkGroup`` line is unchanged when metadata omits ``WorkGroup``.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "          WorkGroup: [1, 2, 3]\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {"MIArchVgpr": " #MV"})
    for line in out.splitlines():
        if "WorkGroup" in line:
            assert "#" not in line


def test_inject_groups_miarch_without_metadata() -> None:
    """``MIArchVgpr`` line is unchanged when metadata omits that key.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "          MIArchVgpr: false\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {"WorkGroup": " #WG"})
    for line in out.splitlines():
        if "MIArchVgpr" in line:
            assert "#" not in line


def test_inject_fork_param_key_not_in_metadata() -> None:
    """Fork-parameter lines whose key is absent from metadata are unchanged.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - ZZOnly: [1]\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {"DepthU": " #x"})
    assert "ZZOnly: [1]" in out
    assert "#x" not in out


def test_inject_groups_idempotent_cms() -> None:
    """Lines that already contain ``CMS —`` are not modified.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - Groups:\n"
        "      - - MatrixInstruction: [1, 2, 3, 4, 5, 6, 7, 8, 9] #CMS — keep\n"
        "    BenchmarkFinalParameters:\n"
    )
    out = M.injectForkParameterInlineComments(src, {})
    assert "#CMS — keep" in out
    assert out.count("#CMS") == 1


def test_inject_fork_skips_groups_key() -> None:
    """The ``Groups`` fork entry does not receive fork-parameter suffixes.

    Returns:
        None.

    Raises:
        None.
    """
    src = (
        "    ForkParameters:\n"
        "    - DepthU: [32]\n"
        "    - Groups:\n"
        "    BenchmarkFinalParameters:\n"
    )
    meta = {"Groups": " # should-not-apply", "DepthU": " # DU"}
    out = M.injectForkParameterInlineComments(src, meta)
    assert "DepthU: [32] # DU" in out.replace("\n", " ")
    assert "should-not-apply" not in out


def test_inject_default_comment_by_key() -> None:
    """Passing ``commentByKey=None`` uses built-in metadata.

    Returns:
        None.

    Raises:
        None.
    """
    src = "    ForkParameters:\n" "    - DepthU: [32]\n" "    BenchmarkFinalParameters:\n"
    out = M.injectForkParameterInlineComments(src, None)
    assert "Default Value:" in out


def test_t_print_respects_log_level(capfd: pytest.CaptureFixture[str]) -> None:
    """``tPrint`` emits only when ``ClientLogLevel`` is high enough.

    Returns:
        None.

    Raises:
        None.
    """
    prev = globalParameters["ClientLogLevel"]
    try:
        globalParameters["ClientLogLevel"] = 0
        M.tPrint(1, "hidden")
        capfd.readouterr()
        globalParameters["ClientLogLevel"] = 2
        M.tPrint(1, "shown")
        captured = capfd.readouterr()
        assert "shown" in captured.out
        assert "hidden" not in captured.out
    finally:
        globalParameters["ClientLogLevel"] = prev


def test_set_global_params_i8_vs_other() -> None:
    """I8 problem type selects different data-init presets.

    Returns:
        None.

    Raises:
        None.
    """
    ver = {"MinimumRequiredVersion": "1.0.0"}
    i8 = M.setGlobalParams(ver, {"DataType": "I8"})
    fp = M.setGlobalParams(ver, {"DataType": "S"})
    assert i8["DataInitTypeA"] == 3
    assert fp["DataInitTypeA"] == 12


def test_form_problem_type_yaml_data_empty() -> None:
    """Empty problem type raises ``RuntimeError``.

    Returns:
        None.

    Raises:
        None.
    """
    with pytest.raises(RuntimeError):
        M.formProblemTypeYamlData({})


def test_form_problem_type_yaml_data_defaults() -> None:
    """Non-default list-valued problem-type keys become ``FlowList`` entries.

    Returns:
        None.

    Raises:
        None.
    """
    from Tensile.SolutionStructs.Problem import _defaultProblemType as dpt

    state = {
        "OperationType": "GEMM",
        "DataType": 7,
        "DestDataType": 7,
        "ComputeDataType": 0,
        "HighPrecisionAccumulate": True,
        "TransposeA": False,
        "TransposeB": False,
        "IndexAssignmentsA": [0, 3, 2],
    }
    assert dpt["IndexAssignmentsA"] != state["IndexAssignmentsA"]
    out = M.formProblemTypeYamlData(state)
    assert out["OperationType"] == "GEMM"
    assert isinstance(out["IndexAssignmentsA"], M.FlowList)


def test_form_groups_empty_and_nonempty() -> None:
    """``formGroups`` nests one mapping under ``Groups``.

    Returns:
        None.

    Raises:
        None.
    """
    empty = M.formGroups({})
    assert empty == {"Groups": [[{}]]}
    row = {"MatrixInstruction": M.FlowList([1])}
    g = M.formGroups(row)
    assert g["Groups"][0][0]["MatrixInstruction"] == M.FlowList([1])


def test_form_9_bit_mi_inst_errors_and_ok() -> None:
    """``form9BitMIInst`` validates MI vectors and returns a group row dict.

    Returns:
        None.

    Raises:
        None.
    """
    bad = {
        "MIBlock": [],
        "MIWaveTile": [1],
        "MIWaveGroup": [1],
        "WorkGroup": [1, 1, 1],
        "MIArchVgpr": False,
    }
    with pytest.raises(RuntimeError):
        M.form9BitMIInst(bad)
    good = {
        "MIBlock": [1, 2, 3, 4, 5, 9],
        "MIWaveTile": [6, 7],
        "MIWaveGroup": [8, 9],
        "WorkGroup": [2, 2, 1],
        "MIArchVgpr": True,
    }
    r = M.form9BitMIInst(good)
    assert list(r["MatrixInstruction"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_form_fork_params_without_enable_matrix_instruction_key() -> None:
    """Missing ``EnableMatrixInstruction`` skips the enable/MI diagnostic block.

    Returns:
        None.

    Raises:
        None.
    """
    sol = {"KernelLanguage": "Assembly"}
    r = M.formForkParams(sol, False)
    assert r["ForkParameters"][-1]["Groups"][0][0] == {}


def test_form_fork_params_skip_mi_and_enable_branches() -> None:
    """Fork params respect ``skipMI`` and ``EnableMatrixInstruction`` logic.

    Returns:
        None.

    Raises:
        None.
    """
    sol = {
        "EnableMatrixInstruction": True,
        "MatrixInstruction": [1],
        "MIBlock": [10, 11, 12, 13, 14, 0],
        "MIWaveTile": [1, 1],
        "MIWaveGroup": [1, 1],
        "WorkGroup": [4, 4, 1],
        "MIArchVgpr": False,
        "KernelLanguage": "Assembly",
    }
    with_mi = M.formForkParams(sol, False)
    assert with_mi["ForkParameters"][-1]["Groups"][0][0]["MatrixInstruction"]
    no_mi = M.formForkParams(sol, True)
    assert no_mi["ForkParameters"][-1]["Groups"][0][0] == {}


def test_form_fork_params_matrix_instruction_disabled_message(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Disabled matrix instruction logs a notice when the flag key exists.

    Returns:
        None.

    Raises:
        None.
    """
    prev = globalParameters["ClientLogLevel"]
    try:
        globalParameters["ClientLogLevel"] = 2
        sol = {
            "EnableMatrixInstruction": True,
            "MatrixInstruction": [],
            "MIBlock": [10, 11, 12, 13, 14, 0],
            "MIWaveTile": [1, 1],
            "MIWaveGroup": [1, 1],
            "WorkGroup": [4, 4, 1],
            "MIArchVgpr": False,
            "KernelLanguage": "Assembly",
        }
        M.formForkParams(sol, False)
        assert "disabled" in capfd.readouterr().out
    finally:
        globalParameters["ClientLogLevel"] = prev


def test_form_problem_size_origami_and_exact() -> None:
    """Origami path uses placeholder sizes; exact logic matches solution index.

    Returns:
        None.

    Raises:
        None.
    """
    prev = globalParameters["ClientLogLevel"]
    try:
        globalParameters["ClientLogLevel"] = 2
        o = M.formProblemSize(None, 0, {"BiasDataTypeList": [0, 7]})
        assert o["BenchmarkFinalParameters"][0]["ProblemSizes"]
        e = M.formProblemSize(
            [([10, 20], [0]), ([99], [1])],
            1,
            {"BiasDataTypeList": [1]},
        )
        assert list(e["BenchmarkFinalParameters"][0]["ProblemSizes"][0]["Exact"]) == [99]
    finally:
        globalParameters["ClientLogLevel"] = prev


def test_form_library_logic() -> None:
    """``formLibraryLogic`` wraps schedule and device names.

    Returns:
        None.

    Raises:
        None.
    """
    d = M.formLibraryLogic("sched", ["dev0", "dev1"], "arch")
    assert isinstance(d["ScheduleName"], M.Quoted)
    assert isinstance(d["DeviceNames"], M.FlowList)


def test_write_to_tensile_yaml_file_basename_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writing with no directory component skips ``os.makedirs``.

    Args:
        tmp_path: Pytest temp directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        None.
    """
    monkeypatch.chdir(tmp_path)
    data = {"GlobalParameters": {"MinimumRequiredVersion": "1.0.0"}}
    assert M.writeToTensileYamlFile("bare_out.yaml", data) == "bare_out.yaml"
    assert (tmp_path / "bare_out.yaml").is_file()


def test_write_to_tensile_yaml_file_success(tmp_path: Path) -> None:
    """Successful write returns the output path.

    Returns:
        None.

    Raises:
        None.
    """
    outp = tmp_path / "sub" / "out.yaml"
    data = {"GlobalParameters": {"MinimumRequiredVersion": "1.0.0"}}
    assert M.writeToTensileYamlFile(str(outp), data) == str(outp)
    assert outp.is_file()


def test_write_to_tensile_yaml_file_oserror() -> None:
    """I/O errors return ``None`` and do not raise.

    Returns:
        None.

    Raises:
        None.
    """
    real_open = builtins.open

    def boom_open(
        path: str, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if "w" in mode:
            raise OSError("simulated")
        return real_open(path, mode, *args, **kwargs)

    with patch("builtins.open", boom_open):
        assert (
            M.writeToTensileYamlFile("/tmp/should_not_write_tly.yaml", {"a": 1}) is None
        )


def test_tensile_lib_logic_to_yaml_golden_matches_fixture() -> None:
    """End-to-end conversion matches ``test_data/TensileLibLogicToYaml_expected_config.yaml``.

    Uses ``test_data/TensileLibLogicToYaml_liblogic.yaml`` as library logic input.

    Returns:
        None.

    Raises:
        None.
    """
    liblogic_body = _read_liblogic_fixture()
    expected_body = _read_expected_config_fixture()

    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        f.write(liblogic_body)
        f.flush()
        lib_path = f.name

    try:
        with tempfile.TemporaryDirectory() as workspace:
            config_yaml = os.path.join(workspace, "config.yaml")
            M.TensileLibLogicToYaml(lib_path, 0, config_yaml, False)

            with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
                f.write(expected_body)
                f.flush()
                expected_path = f.name

            try:
                assert filecmp.cmp(config_yaml, expected_path, shallow=False)
            finally:
                os.unlink(expected_path)
    finally:
        os.unlink(lib_path)


def test_tensile_lib_logic_to_yaml_read_empty_raises() -> None:
    """Empty read result raises ``RuntimeError``.

    Returns:
        None.

    Raises:
        None.
    """
    with patch.object(M.LibraryIO, "readYAML", return_value=""):
        with pytest.raises(RuntimeError, match="empty"):
            M.TensileLibLogicToYaml("/fake/path.yaml", 0, "/tmp/x.yaml", False)


def test_tensile_lib_logic_to_yaml_bad_solution_index_and_entry(
    tmp_path: Path,
) -> None:
    """Invalid ``solutionIndex`` and empty solution entry raise ``RuntimeError``.

    Returns:
        None.

    Raises:
        None.
    """
    fields = (
        {"MinimumRequiredVersion": "1.0.0"},
        "s",
        "a",
        ["d"],
        {"OperationType": "GEMM", "DataType": "S"},
        [""],
        [],
        None,
        [],
        {},
    )
    with patch.object(M.LibraryIO, "readYAML", return_value={"x": 1}):
        with patch.object(M.LibraryIO, "rawLibraryLogic", return_value=fields):
            with pytest.raises(RuntimeError, match="solution idx"):
                M.TensileLibLogicToYaml("/f.yaml", "", "/tmp/o.yaml", False)
            with pytest.raises(RuntimeError, match="matching data"):
                M.TensileLibLogicToYaml("/f.yaml", 0, "/tmp/o.yaml", False)


def test_main_multi_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` writes one file per index when ``-d`` lists multiple values.

    Args:
        tmp_path: Pytest temp directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        None.
    """
    lib = tmp_path / "in.yaml"
    lib.write_text(_read_liblogic_fixture(), encoding="utf-8")
    out = tmp_path / "out.yaml"
    out.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "TensileLibLogicToYaml",
            "-i",
            str(lib.resolve()),
            "-d",
            "0,0",
            "-o",
            str(out.resolve()),
        ],
    )
    M.main()
    out0 = tmp_path / "out_0.yaml"
    assert out0.is_file()
    assert out0.stat().st_size > 0


def test_main_single_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single index uses the output path unchanged.

    Args:
        tmp_path: Pytest temp directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        None.
    """
    lib = tmp_path / "in.yaml"
    lib.write_text(_read_liblogic_fixture(), encoding="utf-8")
    out = tmp_path / "single.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "TensileLibLogicToYaml",
            "-i",
            str(lib.resolve()),
            "-d",
            "0",
            "-o",
            str(out.resolve()),
        ],
    )
    M.main()
    assert out.is_file()
