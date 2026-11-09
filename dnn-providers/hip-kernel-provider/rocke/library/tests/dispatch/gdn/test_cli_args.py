# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU contract of ``dispatch.gdn.cli_args``: the helpers the GDN/KDA decode
tools (driver, sweep, defaults report, benchmarks) share, one definition each."""

from __future__ import annotations

import argparse
import math

import pytest

from dispatch.gdn import cli_args


@pytest.mark.parametrize(
    "text", ["", ",", "1,,2", "1,", "a", "1,x", "0", "-3,1", "1,0"]
)
def test_parse_batches_refuses_anything_but_positive_ints(text):
    with pytest.raises(argparse.ArgumentTypeError):
        cli_args.parse_batches(text)


def test_parse_batches_returns_a_tuple_of_ints():
    assert cli_args.parse_batches("1, 8,64") == (1, 8, 64)


@pytest.mark.parametrize("value", [None, 0, 1, 1024])
def test_rotate_mb_accepts_unset_warm_and_cold(value):
    cli_args.check_rotate_mb(value)
    cli_args.check_rotate_mb(value, argparse.ArgumentParser())


def test_rotate_mb_below_zero_is_one_rule_with_two_reporters(capsys):
    with pytest.raises(ValueError, match=r"^rotate_mb must be >= 0 \(0 = warm\): -1$"):
        cli_args.check_rotate_mb(-1)
    with pytest.raises(ValueError, match=r"^batch_mb must be >= 0"):
        cli_args.check_rotate_mb(-5, name="batch_mb")
    ap = argparse.ArgumentParser()
    with pytest.raises(SystemExit) as err:
        cli_args.check_rotate_mb(-1, ap)
    assert err.value.code == 2
    assert "--rotate-mb must be >= 0 (0 = warm)" in capsys.readouterr().err


def test_usage_error_text_names_a_file_exactly_as_given():
    path = "C:\\Users\\x\\missing.json"
    err = FileNotFoundError(2, "No such file or directory", path)
    assert path.replace("\\", "\\\\") in str(err)  # str() repr-escapes the path
    assert cli_args.usage_error_text(err) == f"{path}: No such file or directory"
    assert cli_args.usage_error_text(OSError("no file named")) == "no file named"
    assert cli_args.usage_error_text(ValueError("bad row")) == "bad row"


def test_nan_safe_max_is_nan_in_any_position():
    assert cli_args.nan_safe_max((1.0, 3.0, 2.0)) == 3.0
    for errs in ((math.nan, 1.0, 2.0), (1.0, math.nan, 2.0), (1.0, 2.0, math.nan)):
        assert math.isnan(cli_args.nan_safe_max(errs))


def _row(index, us, status="ok"):
    return {"index": index, "us": us, "status": status}


def test_fastest_rows_order_by_time_then_index_and_skip_untimed():
    rows = [_row(3, 2.0), _row(1, 2.0), _row(0, 1.0, "fail"), _row(2, 5.0)]
    assert [r["index"] for r in cli_args.fastest_first(rows)] == [1, 3, 2]
    assert cli_args.fastest_timed(rows)["index"] == 1
    assert cli_args.fastest_timed([_row(0, 1.0, "error")]) is None


def test_the_module_needs_no_torch_and_no_upper_layer():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli_args))
    roots = {
        (
            (n.module or "").split(".")[0]
            if isinstance(n, ast.ImportFrom)
            else a.name.split(".")[0]
        )
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    assert roots <= {"__future__", "argparse", "math", "typing"}, roots
