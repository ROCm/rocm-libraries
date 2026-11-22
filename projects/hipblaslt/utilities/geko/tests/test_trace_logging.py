# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real hipblaslt-test producers; run under the project's GPU wrapper, -n 0.

--hipblaslt-path points to a build containing clients/hipblaslt-{test,bench}.
HIPBLASLT_TENSILE_LIBPATH and LD_LIBRARY_PATH choose the host/device libraries.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess

import pytest
import yaml

pytestmark = pytest.mark.slow
DATA = Path(__file__).parent / "test_data" / "trace_logging"
OLD_KEYS = (
    "function M N K lda ldb ldc ldd stride_a stride_b stride_c stride_d "
    "alpha alphai beta betai transA transB batch_count batch_mode scaleA scaleB "
    "scaleC scaleD swizzleA swizzleB scaleAlpha_vector gradient use_e bias_vector "
    "bias_source bias_stride a_type b_type c_type d_type scale_type bias_type "
    "aux_type compute_type activation_type flush any_stride rotating cold_iters "
    "iters solution_index solution_Name kernel_name"
).split()


@pytest.fixture(scope="module")
def producer(request, tmp_path_factory):
    build = Path(request.config.getoption("--hipblaslt-path") or "")
    binary = build / "clients" / "hipblaslt-test"
    assert binary.is_file(), "Supply --hipblaslt-path with the trace producer build"
    directory = tmp_path_factory.mktemp("trace_logging")
    cache = {}

    def run(scenario, mask):
        key = scenario, mask
        if key not in cache:
            path = directory / f"{scenario}-{mask}.yaml"
            child_env = os.environ.copy()
            child_env.pop("HIPBLASLT_LOG_LEVEL", None)
            child_env["HIPBLASLT_LOG_MASK"] = str(mask)
            child_env["HIPBLASLT_LOG_FILE"] = str(path)
            command = [str(binary), f"--gtest_filter=TraceLoggingProducer.{scenario}"]
            result = subprocess.run(command, env=child_env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            (directory / f"{scenario}-{mask}.out").write_text(result.stdout)
            assert result.returncode == 0, result.stdout
            assert "[  PASSED  ] 1 test." in result.stdout, result.stdout
            assert path.is_file(), f"Real producer wrote no mask-{mask} log"
            text = path.read_text()
            records = text.splitlines() if mask == 32 else yaml.safe_load(text)
            assert records, f"Real producer wrote an empty mask-{mask} log"
            cache[key] = records, text
        return cache[key]

    return run


@pytest.mark.parametrize("mask", [32, 64, 128])
def test_sigmoid_is_named_in_each_log_mask(producer, mask):
    """Feature: add Sigmoid to the host converter; removing that case breaks it."""
    records, _ = producer("Sigmoid", mask)
    if mask == 32:
        tokens = shlex.split(records[0])
        activation = tokens[tokens.index("--activation_type") + 1]
    else:
        assert len(records) == 1
        activation = records[0]["activation_type"]
    assert activation == "sigmoid", f"mask-{mask} sigmoid was logged as {activation!r}"


@pytest.mark.parametrize("scenario,want", [("Default", 0), ("Device", 1), ("Vector", 4)])
def test_extended_profile_logs_raw_pointer_mode(producer, scenario, want):
    """Feature: append the raw enum, not SAV/host inference, to mask 128."""
    records, _ = producer(scenario, 128)
    assert len(records) == 1
    assert "pointer_mode" in records[0], "mask-128 record omitted raw pointer_mode"
    assert type(records[0]["pointer_mode"]) is int
    assert records[0]["pointer_mode"] == want
    assert records[0]["call_count"] == 1


def test_extended_profile_logs_clamp_arguments(producer):
    """Feature: log both descriptor arguments; omitting either breaks it."""
    records, _ = producer("Clamp", 128)
    assert len(records) == 1
    row = records[0]
    assert row["activation_type"] == "clamp"
    assert "activation_arg0" in row, "mask-128 clamp record omitted activation_arg0"
    assert "activation_arg1" in row, "mask-128 clamp record omitted activation_arg1"
    assert (row["activation_arg0"], row["activation_arg1"]) == (-1.5, 2.25)


def test_pointer_mode_separates_aggregation_counts(producer):
    """Feature: include pointer_mode in the real argument_profile tuple key."""
    records, _ = producer("PointerPair", 128)
    assert len(records) == 2, f"pointer-mode calls collapsed into {len(records)} record(s)"
    assert {r["pointer_mode"]: r["call_count"] for r in records} == {1: 2, 4: 3}
    left, right = [{k: v for k, v in r.items() if k not in ("pointer_mode", "call_count")}
                   for r in records]
    assert left == right, "Fixture must isolate pointer mode, not a different solution or shape"


@pytest.mark.parametrize("scenario,changed", [
    ("ClampArg0Pair", (-0.5, 2.25)), ("ClampArg1Pair", (-1.5, 3.5)),
])
def test_clamp_arguments_separate_aggregation_counts(producer, scenario, changed):
    """Feature: include both activation arguments in the aggregation tuple key."""
    records, _ = producer(scenario, 128)
    assert len(records) == 2, f"clamp-argument calls collapsed into {len(records)} record(s)"
    assert {(r["activation_arg0"], r["activation_arg1"]): r["call_count"] for r in records} == {
        (-1.5, 2.25): 2, changed: 3,
    }
    left, right = [{k: v for k, v in r.items()
                    if k not in ("activation_arg0", "activation_arg1", "call_count")}
                   for r in records]
    assert left == right, "Fixture must isolate clamp arguments"


def test_extended_profile_keeps_old_key_order(producer):
    """Preservation: append fields; moving/removing an old field breaks consumers."""
    records, _ = producer("Default", 128)
    keys = list(records[0])
    assert [key for key in keys if key in OLD_KEYS] == OLD_KEYS
    assert keys[-1] == "call_count"


@pytest.mark.parametrize("mask", [32, 64, 128])
@pytest.mark.parametrize("scenario,want", [
    ("Default", "none"), ("Relu", "relu"), ("DRelu", "relu"),
    ("Gelu", "gelu"), ("DGelu", "gelu"), ("Silu", "swish"), ("Clamp", "clamp"),
])
def test_other_activation_names_are_preserved(producer, scenario, want, mask):
    """Preservation: changing the Silu mapping to none loses the real swish call."""
    records, _ = producer(scenario, mask)
    if mask == 32:
        tokens = shlex.split(records[0])
        got = tokens[tokens.index("--activation_type") + 1]
    else:
        assert len(records) == 1
        got = records[0]["activation_type"]
    assert got == want, f"mask-{mask} {scenario} was logged as {got!r}, expected {want!r}"


@pytest.mark.parametrize("scenario", ["Default", "Sigmoid"])
def test_mask64_record_is_unchanged_except_sigmoid(producer, scenario):
    """Preservation: changing a profile value/key/order fails the captured oracle."""
    records, _ = producer(scenario, 64)
    expected = yaml.safe_load((DATA / f"mask64-{scenario.lower()}.yaml").read_text())
    assert records == expected, "mask-64 output changed beyond the sigmoid correction"
    assert list(records[0]) == list(expected[0]), "mask-64 field order changed"


def test_logged_sigmoid_bench_command_round_trips(producer, request, tmp_path):
    """Feature: corrected producer line must still be accepted and execute sigmoid."""
    records, _ = producer("Sigmoid", 32)
    assert len(records) == 1
    tokens = shlex.split(records[0])
    assert tokens[tokens.index("--activation_type") + 1] == "sigmoid", (
        "logged bench line cannot round-trip sigmoid: producer logged another activation"
    )
    binary = Path(request.config.getoption("--hipblaslt-path")) / "clients" / "hipblaslt-bench"
    child_env = os.environ.copy()
    child_env.pop("HIPBLASLT_LOG_LEVEL", None)
    child_env["HIPBLASLT_LOG_MASK"] = "0"
    child_env["HIPBLASLT_LOG_FILE"] = str(tmp_path / "roundtrip.yaml")
    # Replay every logged argument verbatim; only add the enforced numerical check.
    result = subprocess.run([str(binary), *tokens[1:], "-v"], env=child_env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (tmp_path / "roundtrip.out").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    lines = result.stdout.splitlines()
    verdicts = []
    for i, line in enumerate(lines[:-1]):
        header = next(csv.reader([line]))
        if "norm_error" in header:
            values = next(csv.reader([lines[i + 1]]))
            verdicts.append(float(values[header.index("norm_error")]))
    assert verdicts, "hipblaslt-bench emitted no enforced norm_error verdict"
    assert all(math.isfinite(error) and error < 0.00001 for error in verdicts), verdicts


def _f32(value):
    """Round a parsed number to binary32, the type of the descriptor argument."""
    return struct.unpack("f", struct.pack("f", value))[0]


ONE_ULP_ABOVE_1 = struct.unpack("f", struct.pack("I", 0x3F800001))[0]


@pytest.mark.parametrize("scenario,arg", [
    ("ClampArg0Ulp", "activation_arg0"), ("ClampArg1Ulp", "activation_arg1"),
])
def test_one_ulp_apart_arguments_log_distinct_exact_values(producer, scenario, arg):
    """Round-trip precision: 6 significant digits would print both values as 1."""
    records, _ = producer(scenario, 128)
    assert len(records) == 2, f"ulp-apart calls collapsed into {len(records)} record(s)"
    values = [r[arg] for r in records]
    assert len(set(values)) == 2, f"ulp-apart {arg} values print identically: {values}"
    assert {_f32(r[arg]): r["call_count"] for r in records} == {1.0: 2, ONE_ULP_ABOVE_1: 3}


def test_exponent_form_arguments_parse_as_exact_floats(producer):
    """'1e+10' is a YAML string; the record must stay a decimal float."""
    records, _ = producer("ClampExponent", 128)
    assert len(records) == 1
    row = records[0]
    for key, want in (("activation_arg0", -2e10), ("activation_arg1", 1e10)):
        assert isinstance(row[key], float), f"{key} parsed as {type(row[key]).__name__}: {row[key]!r}"
        assert _f32(row[key]) == want
