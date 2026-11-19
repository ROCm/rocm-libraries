# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/conv_step_replay.py host-side guards: commands the replay
# cannot size are refused before any allocation, and amd-smi is looked up
# portably. Host-only.

from __future__ import annotations

import os
import stat

import pytest

from benchmarks.common import conv_step_replay as sr
from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

_3D = (
    "-n 1 -c 64 --in_d 10 -H 10 -W 10 -k 64 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 "
    "--conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 "
    "--in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1"
)
_2D = (
    "-n 1 -c 64 -H 10 -W 10 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 "
    "--in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 1 -F 1"
)


def _check(cmd):
    problem, dtype, _ = parse_miopen_cmd(cmd)
    sr.check_replayable("fwd_s1", problem, dtype)


def test_bf16_3d_is_replayable():
    _check(f"MIOpenDriver convbfp16 {_3D}")


@pytest.mark.parametrize(
    "cmd",
    [
        f"MIOpenDriver conv {_3D}",
        f"MIOpenDriver convfp16 {_3D}",
        f"MIOpenDriver convbfp16 {_2D}",
    ],
)
def test_other_commands_are_refused(cmd):
    # The replay sizes and fills operands as bf16 3-D: an fp32 command would
    # bind its kernel on half-size buffers, a 2-D one has no depth.
    with pytest.raises(SystemExit, match="bf16 3-D"):
        _check(cmd)


def _fake_exe(path):
    path.write_text("#!/usr/bin/env python3\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_amd_smi_on_path_and_rocm_path(tmp_path, monkeypatch):
    on_path = tmp_path / "p"
    on_path.mkdir()
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    _fake_exe(rocm / "bin" / "amd-smi")
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.delenv("ROCM_HOME", raising=False)
    assert sr.find_amd_smi() == os.path.join(str(rocm), "bin", "amd-smi")
    exe = _fake_exe(on_path / "amd-smi")
    assert sr.find_amd_smi() == str(exe)
    monkeypatch.setenv("PATH", str(tmp_path / "none"))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "none"))
    assert sr.find_amd_smi() is None
    s = sr.ClockSampler(0)
    s.start()  # no amd-smi: sampling stays off, no thread
    s.stop()
    assert s.summary() is None


def test_graph_check_rejects_nan_and_missing_outputs():
    import numpy as np

    from benchmarks.common.conv_step_replay import GRAPH_CHECK_TOL, graph_max_rel_err

    a = np.array([1.0, -2.0, 4.0], dtype=np.float32)
    assert graph_max_rel_err({"o": a}, {"o": a.copy()}) == 0.0
    off = a.copy()
    off[1] = -2.5
    assert graph_max_rel_err({"o": a}, {"o": off}) == pytest.approx(0.125)
    nan = a.copy()
    nan[2] = np.nan
    for eager, graph in (({"o": a}, {"o": nan}), ({"o": nan}, {"o": a})):
        worst = graph_max_rel_err(eager, graph)
        assert not worst < GRAPH_CHECK_TOL
    assert not graph_max_rel_err({"o": a}, {}) < GRAPH_CHECK_TOL
    inf = a.copy()
    inf[0] = np.inf
    assert not graph_max_rel_err({"o": a}, {"o": inf}) < GRAPH_CHECK_TOL
