# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Existing tune/search/bench CLI dispatch is unchanged by --specialize.

Only --specialize bypasses devices; the other modes keep their runtime options and
log preparation. Break: pass devices=[] at an existing mode boundary. CPU doubles
stop at the GPU workflow boundary; parsing and inline YAML are real.
"""
import pytest

from geko import cli
from geko.bench import log


@pytest.mark.parametrize("mode", ["tune", "search", "bench"])
def test_existing_modes_keep_dispatch_options_and_inline_workload(mode, tmp_path, monkeypatch):
    seen = []
    def workflow(name):
        def run(hipblaslt, workload=None, workdir=None, **kwargs):
            seen.append((name, hipblaslt, workload, workdir, kwargs))
            return 17 if name == "bench" else None
        return run
    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **kwargs: str(tmp_path / "stock"))
    monkeypatch.setattr(cli.device, "main", lambda count: 0)
    for name in ("run_configure", "run_search", "run_bench", "run_optimize"):
        monkeypatch.setattr(cli, name, workflow(name.replace("run_", "")))
    out = tmp_path / "out"
    argv = ["--" + mode, "--inline", "32", "64", "1", "128", "B", "B", "S", "N", "T",
            "--devices", "0", "--arch", "gfx950", "--workdir", str(out)]
    code = cli.main(argv)
    assert code == (17 if mode == "bench" else 0)
    assert [s[0] for s in seen] == ({"tune": ["configure", "optimize"], "search": ["search"], "bench": ["bench"]}[mode])
    assert all(s[4]["devices"] == [0] for s in seen)
    assert all(s[1] == str(tmp_path / "stock") for s in seen)
    synthetic = out / "synthetic_workload.yaml"
    assert seen[0][2] == str(synthetic)
    parsed = log.parse(synthetic)
    assert len(parsed) == 1
    assert (parsed[0]["M"], parsed[0]["N"], parsed[0]["K"], parsed[0]["call_count"]) == (32, 64, 128, 1)
    assert (parsed[0]["a_type"], parsed[0]["d_type"], parsed[0]["compute_type"], parsed[0]["transA"], parsed[0]["transB"]) == ("bf16_r", "bf16_r", "c_f32_r", "N", "T")
    assert not (out / "specialize_manifest.json").exists()
    assert not (out / "logic").exists()
    if mode == "tune":
        assert seen[0][4]["arch"] == "gfx950"
        assert seen[0][4]["keep_thr"] == 0.0
        assert seen[1][4]["n_slots"] == 4
    elif mode == "search":
        assert seen[0][4]["keep_thr"] == 0.1
        assert seen[0][4]["duration"] == 0.04
    else:
        assert seen[0][4]["benchmark_duration"] == 0.5
