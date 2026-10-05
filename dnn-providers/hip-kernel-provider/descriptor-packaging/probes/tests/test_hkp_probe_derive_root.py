"""Tests for hkp_probe_derive_root.py (pure stdlib; no compile, no kpack)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_DP = Path(__file__).resolve().parents[2]
_TOOL = _DP / "tools" / "hkp_probe_derive_root.py"
_PRODUCTION_ROOT = (
    _DP.parent
    / "src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_attention_dense"
)
_PRODUCTION_KDP = "gfx950_attention_dense.kdp.json"
_PRODUCTION_INSTANCE = "attention_dense.bf16_d64_hq8_kv1_c1_bm256_bn64.gfx950"


def _run(src: Path, kdp: str, instance: str, out: Path):
    return subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--from",
            str(src),
            "--kdp",
            kdp,
            "--instance-name",
            instance,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_synthetic_root(root: Path, names: list[str]) -> None:
    root.mkdir()
    kdp = {
        "version": "1.0",
        "id": "kdp-id",
        "arch": "gfx950",
        "kernelDescriptors": [{"name": n, "payload": i} for i, n in enumerate(names)],
    }
    (root / "x.kdp.json").write_text(json.dumps(kdp, indent=2) + "\n")
    (root / "x.kmd.json").write_text('{"id": "kmd"}\n')


def test_instance_absent_exits_2(tmp_path):
    src = tmp_path / "src"
    _write_synthetic_root(src, ["a", "b"])
    r = _run(src, "x.kdp.json", "missing", tmp_path / "out")
    assert r.returncode == 2
    assert (
        r.stderr.strip()
        == "hkp_probe_derive: instance 'missing' not found in x.kdp.json (2 descriptors)"
    )
    assert not (tmp_path / "out").exists()


def test_instance_duplicated_exits_2(tmp_path):
    src = tmp_path / "src"
    _write_synthetic_root(src, ["a", "dup", "dup"])
    r = _run(src, "x.kdp.json", "dup", tmp_path / "out")
    assert r.returncode == 2
    assert "hkp_probe_derive: instance 'dup' matched 2 times" in r.stderr
    assert not (tmp_path / "out").exists()


def test_kdp_absent_exits_2(tmp_path):
    src = tmp_path / "src"
    _write_synthetic_root(src, ["a"])
    r = _run(src, "nope.kdp.json", "a", tmp_path / "out")
    assert r.returncode == 2
    assert r.stderr.strip() == f"hkp_probe_derive: nope.kdp.json not found under {src}"


def test_stale_output_removed(tmp_path):
    src = tmp_path / "src"
    _write_synthetic_root(src, ["a", "b"])
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("stale")
    r = _run(src, "x.kdp.json", "b", out)
    assert r.returncode == 0, r.stderr
    assert not (out / "stale.txt").exists()
    kdp = json.loads((out / "x.kdp.json").read_text())
    assert [d["name"] for d in kdp["kernelDescriptors"]] == ["b"]
    assert (out / "x.kmd.json").read_bytes() == (src / "x.kmd.json").read_bytes()


def test_real_production_root(tmp_path):
    out = tmp_path / "out"
    r = _run(_PRODUCTION_ROOT, _PRODUCTION_KDP, _PRODUCTION_INSTANCE, out)
    assert r.returncode == 0, r.stderr

    src_kdp = json.loads((_PRODUCTION_ROOT / _PRODUCTION_KDP).read_text())
    out_kdp = json.loads((out / _PRODUCTION_KDP).read_text())
    assert [d["name"] for d in out_kdp["kernelDescriptors"]] == [_PRODUCTION_INSTANCE]
    assert list(out_kdp) == list(src_kdp)
    for key in ("id", "engine", "dispatch", "matchers", "arch", "provenance"):
        assert out_kdp[key] == src_kdp[key]

    siblings = sorted(
        p.name for p in _PRODUCTION_ROOT.iterdir() if p.name != _PRODUCTION_KDP
    )
    assert len(siblings) == 5
    for name in siblings:
        assert (out / name).read_bytes() == (_PRODUCTION_ROOT / name).read_bytes()
    assert sorted(p.name for p in out.iterdir()) == sorted(siblings + [_PRODUCTION_KDP])


def test_production_instance_exactly_once():
    kdp = json.loads((_PRODUCTION_ROOT / _PRODUCTION_KDP).read_text())
    names = [d["name"] for d in kdp["kernelDescriptors"]]
    assert names.count(_PRODUCTION_INSTANCE) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
