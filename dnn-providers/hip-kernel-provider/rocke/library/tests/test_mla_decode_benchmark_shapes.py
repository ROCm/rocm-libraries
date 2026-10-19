# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MLA decode shape files resolve through dispatch as the driver expects.

CPU only. ``benchmarks.common.benchmark_mla_decode_live`` verifies and times
whatever these files hold, so a dispatch change that stops serving one of
their shapes -- or starts serving a head count nobody validated -- shows up
here rather than as a quietly shorter benchmark run. The shapes decode refuses
today (a tensor-parallel rank's 16 heads) stay in the files on purpose, and
are pinned as refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common import benchmark_mla_decode_live as live
from dispatch.mla import dispatch_mla
from dispatch.mla.decode import DECODE_NUM_HEADS

ARCHES = ("gfx942", "gfx950")
_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


@pytest.fixture(autouse=True)
def _llvm22(monkeypatch):
    """Decode admission refuses llvm20; pin a flavor it serves."""
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    """CPU only, whatever GPU the runner has: with no device arch the CU count
    and the grid limits take their fallbacks, as on a GPU-less CI box."""
    from rocke.runtime import hip_module

    monkeypatch.setattr(hip_module, "get_device_arch", lambda device=0: None)


def _shapes(arch):
    path = _BENCH / arch / "attention" / "decode" / "mla_shapes.json"
    return live._decode_shapes(path, model=None, limit=None)


@pytest.mark.parametrize("arch", ARCHES)
def test_served_head_counts_dispatch_to_decode(arch):
    served = [s for s in _shapes(arch) if s.num_query_heads in DECODE_NUM_HEADS]
    assert len(served) == 26
    for s in served:
        result = dispatch_mla(s.to_request(arch=arch))
        assert result.candidate.name == f"mla_decode_{arch}", s.signature


@pytest.mark.parametrize("arch", ARCHES)
def test_tensor_parallel_rank_is_refused(arch):
    refused = [s for s in _shapes(arch) if s.num_query_heads not in DECODE_NUM_HEADS]
    assert {s.num_query_heads for s in refused} == {16}
    for s in refused:
        with pytest.raises(ValueError, match="num_heads=16 not in"):
            dispatch_mla(s.to_request(arch=arch))


@pytest.mark.parametrize("arch", ARCHES)
def test_resolve_mode_reports_every_shape(arch, capsys):
    assert live.main(["--arch", arch]) == 0
    out = capsys.readouterr().out
    assert "SUMMARY 26/28 shapes resolved" in out and "2 refused by dispatch" in out


def test_build_mode_builds_the_combine_too(monkeypatch):
    # A split selection is two kernels; --mode build must build both.
    from dispatch.mla import decode

    built = []
    real = decode.MlaDecodeLaunch.build_combine
    monkeypatch.setattr(
        decode.MlaDecodeLaunch,
        "build_combine",
        lambda self: built.append(self.combine_spec) or real(self),
    )
    split = next(s for s in _shapes("gfx942") if s.label == "ds_v3_b1_kv32768")
    row = live._resolve(split, arch="gfx942", build=True)
    assert row["status"] == "ok" and row["num_seg"] > 1
    assert len(built) == 1


@pytest.mark.parametrize("arch", ARCHES)
def test_resolve_mode_fails_when_a_served_head_count_is_refused(
    arch, monkeypatch, capsys
):
    # llvm20 refuses every shape: that is a failure, not 28 skips.
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    assert live.main(["--arch", arch]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_report_fails_a_build_failure_not_a_refusal():
    ok = {
        "shape": "a",
        "status": "ok",
        "candidate": "c",
        "kernels": ["k"],
        "grid": (1, 1, 1),
        "num_seg": 1,
    }
    refused = {
        "shape": "b",
        "status": "unsupported",
        "detail": "num_heads=16 not in (64, 128)",
    }
    broken = {"shape": "c", "status": "build_failed", "detail": "RuntimeError: x"}
    assert live._report([ok, refused], "gfx942") == 0
    assert live._report([ok, refused, broken], "gfx942") == 1
