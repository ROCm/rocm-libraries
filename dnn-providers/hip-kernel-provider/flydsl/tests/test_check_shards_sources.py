# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``tools/check_shards.py`` over multi-launch (``sources``) UKDs: a dQ row whose dBias reduce is a companion object.

The shard is synthesised (a manifest and a packed KDP whose provenance sits inline), so the cases need no pack
and no content tree: each breaks one thing a shipped ``sources`` UKD must get right.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
sys.path.insert(0, str(FLYDSL))
import check_shards  # noqa: E402

OP, TARGET, ARCH = "sdpa_bwd_dq", "gfx11-generic", "gfx1100"
ROW_SHA, COMPANION_SHA = "1" * 64, "2" * 64
REDUCE = "fmha_bwd_dbias_reduce_gfx11_kernel_0"
DIR = f"FlyDSL/{OP}/{TARGET}"

MANIFEST = {
    "instances": [
        {
            "name": "dq_bias",
            "dtype": "bf16",
            "file": "dq_bias.hsaco",
            "symbol": "dq_sym",
            "sha256": ROW_SHA,
            "knobs": {"reduce_symbol": REDUCE},
        }
    ],
    "companions": [
        {
            "spec": f"dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1;arch={TARGET}",
            "file": "aa5ed4007ca3.hsaco",
            "symbol": REDUCE,
            "sha256": COMPANION_SHA,
        }
    ],
}

UKD = {
    "name": "dq_bias",
    "arch": [ARCH],
    "sources": {
        "dq": {"kind": "kpack", "symbol": "dq_sym", "sha256": ROW_SHA},
        "dbias_reduce": {"kind": "kpack", "symbol": REDUCE, "sha256": COMPANION_SHA},
    },
    "provenance": {
        "origin_kind": "hsaco",
        "sources": {
            "dq": {
                "file": f"{DIR}/dq_bias.hsaco",
                "sha256": ROW_SHA,
                "symbol": "dq_sym",
            },
            "dbias_reduce": {
                "file": f"{DIR}/aa5ed4007ca3.hsaco",
                "sha256": COMPANION_SHA,
                "symbol": REDUCE,
            },
        },
    },
}


def _problems(tmp_path: Path, monkeypatch, ukd: dict) -> list[str]:
    content = tmp_path / "FlyDSL"
    folder = content / OP / TARGET
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    shard = tmp_path / "out" / ARCH / "FlyDSL" / OP / TARGET
    shard.mkdir(parents=True)
    (shard / "x.kdp.json").write_text(
        json.dumps({"kernelDescriptors": [ukd]}), encoding="utf-8"
    )
    monkeypatch.setattr(check_shards.provenance_sidecar, "attach", lambda *a: None)
    return check_shards.check(content, tmp_path / "out", [ARCH])


def test_a_sources_ukd_that_is_the_checked_in_objects_passes(tmp_path, monkeypatch):
    assert _problems(tmp_path, monkeypatch, UKD) == []


def _edit(path: tuple[str, ...], value) -> dict:
    ukd = copy.deepcopy(UKD)
    node = ukd
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return ukd


@pytest.mark.parametrize(
    ("ukd", "diagnostic"),
    [
        (
            _edit(("provenance", "sources", "dbias_reduce", "sha256"), "3" * 64),
            "dbias_reduce ships payload",
        ),
        (_edit(("sources", "dq", "sha256"), "3" * 64), "dq ships payload"),
        (
            _edit(("provenance", "sources", "dq", "file"), f"{DIR}/other.hsaco"),
            "dq was packed from",
        ),
        (
            _edit(("provenance", "sources", "dbias_reduce", "symbol"), "other"),
            "dbias_reduce is symbol",
        ),
        (_edit(("provenance", "origin_kind"), "rocke"), "not packed from an hsaco"),
    ],
    ids=["reduce-payload", "dq-payload", "dq-file", "reduce-symbol", "origin"],
)
def test_a_sources_ukd_that_is_not_the_checked_in_objects_is_refused(
    tmp_path, monkeypatch, ukd, diagnostic
):
    problems = _problems(tmp_path, monkeypatch, ukd)
    assert any(diagnostic in p for p in problems), problems


def test_a_launch_missing_from_a_sources_ukd_is_refused(tmp_path, monkeypatch):
    ukd = copy.deepcopy(UKD)
    del ukd["sources"]["dbias_reduce"]
    del ukd["provenance"]["sources"]["dbias_reduce"]
    assert any("ships launches" in p for p in _problems(tmp_path, monkeypatch, ukd))


def test_a_reduce_row_shipped_as_one_launch_is_refused(tmp_path, monkeypatch):
    single = {
        "name": "dq_bias",
        "arch": [ARCH],
        "kernel_source": {"sha256": ROW_SHA},
        "provenance": {"origin_kind": "hsaco", "file": f"{DIR}/dq_bias.hsaco"},
    }
    assert any("one launch" in p for p in _problems(tmp_path, monkeypatch, single))
