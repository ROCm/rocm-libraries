# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``tools/check_shards.py`` over multi-launch (``sources``) UKDs: dQ and merged rows with their companion objects.

A row's launches are its own object, its dBias reduce companion when the manifest lists one for it (gfx11 dQ rows that
store dB), and its tail-zero companion. The shard is synthesised (a manifest and a packed KDP whose provenance sits inline),
so the cases need no pack and no content tree: each breaks one thing a shipped ``sources`` UKD must get right.
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
ROW_SHA, COMPANION_SHA, TAILZERO_SHA = "1" * 64, "2" * 64, "4" * 64
TAILZERO = "fmha_bwd_tailzero_gfx11_kernel_0"
REDUCE = "fmha_bwd_dbias_reduce_gfx11_kernel_0"
GFX950_REDUCE = "fmha_bwd_dbias_reduce_gfx950_kernel_1"
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
        },
        {
            "spec": f"tailzero,bf16,0;arch={TARGET}",
            "file": "bb5ed4007ca3.hsaco",
            "symbol": TAILZERO,
            "sha256": TAILZERO_SHA,
        },
    ],
}

UKD = {
    "name": "dq_bias",
    "arch": [ARCH],
    "sources": {
        "dq": {"kind": "kpack", "symbol": "dq_sym", "sha256": ROW_SHA},
        "dbias_reduce": {"kind": "kpack", "symbol": REDUCE, "sha256": COMPANION_SHA},
        "tailzero": {"kind": "kpack", "symbol": TAILZERO, "sha256": TAILZERO_SHA},
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
            "tailzero": {
                "file": f"{DIR}/bb5ed4007ca3.hsaco",
                "sha256": TAILZERO_SHA,
                "symbol": TAILZERO,
            },
        },
    },
}


def _problems(
    tmp_path: Path,
    monkeypatch,
    ukd: dict,
    manifest: dict = MANIFEST,
    op: str = OP,
    target: str = TARGET,
    arch: str = ARCH,
) -> list[str]:
    content = tmp_path / "FlyDSL"
    folder = content / op / target
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shard = tmp_path / "out" / arch / "FlyDSL" / op / target
    shard.mkdir(parents=True)
    (shard / "x.kdp.json").write_text(
        json.dumps({"kernelDescriptors": [ukd]}), encoding="utf-8"
    )
    monkeypatch.setattr(check_shards.provenance_sidecar, "attach", lambda *a: None)
    return check_shards.check(content, tmp_path / "out", [arch])


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


def test_a_row_missing_its_tailzero_launch_is_refused(tmp_path, monkeypatch):
    ukd = copy.deepcopy(UKD)
    del ukd["sources"]["tailzero"]
    del ukd["provenance"]["sources"]["tailzero"]
    assert any("ships launches" in p for p in _problems(tmp_path, monkeypatch, ukd))


def test_a_pack_with_a_tailzero_launch_the_manifest_lacks_is_refused(
    tmp_path, monkeypatch
):
    manifest = copy.deepcopy(MANIFEST)
    del manifest["companions"][1]
    problems = _problems(tmp_path, monkeypatch, UKD, manifest)
    assert any("ships launches" in p for p in problems), problems


def test_a_row_naming_a_reduce_the_pack_lacks_is_refused(tmp_path, monkeypatch):
    ukd = copy.deepcopy(UKD)
    del ukd["sources"]["dbias_reduce"]
    del ukd["provenance"]["sources"]["dbias_reduce"]
    problems = _problems(tmp_path, monkeypatch, ukd)
    assert any("dbias_reduce" in p and "ships launches" in p for p in problems)


def test_a_reduce_symbol_no_companion_carries_is_refused(tmp_path, monkeypatch):
    manifest = copy.deepcopy(MANIFEST)
    manifest["instances"][0]["knobs"]["reduce_symbol"] = "other_reduce"
    problems = _problems(tmp_path, monkeypatch, UKD, manifest)
    assert any("no dbias_reduce companion" in p for p in problems), problems


def _combined(op: str, main: str, folder_prefix: str, tailzero_file: str) -> tuple:
    """A gfx950 row with its reduce inside its own object: ``{main, dbias_reduce, tailzero}``, the reduce entry naming the row's file."""
    target_dir = f"FlyDSL/{op}/gfx950"
    manifest = {
        "instances": [
            {
                "name": "row",
                "dtype": "bf16",
                "file": "row.hsaco",
                "symbol": "row_sym",
                "sha256": ROW_SHA,
                "knobs": {"reduce_symbol": GFX950_REDUCE},
            }
        ],
        "companions": [
            {
                "spec": "tailzero,bf16,0;arch=gfx950",
                "file": tailzero_file,
                "symbol": "fmha_bwd_tailzero_gfx950_kernel_0",
                "sha256": TAILZERO_SHA,
            }
        ],
    }
    ukd = {
        "name": "row",
        "arch": ["gfx950"],
        "sources": {
            main: {"kind": "kpack", "symbol": "row_sym", "sha256": ROW_SHA},
            "dbias_reduce": {
                "kind": "kpack",
                "symbol": GFX950_REDUCE,
                "sha256": ROW_SHA,
            },
            "tailzero": {"kind": "kpack", "symbol": "x", "sha256": TAILZERO_SHA},
        },
        "provenance": {
            "origin_kind": "hsaco",
            "sources": {
                main: {
                    "file": f"{target_dir}/row.hsaco",
                    "sha256": ROW_SHA,
                    "symbol": "row_sym",
                },
                "dbias_reduce": {
                    "file": f"{target_dir}/row.hsaco",
                    "sha256": ROW_SHA,
                    "symbol": GFX950_REDUCE,
                },
                "tailzero": {
                    "file": f"{folder_prefix}/f69f4ac307db.hsaco",
                    "sha256": TAILZERO_SHA,
                    "symbol": "fmha_bwd_tailzero_gfx950_kernel_0",
                },
            },
        },
    }
    return ukd, manifest


def test_a_gfx950_row_launches_its_reduce_from_its_own_object(tmp_path, monkeypatch):
    ukd, manifest = _combined(
        "sdpa_bwd_dq", "dq", "FlyDSL/sdpa_bwd_dq/gfx950", "f69f4ac307db.hsaco"
    )
    problems = _problems(
        tmp_path, monkeypatch, ukd, manifest, "sdpa_bwd_dq", "gfx950", "gfx950"
    )
    assert problems == []


def test_a_gfx950_row_shipping_no_reduce_launch_is_refused(tmp_path, monkeypatch):
    ukd, manifest = _combined(
        "sdpa_bwd_dq", "dq", "FlyDSL/sdpa_bwd_dq/gfx950", "f69f4ac307db.hsaco"
    )
    del ukd["sources"]["dbias_reduce"]
    del ukd["provenance"]["sources"]["dbias_reduce"]
    problems = _problems(
        tmp_path, monkeypatch, ukd, manifest, "sdpa_bwd_dq", "gfx950", "gfx950"
    )
    assert any("dbias_reduce" in p and "ships launches" in p for p in problems)


def test_a_merged_row_with_a_cross_folder_tailzero_passes(tmp_path, monkeypatch):
    ukd, manifest = _combined(
        "sdpa_bwd_merged",
        "merged",
        "FlyDSL/sdpa_bwd_dq/gfx950",
        "../../sdpa_bwd_dq/gfx950/f69f4ac307db.hsaco",
    )
    problems = _problems(
        tmp_path, monkeypatch, ukd, manifest, "sdpa_bwd_merged", "gfx950", "gfx950"
    )
    assert problems == []


def test_a_merged_row_packing_its_tailzero_from_the_wrong_folder_is_refused(
    tmp_path, monkeypatch
):
    ukd, manifest = _combined(
        "sdpa_bwd_merged",
        "merged",
        "FlyDSL/sdpa_bwd_merged/gfx950",
        "../../sdpa_bwd_dq/gfx950/f69f4ac307db.hsaco",
    )
    problems = _problems(
        tmp_path, monkeypatch, ukd, manifest, "sdpa_bwd_merged", "gfx950", "gfx950"
    )
    assert any("tailzero was packed from" in p for p in problems), problems


def test_a_reduce_row_shipped_as_one_launch_is_refused(tmp_path, monkeypatch):
    single = {
        "name": "dq_bias",
        "arch": [ARCH],
        "kernel_source": {"sha256": ROW_SHA},
        "provenance": {"origin_kind": "hsaco", "file": f"{DIR}/dq_bias.hsaco"},
    }
    assert any("one launch" in p for p in _problems(tmp_path, monkeypatch, single))
