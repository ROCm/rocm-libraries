# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The prep ``sink`` pack: a sink-twin prep row (``sink=1``) launches ``prep`` then ``dsink_reduce``.

The dSink reduce objects are the content op ``sdpa_bwd_dsink_reduce`` (``gen_descriptors.COMPANION_ONLY_OPS``): no slot, no
descriptors of their own, and no companion in the prep manifest, so prep's manifest and ``inputs_sha256`` are what they were
before. ``gen_descriptors.py`` names each object of the row's dtype by a path into that op's folder, ``tools/check_shards.py``
holds the packed shard to it. Host-only: manifests and shards are synthesised and no object is compiled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
sys.path.insert(0, str(FLYDSL))
import check_shards  # noqa: E402
import gen_descriptors as gd  # noqa: E402

ARCH = "gfx950"
PREP, DSINK = "sdpa_bwd_prep", "sdpa_bwd_dsink_reduce"
TWIN_SYMBOL = "fmha_bwd_prep_sink_gfx950_kernel_0"
DSINK_SYMBOL = "fmha_bwd_dsink_reduce_gfx950_kernel_0"
# The sweep's pins of the two dSink reduce objects (``tests/data/sdpa_bwd_reference.csv``): spec hash by dtype.
DSINK_FILES = {"bf16": "34902dd91db1.hsaco", "f16": "185bcc2fd31d.hsaco"}


def _blob(name: str) -> bytes:
    return name.encode()


def _sha(name: str) -> str:
    return hashlib.sha256(_blob(name)).hexdigest()


def _prep_row(name: str, dtype: str, sink: int) -> dict:
    return {
        "name": name,
        "file": f"{name}.hsaco",
        "sha256": _sha(f"{name}.hsaco"),
        "symbol": TWIN_SYMBOL if sink else "fmha_bwd_prep_gfx950_kernel_0",
        "priority": 0,
        "dtype": dtype,
        "knobs": {"head_dim": 64, "padded": 0, "sink": sink, "block_threads": 256},
    }


def _dsink_row(dtype: str) -> dict:
    file = DSINK_FILES[dtype]
    return {
        "name": f"dsink_reduce_{dtype}",
        "file": file,
        "sha256": _sha(file),
        "symbol": DSINK_SYMBOL,
        "priority": 0,
        "dtype": dtype,
        "spec": f"dsink_reduce,{dtype},32,plain,bk,0,0,tail,0,1",
        "knobs": {},
    }


PREP_ROWS = [
    _prep_row("plain_bf16", "bf16", 0),
    _prep_row("twin_bf16", "bf16", 1),
    _prep_row("twin_f16", "f16", 1),
]


def _write(content: Path, op: str, rows: list[dict]) -> None:
    folder = content / op / ARCH
    folder.mkdir(parents=True)
    for row in rows:
        (folder / row["file"]).write_bytes(_blob(row["file"]))
    manifest = {
        "op": op,
        "arch": ARCH,
        "toolchain": {"inputs_sha256": "synthetic"},
        "instances": rows,
    }
    (folder / gd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(gd, "inputs_digest", lambda op, arch: "synthetic")
    monkeypatch.setattr(gd, "describe", lambda data, where, symbol: None)
    monkeypatch.setattr(gd, "verify_arch", lambda *args: None)
    monkeypatch.setattr(gd, "verify_generic", lambda *args: None)


def _entries(
    tmp_path: Path, dsink_dtypes: tuple[str, ...] = ("bf16", "f16")
) -> list[gd.KernelEntry]:
    _write(tmp_path, PREP, copy.deepcopy(PREP_ROWS))
    if dsink_dtypes:
        _write(tmp_path, DSINK, [_dsink_row(d) for d in dsink_dtypes])
    return gd.collect(tmp_path, ARCH)


def _pack(entries: list[gd.KernelEntry]) -> tuple[gd.OpDescriptors, dict]:
    # A slot op takes the split engine's UED id; these tests check packs, not that id.
    op = gd.OpDescriptors(PREP, gd._fields_for_op(entries), ued="split-engine-id")
    return op, gd._arch_documents(entries, op, ARCH)


def test_the_dsink_reduce_op_is_no_slot_row_and_a_twin_row_names_the_object_of_its_dtype(
    tmp_path, stubbed
):
    entries = _entries(tmp_path)
    assert {e.op for e in entries} == {PREP}
    by_name = {e.record["name"]: e for e in entries}
    assert by_name["plain_bf16"].companions is None
    for name, dtype in (("twin_bf16", "bf16"), ("twin_f16", "f16")):
        launch = by_name[name].companions[gd.DSINK_REDUCE_ROLE]
        assert launch["file"] == f"../../{DSINK}/{ARCH}/{DSINK_FILES[dtype]}"
        assert launch["symbol"] == DSINK_SYMBOL


def test_a_twin_row_whose_dtype_has_no_dsink_reduce_object_is_refused(
    tmp_path, stubbed
):
    with pytest.raises(ValueError, match="dsink_reduce object of dtype f16"):
        _entries(tmp_path, dsink_dtypes=("bf16",))


def test_a_twin_row_with_no_dsink_reduce_content_at_all_is_refused(tmp_path, stubbed):
    with pytest.raises(ValueError, match="generate it first"):
        _entries(tmp_path, dsink_dtypes=())


def test_the_twin_rows_are_the_sink_pack_and_the_plain_row_stays_one_launch(
    tmp_path, stubbed
):
    op, pack = _pack(_entries(tmp_path))
    assert sorted(pack) == ["flydsl_sdpa_bwd_prep.kdp.json", f"{ARCH}.sink.kdp.json"]
    (plain,) = pack["flydsl_sdpa_bwd_prep.kdp.json"]["kernelDescriptors"]
    assert (
        "sources" not in plain and plain["kernel_source"]["file"] == "plain_bf16.hsaco"
    )
    sink = pack[f"{ARCH}.sink.kdp.json"]
    assert [k["name"] for k in sink["kernelDescriptors"]] == ["twin_bf16", "twin_f16"]
    for kernel in sink["kernelDescriptors"]:
        assert "kernel_source" not in kernel
        # Launch order: the twin writes the sink terms, the reduce folds them.
        assert list(kernel["sources"]) == ["prep", gd.DSINK_REDUCE_ROLE]
        assert kernel["sources"]["prep"]["symbol"] == TWIN_SYMBOL
    bf16, f16 = sink["kernelDescriptors"]
    assert bf16["sources"][gd.DSINK_REDUCE_ROLE] == {
        "kind": "hsaco",
        "file": f"../../{DSINK}/{ARCH}/{DSINK_FILES['bf16']}",
        "symbol": DSINK_SYMBOL,
    }
    assert f16["sources"][gd.DSINK_REDUCE_ROLE]["file"].endswith(DSINK_FILES["f16"])
    assert (sink["slot"], sink["version"]) == ("prep", "1.1")


def test_the_sink_pack_has_its_own_dispatch_symbol(tmp_path, stubbed):
    op, pack = _pack(_entries(tmp_path))
    udd = op.documents()["flydsl_sdpa_bwd_prep_sink.udd.json"]
    assert (
        udd["dispatch_symbol"] == "hipkernel.flydsl_sdpa_bwd_split.prep_sink.dispatch"
    )
    assert pack[f"{ARCH}.sink.kdp.json"]["dispatch"] == udd["id"]
    plain_udd = op.documents()["flydsl_sdpa_bwd_prep.udd.json"]
    assert (
        pack["flydsl_sdpa_bwd_prep.kdp.json"]["dispatch"]
        == plain_udd["id"]
        != udd["id"]
    )


# ---------------------------------------------------------------------------
# tools/check_shards.py
# ---------------------------------------------------------------------------

DIR = f"FlyDSL/{PREP}/{ARCH}"
DSINK_DIR = f"FlyDSL/{DSINK}/{ARCH}"
TWIN = PREP_ROWS[1]
TWIN_UKD = {
    "name": TWIN["name"],
    "arch": [ARCH],
    "sources": {
        "prep": {"kind": "kpack", "symbol": TWIN_SYMBOL, "sha256": TWIN["sha256"]},
        "dsink_reduce": {
            "kind": "kpack",
            "symbol": DSINK_SYMBOL,
            "sha256": _sha(DSINK_FILES["bf16"]),
        },
    },
    "provenance": {
        "origin_kind": "hsaco",
        "sources": {
            "prep": {
                "file": f"{DIR}/{TWIN['file']}",
                "sha256": TWIN["sha256"],
                "symbol": TWIN_SYMBOL,
            },
            "dsink_reduce": {
                "file": f"{DSINK_DIR}/{DSINK_FILES['bf16']}",
                "sha256": _sha(DSINK_FILES["bf16"]),
                "symbol": DSINK_SYMBOL,
            },
        },
    },
}


def _problems(tmp_path: Path, monkeypatch, ukd: dict, with_dsink: bool = True):
    content = tmp_path / "FlyDSL"
    for op, rows in (
        (PREP, [dict(TWIN)]),
        *(((DSINK, [_dsink_row("bf16")]),) if with_dsink else ()),
    ):
        folder = content / op / ARCH
        folder.mkdir(parents=True)
        manifest = {"op": op, "instances": rows}
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shard = tmp_path / "out" / ARCH / "FlyDSL" / PREP / ARCH
    shard.mkdir(parents=True)
    (shard / "x.kdp.json").write_text(
        json.dumps({"kernelDescriptors": [ukd]}), encoding="utf-8"
    )
    monkeypatch.setattr(check_shards.provenance_sidecar, "attach", lambda *a: None)
    return check_shards.check(content, tmp_path / "out", [ARCH])


def _edit(path: tuple[str, ...], value) -> dict:
    ukd = copy.deepcopy(TWIN_UKD)
    node = ukd
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return ukd


def test_a_twin_row_shipped_with_the_checked_in_dsink_reduce_passes(
    tmp_path, monkeypatch
):
    assert _problems(tmp_path, monkeypatch, TWIN_UKD) == []


@pytest.mark.parametrize(
    ("ukd", "diagnostic"),
    [
        (
            _edit(("provenance", "sources", "dsink_reduce", "sha256"), "3" * 64),
            "dsink_reduce ships payload",
        ),
        (
            _edit(
                ("provenance", "sources", "dsink_reduce", "file"),
                f"{DIR}/{DSINK_FILES['bf16']}",
            ),
            "dsink_reduce was packed from",
        ),
        (
            _edit(("provenance", "sources", "dsink_reduce", "symbol"), "other"),
            "dsink_reduce is symbol",
        ),
    ],
    ids=["payload", "file-in-the-prep-folder", "symbol"],
)
def test_a_twin_row_whose_dsink_reduce_is_not_the_checked_in_object_is_refused(
    tmp_path, monkeypatch, ukd, diagnostic
):
    problems = _problems(tmp_path, monkeypatch, ukd)
    assert any(diagnostic in p for p in problems), problems


def test_a_twin_row_missing_its_dsink_reduce_launch_is_refused(tmp_path, monkeypatch):
    ukd = copy.deepcopy(TWIN_UKD)
    del ukd["sources"]["dsink_reduce"], ukd["provenance"]["sources"]["dsink_reduce"]
    problems = _problems(tmp_path, monkeypatch, ukd)
    assert any("ships launches ['prep']" in p for p in problems), problems


def test_a_twin_row_shipped_as_one_launch_is_refused(tmp_path, monkeypatch):
    ukd = {
        "name": TWIN["name"],
        "arch": [ARCH],
        "kernel_source": {"sha256": TWIN["sha256"]},
        "provenance": {"origin_kind": "hsaco", "file": f"{DIR}/{TWIN['file']}"},
    }
    problems = _problems(tmp_path, monkeypatch, ukd)
    assert any("launches companion objects" in p for p in problems), problems


def test_a_twin_row_whose_dsink_reduce_content_is_absent_is_refused(
    tmp_path, monkeypatch
):
    problems = _problems(tmp_path, monkeypatch, TWIN_UKD, with_dsink=False)
    assert any("no dsink_reduce object of dtype bf16" in p for p in problems), problems
