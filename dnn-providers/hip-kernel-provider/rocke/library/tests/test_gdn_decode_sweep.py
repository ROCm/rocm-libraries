# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the GDN/KDA decode tiles x knobs sweep (``sweep.py``)."""

from __future__ import annotations

import json

import pytest

from builders.gfx950.gdn import sweep
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all
from rocke.dispatch.tuning import sweep_level


def _args(**kw):
    base = dict(
        gate_kind="gdn",
        head_k_dim=128,
        head_v_dim=128,
        state_dtype="f32",
        fuse_conv=False,
        fuse_out_norm=False,
    )
    base.update(kw)
    return type("A", (), base)()


def test_parse_shard_and_geometries():
    assert sweep.parse_shard("2/3") == (2, 3)
    assert sweep.parse_geometries("16/48,8/16") == [(16, 48), (8, 16)]
    for bad in ("3/3", "-1/2", "0/0"):
        with pytest.raises(ValueError):
            sweep.parse_shard(bad)


def test_shards_partition_the_space():
    items = list(range(17))
    parts = [sweep.shard_of(items, (i, 3)) for i in range(3)]
    assert sorted(x for p in parts for x in p) == items
    assert all(set(a).isdisjoint(b) for a in parts for b in parts if a is not b)


def test_space_is_dispatch_all_at_the_level():
    req = sweep.request_for(_args(), 16, 48, 1)
    assert req == GdnDecodeRequest(
        batch=1, arch="gfx950", num_k_heads=16, num_v_heads=48, state_dtype="f32"
    )
    with sweep_level("production"):
        want = dispatch_gdn_decode_all(req)
    got = sweep.space(req, "production")
    assert [r.spec.kernel_spec.kernel_name() for r in got] == [
        r.spec.kernel_spec.kernel_name() for r in want
    ]
    # Every configuration is a distinct kernel, so the compile map loses none.
    names = [r.spec.kernel_spec.kernel_name() for r in got]
    assert len(set(names)) == len(names)
    key = sweep.record_key(got[0])
    assert key["kernel_name"] == names[0] and key["knobs"] == {}


def test_full_level_is_a_superset_of_production():
    req = sweep.request_for(_args(), 16, 48, 1)
    prod = {r.spec.kernel_spec.kernel_name() for r in sweep.space(req, "production")}
    full = {r.spec.kernel_spec.kernel_name() for r in sweep.space(req, "full")}
    assert prod < full


def _rec(name, batch, us, auto=False, status="ok"):
    return {
        "geometry": [16, 48],
        "head_dims": [128, 128],
        "gate_kind": "gdn",
        "state_dtype": "f32",
        "fuse_conv": False,
        "fuse_out_norm": False,
        "cold": 64,
        "kernel_name": name,
        "spec_id": name,
        "knobs": {},
        "waves_per_eu": 0,
        "batch": batch,
        "us": us,
        "auto": auto,
        "status": status,
    }


def test_rank_uses_geomean_of_per_batch_ratios(tmp_path):
    rows = [
        _rec("a", 1, 1.0),
        _rec("a", 2, 4.0),  # 2x the fastest at batch 2
        _rec("b", 1, 2.0),  # 2x the fastest at batch 1
        _rec("b", 2, 2.0),
        _rec("c", 1, 1.5, auto=True),
        _rec("c", 2, 2.5),
        _rec("d", 1, 0.1, status="incorrect"),  # never ranked
        _rec("e", 1, 1.0),  # missing batch 2: dropped
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ((case, (batches, entries)),) = sweep.rank(sweep.load([str(p)])).items()
    assert batches == [1, 2]
    names = [e[2] for e in entries]
    assert names == ["c", "a", "b"]  # geomeans 1.369, 1.414, 1.414 (a worse worst)
    assert entries[0][1] == pytest.approx(1.5)


def test_merge_prints_ratios_only(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "".join(json.dumps(r) + "\n" for r in (_rec("a", 1, 1.25), _rec("b", 1, 2.5)))
    )
    assert sweep.main(["merge", str(p), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "x1.000" in out and "x2.000" in out
    assert "1.25" not in out and "2.5" not in out
