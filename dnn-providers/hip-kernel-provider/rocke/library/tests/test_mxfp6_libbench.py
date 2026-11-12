# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the MXFP6 tile-library model and harness (``builders.gfx950.mxfp6_gemm.libtiles`` / ``libbench``).

The library list, the work model, the benchmark problem sets (exact tile counts, tile multiples, MALL-exceeding
rotation), the roofline bookkeeping, and the harness pipeline end to end on its synthetic backend
(``MXFP6_LIBBENCH_FAKE``: no compiler, no GPU): plan with the lever refusal, sharded compile, GPU stage, resume,
explicit re-measurement, table and status.
"""

from __future__ import annotations

import json
import math

import pytest
from builders.gfx950.mxfp6_gemm import libbench as LB
from builders.gfx950.mxfp6_gemm import libtiles as T


def test_library_list():
    du = [t.du for t in T.LIBRARY]
    assert len(T.LIBRARY) == 70 and du.count(256) == 57 and du.count(512) == 13
    assert len(set(T.LIBRARY)) == 70
    t = T.Tile.parse("128x32x256")  # MT0 = weight rows (N), MT1 = tokens (M)
    assert (t.tile_n, t.tile_m) == (128, 32)
    assert T.FAMILIES["ak"][0] == ("NN", "TN") and T.FAMILIES["am"][0] == ("NT", "TT")


def test_work_model():
    t = T.Tile.parse("256x256x256")
    w = T.work(t, T.default_layout(t))
    assert w.flop == 2.0 * 256 * 256 * 256
    assert math.isclose(w.ai_glob, 2 * 256 * 256 / (512 * T.B_ELEM))
    assert (w.tm, w.tn, w.ks, w.waves) == (8, 8, 2, 4)
    for t in T.LIBRARY:  # every library tile has a default layout that fits the register file
        lay = T.default_layout(t)
        assert T.layout_ok(t, lay) and T.work(t, lay).occ_est >= 1


def test_problem_sets():
    for t in T.LIBRARY:
        for p in T.problems(t):
            assert p.m % t.tile_m == 0 and p.n % t.tile_n == 0 and p.k % t.du == 0
            assert p.batch * (p.m // t.tile_m) * (p.n // t.tile_n) == T.FILL_TILES
            assert p.rotate * p.in_bytes >= 2 * T.MALL_BYTES or p.rotate == 1 and p.in_bytes >= 2 * T.MALL_BYTES
        g = [p for p in T.problems(t) if p.set == "G"][0]
        assert g.m == t.tile_m  # one token tile per grouped entry
    ks = [p.k for p in T.k_sweep(T.LIBRARY[0])]
    assert ks == list(T.K_SWEEP)


def test_roofline_levels():
    t = T.Tile.parse("128x32x256")
    w = T.work(t, T.default_layout(t))
    g = [p for p in T.problems(t) if p.set == "G"][0]
    rf = T.roofline(w, {"lds": 1.0, "l2": 1.0, "mall": 1.0, "hbm": 1.0}, 1, g)
    assert math.isclose(rf["ai_hbm"], g.flop / g.unique_bytes)
    assert rf["ai_fabric"] > 0 and rf["l2"] > 0


def test_wave_rows_and_scale_sel_skip():
    assert LB._wave_rows({"tile_m": 32, "tile_n": 32, "waves_m": 2, "waves_n": 2}) == (16, 16)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("MXFP6_LIBBENCH_FAKE", "1")


def test_pipeline_fake(fake, tmp_path):
    man = {"name": "t", "formats": ["e2m3", "e3m2"], "sets": ["S", "G"],
           "problems": [{"set": "X", "batch": 2, "m": 512, "n": 512, "k": 4096}],
           "configs": [{"id": "a", "tile": "256x256x256", "family": "ak"},
                       {"id": "b", "tile": "64x128x256", "family": "am", "levers": "scalesep,fakelever"},
                       {"id": "bad", "tile": "32x32x512", "family": "ak", "levers": "scalesep,nosuchlever"}]}
    run = tmp_path / "run"
    pl = LB.plan(LB.load_manifest(man), run)
    assert [r["config"] for r in pl["refused"]] == ["bad"]  # unknown direct-compiler lever: refused
    assert len(pl["cells"]) == 2 * 2 * 3
    assert LB.compile_shard(run, 0, 2, jobs=1)["failed"] == 0
    assert LB.compile_shard(run, 1, 2, jobs=1)["failed"] == 0
    assert LB.status(run)["build"] == {"ok": len(pl["builds"])}
    assert LB.gpu_shard(run)["done"] == len(pl["cells"])
    assert LB.gpu_shard(run)["done"] == 0  # resumable: nothing left
    one = pl["cells"][0]["id"]
    assert LB.gpu_shard(run, cells=[one])["done"] == 1  # an explicit list re-measures
    rows = json.loads(LB.table(run, fmt="json"))
    assert len(rows) == len(pl["cells"]) and all(r["verify"] == "ok" for r in rows)
    assert LB.status(run)["cell"] == {"done": len(pl["cells"])}
    assert len(LB.families(run)) >= 1


def test_compile_server_api():
    """The reusable build API other lanes call (searches, WaveScope, the build service)."""
    for m in ("__enter__", "__exit__", "close", "compile", "compile_multi", "_meta"):
        assert callable(getattr(LB.CompileServer, m, None)), m


def test_obj_index(tmp_path):
    """A run's objects through its index (no per-build files), the per-build files as the fallback."""
    run = tmp_path / "run"
    (run / "objs" / "index").mkdir(parents=True)
    obj = tmp_path / "cache" / "k1.hsaco"
    obj.parent.mkdir()
    obj.write_bytes(b"OBJ")
    (run / "objs" / "index" / "h-1.jsonl").write_text(json.dumps(["b1", {"hsaco": str(obj), "name": "k"}]) + "\n")
    assert LB.obj_path(run, "b1") == obj and LB.obj_meta(run, "b1")["name"] == "k"
    assert LB.obj_path(run, "b2") is None and LB.obj_meta(run, "b2") is None
    (run / "objs" / "b2.hsaco").write_bytes(b"X")
    (run / "objs" / "b2.hsaco.json").write_text(json.dumps({"name": "k2"}))
    assert LB.obj_path(run, "b2") == run / "objs" / "b2.hsaco" and LB.obj_meta(run, "b2")["name"] == "k2"
    (run / "objs" / "index" / "h-2.jsonl").write_text(json.dumps(["b3", {"hsaco": str(obj), "name": "k3"}]) + "\n")
    assert LB.obj_meta(run, "b3")["name"] == "k3"  # a second writer's lines are picked up (index re-read)


def test_buildd_fake(fake, tmp_path, monkeypatch):
    """The build service plans + builds a run request in one long-lived process, answers a superseded-stage request
    as stale, rejects a bad request without dying, and exits 75 when the lane's CURRENT pointer moves."""
    import threading
    import time as _t

    man = {"name": "b", "formats": ["e2m3"], "sets": ["S"],
           "configs": [{"id": "a", "tile": "256x256x256", "family": "ak"}]}
    mp = tmp_path / "man.json"
    mp.write_text(json.dumps(man))
    q = tmp_path / "q"
    q.mkdir()
    cur = tmp_path / "CURRENT"
    cur.write_text("stage_sha=AAA\ncompiler_sha=c\n")
    (q / "r1.req").write_text(json.dumps({"type": "run", "run": str(tmp_path / "run"), "manifest": str(mp),
                                          "stage_sha": "AAA"}))
    (q / "r2.req").write_text(json.dumps({"type": "run", "run": str(tmp_path / "run2"), "stage_sha": "OLD"}))
    (q / "r3.req").write_text(json.dumps({"type": "nosuch"}))
    monkeypatch.setenv("LIBBENCH_TREE_SHA", "AAA")
    def mover():
        for _ in range(400):
            if all((q / f"r{i}.{s}").exists() for i, s in ((1, "done"), (2, "stale"), (3, "bad"))):
                break
            _t.sleep(0.02)
        cur.write_text("stage_sha=BBB\ncompiler_sha=c\n")  # a restage: the service must restart

    th = threading.Thread(target=mover)
    th.start()
    assert LB.buildd(q, jobs=1, current=cur, poll_s=0.01, http=False) == 75
    th.join()
    done = json.loads((q / "r1.done").read_text())
    assert done["ok"] and done["failed"] == 0 and done["ok"] >= 1
    assert LB.status(tmp_path / "run")["build"]["ok"] == done["ok"]
    assert "superseded" in json.loads((q / "r2.stale").read_text())["error"]
    assert "unknown request type" in json.loads((q / "r3.bad").read_text())["error"]
    assert json.loads((q / "HEARTBEAT").read_text())["tree"] == "AAA"


def test_buildd_http_fake(fake, tmp_path):
    """The service's HTTP channel: the token is required, a run request answers with its result, a path outside
    the service root is refused, and GET is a liveness probe."""
    import threading
    import time as _t
    import urllib.error
    import urllib.request

    man = {"name": "h", "formats": ["e2m3"], "sets": ["S"],
           "configs": [{"id": "a", "tile": "256x256x256", "family": "ak"}]}
    mp = tmp_path / "man.json"
    mp.write_text(json.dumps(man))
    q = tmp_path / "q"
    th = threading.Thread(target=LB.buildd, args=(q, 1), kwargs={"poll_s": 0.01, "bind": "127.0.0.1"})
    th.start()
    try:
        for _ in range(500):
            if (q / "HEARTBEAT").exists():
                break
            _t.sleep(0.01)
        hb = json.loads((q / "HEARTBEAT").read_text())

        def post(req, token):
            r = urllib.request.Request(hb["url"], data=json.dumps(req).encode(), headers={"X-Token": token})
            try:
                with urllib.request.urlopen(r, timeout=30) as f:
                    return f.status, json.loads(f.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        assert post({"type": "run", "run": str(tmp_path / "run")}, "wrong")[0] == 403
        code, res = post({"type": "run", "run": str(tmp_path / "run"), "manifest": str(mp)}, hb["token"])
        assert code == 200 and res["ok"] >= 1 and res["failed"] == 0, res
        code, res = post({"type": "run", "run": str(tmp_path.parent / "elsewhere")}, hb["token"])
        assert code == 200 and "outside the service root" in res["error"]
        with urllib.request.urlopen(hb["url"], timeout=10) as f:
            assert json.loads(f.read())["pid"] == hb["pid"]
    finally:
        (q / "stop").write_text("")
        th.join(timeout=30)


def test_serve_fake(fake, tmp_path):
    """The persistent GPU worker runs request files from its inbox in one process until ``stop``."""
    import threading
    import time as _t

    man = {"name": "s", "formats": ["e2m3"], "sets": ["S"],
           "configs": [{"id": "a", "tile": "256x256x256", "family": "ak"},
                       {"id": "b", "tile": "64x128x256", "family": "am"}]}
    run = tmp_path / "run"
    pl = LB.plan(LB.load_manifest(man), run)
    LB.compile_shard(run, jobs=1)
    ids = [c["id"] for c in pl["cells"]]
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "r1.req").write_text(json.dumps({"run": str(run), "cells": ids[:1]}))
    (inbox / "r2.req").write_text(json.dumps({"run": str(run), "cells": ids[1:]}))

    def stopper():
        for _ in range(200):
            if (inbox / "r2.done").exists():
                break
            _t.sleep(0.05)
        (inbox / "stop").write_text("")

    th = threading.Thread(target=stopper)
    th.start()
    assert LB.serve(run, inbox, tmp_path / "cache", poll_s=0.01) == 0
    th.join()
    assert json.loads((inbox / "r1.done").read_text()) == {"done": 1}
    assert json.loads((inbox / "r2.done").read_text()) == {"done": len(ids) - 1}
    assert LB.status(run)["cell"] == {"done": len(ids)}
