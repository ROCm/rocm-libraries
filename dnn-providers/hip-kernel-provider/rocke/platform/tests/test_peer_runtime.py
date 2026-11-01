# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of ``rocke.runtime.peer`` (no GPU): argument validation, tag
arithmetic, error-record decoding, the file rendezvous with two local
processes over a fake allocator. The IPC / multi-GPU proofs run in the
multi-process GPU suite (``rocke.examples.gfx950.ccl``)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from rocke.runtime import peer as P


def test_tag_add_skips_zero_and_wraps():
    m = (1 << 32) - 1
    assert P.tag_add(1, 0) == 1
    assert P.tag_add(5, 3) == 8
    assert P.tag_add(m, 1) == 1  # 0 skipped
    assert P.tag_add(m - 1, 2) == 1
    seq = [P.tag_add(m - 3, k) for k in range(8)]
    assert seq == [m - 3, m - 2, m - 1, m, 1, 2, 3, 4]
    assert 0 not in seq
    # composition: (e + a) + b == e + (a + b)
    for e in (1, 7, m - 10, m):
        for a in (0, 1, 9, 100):
            for c in (0, 3, 11):
                assert P.tag_add(P.tag_add(e, a), c) == P.tag_add(e, a + c)
    with pytest.raises(ValueError):
        P.tag_add(0, 1)


def test_device_tag_formula_matches_host():
    """The device form ``s_add_u32 e, u; s_addc_u32 0`` (carry adds one)."""
    m32 = 1 << 32

    def dev(e, u):
        s = e + u
        return (s % m32) + (1 if s >= m32 else 0)

    for e in (1, 2, 1000, m32 - 5000, m32 - 2, m32 - 1):
        for u in (0, 1, 2, 4999, 5000, 5001, 70000):
            assert dev(e, u) == P.tag_add(e, u), (e, u)


def test_error_record_decode():
    w = [0] * P.CTRL_WORDS
    w[P.REC_STATUS] = 1
    w[P.REC_CODE] = (-3) & 0xFFFFFFFF
    w[P.REC_STAGE] = 1
    w[P.REC_USE] = 4
    w[P.REC_PEER] = 1
    w[P.REC_TAG_EXPECTED] = 77
    w[P.REC_TAG_SEEN] = 75
    w[P.REC_RT_LO] = 5
    w[P.REC_RT_HI] = 1
    rec = P.ErrorRecord.from_words(w)
    assert rec.code == -3 and rec.name == "PEER_TIMEOUT"
    assert rec.realtime == (1 << 32) + 5
    err = P.PeerError(0, rec)
    assert "PEER_TIMEOUT" in str(err) and "expected 77 seen 75" in str(err)
    assert set(P.ERROR_NAMES.values()) >= {"PEER_TIMEOUT", "PEER_ABORT", "TAG_AHEAD", "CCL_MISMATCH"}


def test_control_offsets_match_ccl_layout():
    lay = pytest.importorskip("kernels.gfx950.qwen38_mono.ccl.layout")
    assert lay.OFF == {
        "epoch": P.OFF_EPOCH,
        "abort": P.OFF_ABORT,
        "exit": P.OFF_EXIT,
        "claim": P.OFF_CLAIM,
        "launches": P.OFF_LAUNCHES,
        "slotbase": P.OFF_SLOTBASE,
        "gsync": P.OFF_GSYNC,
    }
    assert {v: k for k, v in lay.ERR.items()} == {k: v for k, v in P.ERROR_NAMES.items() if k}


def _fake_alloc(base):
    return lambda nbytes, mem: base


def test_validation_single_rank_fake():
    g = P.PeerGroup(0, 1, 3 << 20, alloc=_fake_alloc(0x1000_0000))
    assert g.nbytes == 4 << 20  # whole 2 MiB pages
    assert g.bases == (0x1000_0000,)
    ka = g.kernargs()
    assert ka["peer0"] == ka["arena"] == 0x1000_0000 and ka["rank"] == 0
    g.close()
    with pytest.raises(ValueError):
        P.PeerGroup(2, 2, alloc=_fake_alloc(1))
    with pytest.raises(ValueError):
        P.PeerGroup(0, 9, alloc=_fake_alloc(1))
    with pytest.raises(ValueError):
        P.PeerGroup(0, 1, 128, alloc=_fake_alloc(1))
    with pytest.raises(ValueError):
        P.PeerGroup(0, 1, mem="cached", alloc=_fake_alloc(1))


_RANK_CODE = """
import json, sys
from rocke.runtime import peer as P
rank, ranks, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
g = P.PeerGroup(rank, ranks, 4 << 20, rendezvous="file:" + path, alloc=lambda n, m: 0x10_0000_0000 * (rank + 1))
print(json.dumps({"rank": rank, "bases": list(g.bases), "ka": g.kernargs()}))
g.barrier()
g.reset(epoch=7)
g.close()
"""


def test_file_rendezvous_two_processes(tmp_path):
    """Two local processes meet through the file rendezvous (fake allocator:
    no GPU) and see each other's arena bases."""
    ranks = 2
    procs = [
        subprocess.Popen([sys.executable, "-c", _RANK_CODE, str(r), str(ranks), str(tmp_path / "rv")],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for r in range(ranks)
    ]
    outs = [p.communicate(timeout=180) for p in procs]
    want = [0x10_0000_0000 * (r + 1) for r in range(ranks)]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, err[-2000:]
        d = json.loads(out.strip().splitlines()[-1])
        assert d["bases"] == want
        assert d["ka"]["arena"] == want[d["rank"]] and d["ka"][f"peer{d['rank']}"] == want[d["rank"]]


def test_file_rendezvous_timeout(tmp_path):
    rv = P.FileRendezvous(str(tmp_path), 0, 2, timeout_s=0.2)
    rv.timeout_s = 0.2
    with pytest.raises(TimeoutError):
        rv.all_gather({"x": 1})


def test_peer_module_not_imported_by_runtime():
    """Byte identity by construction: no existing runtime module imports peer."""
    import pathlib

    root = pathlib.Path(P.__file__).parent
    for f in root.glob("*.py"):
        if f.name == "peer.py":
            continue
        assert "peer" not in [ln.split()[1] for ln in f.read_text().splitlines() if ln.startswith("from .") and len(ln.split()) > 1], f
