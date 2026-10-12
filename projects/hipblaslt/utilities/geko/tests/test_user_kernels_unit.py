# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from geko import user_kernels as uk

msgpack = pytest.importorskip("msgpack")


def _problem_type(trans_a: bool) -> dict:
    return {"transA": trans_a, "transB": False, "aType": "BFloat16", "bType": "BFloat16",
            "dType": "BFloat16"}


def _write_payload(path: Path, compressed: bool = True) -> Path:
    lib = {
        "solutions": [{"index": 7, "problemType": _problem_type(True)},
                      {"index": 3, "problemType": _problem_type(True)}],
        "library": {"rows": [{"library": {
            "properties": [{"type": "FreeSizeA"}, {"type": "FreeSizeB"},
                           {"type": "BatchSize"}, {"type": "BoundSize"}],
            "table": [{"key": [16, 6144, 1, 2048], "index": 7},
                      {"key": [32, 6144, 1, 2048], "index": 3}],
        }}]},
    }
    raw = msgpack.packb(lib)
    dat = path / ("TensileLibrary_X_Contraction_l_Alik_Bljk_gfx950.dat"
                  + (".zlib" if compressed else ""))
    dat.write_bytes(zlib.compress(raw) if compressed else raw)
    dat.with_name(dat.name.split(".dat")[0] + ".co").write_bytes(b"")
    return dat


def test_read_payload_orders_indices_and_reads_sizes(tmp_path: Path) -> None:
    for compressed in (True, False):
        d = tmp_path / str(compressed)
        d.mkdir()
        pl = uk.read_payload(_write_payload(d, compressed))
        assert pl.shard_indices == [3, 7]
        assert pl.co.name == "TensileLibrary_X_Contraction_l_Alik_Bljk_gfx950.co"
        got = [(p.trans, p.m, p.n, p.batch, p.k, shard) for p, shard in pl.sizes]
        assert got == [("TN", 16, 6144, 1, 2048, 7), ("TN", 32, 6144, 1, 2048, 3)]


def test_problem_shapes_are_column_major_operands() -> None:
    p = uk.Problem("TN", 16, 6144, 1, 2048, "BFloat16", "BFloat16", "BFloat16")
    assert [s[:2] for s in p.shapes] == [(2048, 16), (2048, 6144), (16, 6144), (16, 6144)]


def test_geko_winners(tmp_path: Path) -> None:
    assert uk.geko_winners(tmp_path) is None
    results = tmp_path / "results"
    results.mkdir()
    (results / "raw_results.csv").write_text("m\n")
    assert uk.geko_winners(tmp_path) == set()
    (results / "final_results.csv").write_text(
        "transA,transB,batch_count,m,n,k,a_type\nT,N,1,16,6144,2048,bf16_r\n")
    assert uk.geko_winners(tmp_path) == {("TN", 16, 6144, 1, 2048, "bf16_r")}


def test_algo_from_index_round_trips() -> None:
    algo = uk.MatmulAlgo.from_index(3 << 29, 1 << 20)
    assert algo.index == 3 << 29
    assert bytes(algo.data[4:]) == bytes(12)
    assert algo.max_workspace_bytes == 1 << 20


def test_user_kernel_library_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(uk.LIBRARY_ENV, raising=False)
    assert uk.user_kernel_library() is None
    monkeypatch.setenv(uk.LIBRARY_ENV, "/a/lib")
    assert uk.user_kernel_library() == Path("/a/lib")


def test_saved_in_matches_code_object_and_record(tmp_path: Path) -> None:
    co = tmp_path / "k.co"
    co.write_bytes(b"code")
    root = tmp_path / "lib"
    (root / "objects").mkdir(parents=True)
    (root / "gfx950").mkdir()
    assert uk.saved_in(root, co) is False
    (root / "objects" / ("c" * 32 + ".co")).write_bytes(b"code")
    assert uk.saved_in(root, co) is False
    (root / "gfx950" / "journal").write_text(f"R {'d' * 32} {'c' * 32} dat.zlib\n")
    assert uk.saved_in(root, co) is True
    co.write_bytes(b"other")
    assert uk.saved_in(root, co) is False


def test_saved_mappings_keeps_newest_mapping_per_size(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    (root / "objects").mkdir(parents=True)
    (root / "gfx950").mkdir()
    dat = _write_payload(tmp_path)
    (root / "objects" / ("d" * 32 + ".dat.zlib")).write_bytes(dat.read_bytes())
    d, c = "d" * 32, "c" * 32
    (root / "gfx950" / "journal").write_text(
        f"R {d} {c} dat.zlib\nE {d} {c} 7 16 6144 1 2048\nE {d} {c} 3 32 6144 1 2048\n"
        f"E {d} {c} 3 16 6144 1 2048\nE {'e' * 32} {c} 3 8 8 1 8\n")
    got = sorted((arch, p.trans, p.m, shard) for arch, p, _, shard in uk.saved_mappings(root))
    assert got == [("gfx950", "TN", 16, 3), ("gfx950", "TN", 32, 3)]


def test_read_payload_names_solutions(tmp_path: Path) -> None:
    pl = uk.read_payload(_write_payload(tmp_path))
    assert set(pl.names) == {3, 7} and pl.problem_types[7]["transA"] is True


def test_mapped_libraries_finds_libc() -> None:
    assert any("libc" in Path(p).name for p in uk.mapped_libraries("libc"))
