# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests of the tilewright Python bindings."""

import math
import random
import struct
import threading
import zlib
from pathlib import Path

import pytest

import tilewright as tw

HASH = "e7fe4b524851e895"
Q, I, X = 55, 12, 37
LARGE = "Large|Large|LargeK|Bnone"
WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "hipblaslt"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/"


def _lenstr(s):
    b = s.encode("ascii")
    return struct.pack("<H", len(b)) + b


def _f32s(values):
    return struct.pack(f"<{len(values)}f", *values)


def _cell(label, embed=4, hidden=8, inter=4, seed=1, signatures=()):
    rng = random.Random(seed)

    def vec(n, scale):
        return [rng.uniform(-scale, scale) for _ in range(n)]

    def std(n):
        return [0.5 + rng.random() for _ in range(n)]

    out = _lenstr(label) + struct.pack("<IIIf", embed, hidden, inter, 0.9)
    out += _f32s(vec(Q, 2.0)) + _f32s(std(Q))
    out += _f32s(vec(I, 0.5)) + _f32s(std(I))
    out += _f32s(vec(X, 2.0)) + _f32s(std(X))
    out += struct.pack("<I", len(signatures))
    out += b"".join(struct.pack("<8i", *s) for s in signatures)
    h, e, ih = hidden, embed, inter
    for n in (h * Q, h, h * h, h, e * h, e, h * I, h, e * h, e, ih * X, ih, ih, 1):
        out += _f32s(vec(n, 0.3))
    return out


def write_model(cells, splits=(), arch="gfxtest", mi_table=(), feature_hash=HASH):
    """MLREC_v2 bytes of a model with fp32 weights."""
    tail = _lenstr(feature_hash) + _lenstr(arch)
    tail += struct.pack("<ddddd", 4.0, 0.0, 0.01, 0.0, 32.0)
    tail += struct.pack("<I", len(mi_table))
    for m, n, k, dtype, cycles in mi_table:
        tail += struct.pack("<IIIid", m, n, k, dtype, cycles)
    payload = b""
    for parent, axis, threshold, lo, hi in splits:
        payload += _lenstr(parent) + axis.encode("ascii") + b"\0"
        payload += struct.pack("<i", threshold) + _lenstr(lo) + _lenstr(hi)
    payload += b"".join(cells)
    header_size = 52 + len(tail)
    head = b"MLREC_v2" + struct.pack(
        "<IIQIB3xIIIII",
        0x01020304,
        header_size,
        len(payload),
        zlib.crc32(payload) & 0xFFFFFFFF,
        0,
        Q,
        I,
        X,
        len(cells),
        len(splits),
    )
    return head + tail + payload + b"MLRECEND"


def _sig(c):
    return (
        c.mt.m,
        c.mt.n,
        c.mt.k,
        c.mi.m,
        c.mi.n,
        c.mi.k,
        c.cache_hints_a,
        c.cache_hints_b,
    )


def _config(mt_m, mt_n, mt_k=64, index=0):
    return tw.Config(
        mt=tw.Dim3(mt_m, mt_n, mt_k),
        mi=tw.Dim3(16, 16, 32),
        occupancy=1,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
        index=index,
    )


def _problem(m=4096, n=4096, k=4096, batch=1):
    return tw.Problem(
        size=tw.Dim3(m, n, k),
        batch=batch,
        a_transpose=tw.Transpose.T,
        b_transpose=tw.Transpose.N,
        a_dtype=tw.DataType.BFloat16,
        b_dtype=tw.DataType.BFloat16,
        c_dtype=tw.DataType.BFloat16,
        d_dtype=tw.DataType.BFloat16,
        mi_dtype=tw.DataType.BFloat16,
    )


HW = tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=4 << 20)
POOL = [
    _config(m, n, index=10 + i)
    for i, (m, n) in enumerate(
        [(64, 64), (256, 128), (128, 256), (128, 128), (512, 512)]
    )
]


@pytest.fixture(scope="module")
def model():
    split = (LARGE, "M", 2048, LARGE + "#M<=2048", LARGE + "#M>2048")
    cells = [
        _cell(LARGE, seed=1, signatures=[_sig(POOL[1]), _sig(POOL[3]), _sig(POOL[4])]),
        _cell(LARGE + "#M<=2048", embed=5, hidden=7, inter=3, seed=2),
    ]
    return tw.load_model_from_memory(write_model(cells, [split]))


def test_enums_are_ints():
    expected = {
        "Float": 0,
        "Double": 1,
        "Half": 4,
        "Int32": 6,
        "BFloat16": 7,
        "Int8": 8,
        "XFloat32": 11,
        "Float8_fnuz": 12,
        "Float8": 16,
        "BFloat8": 17,
        "Float8BFloat8": 18,
        "BFloat8Float8": 19,
        "Float4": 22,
        "None_": 23,
    }
    for name, value in expected.items():
        assert int(getattr(tw.DataType, name)) == value
        assert tw.DataType(value) == getattr(tw.DataType, name)
    assert tw.DataType.BFloat16 == 7
    assert [int(t) for t in (tw.Transpose.T, tw.Transpose.N)] == [0, 1]
    assert [
        int(w)
        for w in (
            tw.WeightType.Fp32,
            tw.WeightType.Bf16,
            tw.WeightType.Int8,
            tw.WeightType.Int4,
        )
    ] == [0, 1, 2, 3]


def test_keyword_construction_and_defaults():
    p = tw.Problem()
    assert (p.size.m, p.size.n, p.size.k, p.batch) == (0, 0, 0, 1)
    assert p.a_transpose == tw.Transpose.N and p.mi_dtype == tw.DataType.None_
    p = tw.Problem(
        size=tw.Dim3(m=1, n=2, k=3), batch=4, a_dtype=4, mi_dtype=tw.DataType.Float8
    )
    assert p.size == tw.Dim3(1, 2, 3) and p.batch == 4
    assert p.a_dtype == tw.DataType.Half and p.mi_dtype == tw.DataType.Float8
    p.size.m = 77
    p.b_dtype = 7
    assert p.size.m == 77 and p.b_dtype == tw.DataType.BFloat16

    c = tw.Config()
    assert (c.occupancy, c.cache_hints_a, c.grvw_a, c.gwvw_d, c.index) == (
        -1,
        0,
        1,
        1,
        0,
    )
    c = tw.Config(
        mt=tw.Dim3(256, 128, 64),
        mi=tw.Dim3(16, 16, 32),
        occupancy=2,
        cache_hints_b=4,
        index=9,
    )
    assert (c.mt.m, c.mi.k, c.occupancy, c.cache_hints_b, c.index) == (256, 32, 2, 4, 9)
    h = tw.Hardware(N_CU=8, lds_capacity=1024, L2_capacity=2048)
    assert (h.N_CU, h.lds_capacity, h.L2_capacity) == (8, 1024, 2048)
    assert "Dim3(m=1" in repr(tw.Dim3(1, 2, 3))


def test_feature_catalog_hash():
    assert tw.feature_catalog_hash() == HASH


def test_describe_route_and_labels(model):
    info = tw.describe(model)
    assert (info.arch, info.feature_catalog_hash, info.n_cells, info.n_splits) == (
        "gfxtest",
        HASH,
        2,
        1,
    )
    assert info.weight_type == tw.WeightType.Fp32
    assert model.describe().n_cells == 2
    assert tw.cell_label(model, tw.route(model, _problem(m=2048))) == LARGE + "#M<=2048"
    assert tw.cell_label(model, tw.route(model, _problem(m=2049))) == LARGE
    assert tw.route(model, _problem(m=8, n=8, k=8)) == -1
    assert tw.cell_label(model, 99) == ""


def test_load_errors_raise_value_error(tmp_path):
    good = write_model([_cell(LARGE)])
    with pytest.raises(ValueError, match="convert"):
        tw.load_model_from_memory(b"MLREC_v1" + good[8:])
    with pytest.raises(ValueError, match="LFS"):
        tw.load_model_from_memory(LFS_PREFIX + b"v1\noid sha256:00\nsize 1\n")
    with pytest.raises(ValueError, match="CRC"):
        tw.load_model_from_memory(good[:-20] + bytes([good[-20] ^ 1]) + good[-19:])
    with pytest.raises(ValueError, match="hash"):
        tw.load_model_from_memory(write_model([_cell(LARGE)], feature_hash="0" * 16))
    for n in (0, 7, 51, 52, len(good) - 9, len(good) - 1):
        with pytest.raises(ValueError):
            tw.load_model_from_memory(good[:n])
    with pytest.raises(ValueError, match="missing"):
        tw.load_model(str(tmp_path / "missing.bin"))
    with pytest.raises(TypeError):
        tw.CandidateSet(None, POOL)


def test_load_model_and_index(tmp_path):
    data = write_model([_cell(LARGE)])
    (tmp_path / "m.tilewright.bin").write_bytes(data)
    (tmp_path / "bad.tilewright.bin").write_bytes(data[:-1] + b"X")
    (tmp_path / "tilewright_index").write_text(
        "# index\nStem\tm.tilewright.bin\nBad bad.tilewright.bin\n"
    )
    m = tw.load_model(str(tmp_path / "m.tilewright.bin"))
    assert tw.describe(m).n_cells == 1
    assert tw.describe(tw.load_model_by_index("Stem", str(tmp_path))).n_cells == 1
    assert tw.load_model_by_index("Absent", str(tmp_path)) is None
    assert tw.load_model_by_index("Stem", str(tmp_path / "nowhere")) is None
    with pytest.raises(ValueError, match="MLRECEND"):
        tw.load_model_by_index("Bad", str(tmp_path))


def test_rank_contract_tiers_and_candidate_set(model):
    cs = tw.CandidateSet(model, POOL)
    assert len(cs) == len(POOL) and [c.index for c in cs.configs] == [
        c.index for c in POOL
    ]
    r0 = cs.rank(_problem(), HW)
    assert sorted(r.config_index for r in r0 if r.scored) == [1, 3]
    assert [r.config_index for r in r0 if not r.scored] == [0, 2, 4]
    assert r0[0].score >= r0[1].score and all(
        math.isfinite(r.score) for r in r0 if r.scored
    )
    deep = cs.rank(_problem(), HW, min_scored=3)
    assert [r.config_index for r in deep[:2]] == [r.config_index for r in r0[:2]]
    assert sorted(r.config_index for r in deep[2:] if r.scored) == [0, 2]
    assert deep == tw.rank_configs(model, _problem(), HW, POOL, 3)
    for p in (
        _problem(m=100, n=3000, k=700),
        _problem(m=1500, batch=3),
        _problem(m=8, n=8, k=8),
    ):
        for depth in (0, 2, 1000):
            assert cs.rank(p, HW, depth) == tw.rank_configs(model, p, HW, POOL, depth)
    for bad in (tw.Hardware(), tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=0)):
        assert not any(r.scored for r in cs.rank(_problem(), bad))
    assert tw.rank_configs(model, _problem(), HW, []) == []


def test_compute_features(model):
    p = _problem(m=1024, n=2000, k=3000)
    f = tw.compute_features(model, p, POOL[1], HW)
    assert (len(f.query), len(f.item), len(f.interaction)) == (Q, I, X)
    assert f.query[0] == 10.0 and f.query[9] == 1.0 and f.query[10] == 0.0
    assert f.item[0] == 8.0 and f.item[1] == 7.0
    assert f.interaction[33] == 32.0 / 4.0
    empty = tw.compute_features(model, p, POOL[1], tw.Hardware())
    assert (empty.query, empty.item, empty.interaction) == ([], [], [])


def test_rank_from_threads(model):
    cs = tw.CandidateSet(model, POOL)
    problems = [
        _problem(m=100 + 97 * i, n=4096 - 61 * i, k=64 + 33 * i) for i in range(24)
    ]
    expected = [tw.rank_configs(model, p, HW, POOL, 3) for p in problems]
    errors = []

    def work():
        for _ in range(20):
            for p, e in zip(problems, expected):
                if cs.rank(p, HW, 3) != e:
                    errors.append(p)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def _shipped_models():
    if not WEIGHTS.is_dir():
        return [], 0
    models, pointers = [], 0
    for path in sorted(WEIGHTS.rglob("*.tilewright.bin")):
        with open(path, "rb") as f:
            if f.read(len(LFS_PREFIX)) == LFS_PREFIX:
                pointers += 1
                continue
        models.append(path)
    return models, pointers


def test_shipped_models():
    models, pointers = _shipped_models()
    if not models:
        pytest.skip(f"no materialized shipped models ({pointers} Git LFS pointers)")
    pool = [
        _config(m, n, k)
        for m in (64, 128, 256)
        for n in (64, 128, 256)
        for k in (32, 64)
    ]
    for path in models:
        m = tw.load_model(str(path))
        info = tw.describe(m)
        assert info.arch.startswith(path.parent.name[:6]) and info.n_cells > 0
        cs = tw.CandidateSet(m, pool)
        p = _problem(m=4096, n=4096, k=4096)
        assert tw.route(m, p) >= 0
        ranked = cs.rank(
            p, tw.Hardware(N_CU=128, lds_capacity=65536, L2_capacity=4 << 20)
        )
        assert ranked == tw.rank_configs(
            m, p, tw.Hardware(N_CU=128, lds_capacity=65536, L2_capacity=4 << 20), pool
        )
        assert any(r.scored for r in ranked)
