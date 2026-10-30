# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests of the tilewright Python bindings."""

import math
import os
import random
import re
import struct
import subprocess
import sys
import threading
import zlib
from pathlib import Path

import pytest

import tilewright as tw

HASH = "e7fe4b524851e895"
Q, I, X = 55, 12, 37
LARGE = "Large|Large|LargeK|Bnone"
# hipBLASLt ships the models in the <arch>/<subtree>/Tilewright directories of
# its library logic.
WEIGHTS = (
    Path(__file__).resolve().parents[4]
    / "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)
LFS_PREFIX = b"version https://git-lfs.github.com/spec/"
TENSORS = (
    "q_w0",
    "q_b0",
    "q_w2",
    "q_b2",
    "q_w4",
    "q_b4",
    "i_w0",
    "i_b0",
    "i_w2",
    "i_b2",
    "x_w0",
    "x_b0",
    "x_w2",
    "x_b2",
)
NAN, INF = float("nan"), float("inf")


def _lenstr(s):
    b = s.encode("ascii")
    return struct.pack("<H", len(b)) + b


def _f32s(values):
    return struct.pack(f"<{len(values)}f", *values)


def _f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _cell(label, embed=4, hidden=8, inter=4, seed=1, signatures=(), edit=None):
    """A cell record with seeded random tensors; `edit(tensors)` may change
    them (keys: temperature, the whitening vectors and TENSORS)."""
    rng = random.Random(seed)

    def vec(n, scale):
        return [rng.uniform(-scale, scale) for _ in range(n)]

    def std(n):
        return [0.5 + rng.random() for _ in range(n)]

    t = {"temperature": 0.9}
    for tower, n, scale in (("q", Q, 2.0), ("i", I, 0.5), ("x", X, 2.0)):
        t[f"{tower}_mean"] = vec(n, scale)
        t[f"{tower}_std"] = std(n)
    h, e, ih = hidden, embed, inter
    sizes = (h * Q, h, h * h, h, e * h, e, h * I, h, e * h, e, ih * X, ih, ih, 1)
    for name, n in zip(TENSORS, sizes):
        t[name] = vec(n, 0.3)
    if edit is not None:
        edit(t)
    out = _lenstr(label) + struct.pack("<IIIf", embed, hidden, inter, t["temperature"])
    for key in ("q_mean", "q_std", "i_mean", "i_std", "x_mean", "x_std"):
        out += _f32s(t[key])
    out += struct.pack("<I", len(signatures))
    out += b"".join(struct.pack("<8i", *s) for s in signatures)
    for name in TENSORS:
        out += _f32s(t[name])
    return out


def _zero(t):
    for name in TENSORS:
        t[name] = [0.0] * len(t[name])


def _item_probe(j, mean, sd):
    """Weights under which every config scores exactly the whitened item
    feature `j` (stored with `mean` and `sd`)."""

    def edit(t):
        _zero(t)
        t["temperature"] = 1.0
        t["i_w0"][j], t["i_w0"][I + j] = 1.0, -1.0
        t["i_w2"][0], t["i_w2"][1] = 1.0, -1.0
        t["q_b4"][0] = 1.0
        t["i_mean"][j], t["i_std"][j] = mean, sd

    return edit


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


def _config(mt_m, mt_n, mt_k=64, index=0, occupancy=1):
    return tw.Config(
        mt=tw.Dim3(mt_m, mt_n, mt_k),
        mi=tw.Dim3(16, 16, 32),
        occupancy=occupancy,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
        index=index,
    )


def _copy(c, index, attributes=None):
    return tw.Config(
        mt=c.mt,
        mi=c.mi,
        occupancy=c.occupancy,
        cache_hints_a=c.cache_hints_a,
        cache_hints_b=c.cache_hints_b,
        grvw_a=c.grvw_a,
        grvw_b=c.grvw_b,
        gwvw_d=c.gwvw_d,
        index=index,
        attributes=attributes,
    )


def _order(results):
    return [r.config_index for r in results]


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


@pytest.fixture(scope="module")
def flat_model():
    """Every config scores 0."""
    return tw.load_model_from_memory(write_model([_cell(LARGE, edit=_zero)]))


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
    assert [
        int(s) for s in (tw.Schedule.Default, tw.Schedule.Dynamic, tw.Schedule.Auto)
    ] == [0, 1, 2]
    assert [int(t) for t in (tw.TieBreak.PoolOrder, tw.TieBreak.Prior)] == [0, 1]


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
    recorded = info.provenance
    assert isinstance(recorded, tw.Provenance) and "config_id=''" in repr(recorded)
    assert (recorded.training_data_hash, recorded.source_revision) == ("", "")
    veto = info.alignment_veto
    assert isinstance(veto, tw.AlignmentVeto)
    assert (veto.on, veto.k_min, veto.align_bits) == (False, 0, 0)
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
    assert tw.describe(m).min_effective_cus == 0
    assert tw.describe(tw.load_model_by_index("Stem", str(tmp_path))).n_cells == 1
    assert tw.load_model_by_index("Absent", str(tmp_path)) is None
    assert tw.load_model_by_index("Stem", str(tmp_path / "nowhere")) is None
    with pytest.raises(ValueError, match="MLRECEND"):
        tw.load_model_by_index("Bad", str(tmp_path))


def test_load_model_by_index_prefers_the_device_chip_id(tmp_path):
    (tmp_path / "any.bin").write_bytes(write_model([_cell(LARGE)], arch="gfxany"))
    (tmp_path / "chip.bin").write_bytes(write_model([_cell(LARGE)], arch="gfxchip"))
    (tmp_path / "tilewright_index").write_text(
        "Stem\tchip.bin\tchip=0xa2b\nStem\tany.bin\nChipOnly chip.bin chip=0x00FF\n"
    )
    d = str(tmp_path)

    def arch(stem, **kw):
        m = tw.load_model_by_index(stem, d, **kw)
        return None if m is None else tw.describe(m).arch

    assert arch("Stem") == arch("Stem", pci_chip_id=0) == "gfxany"
    assert arch("Stem", pci_chip_id=0xA2B) == "gfxchip"
    assert tw.load_model_by_index("Stem", d, 0xA2B) is not None
    assert arch("Stem", pci_chip_id=0xA2C) == "gfxany"
    assert arch("ChipOnly", pci_chip_id=0xFF) == "gfxchip"
    assert arch("ChipOnly") is None and arch("ChipOnly", pci_chip_id=1) is None
    for bad in (-1, 1 << 32, "0xa2b", 1.5):
        with pytest.raises(TypeError):
            tw.load_model_by_index("Stem", d, pci_chip_id=bad)

    for row in ("Stem any.bin chip=0x0", "Stem any.bin chip=6699", "Stem any.bin x y"):
        (tmp_path / "tilewright_index").write_text(f"Stem\tany.bin\n{row}\n")
        for kw in ({}, {"pci_chip_id": 0xA2B}):
            with pytest.raises(ValueError, match=r"tilewright_index:2: malformed row"):
                tw.load_model_by_index("Stem", d, **kw)
        assert tw.load_model_by_index("Other", d) is None


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


def _scored_prefix(results, k):
    out = []
    for r in results:
        if not r.scored or len(out) == k:
            break
        out.append(r)
    return out


def test_rank_top_is_the_scored_prefix_of_rank(model, flat_model):
    prior = [0.5, NAN, -1.0, 2.0, 0.5]
    exclusive = tw.ExecutionContext()
    problems = (_problem(), _problem(m=100, n=3000, k=700), _problem(m=8, n=8, k=8))
    for m in (model, flat_model):
        for tie_break in (tw.TieBreak.PoolOrder, tw.TieBreak.Prior):
            cs = tw.CandidateSet(m, POOL, tie_break=tie_break)
            for p in problems:
                for k in (0, 1, 2, 3, 1000):
                    top = cs.rank_top(p, HW, k, prior=prior)
                    assert top == _scored_prefix(cs.rank(p, HW, k, prior=prior), k)
                    assert (
                        cs.rank_top(p, HW, k=k, context=exclusive, prior=prior) == top
                    )
    cs = tw.CandidateSet(model, POOL)
    budget = tw.ExecutionContext(cu_budget=1)
    assert cs.rank_top(_problem(), HW, 1000, context=budget) == cs.rank_top(
        _problem(), HW, 1000
    )
    dynamic = tw.ExecutionContext(schedule=tw.Schedule.Dynamic)
    assert cs.rank_top(_problem(), HW, 1000, context=dynamic) == []
    with pytest.raises(ValueError, match="prior"):
        cs.rank_top(_problem(), HW, 1, prior=[0.0])


def test_cache_limit_never_changes_rankings(model):
    saved = tw.cache_limit()
    problems = [
        _problem(m=100 + 211 * i, n=4096 - 97 * i, k=64 + 333 * i) for i in range(12)
    ]
    try:
        tw.set_cache_limit(2**63)
        fresh = tw.load_model_from_memory(
            write_model(
                [
                    _cell(LARGE, seed=1, signatures=[_sig(POOL[1])]),
                    _cell(LARGE + "#M<=2048", embed=5, hidden=7, inter=3, seed=2),
                ],
                [(LARGE, "M", 2048, LARGE + "#M<=2048", LARGE + "#M>2048")],
            )
        )
        info = tw.describe(fresh)
        assert info.packed_bytes > 0 and info.expanded_bytes > info.packed_bytes
        before = tw.cache_usage()
        expect = [tw.CandidateSet(fresh, POOL).rank(p, HW, 3) for p in problems]
        assert tw.cache_usage() > before
        for limit in (0, 1, 2**63):
            tw.set_cache_limit(limit)
            assert tw.cache_limit() == limit
            cs = tw.CandidateSet(fresh, POOL)
            assert [cs.rank(p, HW, 3) for p in problems] == expect
            if limit == 0:
                assert tw.cache_usage() <= info.expanded_bytes
        del cs
        tw.set_cache_limit(0)
        assert tw.cache_usage() == 0
    finally:
        tw.set_cache_limit(saved)


def test_prepare_builds_what_rankings_use(model):
    saved = tw.cache_limit()
    problems = [_problem(), _problem(m=100, n=3000, k=700), _problem(m=1500, batch=3)]
    try:
        tw.set_cache_limit(2**63)
        expect = [tw.CandidateSet(model, POOL).rank(p, HW, 3) for p in problems]
        for how in ("all", "cells", "problem"):
            cs = tw.CandidateSet(model, POOL)
            if how == "all":
                cs.prepare()
            elif how == "cells":
                cs.prepare(cells=[tw.route(model, p) for p in problems] + [-1, 99])
            else:
                for p in problems:
                    cs.prepare(problem=p, hardware=HW)
            usage = tw.cache_usage()
            assert [cs.rank(p, HW, 3) for p in problems] == expect
            assert tw.cache_usage() == usage
        with pytest.raises(ValueError, match="together"):
            cs.prepare(problem=problems[0])
    finally:
        tw.set_cache_limit(saved)


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


def test_config_attributes():
    c = tw.Config(attributes={"workgroup_mapping": 8, "wave_num": -4})
    assert c.attributes == {"workgroup_mapping": 8, "wave_num": -4}
    assert list(c.attributes) == ["workgroup_mapping", "wave_num"]
    assert "attributes={'workgroup_mapping': 8, 'wave_num': -4}" in repr(c)
    assert tw.Config().attributes == {} and "attributes" not in repr(tw.Config())
    c.attributes = {"lo": -(2**63), "hi": 2**63 - 1, "flag": True}
    assert c.attributes == {"lo": -(2**63), "hi": 2**63 - 1, "flag": 1}
    copy = c.attributes
    copy["other"] = 1
    assert "other" not in c.attributes
    c.attributes = None
    assert c.attributes == {}
    for bad in ({"x": 1.5}, {"x": 2**63}, {"x": "1"}, {1: 2}, [("x", 1)], 3):
        with pytest.raises(TypeError):
            tw.Config(attributes=bad)
        with pytest.raises(TypeError):
            c.attributes = bad


def test_attributes_never_change_rankings(model):
    tagged = [
        _copy(c, c.index, {"workgroup_mapping": i, "wave_num": 4, "x": -i})
        for i, c in enumerate(POOL)
    ]
    cs_plain, cs_tagged = tw.CandidateSet(model, POOL), tw.CandidateSet(model, tagged)
    for p in (_problem(), _problem(m=100, n=3000, k=700), _problem(m=1500, batch=3)):
        for depth in (0, 1000):
            expect = cs_plain.rank(p, HW, depth)
            assert cs_tagged.rank(p, HW, depth) == expect
            assert tw.rank_configs(model, p, HW, tagged, depth) == expect
    empty = list(tagged)
    empty[2] = _copy(POOL[2], 0, {"wave_num": 1, "": 2})
    with pytest.raises(
        ValueError, match="config 2 has an attribute with an empty name"
    ):
        tw.CandidateSet(model, empty)
    with pytest.raises(ValueError, match="empty name"):
        tw.rank_configs(model, _problem(), HW, empty)
    with pytest.raises(ValueError, match="empty name"):
        tw.rank_configs(
            model, _problem(), tw.Hardware(), empty, context=tw.ExecutionContext(1)
        )


def test_v2_models_read_no_attributes(model):
    assert tw.attribute_names(model) == []


def test_execution_context_and_supports(model):
    ctx = tw.ExecutionContext()
    assert (ctx.cu_budget, ctx.schedule) == (0, tw.Schedule.Default)
    assert repr(tw.ExecutionContext(cu_budget=8, schedule=tw.Schedule.Auto)) == (
        "ExecutionContext(cu_budget=8, schedule=Auto)"
    )
    ctx.cu_budget, ctx.schedule = 3, tw.Schedule.Dynamic
    assert (ctx.cu_budget, ctx.schedule) == (3, tw.Schedule.Dynamic)
    n_cu = HW.N_CU
    supported = [
        tw.ExecutionContext(),
        tw.ExecutionContext(cu_budget=n_cu),
        tw.ExecutionContext(cu_budget=10 * n_cu, schedule=tw.Schedule.Default),
    ]
    unsupported = [
        tw.ExecutionContext(cu_budget=n_cu - 1),
        tw.ExecutionContext(cu_budget=1),
        tw.ExecutionContext(schedule=tw.Schedule.Dynamic),
        tw.ExecutionContext(cu_budget=n_cu, schedule=tw.Schedule.Auto),
    ]
    assert all(tw.supports(model, c, HW) for c in supported)
    assert not any(tw.supports(model, c, HW) for c in unsupported)

    def fields(c):
        return None if c is None else (c.cu_budget, c.schedule)

    # A v2 model covers every CU only: a smaller budget is scored as every
    # CU; a schedule it does not list is not scored.
    budget_only, schedule_only = unsupported[:2], unsupported[2:]
    for c in supported:
        assert fields(tw.scoring_context(model, c, HW)) == fields(c)
    for c in budget_only:
        assert fields(tw.scoring_context(model, c, HW)) == (0, tw.Schedule.Default)
    for c in schedule_only:
        assert tw.scoring_context(model, c, HW) is None

    cs = tw.CandidateSet(model, POOL)
    for depth in (0, 3):
        exclusive = cs.rank(_problem(), HW, depth)
        assert any(r.scored for r in exclusive)
        for c in supported + budget_only + [None]:
            assert cs.rank(_problem(), HW, depth, context=c) == exclusive
            assert tw.rank_configs(model, _problem(), HW, POOL, depth, c) == exclusive
        for c in schedule_only:
            for r in (
                cs.rank(_problem(), HW, depth, context=c),
                tw.rank_configs(model, _problem(), HW, POOL, depth, context=c),
            ):
                assert _order(r) == list(range(len(POOL)))
                assert not any(x.scored for x in r)


def test_launch_plan():
    p = tw.LaunchPlan()
    assert (p.dynamic_queue, p.grid_wgs, p.partial_reduction) == (
        False,
        0,
        tw.PartialReduction.None_,
    )
    assert [
        int(r)
        for r in (
            tw.PartialReduction.None_,
            tw.PartialReduction.Tree,
            tw.PartialReduction.Parallel,
            tw.PartialReduction.Atomic,
        )
    ] == [0, 1, 2, 3]
    q = tw.LaunchPlan(True, 2**40, tw.PartialReduction.Atomic)
    assert repr(q) == (
        "LaunchPlan(dynamic_queue=True, grid_wgs=1099511627776, "
        "partial_reduction=PartialReduction.Atomic)"
    )
    assert q != p and tw.LaunchPlan(partial_reduction=tw.PartialReduction.Atomic) != q
    q.dynamic_queue, q.grid_wgs, q.partial_reduction = (
        False,
        0,
        tw.PartialReduction.None_,
    )
    assert q == p
    with pytest.raises(TypeError):
        tw.LaunchPlan(grid_wgs=-1)


def test_v2_models_never_ask_a_launch_model(model):
    asked = []

    def launch(config, problem, context):
        asked.append(config.index)
        return tw.LaunchPlan(True, 1, tw.PartialReduction.Tree)

    class Planner:
        def plan(self, config, problem, context):
            return launch(config, problem, context)

        def launch_key(self, config):
            asked.append(config.index)
            return config.index

    plain = tw.CandidateSet(model, POOL)
    plans = [tw.LaunchPlan(True, 3)] * 20
    for source in (launch, Planner(), plans, tuple(plans), None):
        cs = tw.CandidateSet(model, POOL, launch=source)
        for p in (_problem(), _problem(m=100, n=3000, k=700)):
            for depth in (0, 1000):
                assert cs.rank(p, HW, depth) == plain.rank(p, HW, depth)
                assert tw.rank_configs(model, p, HW, POOL, depth, launch=source) == (
                    plain.rank(p, HW, depth)
                )
    assert asked == []
    f = tw.compute_features(model, _problem(), POOL[1], HW)
    g = tw.compute_features(model, _problem(), POOL[1], HW, plan=tw.LaunchPlan(True, 7))
    assert (f.query, f.item, f.interaction) == (g.query, g.item, g.interaction)
    for bad in (3, "plans", [tw.LaunchPlan(), 1]):
        with pytest.raises(TypeError):
            tw.CandidateSet(model, POOL, launch=bad)
        with pytest.raises(TypeError):
            tw.rank_configs(model, _problem(), HW, POOL, launch=bad)


def test_prior_tie_break(flat_model):
    pool = [
        _config(m, m, 32, index=i)
        for i, m in enumerate((256, 128, 128, 64, 256, 32, 64, 128))
    ]
    prior = [3.0, NAN, -1.0, 3.0, INF, -INF, 0.5, -1.0]
    by_prior, in_order = [2, 7, 6, 0, 3, 1, 4, 5], list(range(len(pool)))
    p = _problem()
    cs = tw.CandidateSet(flat_model, pool, tie_break=tw.TieBreak.Prior)
    assert _order(cs.rank(p, HW, prior=prior)) == by_prior
    assert _order(cs.rank(p, HW, 0, None, prior)) == by_prior
    assert _order(cs.rank(p, HW)) == in_order
    prior_rank = tw.rank_configs(
        flat_model, p, HW, pool, tie_break=tw.TieBreak.Prior, prior=prior
    )
    assert _order(prior_rank) == by_prior
    assert (
        _order(tw.CandidateSet(flat_model, pool).rank(p, HW, prior=prior)) == in_order
    )
    assert _order(tw.rank_configs(flat_model, p, HW, pool, prior=prior)) == in_order
    assert all(r.scored and r.score == 0.0 for r in prior_rank)
    for bad in (prior[:-1], prior + [0.0], []):
        with pytest.raises(ValueError, match="prior"):
            cs.rank(p, HW, prior=bad)
        with pytest.raises(ValueError, match="prior"):
            tw.rank_configs(flat_model, p, HW, pool, prior=bad)
    with pytest.raises(TypeError):
        cs.rank(p, HW, prior=["a"] * len(pool))


def test_duplicate_configs_rank_like_distinct_ones(model):
    kind = [(3 * i + 1) % len(POOL) for i in range(3 * len(POOL))]
    copies = [_copy(POOL[k], 100 + i, {"copy": i}) for i, k in enumerate(kind)]
    cs = tw.CandidateSet(model, copies)
    for p in (_problem(), _problem(m=100, n=3000, k=700), _problem(m=1500, batch=3)):
        tier1 = {
            r.config_index for r in tw.rank_configs(model, p, HW, POOL) if r.scored
        }
        base = {r.config_index: r for r in tw.rank_configs(model, p, HW, POOL, 10**6)}
        for depth in (0, 10**6):
            got = cs.rank(p, HW, depth)
            assert got == tw.rank_configs(model, p, HW, copies, depth)
            for r in got:
                b = base[kind[r.config_index]]
                assert r.scored == (b.scored and (depth > 0 or b.config_index in tier1))
                if r.scored:
                    assert r.score == b.score

            def key(i):
                return (kind[i] not in tier1, -base[kind[i]].score, i)

            scored = sorted((r.config_index for r in got if r.scored), key=key)
            assert [r.config_index for r in got if r.scored] == scored


def test_constant_item_features_whiten_to_zero_off_their_training_value():
    one = _f32(1.0 / 9.0)
    pool = [_config(128, 128, occupancy=o, index=o) for o in range(1, 10)]
    p = _problem()
    single = tw.load_model_from_memory(
        write_model([_cell(LARGE, edit=_item_probe(8, _f32(one - 1.87e-4), 1.87e-4))])
    )
    for r in (
        tw.rank_configs(single, p, HW, pool),
        tw.CandidateSet(single, pool).rank(p, HW),
    ):
        scores = {pool[x.config_index].occupancy: x.score for x in r}
        assert scores[1] == pytest.approx(1.0, rel=1e-3)
        assert all(scores[o] == 0.0 for o in range(2, 10))
    varied = tw.load_model_from_memory(
        write_model([_cell(LARGE, edit=_item_probe(8, one, 1e-3))])
    )
    scores = {
        pool[x.config_index].occupancy: x.score
        for x in tw.rank_configs(varied, p, HW, pool)
    }
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx((2.0 / 9.0 - 1.0 / 9.0) / 1e-3, rel=1e-4)


def test_pick_log_reports_the_pool_position(tmp_path):
    path = tmp_path / "flat.tilewright.bin"
    path.write_bytes(write_model([_cell(LARGE, edit=_zero)]))
    code = f"""
import tilewright as tw
m = tw.load_model({str(path)!r})
pool = [tw.Config(mt=tw.Dim3(s, s, 64), mi=tw.Dim3(16, 16, 32), occupancy=1) for s in (64, 128, 256)]
bf16 = tw.DataType.BFloat16
p = tw.Problem(size=tw.Dim3(4096, 4096, 4096), a_dtype=bf16, b_dtype=bf16, c_dtype=bf16,
               d_dtype=bf16, mi_dtype=bf16)
r = tw.rank_configs(m, p, tw.Hardware(64, 65536, 4 << 20), pool,
                    tie_break=tw.TieBreak.Prior, prior=[3.0, 2.0, 1.0])
print(r[0].config_index)
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TILEWRIGHT_")}
    env["TILEWRIGHT_PICK_LOG"] = "1"
    run = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == "2"
    pick = re.fullmatch(
        r"\[TILEWRIGHT_PICK\] m=4096 n=4096 k=4096 b=1 tA=N tB=N leaf=(\S+) "
        r"top1_sig=\(mt_m=256,mt_n=256,mt_k=64,mi_m=16,mi_n=16,mi_k=32,cha=0,chb=0\) "
        r"top1_score=0\.000000 top1_index=2 n_configs=3\n",
        run.stderr,
    )
    assert pick and pick.group(1) == LARGE


def _shipped_models():
    if not WEIGHTS.is_dir():
        return [], 0
    models, pointers = [], 0
    for path in sorted(WEIGHTS.glob("*/*/Tilewright/*.tilewright.bin")):
        with open(path, "rb") as f:
            if f.read(len(LFS_PREFIX)) == LFS_PREFIX:
                pointers += 1
                continue
        models.append(path)
    return models, pointers


class _ShippedLaunch:
    """The launch model a shipped model names in its provenance. The engine
    holds no framework's launch rule, so every config gets the default plan."""

    def __init__(self, name):
        self.id = name

    def plan(self, config, problem, context):
        return tw.LaunchPlan()


def _unsupported_context(model, hw):
    """A context the model scores nothing in, or None: a schedule it does not
    list (MLREC_v2 models: the default only; MLREC_v3 models: those their file
    declares). A CU budget alone never leaves a ranking unscored."""
    for context in (
        tw.ExecutionContext(schedule=tw.Schedule.Dynamic),
        tw.ExecutionContext(schedule=tw.Schedule.Auto),
    ):
        if tw.scoring_context(model, context, hw) is None:
            return context
    return None


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
    twice = [_copy(c, i, {"copy": i // len(pool)}) for i, c in enumerate(pool + pool)]
    hw = tw.Hardware(N_CU=128, lds_capacity=65536, L2_capacity=4 << 20)
    for path in models:
        m = tw.load_model(str(path))
        info = tw.describe(m)
        # <arch>/<subtree>/Tilewright/<model>: the subtree names the arch.
        assert info.arch.startswith(path.parent.parent.name[:6]) and info.n_cells > 0
        name = info.provenance.launch_model
        launch = _ShippedLaunch(name) if name else None
        cs = tw.CandidateSet(m, pool, launch=launch)
        p = _problem(m=4096, n=4096, k=4096)
        assert tw.route(m, p) >= 0
        ranked = cs.rank(p, hw)
        assert ranked == tw.rank_configs(m, p, hw, pool, launch=launch)
        assert any(r.scored for r in ranked)
        doubled = tw.CandidateSet(m, twice, launch=launch).rank(p, hw)
        assert doubled == tw.rank_configs(m, p, hw, twice, launch=launch)
        score = {r.config_index: r.score for r in ranked if r.scored}
        expect = sorted(
            (i for i in range(len(twice)) if i % len(pool) in score),
            key=lambda i: (-score[i % len(pool)], i),
        )
        assert [r.config_index for r in doubled if r.scored] == expect
        assert all(
            r.score == score[r.config_index % len(pool)] for r in doubled if r.scored
        )
        unsupported = _unsupported_context(m, hw)
        if unsupported is not None:
            assert not any(r.scored for r in cs.rank(p, hw, context=unsupported))
        for budget in (1, hw.N_CU // 8, hw.N_CU // 2):
            small = tw.ExecutionContext(cu_budget=budget)
            at = tw.scoring_context(m, small, hw)
            assert at is not None
            assert cs.rank(p, hw, context=small) == cs.rank(p, hw, context=at)


# The training pipeline beside the package, whose lib/feasibility.py mirrors
# the engine's serving rules and whose lib/mlrec.py writes MLREC_v3 files.
PIPELINE = Path(__file__).resolve().parents[2] / "training_pipeline"


def _pipeline():
    if not (PIPELINE / "lib" / "feasibility.py").is_file():
        pytest.skip(f"no training pipeline at {PIPELINE}")
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    from lib import feasibility, features, hardware, mlrec

    return feasibility, features, hardware, mlrec


def _v3_grid_model(features, hardware, mlrec, seed=3):
    """MLREC_v3 bytes of a model with a small random cell per base grid
    label, no whitelist tier and no cascade."""
    import numpy as np

    rng = np.random.default_rng(seed)
    q = features.v3_query_feature_names()
    i = features.v3_item_feature_names(())
    x = features.v3_interaction_feature_names()
    models = {}
    for m in ("Tiny", "Small", "Mid", "Large"):
        for n in ("Tiny", "Small", "Mid", "Large"):
            for k in ("TinyK", "MidK", "LargeK"):
                for b in ("Bnone", "Bany"):
                    shapes = mlrec.tensor_shapes(len(q), len(i), len(x), 4, 6, 5)
                    sd = {
                        name: rng.standard_normal(shape).astype(np.float32)
                        for name, shape in shapes.items()
                    }
                    sd["temperature"] = np.float32([1.0])
                    models[f"{m}|{n}|{k}|{b}"] = {
                        "state_dict": sd,
                        "embed_dim": 4,
                        "hidden_dim": 6,
                        "inter_hidden": 5,
                        "feature_norms": {
                            "q_mean": [0.0] * len(q),
                            "q_std": [1.0] * len(q),
                            "i_mean": [0.0] * len(i),
                            "i_std": [1.0] * len(i),
                            "x_mean": [0.0] * len(x),
                            "x_std": [1.0] * len(x),
                        },
                        "smart_k_signatures": [],
                        "view_key_hashes": [],
                    }
    bundle = {
        "catalog": "v3",
        "attribute_spec": [],
        "contexts": {"min_cu_fraction": 1.0, "max_cu_fraction": 1.0},
        "serving": {"whitelist_tier": False, "cascade_keys": 0},
        "models": models,
    }
    return mlrec.write_model(
        bundle, None, "gfx950", hardware.arch_constants("gfx950"), "fp32"
    )


def test_alignment_veto_matches_the_pipeline_mirror():
    """A model with the alignment veto leaves unscored exactly the configs
    lib/feasibility.py's alignment_vetoed names, unless it names them all,
    over every layout, bf16, f32 and f8 data and K around 512 and the
    128-byte lines; its pick is alignment_veto_served's first of the model
    without the veto's ranking; describe() reads back what mlrec wrote."""
    feasibility, features, hardware, mlrec = _pipeline()
    plain_bytes = _v3_grid_model(features, hardware, mlrec)
    veto = {"k_min": 512, "align_bits": 1024}
    veto_bytes = mlrec.with_alignment_veto(plain_bytes, 512, 1024)
    assert mlrec.read_header(veto_bytes)["serving"]["alignment_veto"] == veto
    assert mlrec.without_alignment_veto(veto_bytes) == plain_bytes
    plain = tw.load_model_from_memory(plain_bytes)
    with_veto = tw.load_model_from_memory(veto_bytes)
    assert not tw.describe(plain).alignment_veto.on
    info = tw.describe(with_veto).alignment_veto
    assert (info.on, info.k_min, info.align_bits) == (True, 512, 1024)
    assert isinstance(info, tw.AlignmentVeto) and "k_min=512" in repr(info)

    pool, kwargs = [], []
    for mt in ((64, 64, 32), (128, 64, 64), (32, 32, 64)):
        for hints in ((0, 0), (4, 0), (0, 4), (4, 4), (1, 0)):
            c = tw.Config(
                mt=tw.Dim3(*mt),
                mi=tw.Dim3(16, 16, 32),
                cache_hints_a=hints[0],
                cache_hints_b=hints[1],
                grvw_a=8,
                grvw_b=8,
                gwvw_d=4,
                index=len(pool),
            )
            pool.append(c)
            kwargs.append({"cache_hints_a": hints[0], "cache_hints_b": hints[1]})
    nt_a = [i for i, k in enumerate(kwargs) if k["cache_hints_a"] == 4]
    names = {
        tw.DataType.BFloat16: "bfloat16",
        tw.DataType.Float: "float",
        tw.DataType.Float8: "float8",
    }
    applied = 0
    for ta in (tw.Transpose.T, tw.Transpose.N):
        for tb in (tw.Transpose.T, tw.Transpose.N):
            for dt, name in names.items():
                for k in (512, 513, 576, 640, 1000, 1024, 4096, 4100):
                    p = tw.Problem(
                        size=tw.Dim3(4096, 4096, k),
                        batch=1,
                        a_transpose=ta,
                        b_transpose=tb,
                        a_dtype=dt,
                        b_dtype=dt,
                        c_dtype=dt,
                        d_dtype=dt,
                        mi_dtype=dt,
                    )
                    problem = {
                        "k": k,
                        "a_transpose": "T" if ta == tw.Transpose.T else "N",
                        "b_transpose": "T" if tb == tw.Transpose.T else "N",
                        "a_dtype": name,
                        "b_dtype": name,
                    }
                    vetoed = {
                        i
                        for i, c in enumerate(kwargs)
                        if feasibility.alignment_vetoed(problem, c, veto)
                    }
                    applied += bool(vetoed)
                    base = tw.rank_configs(plain, p, HW, pool, len(pool))
                    got = tw.rank_configs(with_veto, p, HW, pool, len(pool))
                    assert {r.config_index for r in got if r.scored} == {
                        r.config_index for r in base if r.scored
                    } - vetoed
                    ranked = [kwargs[r.config_index] for r in base if r.scored]
                    served = feasibility.alignment_veto_served(problem, ranked, veto)
                    picks = [r.config_index for r in base if r.scored]
                    assert got[0].scored and got[0].config_index == picks[served[0]]
                    # A pool the veto would empty ranks as without it.
                    only = [pool[i] for i in nt_a]
                    base = tw.rank_configs(plain, p, HW, only, len(only))
                    got = tw.rank_configs(with_veto, p, HW, only, len(only))
                    out = {j for j, i in enumerate(nt_a) if i in vetoed}
                    if len(out) == len(only):
                        assert got == base
                    else:
                        assert {r.config_index for r in got if r.scored} == {
                            r.config_index for r in base if r.scored
                        } - out
    assert applied > 0


def test_alignment_veto_reads_the_width_of_the_operand_it_applies_to():
    """With operands of different widths the engine and lib/feasibility.py
    read the K-contiguous operand's (A in TT, B in NN): at K 576 the 8-bit
    operand's elements fill no whole 128-byte line and the 16-bit one's do."""
    feasibility, features, hardware, mlrec = _pipeline()
    plain_bytes = _v3_grid_model(features, hardware, mlrec)
    plain = tw.load_model_from_memory(plain_bytes)
    with_veto = tw.load_model_from_memory(
        mlrec.with_alignment_veto(plain_bytes, 512, 1024)
    )
    veto = {"k_min": 512, "align_bits": 1024}
    pool, kwargs = [], []
    for mt in ((64, 64, 32), (128, 64, 64)):
        for hints in ((0, 0), (4, 0), (0, 4), (4, 4)):
            pool.append(
                tw.Config(
                    mt=tw.Dim3(*mt),
                    mi=tw.Dim3(16, 16, 32),
                    cache_hints_a=hints[0],
                    cache_hints_b=hints[1],
                    grvw_a=8,
                    grvw_b=8,
                    gwvw_d=4,
                    index=len(pool),
                )
            )
            kwargs.append({"cache_hints_a": hints[0], "cache_hints_b": hints[1]})
    types = {"float8": tw.DataType.Float8, "bfloat16": tw.DataType.BFloat16}
    for ta, tb, a, b, applies in (
        ("T", "T", "float8", "bfloat16", True),
        ("T", "T", "bfloat16", "float8", False),
        ("N", "N", "bfloat16", "float8", True),
        ("N", "N", "float8", "bfloat16", False),
    ):
        p = tw.Problem(
            size=tw.Dim3(4096, 4096, 576),
            batch=1,
            a_transpose=tw.Transpose.T if ta == "T" else tw.Transpose.N,
            b_transpose=tw.Transpose.T if tb == "T" else tw.Transpose.N,
            a_dtype=types[a],
            b_dtype=types[b],
            c_dtype=types[a],
            d_dtype=types[a],
            mi_dtype=types[a],
        )
        problem = {"k": 576, "a_transpose": ta, "b_transpose": tb}
        problem.update(a_dtype=a, b_dtype=b)
        vetoed = {
            i
            for i, c in enumerate(kwargs)
            if feasibility.alignment_vetoed(problem, c, veto)
        }
        assert bool(vetoed) is applies, (ta, tb, a, b)
        base = tw.rank_configs(plain, p, HW, pool, len(pool))
        got = tw.rank_configs(with_veto, p, HW, pool, len(pool))
        assert {r.config_index for r in got if r.scored} == {
            r.config_index for r in base if r.scored
        } - vetoed
