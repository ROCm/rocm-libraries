# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import numpy as np
import pytest

from lib import features as fs
from lib import mlrec
from lib.hardware import DeviceHardware, arch_constants

from test_mlrec import LABELS, random_bundle, shipped_models

HW = DeviceHardware(n_cu=100, lds_bytes=65536, l2_bytes=4 << 20)
DTYPES = [
    "bfloat16",
    "half",
    "float",
    "xfloat32",
    "float8",
    "bfloat8",
    "float8_fnuz",
    "bfloat8_fnuz",
    "float8bfloat8",
    "bfloat8float8_fnuz",
    "float4",
    "float6",
    "bfloat6",
    "int8",
    "int4",
    "int32",
    "int64",
    "int8x4",
    "double",
    "complexfloat",
    "complexdouble",
    "float8bfloat8_fnuz",
    "bfloat8float8",
]
MI_SHAPES = [
    (16, 16, 32),
    (32, 32, 16),
    (32, 32, 8),
    (16, 16, 128),
    (16, 16, 64),
    (32, 32, 64),
    (16, 16, 4),
    (1, 1, 64),
    (32, 16, 128),
    (16, 16, 16),
]


def test_catalog_hash_and_dims():
    assert fs.feature_names_hash() == "e7fe4b524851e895"
    assert len(fs.query_feature_names()) == 55
    assert len(fs.item_feature_names()) == 12
    assert len(fs.interaction_feature_names()) == 37
    assert len(set(fs.feature_names())) == len(fs.feature_names()) == 104


def test_list_is_in_catalog_order():
    kw = dict(m=2048, n=1000, k=512, a_transpose="T", mt_m=128, mt_n=64, mt_k=32)
    c = arch_constants("gfx950")
    d = fs.compute_generic_features(**kw, hardware=HW, constants=c, as_dict=True)
    assert set(d) == set(fs.feature_names())
    lst = fs.compute_generic_features(**kw, hardware=HW, constants=c)
    assert lst == [d[n] for n in fs.feature_names()]
    q, i, x = fs.split_features(lst)
    assert q + i + x == lst


def test_mi_latency_uses_model_table_and_plain_fp8_names():
    c = arch_constants("gfx950")
    assert fs.get_mi_latency(32, 32, 16, "bf16", c) == 32 / 4
    assert fs.get_mi_latency(16, 16, 128, "float8_fnuz", c) == 16 / 4
    assert fs.get_mi_latency(16, 16, 128, "bfloat8_fnuz", c) == 16 / 4
    assert fs.get_mi_latency(16, 16, 77, "bf16", c) == 32 / 4
    g = arch_constants("gfx1250")
    assert fs.get_mi_latency(16, 16, 64, "bfloat8float8_fnuz", g) == 4 / 4
    assert fs.get_mi_latency(16, 16, 128, "f8", g) == 8 / 4


def test_row_helpers():
    row = dict(
        m="64",
        n="128",
        k="256",
        batch_count="2",
        transA="T",
        transB="N",
        a_type="f32_r",
        b_type="f32_r",
        c_type="f32_r",
        d_type="f32_r",
        compute_type="xf32_r",
        mt_m="64",
        occupancy="",
        grvw_a="0",
    )
    p = fs.problem_kwargs_from_row(row)
    assert p["mi_dtype"] == "xfloat32" and p["a_dtype"] == "float"
    assert p["batch"] == 2
    c = fs.config_kwargs_from_row(row)
    assert (
        c["mt_m"] == 64 and c["mt_n"] == 1 and c["occupancy"] == 1 and c["grvw_a"] == 1
    )
    assert (
        fs.problem_kwargs_from_row(dict(row, compute_type="f32_r"))["mi_dtype"]
        == "float"
    )
    with pytest.raises(ValueError):
        fs.problem_kwargs_from_row(dict(row, a_type="q3_r"))


def random_case(rng):
    a, b = rng.choice(DTYPES), rng.choice(DTYPES)
    hw = DeviceHardware(
        n_cu=rng.choice([32, 80, 100, 120, 256, 300]),
        lds_bytes=rng.choice([32768, 65536, 98304, 163840]),
        l2_bytes=rng.choice([2 << 20, 4 << 20, 6 << 20, 16 << 20]),
    )
    prob = dict(
        m=rng.choice(
            [1, 2, 3, 7, 16, 31, 32, 33, 100, 128, 129, 512, 513, 4096, 65536]
        ),
        n=rng.randint(1, 20000),
        k=rng.choice([1, 8, 16, 32, 33, 64, 100, 512, 513, 1024, 4096, 131072]),
        batch=rng.choice([1, 1, 1, 2, 7, 64, 1024]),
        a_transpose=rng.choice("TN"),
        b_transpose=rng.choice("TN"),
        a_dtype=a,
        b_dtype=b,
        c_dtype=rng.choice(DTYPES),
        d_dtype=rng.choice(DTYPES),
        mi_dtype=rng.choice([a, "xfloat32", "float8", "bfloat8_fnuz"]),
    )
    mi = rng.choice(MI_SHAPES)
    cfg = dict(
        mt_m=rng.choice([16, 32, 64, 128, 256, 512]),
        mt_n=rng.choice([16, 32, 64, 128, 256]),
        mt_k=rng.choice([16, 32, 64, 128, 256, 512]),
        mi_m=mi[0],
        mi_n=mi[1],
        mi_k=mi[2],
        occupancy=rng.choice([1, 2, 4]),
        cache_hints_a=rng.choice([0, 0, 1, 2, 3, 4]),
        cache_hints_b=rng.choice([0, 0, 4, 7]),
        grvw_a=rng.choice([1, 2, 4, 8, 16]),
        grvw_b=rng.choice([1, 2, 4, 8]),
        gwvw_d=rng.choice([1, 2, 4, 8]),
    )
    return hw, prob, cfg


def assert_engine_parity(tw, data, n_cases, seed):
    from lib import evaluate as ev

    model = tw.load_model_from_memory(data)
    consts = mlrec.read_model(data)["constants"]
    rng = random.Random(seed)
    for _ in range(n_cases):
        hw, prob, cfg = random_case(rng)
        q, i, x = fs.feature_vectors(prob, cfg, hardware=hw, constants=consts)
        py = np.asarray(q + i + x, dtype=np.float32)
        f = tw.compute_features(
            model,
            ev.make_problem(tw, prob),
            ev.make_config(tw, cfg, 0),
            ev.make_hardware(tw, hw),
        )
        cc = np.asarray(list(f.query) + list(f.item) + list(f.interaction), np.float32)
        assert py.view(np.uint32).tolist() == cc.view(np.uint32).tolist(), (prob, cfg)


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250v0"])
def test_features_match_engine(arch):
    tw = pytest.importorskip("tilewright")
    data = mlrec.write_model(
        random_bundle(LABELS[:1]), None, arch, arch_constants(arch), "fp32"
    )
    assert_engine_parity(tw, data, 1500, seed=sum(arch.encode()))


def test_features_match_engine_on_shipped_models():
    tw = pytest.importorskip("tilewright")
    models = shipped_models()
    picked = {}
    for path, data in models:
        picked.setdefault(mlrec.read_header(data)["arch"], data)
    if not picked:
        pytest.skip("shipped model files are not materialized")
    for n, data in enumerate(picked.values()):
        assert_engine_parity(tw, data, 1500, seed=n)
