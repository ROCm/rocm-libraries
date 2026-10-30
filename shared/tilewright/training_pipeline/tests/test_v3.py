# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import dataclasses
import math
import random
import struct

import numpy as np
import pytest

from lib import evaluate as ev
from lib import features as fs
from lib import hipblaslt_launch as hl
from lib import mlrec
from lib.hardware import DeviceHardware, arch_constants
from test_features import random_case
from test_mlrec import shipped_models

SPEC = fs.DEFAULT_ATTRIBUTE_SPEC
SAMPLE_SPEC = (
    fs.AttributeFeature("param_a"),
    fs.AttributeFeature("param_b", 1, "log2p1"),
    fs.AttributeFeature("param_c", 0, "nonzero"),
    fs.AttributeFeature("param_d", -1, "onehot", (0, 1, 2, 4)),
)
LARGE = "Large|Large|LargeK|Bnone"
LABELS = [LARGE, "Tiny|Small|TinyK|Bany", "Mid|Mid|MidK|Bnone", "Small|Tiny|MidK|Bany"]
HW = DeviceHardware(n_cu=256, lds_bytes=163840, l2_bytes=4 << 20)
BF16_NT = dict(
    a_transpose="N", b_transpose="T", a_dtype="bfloat16", b_dtype="bfloat16",
    c_dtype="bfloat16", d_dtype="bfloat16", mi_dtype="bfloat16",
)  # fmt: skip
BASE_CFG = dict(
    mt_m=128, mt_n=128, mt_k=64, mi_m=16, mi_n=16, mi_k=32, occupancy=1,
    cache_hints_a=0, cache_hints_b=0, grvw_a=8, grvw_b=8, gwvw_d=4,
)  # fmt: skip


def v3_bundle(labels=LABELS, *, spec=SPEC, seed=0, clip=False, **extra):
    rng = np.random.default_rng(seed)
    q = fs.v3_query_feature_names()
    i = fs.v3_item_feature_names(spec)
    x = fs.v3_interaction_feature_names()
    models = {}
    for label in labels:
        shapes = mlrec.tensor_shapes(len(q), len(i), len(x), 4, 6, 5)
        sd = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
        sd["temperature"] = np.float32([1.0])
        entry = {
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
        if clip:
            entry["feature_bounds"] = {
                "q_lo": [-1e6] * len(q),
                "q_hi": [1e6] * len(q),
                "i_lo": [-1e6] * len(i),
                "i_hi": [1e6] * len(i),
                "x_lo": [-1e6] * len(x),
                "x_hi": [1e6] * len(x),
            }
        models[label] = entry
    bundle = {
        "catalog": "v3",
        "attribute_spec": [a.to_dict() for a in spec],
        "contexts": {
            "min_cu_fraction": 0.25,
            "max_cu_fraction": 1.0,
            "schedules": ["default", "dynamic"],
        },
        "clip": clip,
        "serving": {"cascade_keys": 32},
        "models": models,
    }
    bundle.update(extra)
    return bundle


def write(bundle, arch="gfx950", dtype="fp32", provenance=None):
    return mlrec.write_model(
        bundle, None, arch, arch_constants(arch), dtype, provenance
    )


def random_attributes(rng, spec=SAMPLE_SPEC):
    """Attributes of `spec` (some missing) and one no spec reads."""
    attrs = {"unread": rng.choice([0, 1, 7])}
    for a in spec:
        if rng.random() < 0.2:
            continue
        if a.transform == "onehot":
            attrs[a.name] = rng.choice(list(a.values) + [a.default, 99])
        else:
            attrs[a.name] = rng.choice([0, 1, 2, 3, 4, 8, 16, -1, 255])
    return attrs


def test_catalog_hash_matches_engine():
    assert fs.v3_catalog_hash() == "b8b9f4a518add96f"
    tw = pytest.importorskip("tilewright")
    assert fs.v3_catalog_hash() in tw.feature_catalog_hashes()


def test_v3_features_are_the_signature_and_the_launch():
    assert SPEC == ()
    assert fs.V3_ITEM == (
        "log2_mt_m", "log2_mt_n", "log2_mt_k", "log2_mi_m", "log2_mi_n",
        "log2_mi_k", "nta_norm", "ntb_norm",
    )  # fmt: skip
    assert fs.SIGNATURE_FIELDS == (
        "mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k",
        "cache_hints_a", "cache_hints_b",
    )  # fmt: skip
    names = fs.v3_query_feature_names() + fs.v3_item_feature_names()
    names += fs.v3_interaction_feature_names()
    for knob in ("grvw", "gwvw", "occ", "wgm", "stagger", "prefetch", "attr_"):
        assert not any(knob in n for n in names), knob
    assert (len(fs.v3_query_feature_names()), len(fs.v3_item_feature_names())) == (
        59,
        8,
    )
    assert len(fs.v3_interaction_feature_names()) == 44
    assert fs.v3_interaction_feature_names()[-10:] == list(fs.V3_LAUNCH)


def test_attribute_spec_parsing():
    spec = fs.parse_attribute_spec([a.to_dict() for a in SAMPLE_SPEC])
    assert spec == SAMPLE_SPEC
    assert fs.parse_attribute_spec([]) == ()
    assert fs.parse_attribute_spec({"attributes": [{"name": "x"}]})[0].transform == (
        "identity"
    )
    bad = [
        [{"name": "a"}, {"name": "a"}],
        [{"name": "a", "transform": "cube"}],
        [{"name": "a", "transform": "onehot"}],
        [{"name": "a", "transform": "onehot", "values": [1, 1]}],
        [{"name": "a", "values": [1]}],
        [{"name": "a b"}],
        [{"name": "a", "extra": 1}],
    ]
    for raw in bad:
        with pytest.raises(ValueError):
            fs.parse_attribute_spec(raw)
    names = fs.v3_item_feature_names(SAMPLE_SPEC)
    assert len(names) == len(set(names)) == 8 + 3 + 4


@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_round_trip_and_header(spec):
    bundle = v3_bundle(spec=spec, clip=True)
    bundle["models"][LABELS[1]]["view_key_hashes"] = [5, 3, 2**64 - 1]
    data = write(bundle)
    parsed = mlrec.read_model(data)
    assert parsed["version"] == 3 and parsed["attribute_spec"] == spec
    assert parsed["contexts"]["schedules"] == ["default", "dynamic"]
    assert parsed["serving"] == {
        "small_batched_rule": False,
        "dot2_rule": False,
        "non_temporal_rule": False,
        "whitelist_tier": True,
        "launch_keys": False,
        "non_temporal_family_rule": False,
        "bank": False,
        "trained_key_gate": False,
        "cascade_keys": 32,
    }
    assert parsed["trained_view_keys"] is None
    keyed = write(dict(bundle, serving={"launch_keys": True}))
    assert mlrec.read_model(keyed)["serving"]["launch_keys"]
    assert write(mlrec.bundle_from_model(mlrec.read_model(keyed))) == keyed
    cell = {c["label"]: c for c in parsed["cells"]}[LABELS[1]]
    assert cell["view_key_hashes"] == [3, 5, 2**64 - 1]
    again = write(mlrec.bundle_from_model(parsed))
    assert again == data
    header = mlrec.read_header(data)
    item = len(fs.v3_item_feature_names(spec))
    assert header["version"] == 3 and header["dims"] == (59, item, 44)
    assert header["feature_spec_hash"] == mlrec.feature_spec_hash(spec)


def test_writer_rejects_invalid_bundles():
    for contexts in (
        {"min_cu_fraction": 0.0},
        {"min_cu_fraction": 0.5, "max_cu_fraction": 0.4},
        {"schedules": []},
        {"schedules": ["fast"]},
    ):
        with pytest.raises(ValueError):
            write(v3_bundle(contexts=contexts))
    with pytest.raises(ValueError):
        write(v3_bundle(i_names=fs.v3_item_feature_names(SAMPLE_SPEC)))
    with pytest.raises(ValueError):
        write(v3_bundle(spec=SAMPLE_SPEC, i_names=fs.v3_item_feature_names()))
    bad = v3_bundle(clip=True)
    bad["models"][LARGE]["feature_bounds"]["q_lo"][0] = 2e6
    with pytest.raises(ValueError, match="lower bound"):
        write(bad)


def test_reader_rejects_corruption():
    data = write(v3_bundle())
    flipped = bytearray(data)
    flipped[-20] ^= 1
    for bad in (data[:-1], data[:100], bytes(flipped)):
        with pytest.raises(mlrec.FormatError):
            mlrec.read_model(bad)
    reserved = bytearray(data)
    reserved[29] = 1
    with pytest.raises(mlrec.FormatError, match="reserved"):
        mlrec.read_model(bytes(reserved))
    tw = pytest.importorskip("tilewright")
    for bad in (data[:-1], bytes(flipped), bytes(reserved)):
        with pytest.raises(ValueError):
            tw.load_model_from_memory(bad)


def test_engine_loads_python_v3():
    tw = pytest.importorskip("tilewright")
    data = write(v3_bundle(labels=LABELS[:2], clip=True), dtype="bf16")
    info = tw.describe(tw.load_model_from_memory(data))
    assert info.format_version == 3
    assert info.feature_catalog_hash == fs.v3_catalog_hash()
    assert info.feature_spec_hash == mlrec.feature_spec_hash(SPEC)
    assert (info.query_dim, info.item_dim, info.inter_dim) == (59, 8, 44)
    assert (info.min_cu_fraction, info.max_cu_fraction) == (0.25, 1.0)
    assert info.schedules == 0b011 and info.cascade_keys == 32 and info.clip
    assert info.whitelist_tier and not info.non_temporal_rule
    assert not info.small_batched_rule and not info.dot2_rule
    assert not info.launch_keys
    assert info.uncovered_base_labels == 94
    keyed = v3_bundle(labels=LABELS[:2], serving={"launch_keys": True})
    assert tw.describe(tw.load_model_from_memory(write(keyed))).launch_keys


def random_launch_attributes(rng, attrs):
    attrs = dict(attrs)
    for name, values in (
        ("tile_processing_strategy", [0, 1, 2, 2, 2]),
        ("work_assignment", [0, 1, 2, 2, 2]),
        ("stream_k_atomic", [0, 0, 0, 1]),
        ("work_group_x", [64, 128, 256, 1024]),
        ("work_group_y", [1, 1, 2, 4, 1024]),
        ("work_group_z", [1, 1, 1, 2]),
    ):
        if rng.random() < 0.9:
            attrs[name] = rng.choice(values)
    if attrs.get("tile_processing_strategy") == 1:
        attrs["work_assignment"] = 0
    return attrs


def launch_case(rng):
    """A problem on a 256-CU device whose tile count sits near hipBLASLt's
    dynamic thresholds or its k-split ones."""
    mt_m, mt_n, mt_k = (rng.choice(v) for v in ([64, 128, 256], [64, 128], [32, 64]))
    budget = rng.choice([0, 31, 32, 64, 100, 128, 192, 200, 224, 240, 255, 256, 300])
    per_cu = max(budget // 32 * 32, 1) if 0 < budget < 256 else 256
    tiles = rng.choice(
        [1, 4, 16, 64, 255, 300, 479, 480, 481, 1024, 4500]
        + [int(8.41 * per_cu) + d for d in (-1, 0, 1, 2)]
    )
    batch = rng.choice([1, 1, 1, 2, 8])
    n_tiles = rng.choice([1, 2, 3])
    prob = dict(
        BF16_NT,
        m=max(
            mt_m * -(-tiles // (n_tiles * batch)) - rng.choice([0, 0, 1, mt_m // 2]), 1
        ),
        n=mt_n * n_tiles,
        k=rng.choice([1, 64, 512, 2048, 4096, 64 * 300, 64 * 65536]),
        batch=batch,
        a_transpose="T",
        b_transpose="N",
        mi_dtype=rng.choice(["bfloat16", "float", "double", "xfloat32"]),
    )
    cfg = dict(
        BASE_CFG, mt_m=mt_m, mt_n=mt_n, mt_k=mt_k,
        occupancy=rng.choice([-1, 0, 1, 2, 3, 4, 5, 9]),
    )  # fmt: skip
    ctx = fs.ExecutionContext(cu_budget=budget, schedule=rng.choice(fs.SCHEDULES))
    return HW, prob, cfg, ctx


def random_plan(rng, prob, cfg):
    tiles = fs.output_tiles(prob, cfg)
    return fs.LaunchPlan(
        dynamic_queue=rng.random() < 0.3,
        grid_wgs=rng.choice([0, 0, 1, 3, max(tiles - 1, 0), tiles, tiles + 5, 2**40]),
        partial_reduction=rng.choice(fs.PARTIAL_REDUCTIONS),
    )


def same_signature_variant(rng, cfg, attrs):
    """A config that differs from `cfg` only outside the signature and the
    spec: what a v3 model cannot tell apart."""
    other = dict(
        cfg,
        occupancy=rng.choice([1, 2, 4]),
        grvw_a=rng.choice([1, 2, 4, 8]),
        grvw_b=rng.choice([1, 2, 4, 8]),
        gwvw_d=rng.choice([1, 2, 4, 8]),
    )
    return other, dict(attrs, unread=rng.choice([3, 5, 11]))


@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_v3_features_match_engine(spec):
    """Python and the engine agree bit for bit over random problems,
    contexts, attributes and launch plans: hipBLASLt's plans near its
    thresholds and arbitrary ones. A same-signature variant launched alike
    has the same features."""
    tw = pytest.importorskip("tilewright")
    data = write(v3_bundle(labels=LABELS[:1], spec=spec))
    model = tw.load_model_from_memory(data)
    consts = mlrec.read_model(data)["constants"]
    rng = random.Random(len(spec) + 3)

    def both(prob, cfg, attrs, hw, ctx, plan):
        q, it, x = fs.v3_feature_vectors(
            prob, cfg, attrs, hardware=hw, constants=consts, context=ctx,
            spec=spec, plan=plan,
        )  # fmt: skip
        f = tw.compute_features(
            model,
            ev.make_problem(tw, prob),
            ev.make_config(tw, dict(cfg, attributes=attrs), 0),
            ev.make_hardware(tw, hw),
            ev.make_context(tw, ctx),
            plan=ev.make_plan(tw, plan),
        )
        py = np.asarray(q + it + x, dtype=np.float32).view(np.uint32).tolist()
        cc = list(f.query) + list(f.item) + list(f.interaction)
        return py, np.asarray(cc, np.float32).view(np.uint32).tolist()

    dynamic = reductions = 0
    for i in range(3000):
        if i % 2:
            hw, prob, cfg, ctx = launch_case(rng)
            attrs = random_launch_attributes(rng, random_attributes(rng, spec))
            plan = hl.launch_plan(prob, cfg, attrs, ctx, hw.n_cu, "gfx950")
        else:
            hw, prob, cfg = random_case(rng)
            ctx = fs.ExecutionContext(
                cu_budget=rng.choice(
                    [0, 0, 1, 7, 32, hw.n_cu // 2, hw.n_cu - 1, hw.n_cu, 999]
                ),
                schedule=rng.choice(fs.SCHEDULES),
            )
            attrs = random_attributes(rng, spec)
            plan = random_plan(rng, prob, cfg)
        dynamic += plan.dynamic_queue
        reductions += plan.partial_reduction != "none"
        py, cc = both(prob, cfg, attrs, hw, ctx, plan)
        assert py == cc, (prob, cfg, attrs, ctx, plan)
        if i % 10 == 0:
            cfg2, attrs2 = same_signature_variant(rng, cfg, attrs)
            assert both(prob, cfg2, attrs2, hw, ctx, plan) == (py, cc)
    assert 300 < dynamic < 2700 and 300 < reductions < 2700


# MLREC_v3 files list view keys as hashes only, so the kernels behind a
# shipped model's keys are looked up among these signatures: macro tiles of
# multiples of 16 up to 1024, DepthU powers of two from 8 to 2048, the MI
# shapes of the model's MI table and cache hints 0 to 7.
GRID_MT = range(16, 1025, 16)
GRID_DEPTH_U = [2**e for e in range(3, 12)]
GRID_HINTS = range(8)
FNV_OFFSET, FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3


def fnv1a(states, rows, back=False):
    """The FNV-1a 64 `states` after absorbing each row of int64 `rows` as
    `fs.view_key_hash` does, or with `back` the states before it: the prime
    is odd, so each step inverts."""
    data = np.ascontiguousarray(rows, dtype="<i8").view(np.uint8)
    data = data.reshape(len(rows), -1)
    h = np.array(states, dtype=np.uint64)
    if back:
        inverse = np.uint64(pow(FNV_PRIME, -1, 1 << 64))
        for j in reversed(range(data.shape[1])):
            h = (h * inverse) ^ data[:, j]
    else:
        for j in range(data.shape[1]):
            h = (h ^ data[:, j]) * np.uint64(FNV_PRIME)
    return h


def listed_keys(parsed):
    """The view keys an MLREC_v3 model lists: its trained keys and its
    cells' whitelists."""
    keys = set(parsed["trained_view_keys"] or ())
    keys.update(k for cell in parsed["cells"] for k in cell["view_key_hashes"])
    return keys


def listed_signatures(parsed):
    """The grid's signatures whose view keys, with the spec's default
    attributes, the model lists. The two halves of the hash meet: the
    grid's tiles and MI shapes hashed forward, each listed key stepped back
    over every pair of cache hints."""
    keys = listed_keys(parsed)
    if not keys:
        return []
    defaults = [a.default for a in parsed["attribute_spec"]]
    hints = np.array(
        [(a, b, *defaults) for a in GRID_HINTS for b in GRID_HINTS], dtype=np.int64
    )
    want = np.array(sorted(keys), dtype=np.uint64)
    need = fnv1a(np.repeat(want, len(hints)), np.tile(hints, (len(want), 1)), back=True)
    tiles = np.array(
        [(m, n, d) for m in GRID_MT for n in GRID_MT for d in GRID_DEPTH_U],
        dtype=np.int64,
    )
    start = fnv1a(np.full(len(tiles), FNV_OFFSET, dtype=np.uint64), tiles)
    found = {}
    for mi in sorted({entry[:3] for entry in parsed["constants"].mi_table}):
        mid = fnv1a(start, np.tile(np.array(mi, dtype=np.int64), (len(tiles), 1)))
        for i in np.flatnonzero(np.isin(mid, need)).tolist():
            found[int(mid[i])] = tuple(tiles[i].tolist()) + mi
    return sorted(
        found[s] + tuple(hints[j % len(hints), :2].tolist())
        for j, s in enumerate(need.tolist())
        if s in found
    )


def declared_context(rng, contexts, n_cu):
    """A context with a CU budget in, at the ends of or just outside a
    model's declared CU fractions, and any schedule."""
    lo = math.ceil(contexts["min_cu_fraction"] * n_cu)
    hi = max(math.floor(contexts["max_cu_fraction"] * n_cu), lo)
    budget = rng.choice([0, lo, hi, rng.randint(lo, hi), max(lo - 1, 0), n_cu + 1])
    return fs.ExecutionContext(cu_budget=budget, schedule=rng.choice(fs.SCHEDULES))


def test_v3_features_match_engine_on_shipped_models():
    """Python and the engine agree bit for bit on every shipped MLREC_v3
    model, with its own constants and attribute spec, for each kernel it
    lists: in a random problem with a context in or around its declared
    ones and an arbitrary launch plan, and in a problem near hipBLASLt's
    launch thresholds with hipBLASLt's plan on its arch. A model that lists
    no kernel is checked with random ones."""
    tw = pytest.importorskip("tilewright")
    models = [
        (path, data)
        for path, data in shipped_models()
        if mlrec.read_header(data)["version"] == 3
    ]
    if not models:
        pytest.skip("no shipped MLREC_v3 model is materialized")
    for seed, (path, data) in enumerate(models):
        parsed = mlrec.read_model(data)
        spec, consts = parsed["attribute_spec"], parsed["constants"]
        listed = listed_keys(parsed)
        kernels = [dict(zip(fs.SIGNATURE_FIELDS, s)) for s in listed_signatures(parsed)]
        found = {fs.view_key_hash(fs.v3_view_key(k, {}, spec)) for k in kernels}
        assert found <= listed and (found or not listed), path.name
        model = tw.load_model_from_memory(data)
        # Any XCD count serves an arch whose queue count hipBLASLt leaves to
        # the device.
        launch = hl.HipblasltLaunch(parsed["arch"], num_xcd=8)
        rng = random.Random(seed)
        for i in range(2 * max(len(kernels), 300)):
            if i % 2:
                hw, prob, cfg, ctx = launch_case(rng)
                attrs = random_launch_attributes(rng, random_attributes(rng, spec))
            else:
                hw, prob, cfg = random_case(rng)
                ctx = declared_context(rng, parsed["contexts"], hw.n_cu)
                attrs = random_attributes(rng, spec)
            if kernels:
                cfg = dict(cfg, **kernels[i // 2 % len(kernels)])
            if i % 2:
                plan = launch(prob, cfg, attrs, ctx, hw.n_cu)
            else:
                plan = random_plan(rng, prob, cfg)
            q, it, x = fs.v3_feature_vectors(
                prob, cfg, attrs, hardware=hw, constants=consts, context=ctx,
                spec=spec, plan=plan,
            )  # fmt: skip
            f = tw.compute_features(
                model,
                ev.make_problem(tw, prob),
                ev.make_config(tw, dict(cfg, attributes=attrs), 0),
                ev.make_hardware(tw, hw),
                ev.make_context(tw, ctx),
                plan=ev.make_plan(tw, plan),
            )
            py = np.asarray(q + it + x, dtype=np.float32).view(np.uint32).tolist()
            cc = list(f.query) + list(f.item) + list(f.interaction)
            assert len(py) == sum(parsed["dims"]), path.name
            assert py == np.asarray(cc, np.float32).view(np.uint32).tolist(), (
                path.name, prob, cfg, attrs, ctx, plan,
            )  # fmt: skip


def test_exclusive_v3_matches_v2_shared_features():
    consts = arch_constants("gfx950")
    rng = random.Random(7)
    names2 = fs.interaction_feature_names()
    names3 = fs.v3_interaction_feature_names()
    shared = [n for n in names3 if n in names2]
    assert [n for n in names2 if n not in names3] == [
        "log2_grvw_a_bytes",
        "log2_grvw_b_bytes",
        "bw_occ",
    ]
    for _ in range(200):
        _, prob, cfg = random_case(rng)
        q2, i2, x2 = fs.feature_vectors(prob, cfg, hardware=HW, constants=consts)
        q3, i3, x3 = fs.v3_feature_vectors(prob, cfg, hardware=HW, constants=consts)
        assert q3[:55] == q2 and q3[55:] == [1.0, 1.0, 0.0, 0.0]
        assert i3 == i2[:8]
        v3 = dict(zip(names3, x3))
        v2 = dict(zip(names2, x2))
        assert [v3[n] for n in shared] == [v2[n] for n in shared]


def test_launch_features_from_the_plan():
    prob = dict(m=128 * 480, n=128, k=512, batch=1)
    cfg = dict(mt_m=128, mt_n=128)
    x = fs.launch_features(prob, cfg, fs.LaunchPlan(False, 320, "tree"), 224)
    assert dict(zip(fs.V3_LAUNCH, x)) == {
        "is_dynamic": 0.0,
        "log2_grid": math.log2(320),
        "log2_grid_per_cu": math.log2(320 / 224),
        "log2_wg_waves": 1.0,
        "wg_wave_eff": 320 / (2 * 224),
        "log2_tiles_per_wg": math.log2(1.5),
        "uneven_work": 1.0,
        "reduction_tree": 1.0,
        "reduction_parallel": 0.0,
        "reduction_atomic": 0.0,
    }
    default = dict(
        zip(fs.V3_LAUNCH, fs.launch_features(prob, cfg, fs.DEFAULT_LAUNCH, 224))
    )
    assert default == dict(
        zip(fs.V3_LAUNCH, fs.launch_features(prob, cfg, fs.LaunchPlan(False, 480), 224))
    )
    assert default["log2_grid"] == math.log2(480) and default["log2_tiles_per_wg"] == 0
    assert default["uneven_work"] == 0.0 and default["wg_wave_eff"] == 480 / (3 * 224)
    atomic = fs.launch_features(prob, cfg, fs.LaunchPlan(True, 7, "atomic"), 224)
    assert atomic[0] == 1.0 and atomic[-3:] == [0.0, 0.0, 1.0]
    for bad in (
        dict(partial_reduction="sum"),
        dict(grid_wgs=-1),
        dict(grid_wgs=1.5),
        dict(grid_wgs=True),
    ):
        with pytest.raises(ValueError):
            fs.LaunchPlan(**bad)


def test_catalog_reads_its_launch_source():
    consts = arch_constants("gfx950")
    prob = dict(BF16_NT, m=4096, n=1024, k=512, batch=1)
    ctx = fs.ExecutionContext(cu_budget=128, schedule="dynamic")
    calls = []

    def source(p, c, attrs, context, n_cu):
        calls.append((p["m"], c["occupancy"], attrs, context, n_cu))
        return fs.LaunchPlan(True, 7, "tree")

    cat = fs.Catalog("v3", (), source)
    assert cat == fs.Catalog("v3") and cat.names() == fs.Catalog("v3").names()
    got = cat.vectors(
        prob, BASE_CFG, {"unread": 1}, hardware=HW, constants=consts, context=ctx
    )
    assert calls == [(4096, 1, {"unread": 1}, ctx, 256)]
    assert got == fs.v3_feature_vectors(
        prob, BASE_CFG, hardware=HW, constants=consts, context=ctx,
        plan=fs.LaunchPlan(True, 7, "tree"),
    )  # fmt: skip
    assert fs.Catalog("v3").vectors(
        prob, BASE_CFG, hardware=HW, constants=consts, context=ctx
    ) == fs.v3_feature_vectors(
        prob, BASE_CFG, hardware=HW, constants=consts, context=ctx
    )
    assert fs.V2_CATALOG.vectors(
        prob, BASE_CFG, hardware=HW, constants=consts
    ) == fs.feature_vectors(prob, BASE_CFG, hardware=HW, constants=consts)


def test_view_key_is_the_signature_and_the_spec():
    cfg = dict(BASE_CFG)
    attrs = {"param_a": 3, "unread": 9}
    assert fs.v3_view_key(cfg, attrs, SPEC) == (128, 128, 64, 16, 16, 32, 0, 0)
    assert fs.v3_view_key(cfg, attrs, SAMPLE_SPEC) == (
        128, 128, 64, 16, 16, 32, 0, 0, 3, 1, 0, -1,
    )  # fmt: skip
    rng = random.Random(4)
    for _ in range(50):
        other, other_attrs = same_signature_variant(rng, cfg, attrs)
        for spec in (SPEC, SAMPLE_SPEC):
            assert fs.v3_view_key(other, other_attrs, spec) == fs.v3_view_key(
                cfg, attrs, spec
            )
    for field in fs.SIGNATURE_FIELDS:
        assert fs.v3_view_key(dict(cfg, **{field: 2}), attrs, SPEC) != (
            fs.v3_view_key(cfg, attrs, SPEC)
        )


def signature_pool(rng):
    """40 kernels of one signature that differ outside it, then 5 of
    another signature."""
    pool = []
    for _ in range(40):
        cfg, attrs = same_signature_variant(rng, BASE_CFG, {})
        attrs = random_launch_attributes(rng, random_attributes(rng))
        pool.append(dict(cfg, attributes=attrs))
    pool += [dict(BASE_CFG, mt_n=64, attributes={}) for _ in range(5)]
    return pool


@pytest.mark.parametrize("launch_keys", [False, True], ids=["view", "keyed"])
@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_view_key_whitelist_matches_engine(spec, launch_keys):
    """A keyed whitelist admits exactly the kernels whose view key Python
    computes; with the default spec that is every kernel of the signature,
    whatever its launch class. Kernels of one candidate score alike: of one
    launch class with launch keys, of one view key without."""
    tw = pytest.importorskip("tilewright")
    pool = signature_pool(random.Random(11))
    pick = 17
    key = fs.v3_view_key(pool[pick], pool[pick]["attributes"], spec)
    expect = [
        i for i, c in enumerate(pool) if fs.v3_view_key(c, c["attributes"], spec) == key
    ]
    assert (len(expect) == 40) if spec == SPEC else (len(expect) < 40)
    bundle = v3_bundle(labels=[LARGE], spec=spec, serving={"launch_keys": launch_keys})
    bundle["models"][LARGE]["view_key_hashes"] = [fs.view_key_hash(key)]
    model = tw.load_model_from_memory(write(bundle))
    cs = ev.candidate_set(tw, model, pool, HW)
    prob = dict(BF16_NT, m=4096, n=4096, k=4096, batch=1)
    ranked = cs.rank(ev.make_problem(tw, prob), ev.make_hardware(tw, HW), 1)
    scored = [r for r in ranked if r.scored]
    assert sorted(r.config_index for r in scored) == expect
    source = hl.HipblasltLaunch("gfx950") if launch_keys else None
    by_class = {}
    for r in scored:
        c = pool[r.config_index]
        by_class.setdefault(
            fs.v3_candidate_key(c, c["attributes"], spec, source), set()
        ).add(r.score)
    assert all(len(v) == 1 for v in by_class.values())
    assert (len(by_class) > 1) == (launch_keys and spec == SPEC)
    if spec == SPEC:
        assert (len({r.score for r in scored}) > 1) == launch_keys


def test_launch_models_through_the_binding():
    """The engine asks a launch model once per fully scored view key and
    launch key, for the key's first config. A plain callable has launch key
    0, a plan list groups configs with identical plans, an object may define
    launch_key, which only a model that serves launch keys reads; its errors
    reach the caller."""
    tw = pytest.importorskip("tilewright")
    model = tw.load_model_from_memory(
        write(v3_bundle(labels=[LARGE], serving={"launch_keys": True}))
    )
    unkeyed = tw.load_model_from_memory(write(v3_bundle(labels=[LARGE])))
    rng = random.Random(2)
    kernels = []
    for mt_n in (64, 128, 256):
        for _ in range(3):
            cfg, _ = same_signature_variant(rng, dict(BASE_CFG, mt_n=mt_n), {})
            kernels.append(cfg)
    configs = [ev.make_config(tw, k, i) for i, k in enumerate(kernels)]
    p = ev.make_problem(tw, dict(BF16_NT, m=4096, n=4096, k=4096, batch=1))
    h = ev.make_hardware(tw, HW)
    plans = [
        tw.LaunchPlan(i % 2 == 1, 100 + i, tw.PartialReduction(i % 4))
        for i in range(len(kernels))
    ]
    asked = []

    def launch(config, problem, context):
        asked.append(config.index)
        assert (problem.size.m, context.cu_budget) == (4096, 0)
        return plans[config.index]

    class Planner:
        def plan(self, config, problem, context):
            return plans[config.index]

    class FirstOfSignature(Planner):
        def plan(self, config, problem, context):
            return plans[config.index // 3 * 3]

    class PerConfig(Planner):
        def launch_key(self, config):
            return config.index

    ranked = tw.CandidateSet(model, configs, launch=launch).rank(p, h)
    assert sorted(asked) == [0, 3, 6]
    scores = {r.config_index: r.score for r in ranked}
    assert all(r.scored for r in ranked)
    for group in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
        assert len({scores[i] for i in group}) == 1
    assert tw.CandidateSet(model, configs, launch=Planner()).rank(p, h) == ranked
    first = [plans[i // 3 * 3] for i in range(len(plans))]
    for source in (FirstOfSignature(), first, tuple(first)):
        assert tw.CandidateSet(model, configs, launch=source).rank(p, h) == ranked
    alone = tw.CandidateSet(model, configs, launch=PerConfig()).rank(p, h)
    assert alone != ranked
    first_plans = tw.CandidateSet(unkeyed, configs, launch=launch).rank(p, h)
    for source in (PerConfig(), plans, tuple(plans)):
        assert (
            tw.CandidateSet(unkeyed, configs, launch=source).rank(p, h) == first_plans
        )
    for source in (plans, tuple(plans)):
        assert tw.CandidateSet(model, configs, launch=source).rank(p, h) == alone
    plain = tw.CandidateSet(model, configs).rank(p, h)
    defaults = [tw.LaunchPlan()] * len(kernels)
    assert tw.CandidateSet(model, configs, launch=defaults).rank(p, h) == plain
    assert plain != ranked
    # A budget below the model's range is ranked as its smallest budget: the
    # launch model is asked for that budget only.
    budgets = []

    def budgeted(config, problem, context):
        budgets.append(context.cu_budget)
        return plans[config.index]

    at = {
        b: tw.CandidateSet(model, configs, launch=budgeted).rank(
            p, h, context=ev.make_context(tw, fs.ExecutionContext(cu_budget=b))
        )
        for b in (16, 64)
    }
    assert at[16] == at[64]
    assert budgets and set(budgets) == {64}

    def fails(config, problem, context):
        raise ZeroDivisionError("launch")

    with pytest.raises(ZeroDivisionError, match="launch"):
        tw.CandidateSet(model, configs, launch=fails).rank(p, h)
    with pytest.raises(TypeError):
        tw.CandidateSet(model, configs, launch=lambda c, q, x: None).rank(p, h)
    with pytest.raises(IndexError):
        tw.CandidateSet(model, configs, launch=plans[:2]).rank(p, h)


class _Recording(ev.EngineLaunch):
    def __init__(self, *args):
        super().__init__(*args)
        self.asked = []
        self.keys_asked = 0

    def plan(self, config, problem, context):
        self.asked.append(int(config.index))
        return super().plan(config, problem, context)

    def launch_key(self, config):
        self.keys_asked += 1
        return super().launch_key(config)


def mixed_launch_pool(rng, n=160):
    """Kernels over a few signatures with the launch classes of a hipBLASLt
    pool: non-persistent, Stream-K and Hybrid, several occupancies and
    work-group shapes."""
    pool = []
    for _ in range(n):
        cfg = dict(
            BASE_CFG,
            mt_m=rng.choice([64, 128, 256]),
            mt_n=rng.choice([64, 128]),
            occupancy=rng.choice([1, 1, 2, 4]),
            grvw_a=rng.choice([1, 4, 8]),
        )
        strategy, assign = rng.choice([(0, 0), (2, 0), (2, 2), (2, 2), (1, 0)])
        attrs = {
            "tile_processing_strategy": strategy,
            "work_assignment": assign,
            "stream_k_atomic": rng.choice([0, 0, 1]),
            "work_group_x": rng.choice([128, 256]),
            "work_group_y": 1,
            "work_group_z": 1,
            "workgroup_mapping": rng.choice([1, 8, 16]),
        }
        pool.append(dict(cfg, attributes=attrs))
    return pool


@pytest.mark.parametrize("launch_keys", [False, True], ids=["view", "keyed"])
@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_engine_groups_configs_as_python_does(spec, launch_keys):
    """On random pools with mixed launch classes the engine plans exactly
    the first config of each `v3_candidate_key` group under the launch model
    the model's serving flag makes it group by (`ModelView.key_launch`, as
    the subset simulation and the identity ceiling group), configs of a
    group score alike, and the result equals scoring every config with its
    group's plan. Only a model that serves launch keys asks for them."""
    tw = pytest.importorskip("tilewright")
    from lib import identity_ceiling as ic
    from lib import shipped_subset as ss

    data = write(
        v3_bundle(
            labels=[LARGE],
            spec=spec,
            serving={"cascade_keys": 0, "launch_keys": launch_keys},
        )
    )
    model = tw.load_model_from_memory(data)
    source = hl.HipblasltLaunch("gfx950")
    grouping = ss.model_view(data, HW, tw).key_launch
    assert (grouping is not None) == launch_keys
    if launch_keys:
        assert grouping.id == source.id
    h = ev.make_hardware(tw, HW)
    rng = random.Random(len(spec) + 31)
    for trial in range(6):
        pool = mixed_launch_pool(rng)
        if spec:
            for k in pool:
                k["attributes"].update(random_attributes(rng, spec))
        groups = {}
        for i, k in enumerate(pool):
            groups.setdefault(
                fs.v3_candidate_key(k, k["attributes"], spec, grouping), []
            ).append(i)
        by_ceiling = {}
        for i, k in enumerate(pool):
            key = ic.engine_grouping(spec, grouping)(k)
            by_ceiling.setdefault(key, []).append(i)
        assert sorted(by_ceiling.values()) == sorted(groups.values())
        u = ss.build_universe(
            [dict(k, sol_idx_global=i) for i, k in enumerate(pool)], [], spec, grouping
        )
        assert sorted([int(i) for i in row[row >= 0]] for row in u.members) == sorted(
            groups.values()
        )
        firsts = sorted(g[0] for g in groups.values())
        n_views = len({fs.v3_view_key(k, k["attributes"], spec) for k in pool})
        if spec == SPEC:
            assert (n_views < len(groups)) == launch_keys
        if not launch_keys:
            assert n_views == len(groups)
        configs = [ev.make_config(tw, k, i) for i, k in enumerate(pool)]
        ctx = fs.ExecutionContext(cu_budget=rng.choice([0, 128, 224]))
        prob = dict(BF16_NT, m=rng.choice([512, 4096]), n=4096, k=4096, batch=1)
        p = ev.make_problem(tw, prob)
        rec = _Recording(tw, pool, source, HW.n_cu)
        ranked = tw.CandidateSet(model, configs, launch=rec).rank(
            p, h, context=ev.make_context(tw, ctx)
        )
        assert sorted(rec.asked) == firsts, trial
        assert (rec.keys_asked == len(pool)) if launch_keys else rec.keys_asked == 0
        score = {r.config_index: r.score for r in ranked}
        assert all(r.scored for r in ranked)
        for g in groups.values():
            assert len({score[i] for i in g}) == 1
        # Every config planned on its own with its group's plan ranks the
        # same.
        first_of = {i: g[0] for g in groups.values() for i in g}
        plans = [
            ev.make_plan(
                tw,
                source(
                    prob, pool[first_of[i]], pool[first_of[i]]["attributes"], ctx,
                    HW.n_cu,
                ),
            )
            for i in range(len(pool))
        ]  # fmt: skip
        alone = tw.CandidateSet(
            model,
            [ev.make_config(tw, k, i) for i, k in enumerate(pool)],
            launch=plans,
        ).rank(p, h, context=ev.make_context(tw, ctx))
        assert [(r.config_index, r.score) for r in alone] == [
            (r.config_index, r.score) for r in ranked
        ]
        again = tw.rank_configs(
            model, p, h, configs, context=ev.make_context(tw, ctx), launch=rec
        )
        assert again == ranked
        engine = ev.candidate_set(tw, model, pool, HW)
        assert engine.rank(p, h, context=ev.make_context(tw, ctx)) == ranked


def test_a_plan_list_groups_configs_with_identical_plans():
    """For a model that serves launch keys; any other plans a view key by
    its first config and never asks for a key."""
    tw = pytest.importorskip("tilewright")
    model = tw.load_model_from_memory(
        write(
            v3_bundle(labels=[LARGE], serving={"cascade_keys": 0, "launch_keys": True})
        )
    )
    unkeyed = tw.load_model_from_memory(
        write(v3_bundle(labels=[LARGE], serving={"cascade_keys": 0}))
    )
    configs = [ev.make_config(tw, BASE_CFG, i) for i in range(6)]
    plans = [tw.LaunchPlan(i % 3 == 0, 100 + i % 3) for i in range(6)]
    p = ev.make_problem(tw, dict(BF16_NT, m=4096, n=4096, k=4096, batch=1))
    h = ev.make_hardware(tw, HW)
    ranked = tw.CandidateSet(model, configs, launch=plans).rank(p, h)
    score = {r.config_index: r.score for r in ranked}
    assert score[0] == score[3] and score[1] == score[4] and score[2] == score[5]
    assert len(set(score.values())) == 3
    assert tw.rank_configs(model, p, h, configs, launch=plans) == ranked

    class Keyed:
        asked = []

        def plan(self, config, problem, context):
            self.asked.append(config.index)
            return plans[config.index]

        def launch_key(self, config):
            return config.index % 3 - 1

    assert tw.CandidateSet(model, configs, launch=Keyed()).rank(p, h) == ranked
    assert sorted(Keyed.asked) == [0, 1, 2]

    class BadKey(Keyed):
        def launch_key(self, config):
            return "x"

    with pytest.raises(TypeError):
        tw.CandidateSet(model, configs, launch=BadKey())
    with pytest.raises(TypeError):
        tw.rank_configs(model, p, h, configs, launch=BadKey())

    class FailingKey(Keyed):
        def launch_key(self, config):
            raise ZeroDivisionError("key")

    with pytest.raises(ZeroDivisionError):
        tw.CandidateSet(model, configs, launch=FailingKey())

    first = tw.CandidateSet(unkeyed, configs, launch=[plans[0]] * 6).rank(p, h)
    assert len({r.score for r in first}) == 1
    for source in (plans, Keyed(), BadKey(), FailingKey()):
        assert tw.CandidateSet(unkeyed, configs, launch=source).rank(p, h) == first
        assert tw.rank_configs(unkeyed, p, h, configs, launch=source) == first


def _gemm(cu_budget=None, schedule=None):
    g = {
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "batch_count": 1,
        "transA": "N",
        "transB": "T",
        "a_type": "bf16_r",
        "b_type": "bf16_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "compute_type": "f32_r",
        "winner_us": 10.0,
    }
    if cu_budget is not None:
        g["cu_budget"] = cu_budget
    if schedule is not None:
        g["schedule"] = schedule
    cands = []
    for sid, (mt, occ) in enumerate([(64, 1), (128, 2), (256, 1)]):
        cands.append(
            dict(
                sol_idx_global=sid,
                us=10.0 + sid,
                is_skip=0,
                rank=sid,
                is_origami_pick=1 if sid == 2 else 0,
                mt_m=mt, mt_n=mt, mt_k=64, mi_m=16, mi_n=16, mi_k=32,
                occupancy=occ, cache_hints_a=0, cache_hints_b=0,
                grvw_a=8, grvw_b=8, gwvw_d=4,
                attributes={"tile_processing_strategy": sid % 3, "work_assignment": int(sid % 3 != 1)},
            )  # fmt: skip
        )
    g["candidates"] = cands
    return g


def test_context_of_and_gemm_key():
    assert ev.context_of({}) == fs.EXCLUSIVE
    assert ev.context_of({"cu_budget": "", "schedule": ""}) == fs.EXCLUSIVE
    assert ev.context_of({"cu_budget": "64", "schedule": "Dynamic"}) == (
        fs.ExecutionContext(64, "dynamic")
    )
    assert ev.gemm_key(_gemm()) == ev.gemm_key(_gemm(0, "default"))
    assert ev.gemm_key(_gemm()) != ev.gemm_key(_gemm(64))
    with pytest.raises(ValueError):
        ev.context_of({"schedule": "fast"})


def test_ranking_through_engine_with_cu_budget():
    tw = pytest.importorskip("tilewright")
    consts = arch_constants("gfx950")
    gemms = [_gemm(), _gemm(64), _gemm(128, "dynamic"), _gemm(16), _gemm(64, "auto")]
    bundle = v3_bundle(labels=[LARGE])
    evaluation = ev.evaluate_bundle(
        bundle, gemms, arch="gfx950", constants=consts, hardware=HW,
        weight_dtype="fp32", tw=tw,
    )  # fmt: skip
    served = [r.served_by for r in evaluation.results]
    # 16 of 256 CUs is below the model's 0.25 fraction: ranked as 64 of 256;
    # auto is not declared.
    assert served == ["model", "model", "model", "model", "origami_fallback"]
    assert all(r.n_scored == 3 for r in evaluation.results[:4])

    model = tw.load_model_from_memory(write(bundle))
    kernels = [ev._kernel(c) for c in gemms[0]["candidates"]]
    with pytest.raises(ValueError):
        ev.candidate_set(tw, model, kernels)
    cs = ev.candidate_set(tw, model, kernels, HW)
    prob = fs.problem_kwargs_from_row(gemms[0])
    p = ev.make_problem(tw, prob)
    h = ev.make_hardware(tw, HW)
    full = cs.rank(p, h, 3)
    budget_ctx = fs.ExecutionContext(64)
    budget = cs.rank(p, h, 3, ev.make_context(tw, budget_ctx))
    assert all(r.scored for r in full) and all(r.scored for r in budget)
    assert [r.score for r in full] != [r.score for r in budget]

    # The evaluation's launch model gives the engine hipBLASLt's plans.
    configs = [ev.make_config(tw, k, i) for i, k in enumerate(kernels)]
    for ctx, ranked in ((fs.EXCLUSIVE, full), (budget_ctx, budget)):
        plans = [
            ev.make_plan(
                tw, hl.launch_plan(prob, k, k["attributes"], ctx, HW.n_cu, "gfx950")
            )
            for k in kernels
        ]
        table = tw.CandidateSet(model, configs, launch=plans)
        again = table.rank(p, h, 3, ev.make_context(tw, ctx))
        assert [(r.config_index, r.score) for r in again] == [
            (r.config_index, r.score) for r in ranked
        ]
    plain = tw.CandidateSet(model, configs).rank(p, h, 3)
    assert [r.score for r in plain] != [r.score for r in full]

    v2 = ev.evaluate_bundle(
        mlrec.bundle_from_model(mlrec.read_model(_v2_model())), gemms[:2],
        arch="gfx950", constants=consts, hardware=HW, weight_dtype="fp32", tw=tw,
    )  # fmt: skip
    # A v2 model covers every CU only: a budget is ranked as every CU.
    assert [r.served_by for r in v2.results] == ["model", "model"]


def _v2_model():
    from test_mlrec import random_bundle

    bundle = random_bundle([LARGE], dims=(4, 6, 3), signatures={LARGE: []})
    return mlrec.write_model(bundle, None, "gfx950", arch_constants("gfx950"), "fp32")


PROVENANCE = {
    "date": "2000-01-01T23:11:00Z",
    "config_id": "gfx950_bbs_tn_int4",
    "training_data_hash": "sha256:" + "0f" * 8,
    "source_revision": "0123abc-dirty",
    "launch_model": "hipblaslt:gfx950:1",
}


def test_provenance_round_trip():
    plain = write(v3_bundle())
    data = write(v3_bundle(provenance=PROVENANCE))
    parsed = mlrec.read_model(data)
    assert list(parsed["provenance"]) == list(mlrec.PROVENANCE_KEYS)
    assert parsed["provenance"] == PROVENANCE
    assert mlrec.read_header(data)["provenance"] == PROVENANCE
    assert mlrec.read_model(plain)["provenance"] == {}
    assert write(mlrec.bundle_from_model(parsed)) == data
    assert write(mlrec.bundle_from_model(mlrec.read_model(plain))) == plain
    arch, consts = "gfx950", arch_constants("gfx950")
    given = mlrec.write_model(v3_bundle(), None, arch, consts, "fp32", PROVENANCE)
    assert given == data
    partial = {"config_id": "c", "date": None, "source_revision": ""}
    assert mlrec.read_model(write(v3_bundle(provenance=partial)))["provenance"] == {
        "config_id": "c"
    }
    for bad in (
        {"trainer": "x"},
        {"date": "yesterday"},
        {"config_id": "a b"},
        {"source_revision": "r\u00e9v"},
    ):
        with pytest.raises(ValueError):
            write(v3_bundle(provenance=bad))
    v2 = mlrec.bundle_from_model(mlrec.read_model(_v2_model()))
    assert mlrec.write_model(v2, None, arch, consts, "fp32", PROVENANCE) == (
        mlrec.write_model(v2, None, arch, consts, "fp32")
    )

    tw = pytest.importorskip("tilewright")
    info = tw.describe(tw.load_model_from_memory(data))
    recorded = info.provenance
    assert (
        recorded.config_id,
        recorded.training_data_hash,
        recorded.source_revision,
        recorded.date,
        recorded.launch_model,
    ) == tuple(PROVENANCE[k] for k in mlrec.PROVENANCE_KEYS)
    assert tw.describe(tw.load_model_from_memory(plain)).provenance.date == ""


def test_v3_model_ranks_only_with_its_launch_model():
    tw = pytest.importorskip("tilewright")
    want = hl.HipblasltLaunch("gfx950").id
    bundle = v3_bundle(labels=[LARGE])
    named = tw.load_model_from_memory(write(bundle, provenance={"launch_model": want}))
    plain = tw.load_model_from_memory(write(bundle))
    gemm = _gemm()
    kernels = [ev._kernel(c) for c in gemm["candidates"]]
    configs = [ev.make_config(tw, k, i) for i, k in enumerate(kernels)]
    p = ev.make_problem(tw, fs.problem_kwargs_from_row(gemm))
    h = ev.make_hardware(tw, HW)

    # lib.evaluate passes hipBLASLt's launch model, with its id.
    got = ev.candidate_set(tw, named, kernels, HW).rank(p, h, 3)
    same = ev.candidate_set(tw, plain, kernels, HW).rank(p, h, 3)
    assert [(r.config_index, r.score) for r in got] == [
        (r.config_index, r.score) for r in same
    ]
    assert len(got) == len(kernels) and all(r.scored for r in got)

    def plan(config, problem, context):
        return tw.LaunchPlan()

    class Planner:
        def __init__(self, id):
            self.id = id

        def plan(self, config, problem, context):
            return tw.LaunchPlan()

    tw.CandidateSet(named, configs, launch=Planner(want))
    plan.id = want
    tw.CandidateSet(named, configs, launch=plan)
    del plan.id
    other = f"hipblaslt:gfx942:{hl.RULE_VERSION}"
    for launch, shown in (
        (None, "has none"),
        (plan, "launch model is ''"),
        (Planner(None), "launch model is ''"),
        (Planner(other), f"launch model is '{other}'"),
        ([tw.LaunchPlan()] * len(configs), "launch model is ''"),
    ):
        with pytest.raises(ValueError) as error:
            tw.CandidateSet(named, configs, launch=launch)
        assert f"launch model '{want}'" in str(error.value)
        assert shown in str(error.value)
        with pytest.raises(ValueError) as ad_hoc:
            tw.rank_configs(named, p, h, configs, launch=launch)
        assert str(ad_hoc.value) == str(error.value)
        tw.CandidateSet(plain, configs, launch=launch)
        tw.rank_configs(plain, p, h, configs, launch=launch)
    with pytest.raises(TypeError, match="id must be a str"):
        tw.CandidateSet(named, configs, launch=Planner(1))
    with pytest.raises(TypeError, match="id must be a str"):
        tw.rank_configs(named, p, h, configs, launch=Planner(1))

    # An id and a launch key together: the key still splits the pool.
    class KeyedPlanner(Planner):
        def plan(self, config, problem, context):
            return tw.LaunchPlan(grid_wgs=7 * (config.index % 2) + 1)

        def launch_key(self, config):
            return config.index % 2

    keyed = KeyedPlanner(want)
    ranked = tw.CandidateSet(named, configs, launch=keyed).rank(p, h)
    assert ranked == tw.CandidateSet(plain, configs, launch=KeyedPlanner("")).rank(p, h)
    assert ranked == tw.rank_configs(named, p, h, configs, launch=keyed)
    engine = ev.engine_launch(tw, kernels, hl.HipblasltLaunch("gfx950"), HW.n_cu)
    assert engine.id == want
    assert engine.launch_key(configs[0]) == hl.launch_key(
        kernels[0], kernels[0].get("attributes")
    )

    # Bundle provenance reaches the model file the evaluation ranks.
    consts = arch_constants("gfx950")
    kw = dict(arch="gfx950", constants=consts, hardware=HW, weight_dtype="fp32", tw=tw)
    for recorded in (want, None):
        ranked = ev.evaluate_bundle(
            dict(bundle, provenance={"launch_model": recorded}), [gemm], **kw
        )
        assert [r.served_by for r in ranked.results] == ["model"]
    stale = f"hipblaslt:gfx950:{hl.RULE_VERSION + 1}"
    with pytest.raises(ValueError, match=stale):
        ev.evaluate_bundle(
            dict(bundle, provenance={"launch_model": stale}), [gemm], **kw
        )


def test_a_model_of_launch_rule_version_1_ranks_with_its_plans(monkeypatch):
    """A model trained on launch rule version 1 ranks with version 1's plans,
    as TensileLite serves it, beside models of the version gfx950 trains on
    now; the engine refuses a version the arch does not serve."""
    tw = pytest.importorskip("tilewright")
    assert hl.rule_version("gfx950") == 2
    bundle = v3_bundle(labels=[LARGE])
    gemm = _gemm()
    kernels = [ev._kernel(c) for c in gemm["candidates"]]
    p = ev.make_problem(tw, fs.problem_kwargs_from_row(gemm))
    h = ev.make_hardware(tw, HW)
    real = ev.engine_launch
    seen = []

    def engine_launch(tw_, pool, launch, n_cu):
        seen.append(launch)
        return real(tw_, pool, launch, n_cu)

    monkeypatch.setattr(ev, "engine_launch", engine_launch)
    ranked = {}
    for recorded in ("hipblaslt:gfx950:1", "hipblaslt:gfx950:2"):
        model = tw.load_model_from_memory(
            write(bundle, provenance={"launch_model": recorded})
        )
        ranked[recorded] = ev.candidate_set(tw, model, kernels, HW).rank(p, h, 3)
        assert seen[-1].id == recorded
        assert seen[-1].version == int(recorded[-1])
    # Two K iterations per tile and more: both versions plan alike.
    assert ranked["hipblaslt:gfx950:1"] == ranked["hipblaslt:gfx950:2"]
    consts = arch_constants("gfx950")
    kw = dict(arch="gfx950", constants=consts, hardware=HW, weight_dtype="fp32", tw=tw)
    first = dict(bundle, provenance={"launch_model": "hipblaslt:gfx950:1"})
    assert [r.served_by for r in ev.evaluate_bundle(first, [gemm], **kw).results] == [
        "model"
    ]
    assert seen[-1].version == 1
    # The engine itself still takes only the launch model a model records.
    old = tw.load_model_from_memory(
        write(bundle, provenance={"launch_model": "hipblaslt:gfx950:1"})
    )
    with pytest.raises(ValueError, match="launch model is 'hipblaslt:gfx950:2'"):
        tw.CandidateSet(
            old,
            [ev.make_config(tw, k, i) for i, k in enumerate(kernels)],
            launch=real(tw, kernels, hl.HipblasltLaunch("gfx950"), HW.n_cu),
        )


# The engine pins the same values (tests/test_v3.cpp).
ENGINE_SPEC = (
    fs.AttributeFeature("param_a", 0, "log2p1"),
    fs.AttributeFeature("param_b", 8, "identity"),
    fs.AttributeFeature("param_c", 0, "nonzero"),
    fs.AttributeFeature("param_d", -1, "onehot", (-1, 1, 2, 4, 8)),
)
HASH_EMPTY = 0x640B5D2E92A9F47F
HASH_ENGINE_SPEC = 0x40D53A0BF908F951


def test_feature_spec_hash_is_frozen_and_detects_feature_changes(monkeypatch):
    assert mlrec.feature_spec_hash(()) == HASH_EMPTY
    assert mlrec.feature_spec_hash(SPEC) == HASH_EMPTY
    assert mlrec.feature_spec_hash(ENGINE_SPEC) == HASH_ENGINE_SPEC
    assert mlrec.feature_spec_hash(ENGINE_SPEC[1:]) != HASH_ENGINE_SPEC
    assert mlrec.feature_spec_hash(SPEC) != fs.attribute_spec_hash(SPEC)
    good = write(v3_bundle(labels=[LARGE]))
    real = fs.v3_feature_vectors

    def drifted(*args, **kwargs):
        q, i, x = real(*args, **kwargs)
        return q, i, x[:-1] + [x[-1] + 0.5]

    monkeypatch.setattr(fs, "v3_feature_vectors", drifted)
    assert mlrec.feature_spec_hash(SPEC) != HASH_EMPTY
    drifted_model = write(v3_bundle(labels=[LARGE]))
    with pytest.raises(mlrec.FormatError, match="feature_spec_hash"):
        mlrec.read_model(good)
    monkeypatch.setattr(fs, "v3_feature_vectors", real)
    with pytest.raises(mlrec.FormatError, match="feature_spec_hash"):
        mlrec.read_model(drifted_model)
    tw = pytest.importorskip("tilewright")
    tw.load_model_from_memory(good)
    with pytest.raises(ValueError, match="feature_spec_hash"):
        tw.load_model_from_memory(drifted_model)


def test_feature_spec_hash_covers_launch_and_context_features(monkeypatch):
    real_launch = fs.launch_features
    plans = [case[4] for case in mlrec._GOLDEN_CASES]
    contexts = [fs.ExecutionContext(*case[3]) for case in mlrec._GOLDEN_CASES]
    assert fs.DEFAULT_LAUNCH in plans
    assert any(p.dynamic_queue and p.partial_reduction != "none" for p in plans)
    assert any(c != fs.EXCLUSIVE for c in contexts)

    # Each feature, changed for one kind of launch only, changes the hash.
    for index in range(len(fs.V3_LAUNCH)):
        for picked in (
            lambda p: p == fs.DEFAULT_LAUNCH,
            lambda p: p.dynamic_queue and p.partial_reduction != "none",
        ):

            def drifted(problem, config, plan, n_eff, index=index, picked=picked):
                x = real_launch(problem, config, plan, n_eff)
                if picked(plan):
                    x[index] += 0.25
                return x

            monkeypatch.setattr(fs, "launch_features", drifted)
            assert mlrec.feature_spec_hash(SPEC) != HASH_EMPTY, fs.V3_LAUNCH[index]
    monkeypatch.setattr(fs, "launch_features", real_launch)

    # So do the golden cases' plans and contexts.
    cases = mlrec._GOLDEN_CASES
    for edited in (
        tuple(case[:4] + (fs.DEFAULT_LAUNCH,) for case in cases),
        tuple(case[:3] + ((0, "default"),) + case[4:] for case in cases),
    ):
        monkeypatch.setattr(mlrec, "_GOLDEN_CASES", edited)
        assert mlrec.feature_spec_hash(SPEC) != HASH_EMPTY


def test_spec_hash_text():
    spec = (
        fs.AttributeFeature("a", 1, "log2p1"),
        fs.AttributeFeature("b", -2, "onehot", (3, -4)),
    )
    text = fs.v3_catalog_hash() + "\na\t1\tlog2p1\nb\t-2\tonehot\t3,-4\n"
    h = 0xCBF29CE484222325
    for ch in text.encode():
        h = ((h ^ ch) * 0x100000001B3) % 2**64
    assert fs.attribute_spec_hash(spec) == h
    assert fs.view_key_hash([1, -1]) == fs._fnv1a(
        struct.pack("<Q", 1) + struct.pack("<Q", 2**64 - 1)
    )


def test_v3_fallback_routing_matches_engine():
    tw = pytest.importorskip("tilewright")
    from lib import grid
    from lib import subcells as sc

    rng = random.Random(21)
    base = grid.all_cell_labels()
    for trial in range(12):
        picked = rng.sample(base, rng.randint(1, 6))
        labels = []
        for b in picked:
            r = rng.random()
            if r < 0.3:
                labels += [b + "#M<=300", b + "#M>300"]
            elif r < 0.45:
                labels += [b + "#N<=900"]
            elif r < 0.6:
                labels += [b, b + "#K>200"]
            else:
                labels.append(b)
        model = tw.load_model_from_memory(write(v3_bundle(labels=labels, seed=trial)))
        tree = sc.split_tree_from_labels(labels)
        for _ in range(300):
            m, n, k = (rng.choice([1, 7, 64, 200, 301, 900, 1500, 9000]) for _ in "mnk")
            batch = rng.choice([1, 1, 3])
            prob = dict(
                m=m, n=n, k=k, batch=batch, a_transpose="N", b_transpose="N",
                a_dtype="half", b_dtype="half", c_dtype="half", d_dtype="half",
                mi_dtype="half",
            )  # fmt: skip
            engine = ev.model_cell_label(tw, model, ev.make_problem(tw, prob))
            assert engine == sc.v3_serving_cell(m, n, k, batch, tree, set(labels)), (
                labels,
                prob,
            )


def test_with_soft_rules_rewrites_only_the_serving_flags():
    bundle = v3_bundle(
        labels=[LARGE],
        serving={"non_temporal_rule": True, "launch_keys": True, "cascade_keys": 8},
    )
    data = write(bundle)
    out = mlrec.with_soft_rules(data, {"non_temporal_rule": False, "dot2_rule": True})
    assert len(out) == len(data)
    assert sum(a != b for a, b in zip(out, data)) == 1
    serving = mlrec.read_model(out)["serving"]
    assert serving["non_temporal_rule"] is False and serving["dot2_rule"] is True
    assert serving["launch_keys"] and serving["cascade_keys"] == 8
    assert not serving["small_batched_rule"]
    assert mlrec.with_soft_rules(out, {"non_temporal_rule": True}) != data
    back = mlrec.with_soft_rules(out, {"non_temporal_rule": True, "dot2_rule": False})
    assert back == data
    with pytest.raises(ValueError, match="leading-free"):
        mlrec.with_soft_rules(data, {"leading_free_rule": True})
    with pytest.raises(ValueError, match="unknown"):
        mlrec.with_soft_rules(data, {"tiny_rule": True})
    with pytest.raises(mlrec.FormatError, match="MLREC_v3"):
        mlrec.with_soft_rules(b"MLREC_v2" + bytes(64), {})
    tw = pytest.importorskip("tilewright")
    info = tw.describe(ev.load_model_bytes(out, tw))
    assert int(info.format_version) == 3


# ── family-wise rule, bank and trained-key gate ─────────────────────────────


def test_serving_extensions_round_trip():
    keys = [9, 3, 2**64 - 1, 3]
    bundle = v3_bundle(
        labels=[LARGE],
        serving={"non_temporal_rule": True, "non_temporal_family_rule": True,
                 "bank": True, "trained_key_gate": True, "cascade_keys": 8},
        trained_view_keys=keys,
    )  # fmt: skip
    data = write(bundle, provenance=PROVENANCE)
    parsed = mlrec.read_model(data)
    serving = parsed["serving"]
    assert serving["non_temporal_family_rule"] and serving["bank"]
    assert serving["trained_key_gate"] and serving["cascade_keys"] == 8
    assert parsed["trained_view_keys"] == [3, 9, 2**64 - 1]
    assert parsed["provenance"] == mlrec.check_provenance(PROVENANCE)
    header = mlrec.read_header(data)
    assert header["trained_view_keys"] == [3, 9, 2**64 - 1]
    assert write(mlrec.bundle_from_model(parsed), provenance=PROVENANCE) == data
    flags = struct.unpack_from(
        "<I", data, mlrec._v3_header(data, len(data))["serving_offset"]
    )
    assert flags[0] == 0x04 | 0x20 | 0x40 | 0x80
    tw = pytest.importorskip("tilewright")
    info = tw.describe(ev.load_model_bytes(data, tw))
    assert info.non_temporal_family_rule and info.bank and info.trained_key_gate
    assert info.trained_keys == 3 and info.provenance.date == PROVENANCE["date"]


def test_serving_extensions_are_checked():
    with pytest.raises(ValueError, match="non-temporal rule"):
        write(v3_bundle(serving={"non_temporal_family_rule": True}))
    with pytest.raises(ValueError, match="trained key"):
        write(v3_bundle(serving={"trained_key_gate": True}))
    with pytest.raises(ValueError, match="trained key"):
        write(v3_bundle(serving={"trained_key_gate": True}, trained_view_keys=[]))
    data = write(
        v3_bundle(serving={"trained_key_gate": True}, trained_view_keys=[5, 7])
    )
    at = mlrec._v3_header(data, len(data))["header_end"] - 16
    assert struct.unpack_from("<QQ", data, at) == (5, 7)
    swapped = data[:at] + struct.pack("<QQ", 7, 5) + data[at + 16 :]
    with pytest.raises(mlrec.FormatError, match="ascending"):
        mlrec.read_model(swapped)
    at = mlrec._v3_header(data, len(data))["serving_offset"]
    family = bytearray(data)
    struct.pack_into("<I", family, at, 0x20 | 0x80)
    with pytest.raises(mlrec.FormatError, match="non-temporal rule"):
        mlrec.read_model(bytes(family))
    tw = pytest.importorskip("tilewright")
    for bad in (swapped, bytes(family)):
        with pytest.raises(ValueError):
            tw.load_model_from_memory(bad)


def test_serving_flags_can_be_rewritten_without_retraining():
    data = write(v3_bundle(labels=[LARGE], serving={"non_temporal_rule": True}),
                 provenance=PROVENANCE)  # fmt: skip
    family = mlrec.with_soft_rules(data, {"non_temporal_family_rule": True})
    assert len(family) == len(data)
    assert mlrec.read_header(family)["serving"]["non_temporal_family_rule"]
    with pytest.raises(ValueError, match="non-temporal rule"):
        mlrec.with_soft_rules(family, {"non_temporal_rule": False})
    bank = mlrec.with_bank(data, True)
    assert mlrec.read_header(bank)["serving"]["bank"]
    assert mlrec.with_bank(bank, False) == data
    gated = mlrec.with_trained_keys(data, [7, 3, 7])
    parsed = mlrec.read_model(gated)
    assert (
        parsed["trained_view_keys"] == [3, 7] and parsed["serving"]["trained_key_gate"]
    )
    assert parsed["provenance"] == mlrec.read_model(data)["provenance"]
    assert parsed["payload_crc32"] == mlrec.read_model(data)["payload_crc32"]
    assert mlrec.read_model(mlrec.with_trained_keys(gated, [1]))[
        "trained_view_keys"
    ] == [1]
    assert mlrec.with_trained_keys(gated, None) == data
    with pytest.raises(ValueError, match="trained key"):
        mlrec.with_trained_keys(data, [])
    tw = pytest.importorskip("tilewright")
    assert tw.describe(ev.load_model_bytes(gated, tw)).trained_keys == 2


FLOORED = {
    "min_cu_fraction": 0.25,
    "max_cu_fraction": 1.0,
    "schedules": ["default", "dynamic"],
    "min_effective_cus": 64,
}


def test_cu_floor_round_trip_and_rewrite():
    bundle = v3_bundle(
        labels=[LARGE], contexts=FLOORED,
        serving={"trained_key_gate": True, "cascade_keys": 8},
        trained_view_keys=[9, 5],
    )  # fmt: skip
    data = write(bundle, provenance=PROVENANCE)
    parsed = mlrec.read_model(data)
    assert parsed["contexts"] == FLOORED
    assert mlrec.read_header(data)["contexts"] == FLOORED
    assert parsed["trained_view_keys"] == [5, 9]
    assert parsed["provenance"] == mlrec.check_provenance(PROVENANCE)
    assert write(mlrec.bundle_from_model(parsed), provenance=PROVENANCE) == data
    at = mlrec._v3_header(data, len(data))["serving_offset"]
    assert struct.unpack_from("<I", data, at)[0] == 0x80 | 0x100
    # The floor follows the trained keys and precedes the provenance.
    end = mlrec._v3_header(data, len(data))["header_end"]
    tail = mlrec._provenance_bytes(mlrec.check_provenance(PROVENANCE))
    assert data[end - len(tail) - 4 - 16 : end - len(tail)] == struct.pack(
        "<QQI", 5, 9, 64
    )

    # Without the floor the file is byte for byte what a writer without it
    # wrote, and the floor can be added or removed without retraining.
    plain = {k: v for k, v in FLOORED.items() if k != "min_effective_cus"}
    old = write(dict(bundle, contexts=plain), provenance=PROVENANCE)
    assert mlrec.read_model(old)["contexts"] == plain
    assert struct.unpack_from("<I", old, at)[0] == 0x80
    assert len(old) == len(data) - 4
    assert mlrec.with_cu_floor(old, 64) == data
    assert mlrec.with_cu_floor(data, None) == old
    assert mlrec.with_cu_floor(mlrec.with_cu_floor(data, 32), 64) == data
    assert (
        mlrec.read_model(mlrec.with_cu_floor(data, 1))["contexts"]["min_effective_cus"]
        == 1
    )
    ungated = mlrec.with_trained_keys(data, None)
    assert mlrec.read_model(ungated)["contexts"] == FLOORED
    assert mlrec.with_cu_floor(ungated, None) == mlrec.with_trained_keys(old, None)
    assert (
        mlrec.read_model(data)["payload_crc32"]
        == mlrec.read_model(old)["payload_crc32"]
    )
    with pytest.raises(mlrec.FormatError, match="MLREC_v3"):
        mlrec.with_cu_floor(b"MLREC_v2" + data[8:], 64)


@pytest.mark.parametrize("bad", [0, -1, 2**32, True, 1.5, "64", None])
def test_cu_floor_values_are_checked(bad):
    if bad is None:
        assert (
            "min_effective_cus" not in mlrec.read_model(write(v3_bundle()))["contexts"]
        )
        return
    with pytest.raises(ValueError, match="CU floor"):
        write(v3_bundle(contexts=dict(FLOORED, min_effective_cus=bad)))
    with pytest.raises(ValueError, match="CU floor"):
        mlrec.with_cu_floor(write(v3_bundle()), bad)


def test_cu_floor_defects_are_rejected_by_reader_and_engine():
    data = write(v3_bundle(labels=[LARGE], contexts=FLOORED))
    end = mlrec._v3_header(data, len(data))["header_end"]
    assert struct.unpack_from("<I", data, end - 4) == (64,)
    zero = data[: end - 4] + struct.pack("<I", 0) + data[end:]
    with pytest.raises(mlrec.FormatError, match="at least 1"):
        mlrec.read_model(zero)
    at = mlrec._v3_header(data, len(data))["serving_offset"]
    unflagged = bytearray(data)
    struct.pack_into("<I", unflagged, at, 0)
    with pytest.raises(mlrec.FormatError):
        mlrec.read_model(bytes(unflagged))
    unknown = bytearray(data)
    struct.pack_into("<I", unknown, at, 0x1100)
    with pytest.raises(mlrec.FormatError, match="unknown serving flags"):
        mlrec.read_model(bytes(unknown))
    tw = pytest.importorskip("tilewright")
    for bad, why in ((zero, "at least 1"), (bytes(unknown), "unknown serving flags")):
        with pytest.raises(ValueError, match=why):
            tw.load_model_from_memory(bad)
    with pytest.raises(ValueError):
        tw.load_model_from_memory(bytes(unflagged))


def test_cu_floor_through_the_binding():
    """The engine ranks a context with fewer usable CUs than the floor on
    any device, a compute partition included, as the floor (as the whole
    device when it has fewer CUs than the floor), and the others as the
    model without the floor does; a schedule it does not list stays
    unscored."""
    tw = pytest.importorskip("tilewright")
    floored = write(v3_bundle(labels=[LARGE], contexts=FLOORED))
    model = ev.load_model_bytes(floored, tw)
    plain = ev.load_model_bytes(mlrec.with_cu_floor(floored, None), tw)
    assert tw.describe(model).min_effective_cus == 64
    assert tw.describe(plain).min_effective_cus == 0
    pool = [dict(BASE_CFG, sol_idx_global=0), dict(BASE_CFG, mt_m=64, sol_idx_global=1)]
    prob = dict(BF16_NT, m=4096, n=4096, k=4096, batch=1)
    p = ev.make_problem(tw, prob)
    for n_cu in (256, 128, 64, 48, 32):
        device = dataclasses.replace(HW, n_cu=n_cu)
        hw = ev.make_hardware(tw, device)
        cs = ev.candidate_set(tw, model, pool, device)
        free = ev.candidate_set(tw, plain, pool, device)
        for budget in (0, 16, 32, 48, 63, 64, 96, 128, 200, 300):
            for schedule in ("default", "dynamic", "auto"):
                ctx = fs.ExecutionContext(budget, schedule)
                tctx = ev.make_context(tw, ctx)
                usable = ctx.effective_cus(n_cu)
                want = tw.supports(plain, tctx, hw) and usable >= 64
                assert tw.supports(model, tctx, hw) == want, (n_cu, budget, schedule)
                ranked = cs.rank(p, hw, 2, context=tctx)
                at = tw.scoring_context(model, tctx, hw)
                if want:
                    assert ranked == free.rank(p, hw, 2, context=tctx)
                    assert any(r.scored for r in ranked)
                elif schedule == "auto":
                    assert at is None
                    assert not any(r.scored for r in ranked)
                else:
                    assert at is not None
                    assert at.cu_budget == (64 if n_cu > 64 else 0), (n_cu, budget)
                    assert ranked == cs.rank(p, hw, 2, context=at)
                    assert ranked == free.rank(p, hw, 2, context=at)
                    assert any(r.scored for r in ranked)
    # The model covers a partition of 64 CUs whole, and not one of 32 (which
    # it ranks whole, the nearest it can).
    assert tw.supports(model, ev.make_context(tw, fs.EXCLUSIVE),
                       ev.make_hardware(tw, dataclasses.replace(HW, n_cu=64)))  # fmt: skip
    assert not tw.supports(model, ev.make_context(tw, fs.EXCLUSIVE),
                           ev.make_hardware(tw, dataclasses.replace(HW, n_cu=32)))  # fmt: skip
    assert tw.supports(plain, ev.make_context(tw, fs.EXCLUSIVE),
                       ev.make_hardware(tw, dataclasses.replace(HW, n_cu=32)))  # fmt: skip


def _gate_pool():
    """Kernels of three view keys: two of one signature that differ in an
    attribute the spec reads, and one of two configs. The first signature
    fits the LDS in bf16, not in fp32."""
    out = []
    for i, (mt, a) in enumerate([((256, 256, 128), 0), ((256, 256, 128), 4),
                                 ((64, 64, 64), 0), ((64, 64, 64), 0)]):  # fmt: skip
        k = dict(BASE_CFG, mt_m=mt[0], mt_n=mt[1], mt_k=mt[2], sol_idx_global=i)
        k["attributes"] = {"param_a": a}
        out.append(k)
    return out


def test_trained_key_gate_through_the_binding_and_evaluate():
    tw = pytest.importorskip("tilewright")
    pool = _gate_pool()
    trained = [
        fs.view_key_hash(fs.v3_view_key(pool[0], pool[0]["attributes"], SAMPLE_SPEC))
    ]
    bundle = v3_bundle(
        labels=[LARGE], spec=SAMPLE_SPEC,
        serving={"trained_key_gate": True, "whitelist_tier": False, "cascade_keys": 0},
        trained_view_keys=trained,
    )  # fmt: skip
    data = write(bundle)
    model = ev.load_model_bytes(data, tw)
    cs = ev.candidate_set(tw, model, pool, HW)
    assert (cs.untrained_keys, cs.untrained_configs) == (2, 3)
    assert ev.trained_mask(pool, mlrec.read_header(data)) == [True, False, False, False]
    h = ev.make_hardware(tw, HW)
    bf16 = fs.problem_kwargs_from_row(_gemm())
    ranked = cs.rank(ev.make_problem(tw, bf16), h, len(pool))
    assert [r.config_index for r in ranked if r.scored] == [0]
    # fp32: no trained kernel fits the LDS, so every feasible kernel is scored.
    fp32 = dict(bf16, a_dtype="float", b_dtype="float")
    ranked = cs.rank(ev.make_problem(tw, fp32), h, len(pool))
    assert sorted(r.config_index for r in ranked if r.scored) == [2, 3]

    gemm = _gemm()
    gemm["candidates"] = [dict(k, us=10.0 + i, is_skip=0, rank=i, is_origami_pick=0)
                          for i, k in enumerate(pool)]  # fmt: skip
    out = ev.evaluate_model(
        data, [gemm], hardware=HW, pools={ev.pool_key(gemm): pool}, tw=tw
    )
    assert out.results[0].pick_index == 0 and out.results[0].pick_untrained is False
    assert out.n_untrained_configs == {"|".join(ev.pool_key(gemm)): 3}
    assert ev.summarize(out.results)["n_untrained_picks"] == 0
    plain = write(dict(bundle, serving={"whitelist_tier": False, "cascade_keys": 0}))
    ungated = ev.evaluate_model(plain, [gemm], hardware=HW, pools={ev.pool_key(gemm): pool},
                                tw=tw)  # fmt: skip
    assert (
        ungated.results[0].pick_untrained is None and ungated.n_untrained_configs == {}
    )


DRIFT_KEYS = [11, 22, 33]


def _drift_bundle(**serving):
    """A v3 bundle whose cells list view keys, gated, floored and with
    provenance: every optional header field beside the drift fallback."""
    bundle = v3_bundle(
        labels=[LARGE, "Mid|Mid|MidK|Bnone"], contexts=FLOORED,
        serving={"trained_key_gate": True, "cascade_keys": 8, **serving},
        trained_view_keys=[22, 11, 33, 44],
    )  # fmt: skip
    for entry in bundle["models"].values():
        entry["view_key_hashes"] = list(DRIFT_KEYS)
    return bundle


def test_drift_fallback_round_trip_and_rewrite():
    data = write(_drift_bundle(drift_keep_ppm=900000), provenance=PROVENANCE)
    parsed = mlrec.read_model(data)
    assert parsed["serving"]["drift_keep_ppm"] == 900000
    assert mlrec.read_header(data)["serving"]["drift_keep_ppm"] == 900000
    assert write(mlrec.bundle_from_model(parsed), provenance=PROVENANCE) == data
    h = mlrec._v3_header(data, len(data))
    assert (
        struct.unpack_from("<I", data, h["serving_offset"])[0] == 0x80 | 0x100 | 0x400
    )
    # The field follows the trained keys and the CU floor and precedes the
    # provenance.
    tail = mlrec._provenance_bytes(mlrec.check_provenance(PROVENANCE))
    end = h["header_end"]
    assert data[end - len(tail) - 4 - 4 - 32 : end - len(tail)] == struct.pack(
        "<QQQQII", 11, 22, 33, 44, 64, 900000
    )
    # Without it the file is byte for byte what a writer without it wrote, and
    # the fallback can be added, changed or removed without retraining.
    old = write(_drift_bundle(), provenance=PROVENANCE)
    assert "drift_keep_ppm" not in mlrec.read_model(old)["serving"]
    assert len(old) == len(data) - 4
    # The parser places the field where a rewrite writes it: after the field
    # of every flag before it.
    at = mlrec._v3_header(old, len(old))["drift_offset"]
    assert at == h["drift_offset"] == end - len(tail) - 4
    assert data[at : at + 4] == struct.pack("<I", 900000)
    assert mlrec.with_drift_fallback(old, 900000) == data
    assert mlrec.with_drift_fallback(data, None) == old
    assert mlrec.without_drift_fallback(data) == old
    assert mlrec.without_drift_fallback(old) == old
    assert mlrec.with_drift_fallback(mlrec.with_drift_fallback(data, 1), 900000) == data
    assert (
        mlrec.read_model(data)["payload_crc32"]
        == mlrec.read_model(old)["payload_crc32"]
    )
    # The other header editors keep it.
    assert (
        mlrec.read_header(mlrec.with_cu_floor(data, None))["serving"]["drift_keep_ppm"]
        == 900000
    )
    assert mlrec.with_cu_floor(mlrec.with_cu_floor(data, None), 64) == data
    ungated = mlrec.with_trained_keys(data, None)
    assert mlrec.read_header(ungated)["serving"]["drift_keep_ppm"] == 900000
    assert mlrec.with_bank(data, False) == data
    with pytest.raises(mlrec.FormatError, match="MLREC_v3"):
        mlrec.with_drift_fallback(b"MLREC_v2" + data[8:], 1)


def test_drift_fallback_follows_the_alignment_veto():
    """With the alignment veto the drift field follows the veto's two fields:
    each rewrite places its fields where the writer does, either serving
    option is added or removed without the other, and the engine reads
    both."""
    veto = {"k_min": 512, "align_bits": 1024}
    both = write(
        _drift_bundle(alignment_veto=veto, drift_keep_ppm=900000),
        provenance=PROVENANCE,
    )
    parsed = mlrec.read_model(both)
    assert parsed["serving"]["alignment_veto"] == veto
    assert parsed["serving"]["drift_keep_ppm"] == 900000
    assert write(mlrec.bundle_from_model(parsed), provenance=PROVENANCE) == both
    h = mlrec._v3_header(both, len(both))
    at = h["serving_offset"]
    assert struct.unpack_from("<I", both, at)[0] == 0x80 | 0x100 | 0x200 | 0x400
    # The trained keys, the CU floor, the veto's K and line, keep_ppm, then the
    # provenance.
    tail = mlrec._provenance_bytes(mlrec.check_provenance(PROVENANCE))
    end = h["header_end"]
    assert both[end - len(tail) - 48 : end - len(tail)] == struct.pack(
        "<QQQQIIII", 11, 22, 33, 44, 64, 512, 1024, 900000
    )
    assert h["drift_offset"] == end - len(tail) - 4
    veto_only = write(_drift_bundle(alignment_veto=veto), provenance=PROVENANCE)
    drift_only = write(_drift_bundle(drift_keep_ppm=900000), provenance=PROVENANCE)
    assert mlrec.with_drift_fallback(veto_only, 900000) == both
    assert mlrec.without_drift_fallback(both) == veto_only
    assert mlrec.with_alignment_veto(drift_only, 512, 1024) == both
    assert mlrec.without_alignment_veto(both) == drift_only
    assert mlrec.with_cu_floor(mlrec.with_cu_floor(both, None), 64) == both
    # Either flag cleared leaves its fields as bytes the header does not expect.
    for flags in (0x80 | 0x100 | 0x400, 0x80 | 0x100 | 0x200):
        cleared = bytearray(both)
        struct.pack_into("<I", cleared, at, flags)
        with pytest.raises(mlrec.FormatError):
            mlrec.read_model(bytes(cleared))
    tw = pytest.importorskip("tilewright")
    info = tw.describe(tw.load_model_from_memory(both))
    assert info.drift_keep_ppm == 900000
    assert (info.alignment_veto.on, info.alignment_veto.k_min) == (True, 512)
    assert info.alignment_veto.align_bits == 1024


@pytest.mark.parametrize("bad", [0, -1, 1000001, 2**32, True, 0.5, "900000"])
def test_drift_fallback_values_are_checked(bad):
    with pytest.raises(ValueError, match="keep_ppm"):
        write(_drift_bundle(drift_keep_ppm=bad))
    with pytest.raises(ValueError, match="keep_ppm"):
        mlrec.with_drift_fallback(write(_drift_bundle()), bad)


def test_drift_fallback_is_refused_where_the_engine_refuses_it():
    for serving, why in (
        ({"whitelist_tier": False}, "without the whitelist tier"),
        ({"launch_keys": True}, "with launch keys"),
    ):
        with pytest.raises(ValueError, match=why):
            write(_drift_bundle(drift_keep_ppm=1000000, **serving))
        with pytest.raises(ValueError, match=why):
            mlrec.with_drift_fallback(write(_drift_bundle(**serving)), 1000000)
    signed = _drift_bundle(drift_keep_ppm=1000000)
    signed["models"][LARGE]["smart_k_signatures"] = [[256, 256, 64, 16, 16, 32, 0, 0]]
    with pytest.raises(ValueError, match="signatures"):
        write(signed)
    signed["serving"].pop("drift_keep_ppm")
    with pytest.raises(ValueError, match="signatures"):
        mlrec.with_drift_fallback(write(signed), 1000000)
    data = write(_drift_bundle(drift_keep_ppm=1000000))
    with pytest.raises(ValueError, match="whitelist tier"):
        mlrec._with_flags(data, lambda f: f | mlrec.V3_NO_WHITELIST_TIER)
    with pytest.raises(ValueError, match="with_drift_fallback"):
        mlrec._with_flags(data, lambda f: f & ~mlrec.V3_DRIFT_FALLBACK)
    with pytest.raises(ValueError, match="with_drift_fallback"):
        mlrec._with_flags(write(_drift_bundle()), lambda f: f | mlrec.V3_DRIFT_FALLBACK)


def test_drift_fallback_defects_are_rejected_by_reader_and_engine():
    data = write(_drift_bundle(drift_keep_ppm=1000000))
    h = mlrec._v3_header(data, len(data))
    end, at = h["header_end"], h["serving_offset"]
    assert struct.unpack_from("<I", data, end - 4) == (1000000,)
    zero = data[: end - 4] + struct.pack("<I", 0) + data[end:]
    over = data[: end - 4] + struct.pack("<I", 1000001) + data[end:]
    for bad in (zero, over):
        with pytest.raises(mlrec.FormatError, match="outside 1 to 1000000"):
            mlrec.read_model(bad)
    unflagged = bytearray(data)
    struct.pack_into("<I", unflagged, at, 0x80 | 0x100)
    with pytest.raises(mlrec.FormatError):
        mlrec.read_model(bytes(unflagged))
    tierless = bytearray(data)
    struct.pack_into("<I", tierless, at, 0x80 | 0x100 | 0x400 | 0x08)
    with pytest.raises(mlrec.FormatError, match="whitelist tier"):
        mlrec.read_model(bytes(tierless))
    unknown = bytearray(data)
    struct.pack_into("<I", unknown, at, 0x80 | 0x100 | 0x400 | 0x1000)
    with pytest.raises(mlrec.FormatError, match="unknown serving flags"):
        mlrec.read_model(bytes(unknown))
    tw = pytest.importorskip("tilewright")
    for bad, why in (
        (zero, "outside 1 to 1000000"),
        (over, "outside 1 to 1000000"),
        (bytes(tierless), "whitelist tier"),
        (bytes(unknown), "unknown serving flags"),
    ):
        with pytest.raises(ValueError, match=why):
            tw.load_model_from_memory(bad)
    with pytest.raises(ValueError):
        tw.load_model_from_memory(bytes(unflagged))
    model = tw.load_model_from_memory(data)
    assert tw.describe(model).drift_keep_ppm == 1000000
    assert (
        tw.describe(
            tw.load_model_from_memory(mlrec.without_drift_fallback(data))
        ).drift_keep_ppm
        == 0
    )
