# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import pytest

from lib import features as fs
from lib import hipblaslt_launch as hl


def _plan(m, n, k, occupancy, attrs, context, arch="gfx950", batch=1):
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=occupancy)
    attrs = dict({"work_group_x": 256, "work_group_y": 1, "work_group_z": 1}, **attrs)
    p = hl.launch_plan(prob, cfg, attrs, context, 256, arch)
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


HYBRID = {"tile_processing_strategy": 2, "work_assignment": 2}


def test_hybrid_thresholds():
    sm224 = fs.ExecutionContext(cu_budget=224)
    assert _plan(128 * 480, 128, 512, 2, HYBRID, sm224) == (False, 320, "tree")
    assert _plan(128 * 481, 128, 512, 2, HYBRID, sm224) == (True, 481, "none")
    assert _plan(4096, 4096, 512, 3, HYBRID, sm224) == (True, 768, "none")
    assert _plan(4096, 4096, 512, 4, HYBRID, sm224) == (False, 1024, "none")
    assert _plan(128 * 1883, 128, 512, 4, HYBRID, sm224) == (False, 1883, "none")
    assert _plan(128 * 1884, 128, 512, 4, HYBRID, sm224) == (True, 768, "none")


def test_sm_target_and_schedules():
    def plan(budget, schedule="default", arch="gfx950"):
        ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
        return _plan(4096, 4096, 512, 2, HYBRID, ctx, arch)

    assert plan(256) == (False, 512, "none")
    assert plan(224) == plan(240) == plan(255) == (True, 512, "none")
    assert plan(31) == plan(0) == (False, 512, "none")
    assert plan(224, "auto") == (True, 512, "none")
    assert plan(0, "auto") == (False, 512, "none")
    assert plan(0, "dynamic") == (True, 512, "none")
    assert plan(224, arch="gfx942") == (False, 1024, "none")
    assert plan(224, "dynamic", "gfx942") == (True, 512, "none")
    assert hl.hybrid_runtime_choice("gfx950:xnack-")
    assert not hl.hybrid_runtime_choice("gfx9500")


def test_stream_k_grids():
    excl = fs.EXCLUSIVE
    assert _plan(256, 256, 64 * 64, 1, HYBRID, excl) == (False, 32, "parallel")
    assert _plan(256, 256, 64 * 32, 1, HYBRID, excl) == (False, 16, "tree")
    atomic = dict(HYBRID, stream_k_atomic=1)
    assert _plan(256, 256, 64 * 32, 1, atomic, excl) == (False, 16, "atomic")
    dp = {"tile_processing_strategy": 1, "work_assignment": 0}
    assert _plan(4096, 4096, 512, 2, dp, excl) == (False, 512, "none")
    queue = {"tile_processing_strategy": 2, "work_assignment": 1}
    assert _plan(4096, 4096, 512, 5, queue, excl) == (True, 768, "none")
    assert _plan(4096, 4096, 512, 2, {}, excl) == (False, 1024, "none")
    huge = dict(HYBRID, work_group_x=1024, work_group_y=1024)
    assert _plan(128 * 300, 128, 64 * 300, 1, HYBRID, excl) == (False, 200, "tree")
    assert _plan(128 * 300, 128, 64 * 300, 1, huge, excl) == (False, 300, "none")
    assert _plan(128 * 4500, 128, 512, 9, huge, excl) == (False, 2048, "tree")
    assert _plan(1024, 1024, 512, 2, HYBRID, excl, batch=16) == (False, 512, "none")


# (m, n, k, batch, mt_m, mt_n, mt_k, work_group, cu_budget, schedule) of
# occupancy-8 Hybrid gfx950 kernels whose dynamic launch overflows the tree
# fix-up and is clamped to the flags after the per-XCD queue counters.
QUEUE_CLAMPED = [
    (36450, 36788, 31014, 16, 16, 16, 32, (16, 4, 1), 224, "dynamic"),
    (14498, 19072, 8121, 32, 16, 16, 64, (16, 4, 2), 96, "auto"),
    (30965, 13895, 63840, 64, 16, 16, 64, (16, 4, 2), 96, "default"),
    (12387, 60136, 977, 32, 16, 32, 64, (16, 8, 1), 224, "dynamic"),
    (2237, 60189, 2130, 64, 16, 16, 64, (16, 4, 2), 96, "default"),
    (16845, 22929, 6, 64, 16, 32, 64, (16, 8, 1), 64, "auto"),
    (56979, 7497, 10, 64, 16, 64, 32, (16, 16, 1), 224, "auto"),
]


@pytest.mark.parametrize("row", QUEUE_CLAMPED)
def test_dynamic_queues_take_the_head_of_the_flag_region(row):
    m, n, k, batch, mt_m, mt_n, mt_k, wg, budget, schedule = row
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=mt_m, mt_n=mt_n, mt_k=mt_k, occupancy=8)
    attrs = dict(HYBRID, work_group_x=wg[0], work_group_y=wg[1], work_group_z=wg[2])
    ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
    flags = hl.STREAM_K_FLAG_ELEMENTS - 8 * hl.CACHE_LINE_BYTES // 4
    assert hl.launch_plan(prob, cfg, attrs, ctx, 256, "gfx950") == (
        fs.LaunchPlan(True, flags, "none")
    )


def test_flag_region_clamp():
    excl = fs.EXCLUSIVE
    dyn = fs.ExecutionContext(schedule="dynamic")
    wg = {"work_group_x": 1024, "work_group_y": 1024}
    hybrid, queue = dict(HYBRID, **wg), dict(HYBRID, work_assignment=1, **wg)
    assert _plan(128 * 4500, 128, 512, 9, hybrid, excl) == (False, 2048, "tree")
    assert _plan(128 * 4500, 128, 512, 9, hybrid, dyn) == (True, 1792, "none")
    assert _plan(128 * 4500, 128, 512, 9, queue, excl) == (True, 1792, "none")
    assert _plan(128 * 4500, 128, 512, 9, queue, excl, "gfx942") == (
        True,
        1792,
        "none",
    )
    assert _plan(128 * 4500, 128, 512, 9, queue, excl, "gfx90a") == (
        True,
        2048 - 32,
        "none",
    )
    assert _plan(128 * 4500, 128, 512, 9, hybrid, excl, "gfx1250") == (
        False,
        2048,
        "tree",
    )
    small = dict(HYBRID, work_assignment=1)
    assert _plan(4096, 4096, 512, 2, small, excl, "gfx1250") == (True, 512, "none")
    with pytest.raises(ValueError, match="gfx1250"):
        _plan(4096, 4096, 512, 5, small, excl, "gfx1250")


# An architecture without a fixed queue count, made up for these tests; the
# XCD counts given with it are made up too.
UNLISTED = "gfx9990"


def test_the_device_xcd_count_sets_the_queues_where_the_arch_keeps_none():
    assert hl.queue_count("gfx950") == hl.queue_count("gfx950:xnack-", 3) == 8
    for arch in ("gfx1250", "gfx1250-strict", "gfx1250v0", UNLISTED):
        assert hl.queue_count(arch) is None
    assert hl.queue_count(UNLISTED, 3) == hl.queue_count(UNLISTED + ":xnack-", 3) == 3
    prob = dict(m=128 * 4500, n=128, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=9)
    attrs = dict(HYBRID, work_assignment=1, work_group_x=1024, work_group_y=1024)
    attrs["work_group_z"] = 1
    excl = fs.EXCLUSIVE
    for n in (1, 3):
        got = hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, n)
        flags = hl.STREAM_K_FLAG_ELEMENTS - n * hl.CACHE_LINE_BYTES // 4
        assert got == fs.LaunchPlan(True, flags, "none")
    assert hl.launch_plan(prob, cfg, attrs, excl, 256, "gfx90a", 3) == (
        hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, 1)
    )
    for arch in (UNLISTED, "gfx1250-strict"):
        with pytest.raises(ValueError, match="hardware.num_xcd"):
            hl.launch_plan(prob, cfg, attrs, excl, 256, arch)
    source = hl.HipblasltLaunch(UNLISTED, num_xcd=3)
    assert repr(source) == f"HipblasltLaunch('{UNLISTED}', num_xcd=3)"
    assert source.id == hl.HipblasltLaunch(UNLISTED).id
    assert source(prob, cfg, attrs, excl, 256) == (
        hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, 3)
    )
    stem = f"TensileLibrary_X_Contraction_l_Alik_Bljk_Cijk_Dijk_{UNLISTED}"
    assert hl.HipblasltLaunch.for_library(stem, 3).num_xcd == 3
    assert hl.HipblasltLaunch.for_library(stem).num_xcd is None


QUEUE = {"tile_processing_strategy": 2, "work_assignment": 1}


@pytest.mark.parametrize(
    "attrs, arch, schedules, reduced, takes",
    [
        ({}, "gfx1250", ("default", "dynamic"), True, False),
        ({"tile_processing_strategy": 1}, "gfx1250", ("dynamic",), True, False),
        ({"tile_processing_strategy": 2}, "gfx1250", ("dynamic",), True, False),
        (QUEUE, "gfx1250", ("default",), False, True),
        (dict(QUEUE, stream_k_atomic=1), "gfx1250", ("default",), True, False),
        (HYBRID, "gfx1250", ("default",), True, False),
        (HYBRID, "gfx1250", ("default", "auto"), True, False),
        (HYBRID, "gfx1250", ("default", "dynamic"), False, True),
        (HYBRID, "gfx950", ("default",), False, False),
        (HYBRID, "gfx950", ("default",), True, True),
    ],
)
def test_which_kernels_take_a_dynamic_queue(attrs, arch, schedules, reduced, takes):
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    assert hl.takes_dynamic_queue(cfg, attrs, arch, schedules, reduced) is takes


# A CU count no device of these tests has.
MADE_UP_CUS = 300


def test_a_kernel_that_takes_no_dynamic_queue_needs_no_queue_count():
    rng = random.Random(7)
    kernels = [
        dict(HYBRID, stream_k_atomic=a, work_assignment=w, work_group_x=wg)
        for a in (0, 1)
        for w in (0, 1, 2)
        for wg in (64, 256, 1024)
    ]
    for schedules, reduced in (
        (("default",), False),
        (("default",), True),
        (("default", "auto"), True),
        (("default", "dynamic"), False),
    ):
        budgets = (0, 64, 128, 224) if reduced else (0,)
        for attrs in kernels:
            attrs = dict(attrs, work_group_y=1, work_group_z=1)
            for occupancy in (1, 2, 4, 8):
                cfg = dict(mt_m=16, mt_n=16, mt_k=64, occupancy=occupancy)
                if hl.takes_dynamic_queue(cfg, attrs, "gfx1250", schedules, reduced):
                    continue
                for _ in range(40):
                    prob = dict(
                        m=rng.randint(1, 1 << 16),
                        n=rng.randint(1, 1 << 16),
                        k=rng.randint(1, 1 << 16),
                        batch=rng.choice((1, 1, 4)),
                        mi_dtype="bfloat16",
                    )
                    ctx = fs.ExecutionContext(
                        cu_budget=rng.choice(budgets), schedule=rng.choice(schedules)
                    )
                    hl.launch_plan(prob, cfg, attrs, ctx, MADE_UP_CUS, "gfx1250")


@pytest.mark.parametrize("assign", [1, 2])
def test_data_parallel_kernels_take_a_static_grid_only(assign):
    dp = {"tile_processing_strategy": 1, "work_assignment": assign}
    with pytest.raises(ValueError, match="StaticGrid"):
        _plan(4096, 4096, 512, 2, dp, fs.EXCLUSIVE)


def test_launch_arch_is_the_compiler_target_of_the_library_suffix():
    stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
    assert hl.launch_arch(stem + "_gfx950") == "gfx950"
    assert hl.launch_arch(stem + "_CU256_ID75a8_gfx950") == "gfx950"
    assert hl.launch_arch(f"/lib/gfx950/{stem}_gfx950.dat.zlib") == "gfx950"
    assert hl.launch_arch(stem + "_gfx1250") == "gfx1250"
    assert hl.launch_arch(stem + "_gfx1250-strict") == "gfx1250"
    for bad in ("TensileLibrary", "TensileLibrary_"):
        with pytest.raises(ValueError):
            hl.launch_arch(bad)
    ids = {
        hl.HipblasltLaunch.for_library(stem + "_gfx1250").id,
        hl.HipblasltLaunch.for_library(stem + "_gfx1250-strict").id,
        hl.HipblasltLaunch("gfx1250v0").id,
        hl.HipblasltLaunch("gfx1250:xnack+").id,
    }
    assert ids == {f"hipblaslt:gfx1250:{hl.RULE_VERSION}"}


def test_non_persistent_kernels_launch_one_workgroup_per_tile():
    for attrs in ({}, {"tile_processing_strategy": 0, "work_assignment": 2}):
        assert _plan(4096, 1000, 512, 2, attrs, fs.EXCLUSIVE) == (False, 256, "none")
    prob = dict(m=4096, n=4096, k=512, batch=1)
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    assert hl.launch_plan(prob, cfg, None, fs.EXCLUSIVE, 0, "gfx950") == (
        fs.LaunchPlan(False, 1024, "none")
    )


def test_hipblaslt_launch_is_a_launch_source():
    source = hl.HipblasltLaunch("gfx942")
    assert repr(source) == "HipblasltLaunch('gfx942')"
    assert isinstance(hl.RULE_VERSION, int) and hl.RULE_VERSION >= 1
    assert source.id == f"hipblaslt:gfx942:{hl.RULE_VERSION}"
    assert hl.HipblasltLaunch("gfx950").id != source.id
    prob = dict(m=4096, n=4096, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    ctx = fs.ExecutionContext(cu_budget=224)
    assert source(prob, cfg, HYBRID, ctx, 256) == hl.launch_plan(
        prob, cfg, HYBRID, ctx, 256, "gfx942"
    )
    assert fs.Catalog("v3", (), source).launch is source
    assert not set(hl.LAUNCH_ATTRIBUTES) & {a.name for a in fs.DEFAULT_ATTRIBUTE_SPEC}


@pytest.mark.parametrize("name", hl.LAUNCH_ATTRIBUTES)
def test_launch_attributes_default_to_zero(name):
    prob = dict(m=4096, n=4096, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    attrs = dict(HYBRID, work_group_x=256, work_group_y=1, work_group_z=1)
    attrs[name] = 0
    without = {k: v for k, v in attrs.items() if k != name}
    assert hl.launch_plan(prob, cfg, attrs, fs.EXCLUSIVE, 256, "gfx950") == (
        hl.launch_plan(prob, cfg, without, fs.EXCLUSIVE, 256, "gfx950")
    )


SIGNATURE = ("mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k")


def _random_kernel(rng):
    cfg = dict(
        mt_m=rng.choice([16, 64, 128, 256]),
        mt_n=rng.choice([16, 64, 128, 256]),
        mt_k=rng.choice([16, 32, 64, 128]),
        mi_m=16, mi_n=16, mi_k=32, cache_hints_a=0, cache_hints_b=0,
        occupancy=rng.choice([-1, 0, 1, 2, 3, 4, 5]),
        grvw_a=rng.choice([1, 4, 8]), grvw_b=rng.choice([1, 4, 8]),
        gwvw_d=rng.choice([1, 4]),
    )  # fmt: skip
    attrs = {"unrelated": rng.choice([0, 1, 9])}
    for name, values in (
        ("tile_processing_strategy", [0, 1, 2, 2, 3]),
        ("work_assignment", [0, 1, 2]),
        ("stream_k_atomic", [0, 1]),
        ("work_group_x", [64, 128, 256]),
        ("work_group_y", [1, 2, 4]),
        ("work_group_z", [1, 2]),
    ):
        if rng.random() < 0.9:
            attrs[name] = rng.choice(values)
    return cfg, _loadable(attrs)


def _loadable(attrs):
    """`attrs` as TensileLite loads them: a DataParallel kernel takes the
    static grid only."""
    if attrs.get("tile_processing_strategy") == 1:
        attrs["work_assignment"] = 0
    return attrs


def _variant(rng, cfg, attrs):
    """`cfg`/`attrs` with the signature kept and anything else perhaps
    changed, launch parameters included."""
    other_cfg, other_attrs = _random_kernel(rng)
    other_cfg.update({k: cfg[k] for k in SIGNATURE})
    other_cfg.update(
        cache_hints_a=cfg["cache_hints_a"], cache_hints_b=cfg["cache_hints_b"]
    )
    if rng.random() < 0.6:
        other_cfg["occupancy"] = cfg["occupancy"]
        for name in hl.LAUNCH_ATTRIBUTES:
            if rng.random() < 0.8:
                other_attrs.pop(name, None)
                if name in attrs:
                    other_attrs[name] = attrs[name]
    return other_cfg, _loadable(other_attrs)


def _random_problem(rng):
    return dict(
        m=rng.choice([1, 64, 100, 1024, 4096, 128 * 481, 65536]),
        n=rng.choice([1, 64, 128, 1000, 4096]),
        k=rng.choice([1, 64, 512, 4096, 64 * 300, 64 * 65536]),
        batch=rng.choice([1, 1, 2, 16]),
        mi_dtype=rng.choice(["bfloat16", "float", "double"]),
    )


def test_equal_launch_keys_launch_alike():
    rng = random.Random(17)
    equal = differing = 0
    for _ in range(4000):
        cfg, attrs = _random_kernel(rng)
        other_cfg, other_attrs = _variant(rng, cfg, attrs)
        same = hl.launch_key(cfg, attrs) == hl.launch_key(other_cfg, other_attrs)
        plans = []
        for _ in range(6):
            prob = _random_problem(rng)
            ctx = fs.ExecutionContext(
                cu_budget=rng.choice([0, 31, 64, 200, 224, 255, 256, 999]),
                schedule=rng.choice(fs.SCHEDULES),
            )
            n_cu = rng.choice([0, 64, 256, 304])
            arch = rng.choice(["gfx950", "gfx942", "gfx950:xnack-"])
            a = hl.launch_plan(prob, cfg, attrs, ctx, n_cu, arch)
            b = hl.launch_plan(prob, other_cfg, other_attrs, ctx, n_cu, arch)
            plans.append(a == b)
        if same:
            equal += 1
            assert all(plans), (cfg, attrs, other_cfg, other_attrs)
        elif not all(plans):
            differing += 1
    assert equal > 1000 and differing > 500


def test_launch_keys_merge_what_the_rule_cannot_tell_apart():
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    plain = {"tile_processing_strategy": 0, "work_assignment": 2}
    assert hl.launch_key(cfg, plain) == hl.launch_key(dict(cfg, occupancy=4), {})
    sk = dict(HYBRID, work_group_x=256, work_group_y=1, work_group_z=1)
    assert hl.launch_key(
        cfg, dict(sk, work_group_x=128, work_group_y=2)
    ) == hl.launch_key(cfg, sk)
    assert hl.launch_key(dict(cfg, occupancy=0), sk) == hl.launch_key(
        dict(cfg, occupancy=1), sk
    )
    dp = {"tile_processing_strategy": 1, "work_assignment": 0}
    assert hl.launch_key(cfg, dict(dp, stream_k_atomic=1)) == hl.launch_key(cfg, dp)
    assert hl.launch_key(cfg, dict(sk, stream_k_atomic=1)) != hl.launch_key(cfg, sk)
    assert hl.launch_key(dict(cfg, occupancy=3), sk) != hl.launch_key(cfg, sk)
    source = hl.HipblasltLaunch("gfx950")
    assert source.launch_key(cfg, sk) == hl.launch_key(cfg, sk)
    assert 0 <= hl.launch_key(cfg, sk) < 2**64


class _Strict(dict):
    """A mapping that fails on reading any key outside `allowed`."""

    def __init__(self, data, allowed):
        super().__init__(data)
        self.allowed = set(allowed)

    def _check(self, key):
        if key not in self.allowed:
            raise AssertionError(f"launch_plan read {key!r} outside its launch key")

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._check(key)
        return super().__contains__(key)

    def __iter__(self):
        raise AssertionError("launch_plan iterated a kernel's parameters")

    def items(self):
        raise AssertionError("launch_plan iterated a kernel's parameters")

    keys = values = items


def test_launch_plan_reads_the_kernel_only_through_launch_inputs(monkeypatch):
    """Everything launch_plan reads of a kernel beyond the signature's tile
    sizes must pass through `launch_inputs`, which the launch key covers."""
    real = hl.launch_inputs
    rng = random.Random(3)
    cases = [
        (
            _random_problem(rng),
            fs.ExecutionContext(
                cu_budget=rng.choice([0, 64, 224]), schedule=rng.choice(fs.SCHEDULES)
            ),
            rng.choice([0, 256]),
        )
        for _ in range(8)
    ]
    for _ in range(500):
        cfg, attrs = _random_kernel(rng)
        expect = [hl.launch_plan(p, cfg, attrs, c, n, "gfx950") for p, c, n in cases]
        monkeypatch.setattr(hl, "launch_inputs", lambda c, a: real(cfg, attrs))
        strict_cfg = _Strict(cfg, hl.LAUNCH_SIGNATURE_FIELDS)
        strict_attrs = _Strict(attrs, ())
        got = [
            hl.launch_plan(p, strict_cfg, strict_attrs, c, n, "gfx950")
            for p, c, n in cases
        ]
        monkeypatch.setattr(hl, "launch_inputs", real)
        assert got == expect
    real(
        _Strict(dict(occupancy=2), hl.LAUNCH_CONFIG_FIELDS),
        _Strict(dict(HYBRID), hl.LAUNCH_ATTRIBUTES),
    )
