# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CU shares from an eighth to all of the device's CUs.

The production configs give their CU budgets as shares of the bench
device's CUs: three quarters, half, a quarter and an eighth, each a whole
multiple of the budget granule and at least one, so that their models train
and serve CU budgets from one granule up to the whole device; one key turns
the eighth off for its bench time. The gfx1250 sanity configs train half and
an eighth. A run sizes the shares for the device it benches on, a compute
partition included, and a model declares the smallest share it trained, so
that serving falls back below it. Its audit set must have been built with
the budgets it resolves."""
import copy
import dataclasses
import json
import re
import types
from pathlib import Path

import pytest

import run_orchestrator as ro
import stage05_train as s5
from lib import audit as auditlib
from lib import contexts as ctxlib

GRANULE = ctxlib.CU_BUDGET_GRANULE
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
# The production configs: one per gfx950 library with a shipped model, and
# one per data type of the devices that report gfx1250.
PRODUCTION_CONFIGS = [
    p for p in CONFIGS if re.fullmatch(r"gfx950_\w+_origami_v3|gfx1250_\w+_v3", p.stem)
]
# The configs of a sanity run of a new host or partition.
SANITY_CONFIGS = [p for p in CONFIGS if p.stem.endswith("_sanity")]
SHARES = (0.75, 0.5, 0.25)
EIGHTH = 0.125
# A device of eight budget granules gives every share a budget of its own.
WHOLE = 8 * GRANULE


def _load(path):
    return ro.load_config(path)[0]


def _turn_off_the_eighth(cfg):
    """`cfg` with its eighth turned off (`enabled: false`)."""
    out = copy.deepcopy(cfg)
    (eighth,) = [m for m in out["contexts"]["list"] if m.get("cu_fraction") == EIGHTH]
    eighth["enabled"] = False
    return out


def _trained(cfg, n_cu):
    """The trained contexts `cfg` benches on a device of `n_cu` CUs."""
    resolved, _ = ctxlib.resolve_fractions(cfg, n_cu)
    return [c for c in ctxlib.contexts_from_config(resolved) if not c.holdout]


def _declared(cfg, n_cu):
    """The contexts a model declares that a run of `cfg` on a device of
    `n_cu` CUs trains (stage05 `declared_contexts` over GEMMs of every
    trained context)."""
    gemms = [{}] + [{"cu_budget": c.cu_budget} for c in _trained(cfg, n_cu)]
    floor = ro.cfg_get(cfg, "train.min_cu_fraction")
    return s5.declared_contexts(gemms, n_cu, floor)


def _trains_an_eighth_to_all(cfg):
    """A config trains every share down to an eighth: its smallest trained
    share is the eighth, the models may declare it, and on a device of
    eight granules every share gets a budget of its own, the eighth one
    granule, which the models declare."""
    shares = [
        float(m["cu_fraction"])
        for m in ctxlib.trained_entries(cfg)
        if m.get("cu_fraction") is not None
    ]
    assert min(shares) == EIGHTH
    floor = ro.cfg_get(cfg, "train.min_cu_fraction")
    assert floor is None or floor <= EIGHTH
    budgets = [c.cu_budget for c in _trained(cfg, WHOLE)]
    assert min(budgets) == GRANULE and len(set(budgets)) == len(budgets)
    assert _declared(cfg, WHOLE)["min_cu_fraction"] == EIGHTH


def _contexts_comment(path):
    """The comment lines between `contexts:` and its `list:`, joined."""
    lines = path.read_text().splitlines()
    at = lines.index("contexts:")
    out = []
    for ln in lines[at + 1 :]:
        if not ln.strip().startswith("#"):
            break
        out.append(ln.strip().lstrip("#").strip())
    return " ".join(out)


def test_the_production_configs_are_gfx950_and_gfx1250_ones():
    families = sorted({p.stem.split("_")[0] for p in PRODUCTION_CONFIGS})
    assert families == ["gfx1250", "gfx950"]
    assert len([p for p in PRODUCTION_CONFIGS if p.stem.startswith("gfx950_")]) == 9


@pytest.mark.parametrize("path", PRODUCTION_CONFIGS, ids=lambda p: p.name)
def test_production_configs_train_every_share_down_to_the_eighth(path, config_env):
    """Three quarters, half, a quarter and an eighth of the bench device's
    CUs, each on a tenth of a cell's shapes, the eighth on by its own key;
    no budget names a CU count, the budget granule is the default one, and
    no floor keeps the models from declaring the eighth."""
    cfg = _load(path)
    exclusive, *entries = cfg["contexts"]["list"]
    assert exclusive == {"cu_budget": 0} and cfg["contexts"]["holdout"] == []
    assert [m["cu_fraction"] for m in entries] == [*SHARES, EIGHTH]
    assert [ctxlib.enabled(m) for m in entries] == [True] * 4
    assert entries[-1]["enabled"] is True
    assert all(m["cotenant"] and "cu_budget" not in m for m in entries)
    assert [m["fraction"] for m in entries] == [0.1] * 4
    assert "granule" not in cfg["contexts"]
    assert ro.cfg_get(cfg, "train.min_cu_fraction") is None
    assert ro.validate_config(cfg) == ([], [])
    _trains_an_eighth_to_all(cfg)
    comment = _contexts_comment(path)
    assert "enabled: false turns the eighth off" in comment
    assert "granule" in comment and "one XCD's worth" in comment
    assert "partition" in comment and "audit.dir at a new one" in comment


@pytest.mark.parametrize("path", PRODUCTION_CONFIGS, ids=lambda p: p.name)
def test_production_shares_reach_one_granule_on_devices_of_256_and_192_cus(
    path, config_env
):
    """The shares resolve to 192, 128, 64 and 32 CUs on a device of 256 CUs
    and, an eighth of it being less than one granule, to 160, 96, 64 and 32
    on a device of 192: the smallest budget is one granule on both, and the
    models declare it."""
    cfg = _load(path)
    for n_cu, want in ((256, [192, 128, 64, 32]), (192, [160, 96, 64, 32])):
        assert [c.cu_budget for c in _trained(cfg, n_cu)] == want, n_cu
        assert ctxlib.context_errors(cfg, n_cu) == [], n_cu
        assert _declared(cfg, n_cu)["min_cu_fraction"] == GRANULE / n_cu, n_cu


@pytest.mark.parametrize("path", PRODUCTION_CONFIGS, ids=lambda p: p.name)
def test_turning_the_eighth_off_is_one_key(path, config_env):
    cfg = _load(path)
    off = _turn_off_the_eighth(cfg)
    assert ro.validate_config(off) == ([], [])
    budgets = [c.cu_budget for c in _trained(off, WHOLE)]
    assert budgets == [6 * GRANULE, 4 * GRANULE, 2 * GRANULE]
    assert _declared(off, WHOLE)["min_cu_fraction"] == 0.25
    assert [c.cu_budget for c in _trained(cfg, WHOLE)] == [*budgets, GRANULE]


@pytest.mark.parametrize("path", SANITY_CONFIGS, ids=lambda p: p.name)
def test_sanity_configs_train_every_share_down_to_an_eighth(path, config_env):
    _trains_an_eighth_to_all(_load(path))


@pytest.mark.parametrize("eighth", [False, True], ids=["eighth_off", "eighth_on"])
@pytest.mark.parametrize("path", PRODUCTION_CONFIGS, ids=lambda p: p.name)
def test_models_declare_the_smallest_share_they_trained(path, eighth, config_env):
    """On every device of two to sixteen granules the shares resolve to
    distinct multiples of the granule below the device's CUs, largest
    first, and a model declares exactly the smallest of them (every CU
    when the device gives none)."""
    cfg = _load(path)
    cfg = cfg if eighth else _turn_off_the_eighth(cfg)
    for n_cu in range(2 * GRANULE, 16 * GRANULE + 1):
        budgets = [c.cu_budget for c in _trained(cfg, n_cu)]
        assert budgets == sorted(set(budgets), reverse=True), n_cu
        assert all(b % GRANULE == 0 and 0 < b < n_cu for b in budgets), n_cu
        lowest = min(budgets, default=n_cu)
        assert _declared(cfg, n_cu)["min_cu_fraction"] == lowest / n_cu, n_cu


def _with_device_values(root):
    """The fake KFD topology at `root` with an LDS size and an L2 cache on
    every GPU node, as a device's KFD node gives them."""
    nodes = root / "class" / "kfd" / "kfd" / "topology" / "nodes"
    for node in nodes.iterdir():
        props = node / "properties"
        if "simd_per_cu" not in props.read_text():
            continue
        props.write_text(props.read_text() + "lds_size_in_kb 64\n")
        cache = node / "caches" / "0"
        cache.mkdir(parents=True)
        (cache / "properties").write_text("level 2\nsize 4096\n")
    return root


@pytest.mark.parametrize(
    "mode, parts", [("SPX", 1), ("DPX", 2), ("QPX", 4), ("CPX", 8)]
)
def test_a_run_sizes_the_shares_for_the_partition_it_benches_on(
    mode, parts, tmp_path, monkeypatch, config_env
):
    """Each compute partition is a device of its own: a run on one resolves
    the shares from the partition's CUs, which its KFD node gives, says
    which partition they are shares of, and keeps the shares the partition
    can give, one granule the smallest; one of a single granule trains the
    exclusive context alone."""
    import gpu_fakes as gf
    from lib import kfd

    for var in (*kfd.HIP_VARS, kfd.ROCR_VAR):
        monkeypatch.delenv(var, raising=False)
    n_cu = WHOLE // parts
    root = gf.write_sysfs(
        tmp_path / "sys", gf.GPUS[:2], xcc=[(1, mode)] * 2, cus=[n_cu] * 2
    )
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(_with_device_values(root)))
    cfg = _load(CONFIG_DIR / "gfx950_bbs_tn_origami_v3.yaml")
    out, lines = ro.resolve_contexts(cfg, 1)
    where = "" if mode == kfd.FULL_DEVICE_PARTITION else f", one {mode} partition"
    assert lines[0] == (
        f"contexts: CU shares of the {n_cu} CUs of bench device 1"
        + (f"{where} of its card" if where else "")
        + " (its KFD node)"
    )
    want = {
        1: [6 * GRANULE, 4 * GRANULE, 2 * GRANULE, GRANULE],
        2: [3 * GRANULE, 2 * GRANULE, GRANULE],
        4: [GRANULE],
        8: [],
    }[parts]
    assert [c.cu_budget for c in ctxlib.contexts_from_config(out)] == want
    assert len([ln for ln in lines if ln.endswith("left out")]) == 4 - len(want)
    assert not [ln for ln in lines if " is off " in ln]
    assert _declared(cfg, n_cu)["min_cu_fraction"] == min(want, default=n_cu) / n_cu
    pinned = dict(cfg, hardware={"n_cu": n_cu})
    _, pinned_lines = ro.resolve_contexts(pinned, 1)
    assert pinned_lines[0].endswith(" (hardware.n_cu)") and where in pinned_lines[0]
    off, off_lines = ro.resolve_contexts(_turn_off_the_eighth(cfg), 1)
    budgets = [c.cu_budget for c in ctxlib.contexts_from_config(off)]
    assert budgets == (want[:-1] if parts == 1 else want)
    (line,) = [ln for ln in off_lines if " is off " in ln]
    assert line == "contexts.list[4]: cu_fraction 0.125 is off (enabled: false)"


@pytest.mark.parametrize("eighth", [False, True], ids=["eighth_off", "eighth_on"])
@pytest.mark.parametrize("parts", [1, 2], ids=["whole", "half"])
def test_a_model_covers_budgets_down_to_the_smallest_trained_share(
    eighth, parts, config_env
):
    """A model covers every budget from the smallest one its run trained up
    to every CU, and none below it, on the device the run benched on; the
    engine ranks a budget below it as that smallest budget."""
    tw = pytest.importorskip("tilewright")
    from lib import evaluate as ev
    from lib import features as fs
    from test_v3 import HW, v3_bundle, write

    cfg = _load(CONFIG_DIR / "gfx950_bbs_tn_origami_v3.yaml")
    cfg = cfg if eighth else _turn_off_the_eighth(cfg)
    n_cu = WHOLE // parts
    lowest = min(c.cu_budget for c in _trained(cfg, n_cu))
    contexts = _declared(cfg, n_cu)
    model = tw.load_model_from_memory(write(v3_bundle(contexts=contexts)))
    assert tw.describe(model).min_cu_fraction == lowest / n_cu
    hw = ev.make_hardware(tw, dataclasses.replace(HW, n_cu=n_cu))
    served = [
        b
        for b in range(n_cu + 2)
        if tw.supports(model, ev.make_context(tw, fs.ExecutionContext(b)), hw)
    ]
    assert served == [0] + list(range(lowest, n_cu + 2))
    for b in range(1, lowest):
        at = tw.scoring_context(model, ev.make_context(tw, fs.ExecutionContext(b)), hw)
        assert at is not None and at.cu_budget == lowest, (b, at)
    assert lowest == (GRANULE if eighth or parts > 1 else 2 * GRANULE)


@pytest.mark.parametrize("eighth", [False, True], ids=["eighth_off", "eighth_on"])
def test_a_smaller_device_covers_no_fewer_cus_than_the_run_trained(eighth, config_env):
    """On a compute partition of the bench device, which looks to the model
    like a whole device of fewer CUs, it covers every budget and the whole
    partition down to the fewest CUs its run trained (its CU floor) and
    none below, however large a share of the partition that is; the
    engine ranks those as the floor, or as the whole partition when it has
    fewer CUs. Without the floor only the share would count."""
    tw = pytest.importorskip("tilewright")
    from lib import evaluate as ev
    from lib import features as fs
    from lib import mlrec
    from test_v3 import HW, v3_bundle, write

    cfg = _load(CONFIG_DIR / "gfx950_bbs_tn_origami_v3.yaml")
    cfg = cfg if eighth else _turn_off_the_eighth(cfg)
    lowest = min(c.cu_budget for c in _trained(cfg, WHOLE))
    contexts = _declared(cfg, WHOLE)
    assert (
        contexts["min_effective_cus"] == lowest == (GRANULE if eighth else 2 * GRANULE)
    )
    data = write(v3_bundle(contexts=contexts))
    model = tw.load_model_from_memory(data)
    unfloored = tw.load_model_from_memory(mlrec.with_cu_floor(data, None))
    assert tw.describe(model).min_effective_cus == lowest

    def served(m, n_cu):
        hw = ev.make_hardware(tw, dataclasses.replace(HW, n_cu=n_cu))
        return [
            b
            for b in range(n_cu + 2)
            if tw.supports(m, ev.make_context(tw, fs.ExecutionContext(b)), hw)
        ]

    half, quarter, eighth_cus = WHOLE // 2, WHOLE // 4, WHOLE // 8
    if eighth:
        assert served(model, half) == [0] + list(range(GRANULE, half + 2))
        assert served(model, quarter) == [0] + list(range(GRANULE, quarter + 2))
        assert served(model, eighth_cus) == [0, GRANULE, GRANULE + 1]
    else:
        assert served(model, half) == [0] + list(range(2 * GRANULE, half + 2))
        assert served(model, quarter) == [0, quarter, quarter + 1]
        assert served(model, eighth_cus) == []
    for n_cu in (half, quarter, eighth_cus):
        share = round(contexts["min_cu_fraction"] * n_cu)
        assert served(unfloored, n_cu) == [0] + list(range(share, n_cu + 2))
    assert served(model, WHOLE) == served(unfloored, WHOLE)


def _audit_set(cfg, n_cu):
    """The complete audit set in `cfg`'s audit.dir, as `cfg` built it on a
    device of `n_cu` CUs (its manifest and completion marker)."""
    built, _ = ctxlib.resolve_fractions(cfg, n_cu)
    d = auditlib.config_dir(built)
    d.mkdir(parents=True)
    binding = ro.audit_binding(built, auditlib.bench_config(built))
    manifest = {
        "schema": auditlib.SCHEMA_VERSION,
        "binding": binding,
        "generations": [],
    }
    (d / auditlib.MANIFEST).write_text(json.dumps(manifest))
    (d / auditlib.COMPLETE_MARKER).write_text("{}")
    return d


@pytest.mark.parametrize("eighth", [False, True], ids=["eighth_off", "eighth_on"])
@pytest.mark.parametrize("parts", [1, 2], ids=["whole", "half"])
def test_a_run_uses_no_audit_set_built_with_other_budgets(
    parts, eighth, tmp_path, monkeypatch, config_env
):
    """The audit set's groups were benched in the budgets the shares
    resolved to where it was built: a run on a partition of half the
    device's CUs, or with the eighth turned off, resolves others and refuses
    the whole device's set, complete or not and whatever audit.build says,
    in preflight and, with --skip-preflight, when it starts; the same config
    on the whole device uses it."""
    cfg = _load(CONFIG_DIR / "gfx950_bbs_tn_origami_v3.yaml")
    d = _audit_set(cfg, WHOLE)
    run = cfg if eighth else _turn_off_the_eighth(cfg)
    run_cfg, _ = ctxlib.resolve_fractions(run, WHOLE // parts)
    paths = ro.derive_paths(run_cfg)
    monkeypatch.setattr(ro, "audit_status", lambda cfg, paths: ("ok", None))
    errs = ro._audit_build_errors(run_cfg, paths)
    pipeline = types.SimpleNamespace(
        cfg=run_cfg,
        paths=paths,
        run_dir=tmp_path / "run",
        get=lambda key: ro.cfg_get(run_cfg, key),
    )
    args = types.SimpleNamespace(skip_preflight=True, allow_code_change=False)
    build = ro.AuditBuild(pipeline, args)
    if parts == 1 and eighth:
        assert errs == [] and build.prepare() == 0
        return
    was, now = (
        f"[{', '.join(ctx.label for ctx in _trained(c, n))}]"
        for c, n in ((cfg, WHOLE), (run, WHOLE // parts))
    )
    (err,) = errs
    assert err.startswith(f"{d} was built with contexts={was}; this config has {now}")
    assert err.endswith("point audit.dir at a new audit set for other audit settings)")
    assert build.prepare() == 2
    monkeypatch.setattr(ro, "audit_status", lambda cfg, paths: ("incomplete", "x"))
    off = dict(run_cfg, audit=dict(run_cfg["audit"], build="off"))
    assert ro._audit_build_errors(off, paths) == errs
