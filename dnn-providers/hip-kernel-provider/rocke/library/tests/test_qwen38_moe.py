# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Qwen3.8 mono MoE stages in rocKE (``kernels.gfx950.qwen38_mono.moe``) and
their ABI mirror (``abi``) and stage functions (``stage_moe``). CPU only.

* ``abi``: the mailbox regions, offsets, weight-block sizes and the mono MoE
  placement against the ATOM model code when it is importable (``Q38_ATOM_DIR``
  or an installed ``atom``), else against pinned values.
* plans: every knob combination the sweep uses validates or is refused with
  a reason; the work model covers every unit exactly once (host simulation of
  the ``mono`` and ``balanced`` maps for every union size).
* builds: every (rows, knobs) build lowers at llvm22 and llvm23; kernel names
  name only deviating knobs and carry ``_pyonly``; the knob-off build is the
  same IR whatever the off-knobs' spelling; the Python-only ops used are the
  expected ones; LDS fits the mono arena per variant.
* stage functions: the ABI and symbols the ATOM side checks; ``m4`` alone is
  refused; the pairing sidecar fields; the serving set (``--knobs best``: per
  layer kind, only the cells that beat the mono); the mono LDS arena check
  refuses to run blind.
* golden: sha256 of the default-knob builds (suffix rows 1 / 2 / 4, stage
  functions r1 / r2s / r4s): knobs off = the mono's schedule, byte for byte.

The GPU checks (bitwise vs the FlyDSL stages on the node) are
``hackathon-prep/qwen38/moe/moe_ab.py check`` (track harness).
"""

import importlib.util
import json
import os
import sys

import pytest
from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import moe as MO
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVORS = ("llvm22", "llvm23")


def _atom_layout():
    root = os.environ.get("Q38_ATOM_DIR")
    try:
        if root:
            from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

            return _import_atom("atom.models.qwen4_exp_mono.kernels.layout", root)
        if importlib.util.find_spec("atom") is None:
            return None
        from atom.models.qwen4_exp_mono.kernels import layout

        return layout
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------- abi


def test_abi_pinned_values():
    assert A.SCRATCH_BYTES == 1274880
    assert A.ROW_PAIRS == 159360
    assert A.SC["lg"] % 256 == 0 and A.SC["dn"] % 256 == 0
    assert A.DN_ROW == 180 and A.DN_X8 == 160
    assert A.GU_W_BYTES == 2 * 48 * 20 * 1024 and A.GU_S_BYTES == 48 * 10 * 256
    assert A.DN_W_BYTES == 160 * 6 * 1024 and A.DN_S_BYTES == 80 * 3 * 256
    assert len(A.LAYER_ARGS) == 41
    want = {
        1: ({"m3": 65, "m4": 110, "m5": 160}, {"m3": 190, "m4": 156, "m5": 0}),
        2: ({"m3": 130, "m4": 210, "m5": 160}, {"m3": 204, "m4": 156, "m5": 0}),
        3: ({"m3": 195, "m4": 155, "m5": 160}, {"m3": 218, "m4": 156, "m5": 0}),
        4: ({"m3": 130, "m4": 205, "m5": 160}, {"m3": 232, "m4": 156, "m5": 0}),
    }
    for rows, (tasks, base) in want.items():
        L = A.mono_moe_layout(rows)
        assert L.tasks == tasks and L.base == base


def test_abi_matches_atom():
    ly = _atom_layout()
    if ly is None:
        pytest.skip("ATOM layout not importable (set Q38_ATOM_DIR)")
    assert A.REGIONS == ly.REGIONS
    assert A.SC == ly.SC and A.SCRATCH_BYTES == ly.SCRATCH_BYTES
    for rows in range(1, 5):
        L, M = ly.for_rows(rows, seq=rows > 1), A.mono_moe_layout(rows)
        for s in ("m3", "m4", "m5"):
            assert (L.tasks[s], L.base[s]) == (M.tasks[s], M.base[s]), (rows, s)
        assert (L.m3_task_rows, L.m4_slices) == (M.m3_task_rows, M.m4_slices)


# ----------------------------------------------------------------- plans

KNOB_SETS = [
    {},
    {"dedupe": True},
    {"dedupe": True, "union_order": "first"},
    {"dedupe": True, "m4_map": "balanced"},
    {"dedupe": True, "m4_waves": 4, "m4_map": "balanced"},
    {"m4_waves": 4, "m4_map": "balanced"},
    {"dedupe": True, "m5_issue": "early"},
    {"dedupe": True, "m5_issue": "late"},
    {"m4_map": "balanced", "m4_j": 2},
    {"dedupe": True, "shared_first": True},
    {"shared_first": True},
    {"w_cpol": 2},
    {"m3_ept": 4, "m3_rows": 4},
    # codegen / schedule levers (bitwise: same arithmetic in the same order)
    {"m5_ranks": "pre"},
    {"m5_ranks": "pre", "a_pre": True},
    {"m4_guard": "none"},
    {"m4_guard": "none", "m4_interleave": True},
    {"m4_guard": "none", "m4_interleave": True, "m4_apipe": 1, "sched_issue": True},
    {"m5_ranks": "pre", "slots": "arith"},
    {"m5_ranks": "pre", "slots": "arith", "shared_first": True, "m5_extras": "first"},
    {"m5_extras": "m4"},
    {"dedupe": True, "m4_waves": 4, "m4_map": "balanced", "m5_ranks": "pre", "a_pre": True,
     "m4_guard": "none", "m4_interleave": True, "m4_apipe": 1, "m5_extras": "m4"},
    {"shared_early": True},
    {"shared_early": True, "m5_ranks": "pre", "slots": "arith", "m4_waves": 4, "m4_map": "balanced"},
    {"dedupe": True, "shared_first": True, "shared_early": True},
    {"dedupe": True, "m4_base": 108, "m5_issue": "early"},
    {"route_pause": "s_sleep1"},
    {"route_gate": "pre_poll"},
    {"route_gate": "pre_poll", "m5_extras": "first", "m5_ranks": "pre", "slots": "arith", "shared_early": True},
    {"m4_weights": "late"},
    {"m4_weights": "late", "dedupe": True, "m4_waves": 4, "m4_map": "balanced", "m5_extras": "m4"},
]


def _units_by_cta(plan, n_units):
    """Host model of :class:`moe._Unit`: the units each CTA task runs."""
    G, J = plan.groups, plan.j
    out = []
    for task in range(plan.m4_tasks_max):
        got = []
        for k in range(G * J):
            if plan.knobs.m4_map == "mono":
                u = task * G * J + k
                ok = u < n_units
            else:
                start, end = task * n_units >> 8, (task + 1) * n_units >> 8
                u = start + k
                ok = u < end
            if ok:
                got.append(u)
        out.append(got)
    return out


@pytest.mark.parametrize("knobs", KNOB_SETS, ids=lambda k: json.dumps(k, sort_keys=True))
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_plan_covers_every_unit_once(rows, knobs):
    if knobs.get("m3_rows", 0) > rows or rows % max(knobs.get("m3_rows", 1), 1):
        pytest.skip("m3_rows does not divide rows")
    plan = MO.MoePlan(rows, 512, MO.MoeKnobs(**knobs), MO.STAGES)
    plan.validate()
    lo = A.TOP_K + 1 if rows == 1 else A.TOP_K + 1  # U >= 10
    for n_slots in range(lo, plan.max_slots + 1):
        if not knobs.get("dedupe") and n_slots != plan.max_slots:
            continue
        n_units = n_slots * plan.units_per_slot
        seen = [u for cta in _units_by_cta(plan, n_units) for u in cta]
        assert sorted(seen) == list(range(n_units)), (rows, n_slots)
    assert plan.entries <= 64


def test_plan_mono_map_is_mono_tasks():
    for rows in range(1, 5):
        plan = MO.MoePlan(rows, 512, MO.MoeKnobs(), MO.STAGES)
        assert plan.m4_tasks_max == A.mono_moe_layout(rows).tasks["m4"]
        assert plan.j == A.mono_moe_layout(rows).m4_slices


def test_plan_refusals():
    with pytest.raises(ValueError):
        MO.MoeKnobs(m4_waves=2)
    with pytest.raises(ValueError):
        MO.MoeKnobs(compute="mfma_scaled_32x32x64")
    with pytest.raises(ValueError):
        MO.MoePlan(4, 512, MO.MoeKnobs(m4_map="balanced", m4_j=1), MO.STAGES).validate()
    with pytest.raises(ValueError):
        MO.MoePlan(4, 512, MO.MoeKnobs(), ("m5",)).validate()
    for bad in (
        {"m5_ranks": "x"},
        {"m4_guard": "x"},
        {"m4_interleave": True},  # needs m4_guard=none
        {"m4_guard": "none", "m4_apipe": 1},  # needs m4_interleave
        {"slots": "arith"},  # needs m5_ranks=pre
        {"slots": "arith", "m5_ranks": "pre", "dedupe": True},
        {"m5_extras": "x"},
        {"shared_early": True, "dedupe": True},  # needs shared_first + mono map
        {"shared_early": True, "dedupe": True, "shared_first": True, "m4_map": "balanced"},
        {"route_pause": "x"},
        {"route_gate": "x"},
        {"m4_weights": "x"},
    ):
        with pytest.raises(ValueError):
            MO.MoeKnobs(**bad)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("knobs", [
    {"shared_early": True},
    {"shared_early": True, "shared_first": True},
    {"shared_early": True, "m4_waves": 4, "m4_map": "balanced"},
    {"shared_early": True, "dedupe": True, "shared_first": True},
], ids=lambda k: json.dumps(k, sort_keys=True))
def test_shared_early_tasks_are_shared_only(rows, knobs):
    """Host model of ``_m4_shared_only``: a task is early iff every unit it runs
    is the shared expert's (for every union size when deduped)."""
    plan = MO.MoePlan(rows, 512, MO.MoeKnobs(**knobs), MO.STAGES)
    plan.validate()
    ups = plan.units_per_slot
    sh = 0 if knobs.get("shared_first") else rows * A.TOP_K
    for n_slots in range(A.TOP_K + 1, plan.max_slots + 1):
        if not knobs.get("dedupe") and n_slots != plan.max_slots:
            continue
        n_units = n_slots * ups
        model_n = 10**6 if knobs.get("dedupe") else plan.max_slots * ups
        for task, units in enumerate(_units_by_cta(plan, n_units)):
            gj = plan.groups * plan.j
            if plan.knobs.m4_map == "mono":
                first, end = task * gj, min(task * gj + gj, model_n)
            else:
                first, end = task * model_n >> 8, (task + 1) * model_n >> 8
            early = first >= sh * ups and end <= (sh + 1) * ups and first < end
            if early:
                assert units and all(sh * ups <= u < (sh + 1) * ups for u in units), (task, units)
        # the shared slot's units all land on early tasks when not deduped
        if not knobs.get("dedupe"):
            n_early = sum(
                1 for t in range(plan.m4_tasks_max)
                if all(sh * ups <= u < (sh + 1) * ups for u in _units_by_cta(plan, n_units)[t])
                and _units_by_cta(plan, n_units)[t]
            )
            assert n_early >= 1


# ----------------------------------------------------------------- builds


@pytest.mark.parametrize("knobs", KNOB_SETS, ids=lambda k: json.dumps(k, sort_keys=True))
@pytest.mark.parametrize("rows", [1, 4])
def test_build_lowers(rows, knobs):
    if knobs.get("m3_rows", 0) > rows:
        pytest.skip("m3_rows > rows")
    spec = MO.MoeSuffixSpec(rows=rows, experts=512, knobs=MO.MoeKnobs(**knobs))
    kern = MO.build_moe_suffix(spec)
    assert kern.name.endswith("_pyonly")
    used = set(uses_python_only_ops(kern))
    assert {"math.exp", "tile.int_to_ptr", "tile.mfma_scale_mx"} <= used
    for fl in FLAVORS:
        ir = _lower_kernel_to_llvm_python(kern, llvm_flavor=fl, arch="gfx950")
        assert "@llvm.exp.f32" in ir and "mfma.scale.f32.16x16x128" in ir
    off, stride, words = MO.lds_layout(spec.plan)
    assert words <= MO.LDS_LIMIT_WORDS


def test_kernel_name_names_only_deviations():
    base = MO.MoeSuffixSpec(rows=4).kernel_name()
    assert base == "qwen38_moe_m3_m4_m5_e512_r4_pyonly"
    dd = MO.MoeSuffixSpec(rows=4, knobs=MO.MoeKnobs(dedupe=True, m4_waves=4, m4_map="balanced")).kernel_name()
    assert dd == "qwen38_moe_m3_m4_m5_e512_r4_dd_w4_mapbalanced_pyonly"
    # every knob value has a distinct name
    names = {MO.MoeSuffixSpec(rows=4, knobs=MO.MoeKnobs(**k)).kernel_name() for k in KNOB_SETS}
    assert len(names) == len(KNOB_SETS)


def test_default_knobs_spelled_out_build_same_ir():
    a = MO.build_moe_suffix(MO.MoeSuffixSpec(rows=4))
    b = MO.build_moe_suffix(
        MO.MoeSuffixSpec(rows=4, knobs=MO.MoeKnobs(
            dedupe=False, m4_waves=8, m4_map="mono", m5_issue="mono", m5_ranks="lds", a_pre=False,
            m4_guard="branch", sched_issue=False, m4_interleave=False, m4_apipe=0, slots="table",
            m5_extras="after", shared_early=False, route_pause="", route_gate="none", m4_weights="route",
        ))  # fmt: skip
    )
    fa = _lower_kernel_to_llvm_python(a, llvm_flavor="llvm23", arch="gfx950")
    fb = _lower_kernel_to_llvm_python(b, llvm_flavor="llvm23", arch="gfx950")
    assert fa == fb


def _comgr_ok():
    try:
        from rocke.runtime import comgr

        return any("gfx950" in i for i in comgr.supported_isas())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("knobs", KNOB_SETS, ids=lambda k: json.dumps(k, sort_keys=True))
def test_build_compiles(knobs):
    """Every knob set code-generates (comgr, gfx950; rows 4 and the rows-1 build of
    the codegen knobs): a lowering that only fails in LLVM (an i1 ballot
    predicate did) is caught on the CPU."""
    if not _comgr_ok():
        pytest.skip("no comgr with gfx950")
    from rocke.helpers.compile import compile_kernel

    for rows in (4, 1):
        if knobs.get("m3_rows", 0) > rows:
            continue
        kern = MO.build_moe_suffix(MO.MoeSuffixSpec(rows=rows, experts=512, knobs=MO.MoeKnobs(**knobs)))
        art = compile_kernel(kern, arch="gfx950", backend="python", capture_ir_text=False)
        assert art.hsaco


def test_stages_m4_m5_build():
    spec = MO.MoeSuffixSpec(rows=4, stages=("m4", "m5"), knobs=MO.MoeKnobs(dedupe=True))
    kern = MO.build_moe_suffix(spec)
    ir = _lower_kernel_to_llvm_python(kern, llvm_flavor="llvm23", arch="gfx950")
    assert "qwen38_moe_m4_m5_e512_r4_dd_pyonly" in ir


# ----------------------------------------------------------------- stage functions


def _arena_args():
    """Generator arguments for the mono LDS arena check: ATOM's layout when importable, else explicit."""
    if _atom_layout() is not None:
        return []
    return ["--no-arena-check"]


def test_stage_functions_abi(tmp_path, monkeypatch):
    from kernels.gfx950.qwen38_mono import stage_moe as SM
    from kernels.gfx950.qwen38_mono.stage_fn import ABI_PARAMS

    if _atom_layout() is None:
        monkeypatch.setattr(SM, "mono_lds_words", lambda rows, seq: MO.LDS_LIMIT_WORDS)
    rc = SM.main(["--out", str(tmp_path), "--variants", "r1,r4s", "--flavor", "llvm23",
                  "--knobs", '{"dedupe": true}'])  # fmt: skip
    assert rc == 0
    pairs = set()
    for stage in SM.MOE_STAGES:
        for v in ("r1", "r4s"):
            meta = json.loads((tmp_path / f"{stage}_{v}.json").read_text())
            assert meta["symbol"] == f"rocke_q38_{stage}_{v}"
            assert meta["abi"] == [list(p) for p in ABI_PARAMS] + [["lds_base", "i32"]]
            assert meta["knobs"]["inline"] == "always"
            ir = (tmp_path / f"{stage}_{v}.ll").read_text()
            assert f"define void @rocke_q38_{stage}_{v}(" in ir
            assert "alwaysinline" in ir
            assert "amdgpu_kernel" not in ir
            # pairing (ATOM extern_stages refuses m4 / a reusing m5 without its partner)
            req = meta.get("requires")
            if stage == "m4":
                assert req == {"m5": None}
            elif stage == "m5":
                assert req == {"m4": meta["pair"]}
            else:
                assert req is None
            pairs.add((v, meta["pair"]))
    assert len(pairs) == 2  # one generation id per variant
    # m5 alone routes for itself: no requirement
    rc = SM.main(["--out", str(tmp_path / "m5"), "--stages", "m5", "--variants", "r4s", "--flavor", "llvm23"])
    assert rc == 0
    assert "requires" not in json.loads((tmp_path / "m5" / "m5_r4s.json").read_text())
    with pytest.raises(SystemExit):
        SM.main(["--out", str(tmp_path), "--stages", "m4", "--variants", "r1", "--flavor", "llvm23"])


def test_arena_check_does_not_run_blind(tmp_path, monkeypatch):
    """Without ATOM the generator refuses unless told the arena or told to skip
    the check (the check is what keeps a set past the arena from NaN)."""
    import builtins

    from kernels.gfx950.qwen38_mono import stage_moe as SM

    real = builtins.__import__

    def no_atom(name, *args, **kw):
        if name.startswith("atom"):
            raise ImportError(name)
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_atom)
    monkeypatch.delenv("Q38_ATOM_DIR", raising=False)
    with pytest.raises(RuntimeError, match="needs ATOM"):
        SM.mono_lds_words(4, True)
    with pytest.raises(RuntimeError, match="needs ATOM"):
        SM.build_function("m3", 1, False, MO.MoeKnobs())
    args = ["--out", str(tmp_path), "--stages", "m3", "--variants", "r1", "--flavor", "llvm23"]
    with pytest.raises(SystemExit, match="needs ATOM"):
        SM.main(args)
    assert SM.main(args + ["--no-arena-check"]) == 0
    assert SM.main(args + ["--mono-lds-words", str(MO.LDS_LIMIT_WORDS)]) == 0
    with pytest.raises(ValueError, match="LDS arena"):
        SM.main(args + ["--mono-lds-words", "16"])


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_best_knobs_build(rows):
    """The tuned per-variant sets validate, fit the LDS limit and lower as stage functions."""
    from kernels.gfx950.qwen38_mono import stage_moe as SM

    kn = SM.best_knobs(rows)
    plan = MO.MoePlan(rows, 512, kn, MO.STAGES)
    plan.validate()
    assert MO.lds_layout(plan)[2] <= MO.LDS_LIMIT_WORDS
    assert kn.mailbox.impl == "ir"
    for stage in SM.MOE_STAGES:
        SM.build_function(stage, rows, rows > 1, kn, m5_reuse=True, arena_words=MO.LDS_LIMIT_WORDS)


def test_serving_set_shape():
    """``BEST``: every kind and variant has an entry; a cell with a function swaps
    m4 only with m5; the serving plan lists exactly the non-``None`` cells."""
    from kernels.gfx950.qwen38_mono import stage_moe as SM

    variants = {"r1", "r2", "r3", "r4", "r2s", "r3s", "r4s"}
    assert set(SM.KINDS) == {"delta", "delta_ple", "qsa"}
    n = 0
    for kind in SM.KINDS:
        assert set(SM.BEST[kind]) == variants
        for v, entry in SM.BEST[kind].items():
            if entry is None:
                continue
            n += 1
            stages, kn = entry
            assert set(stages) <= set(SM.MOE_STAGES) and stages
            assert "m4" not in stages or "m5" in stages
            SM.knobs_from_json(kn)
    assert len(SM.serving_plan()) == n


def test_serving_set_files(tmp_path, monkeypatch):
    """``--knobs best`` writes ``<stage>_<kind>_<variant>`` for the serving cells only (a ``None``
    cell keeps the FlyDSL stages through ATOM_Q38_EXTERN_MISSING=mono), with the pairing fields."""
    from kernels.gfx950.qwen38_mono import stage_moe as SM

    if _atom_layout() is None:
        monkeypatch.setattr(SM, "mono_lds_words", lambda rows, seq: MO.LDS_LIMIT_WORDS)
    rc = SM.main(["--out", str(tmp_path), "--variants", "r1,r2s,r4s", "--flavor", "llvm23", "--knobs", "best"])
    assert rc == 0
    want = {f"{s}_{k}_{v}.ll" for k, v, st, _kn in SM.serving_plan(variants={"r1", "r2s", "r4s"}) for s in st}
    assert {p.name for p in tmp_path.glob("*.ll")} == want
    for kind in SM.KINDS:
        if SM.BEST[kind]["r2s"] is None:
            assert not list(tmp_path.glob(f"*_{kind}_r2s.ll"))
    for f in tmp_path.glob("m5_*.json"):
        meta = json.loads(f.read_text())
        m4 = f.with_name(f.name.replace("m5_", "m4_", 1))
        if m4.exists():
            assert meta["requires"] == {"m4": json.loads(m4.read_text())["pair"]}
            assert meta["kind"] in SM.KINDS


def test_stage_function_refuses_lds_past_the_mono_arena(monkeypatch):
    """A stage function's LDS is the mono kernel's arena (``lds="arg"``): a knob
    set whose arena does not fit is refused at build time (it ran with NaN in
    ``h_out`` on the GPU before the check)."""
    from kernels.gfx950.qwen38_mono import stage_moe as SM

    kn = MO.MoeKnobs(m3_rows=4, m3_ept=4, m4_weights="late")
    words = MO.lds_layout(MO.MoePlan(4, 512, kn, MO.STAGES))[2]
    monkeypatch.setattr(SM, "mono_lds_words", lambda rows, seq: words - 1)
    with pytest.raises(ValueError, match="LDS arena"):
        SM.build_function("m5", 4, True, kn)
    monkeypatch.setattr(SM, "mono_lds_words", lambda rows, seq: words)
    SM.build_function("m5", 4, True, kn)


# ----------------------------------------------------------------- golden

# sha256 of the default-knob (= the mono's schedule) LLVM text, llvm23 / gfx950. Every MoE knob is
# default-off: a change here means a knob-off build changed. Re-pin only when the change is
# understood (e.g. rocKE core lowering changed for every kernel), with a GPU bitwise re-check.
GOLDEN_SUFFIX = {
    1: "382a7c7a46d4e977",
    2: "a37c5c94ff59d3f6",
    4: "79069c3561936f02",
}
GOLDEN_FUNCTIONS = {
    ("m3", "r1"): "04c8e6669d1e9559", ("m4", "r1"): "e52eb908845b61d7", ("m5", "r1"): "f260b0b9131838e7",
    ("m3", "r2s"): "ddb3d522ca0dcbae", ("m4", "r2s"): "5247e48d61030045", ("m5", "r2s"): "2c509237ec4b9cb8",
    ("m3", "r4s"): "1f70f0057afec014", ("m4", "r4s"): "6f78b5b9f299e4df", ("m5", "r4s"): "e1b16c7e47098c88",
}  # fmt: skip


def golden_hashes():
    """(suffix hashes by rows, stage-function hashes by (stage, variant)) of the default knobs."""
    import hashlib

    from kernels.gfx950.qwen38_mono import stage_moe as SM
    from kernels.gfx950.qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function
    from kernels.gfx950.qwen38_mono.stage_fn import parse_variant

    suf = {}
    for rows in GOLDEN_SUFFIX:
        ir = _lower_kernel_to_llvm_python(MO.build_moe_suffix(MO.MoeSuffixSpec(rows=rows)), llvm_flavor="llvm23",
                                          arch="gfx950")  # fmt: skip
        suf[rows] = hashlib.sha256(ir.encode()).hexdigest()[:16]
    fns = {}
    for stage, v in GOLDEN_FUNCTIONS:
        rows, seq, _g = parse_variant(v)
        kern, _plan = SM.build_function(stage, rows, seq, MO.MoeKnobs(), m5_reuse=True, arena_words=None)
        fn = lower_device_function(kern, arch="gfx950", symbol=f"rocke_q38_{stage}_{v}", llvm_flavor="llvm23",
                                   knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
        fns[(stage, v)] = fn.sha256()[:16]
    return suf, fns


def test_default_knobs_golden():
    suf, fns = golden_hashes()
    assert suf == GOLDEN_SUFFIX
    assert fns == GOLDEN_FUNCTIONS


if __name__ == "__main__":
    if sys.argv[1:] == ["--print-golden"]:
        print(golden_hashes())
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-q"]))
