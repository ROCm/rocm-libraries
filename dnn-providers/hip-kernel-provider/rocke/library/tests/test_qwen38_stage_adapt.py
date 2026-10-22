# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hybrid stage-function bodies as rocKE layer-kernel stages
(``kernels.gfx950.qwen38_mono.stage_adapt``, framework ``per_cta`` stages).

CPU: every registered stage function (q2, b3, m3..m5: the tracks' bodies,
unchanged) builds and lowers inside framework layer kernels of each kind
it belongs to, at rows 1 and the widest seq variant, within one CU's LDS;
the self-placed MoE bodies refuse a non-ATOM placement; a ``per_cta`` stage
calls its emitter once per CTA and layer, unguarded.

GPU (gfx950): a probe body registered in place of q2 writes the ABI values
the adapter hands it (tag, task, layer, step pointer, the QSA lengths
``l0..l3``, the scratch argument) per CTA; they must equal the mono's
(``launch_tag``, ATOM's task map, ``qsa_len``). The real bodies compile to
code objects (comgr) on the node.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from kernels.gfx950.qwen38_mono import stage_adapt as SA
from kernels.gfx950.qwen38_mono.atom_layout import BASE, ORDERS, TASKS
from kernels.gfx950.qwen38_mono.framework import (
    LayerShape,
    ScratchLayout,
    StageSpec,
    build_layer,
    build_layer_kernel,
)
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.lds_arena import CU_LDS_WORDS_GFX950
from rocke.helpers.task_schedule import Stage, TaskTable

FLAVOR = "llvm22"


def _layout():
    d = os.environ.get("Q38_ATOM_DIR")
    if not d or not os.path.isdir(d):
        pytest.skip("Q38_ATOM_DIR not set (the mono layout comes from the ATOM tree)")
    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    return from_atom(d)


def _kinds(stage):
    return [k for k in ("delta", "qsa", "delta_ple") if stage in ORDERS[k]]


def _shapes():
    return [LayerShape(), LayerShape(rows=4, seq=True)]


@pytest.mark.parametrize("stage", ["q2", "b3", "m3", "m4", "m5"])
def test_stage_fn_bodies_build_in_layer_kernels(stage):
    lay = _layout()
    have = SA.available_stage_fns()
    if stage not in have:
        pytest.skip(f"{stage} stage function not importable: {SA.import_errors()}")
    f = SA.stage_fn_factory(stage, lay)
    for kind in _kinds(stage):
        for shape in _shapes():
            k = build_layer(kind, shape, stages=(stage,), factories={stage: f},
                            knobs=SA.adapted_knobs())  # fmt: skip
            ll = lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
            assert SA.lds_words(ll) <= CU_LDS_WORDS_GFX950, (stage, kind, shape)
            assert k.name.endswith("_lmin0")
            if stage == "q2":
                # q2's norm takes an f64 square root (Python engine only):
                # reported, so the layer kernel is routed to the Python engine
                assert "math.sqrt.f64" in uses_python_only_ops(k), (kind, shape)


def test_failed_stage_module_import_is_reported(monkeypatch):
    """A stage module that fails to import is not dropped silently: a
    RuntimeWarning names it, ``import_errors`` keeps it and the factory's
    KeyError quotes it."""
    monkeypatch.setattr(SA, "_EXTRA_MODULES", SA._EXTRA_MODULES + ("stage_does_not_exist",))
    monkeypatch.setattr(SA, "_IMPORT_ERRORS", {})
    with pytest.warns(RuntimeWarning, match="stage_does_not_exist failed to import"):
        SA.available_stage_fns()
    assert "stage_does_not_exist" in SA.import_errors()
    assert "ModuleNotFoundError" in SA.import_errors()["stage_does_not_exist"]
    with pytest.raises(KeyError, match="stage_does_not_exist failed to import"):
        SA.stage_fn_factory("no_such_stage", None)


def test_moe_suffix_run_builds_and_fits():
    lay = _layout()
    if not {"m3", "m4", "m5"} <= set(SA.available_stage_fns()):
        pytest.skip("MoE stage functions not importable")
    facts = {n: SA.stage_fn_factory(n, lay) for n in ("m3", "m4", "m5")}
    for kind in ("delta", "qsa"):
        for shape in _shapes():
            k = build_layer(kind, shape, stages=("m3", "m4", "m5"), factories=facts,
                            knobs=SA.adapted_knobs())  # fmt: skip
            ll = lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
            assert SA.lds_words(ll) <= CU_LDS_WORDS_GFX950


def test_self_placed_bodies_refuse_other_placements():
    lay = _layout()
    if "m4" not in SA.available_stage_fns():
        pytest.skip("MoE stage functions not importable")
    f = SA.stage_fn_factory("m4", lay)
    with pytest.raises(ValueError, match="atom placement only"):
        f("delta", LayerShape(), Stage("m4", TASKS["m4"], base=(BASE["m4"] + 1) % 256))
    f("delta", LayerShape(), Stage("m4", TASKS["m4"], base=BASE["m4"]))
    with pytest.raises(KeyError, match="no stage function"):
        SA.stage_fn_factory("nope", lay)


def _dummy(kind, shape, st):
    """Stand-in for a stage no track has written yet: polls one pair of every
    region it reads, puts one pair into every region it writes."""
    from kernels.gfx950.qwen38_mono import atom_layout as A
    from rocke.helpers.lds_arena import LdsEntry

    def emit(b, ctx, p, t):
        with b.scf_if(b.cmp_eq(ctx.wave, b.const_i32(0))):
            specs = [(ctx.mb(r, local=True), 0, 1) for r in st.reads if r in A.REGIONS]
            if specs:
                ctx.poll(specs)
            with b.scf_if(b.cmp_eq(ctx.lane, b.const_i32(0))):
                for r in st.writes:
                    if r in A.REGIONS:
                        ctx.put(ctx.mb(r), t, t)

    return StageSpec(st, emit, lds=(LdsEntry(f"lds_{st.name}", 64),))


@pytest.mark.parametrize("kind", ["delta", "delta_ple", "qsa"])
@pytest.mark.parametrize("stamps", [False, True], ids=["plain", "tl"])
def test_whole_layer_from_track_emitters_is_resident(kind, stamps):
    """A whole layer per kind: the gdn-b3 track's native framework B3
    (``stage_b3.b3_stage_spec``), the hybrid q2 / m3..m5 bodies through the
    adapter, stand-ins for the rest; padded to one CTA per CU within the
    LDS (``resident_build``)."""
    from functools import partial

    from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs
    from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

    lay = _layout()
    have = set(SA.available_stage_fns())
    facts = {n: _dummy for n in ORDERS[kind]}
    for n in ("q2", "m3", "m4", "m5"):
        if n in have and n in ORDERS[kind]:
            facts[n] = SA.stage_fn_factory(n, lay)
    if "b3" in ORDERS[kind]:
        try:
            from kernels.gfx950.qwen38_mono import stage_b3 as S

            facts["b3"] = lambda k, sh, st: S.b3_stage_spec(
                S.B3Shape(sh.rows, sh.seq, sh.group), base=st.base, phase=st.lds_phase)
        except Exception:  # noqa: BLE001 - another track's module mid-edit
            pass
    for shape in _shapes():
        build = partial(build_layer, kind, shape, factories=facts)
        k, w = SA.resident_build(lambda kn: build(knobs=kn), FrameworkKnobs(stamps=stamps),
                                 flavor=FLAVOR)  # fmt: skip
        assert ONE_CTA_PER_CU_WORDS <= w <= CU_LDS_WORDS_GFX950, (kind, shape, w)
        assert k.name.startswith(f"rk_q38_{kind}_")


def test_per_cta_stage_runs_once_per_layer_unguarded():
    calls = []

    def emit(b, ctx, p, t):
        calls.append((ctx.li, t))
        b.global_store(p.ptr(b, "h_out"), ctx.bid, t)

    st = Stage("s0", 8, base=3)
    table = TaskTable((st,))
    k = build_layer_kernel("pc", [StageSpec(st, emit, per_cta=True)], table,
                           ScratchLayout.of({}, parity_slots=2), layers=2)  # fmt: skip
    assert [c[0] for c in calls] == [0, 1]
    ll = lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
    assert ll.count("store i32") >= 2
    with pytest.raises(ValueError, match="tasks > grid"):
        big = Stage("s0", 300)
        build_layer_kernel("pc2", [StageSpec(big, emit, per_cta=True)], TaskTable((big,), grid=256),
                           ScratchLayout.of({}))  # fmt: skip


# ------------------------------------------------------------------ GPU


def _gpu():
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    prob = H.env_problem()
    if prob:
        pytest.skip(prob)
    from rocke.runtime.hip_module import get_device_arch

    if (get_device_arch() or "").split(":")[0] != "gfx950":
        pytest.skip("needs gfx950")
    return H


@pytest.fixture(scope="module")
def dev():
    H = _gpu()
    d = H.Dev(watchdog_s=120)
    yield d
    d.close()


PROBE_WORDS = 10


def _probe_body(b, a, layout, spec):
    """Writes what it was handed (CTA ``bid``: words ``[10 bid, 10 bid + 10)``
    of ``h_out``)."""
    from rocke.core.ir import I32
    from rocke.helpers import device_ops as dops

    with b.scf_if(b.cmp_eq(a.tid, b.const_i32(0))):
        bid = b.readfirstlane(b.block_id_x())
        ptr = dops.ptr_from_i64(b, a.h_out)
        vals = [a.tag, a.task, a.layer, b.trunc(a.step, I32), a.l0, a.l1, a.l2, a.l3,
                b.trunc(a.scratch, I32), b.add(a.lane, a.wave)]  # fmt: skip
        for i, v in enumerate(vals):
            b.global_store(ptr, b.add(b.mul(bid, b.const_i32(PROBE_WORDS)), b.const_i32(i)), v)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [LayerShape(rows=2), LayerShape(rows=4, seq=True)],
                         ids=["r2", "r4s"])  # fmt: skip
def test_gpu_adapter_hands_the_mono_abi_values(dev, shape, monkeypatch):
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B
    from kernels.gfx950.qwen38_mono import stage_fn as SF
    from rocke.helpers import mailbox as mbx

    lay = _layout()
    SA.available_stage_fns()
    monkeypatch.setitem(SF.STAGES, "q2", (_probe_body, lambda s, l: True))
    f = SA.stage_fn_factory("q2", lay)
    k = build_layer("qsa", shape, stages=("q2",), factories={"q2": f},
                    knobs=SA.adapted_knobs())  # fmt: skip
    H = _gpu()
    art, used = H.compile_for(k, "gfx950")
    fn = dev.function(art)
    n = len(lay.qsa_state)
    pos = lay.state_index("pos")
    lens = [1000 + 37 * q for q in range(4)]
    lbufs = [dev.put(np.array([v], np.int32)) for v in lens]
    tab = np.zeros(4 * n, np.int64)
    for q in range(4):
        tab[q * n + pos] = lbufs[q]
    qsa = dev.put(tab)
    out = dev.alloc(256 * PROBE_WORDS * 4)
    step = dev.put(np.array([77], np.int32))
    scratch = dev.alloc(1 << 20)
    layer = 5
    dev.launch(fn, 256, 512, B.pack_args(step, layer, qsa=qsa, h_out=out, scratch=scratch))
    dev.sync()
    got = dev.get(out, 256 * PROBE_WORDS * 4, np.int32).reshape(256, PROBE_WORDS)
    tag = 77 * mbx.LAYER_SLOTS + layer + 1
    if shape.seq:
        want_ls = [lens[0] + q for q in range(shape.rows)]
    else:
        want_ls = lens[: shape.rows]
    want_ls += [0] * (4 - len(want_ls))
    for bid in range(256):
        row = got[bid].tolist()
        assert row[0] == tag and row[2] == layer
        assert row[1] == (bid + 256 - BASE["q2"]) & 255
        assert (row[3] & 0xFFFFFFFF) == (step & 0xFFFFFFFF)
        assert row[4:8] == want_ls, (bid, row)
        assert (row[8] & 0xFFFFFFFF) == (scratch & 0xFFFFFFFF)
        assert row[9] == 0  # lane 0 of wave 0


@pytest.mark.gpu
@pytest.mark.parametrize("stages,kind", [(("q2",), "qsa"), (("b3",), "delta"),
                                         (("m3", "m4", "m5"), "delta")],
                         ids=["q2", "b3", "moe"])  # fmt: skip
def test_gpu_adapted_bodies_compile(dev, stages, kind):
    lay = _layout()
    if not set(stages) <= set(SA.available_stage_fns()):
        pytest.skip("stage functions not importable")
    H = _gpu()
    facts = {n: SA.stage_fn_factory(n, lay) for n in stages}
    for shape in _shapes():
        k = build_layer(kind, shape, stages=stages, factories=facts, knobs=SA.adapted_knobs())
        art, used = H.compile_for(k, "gfx950")
        assert art.hsaco
        if uses_python_only_ops(k):
            assert used == "python"
        dev.function(art)  # the code object loads
