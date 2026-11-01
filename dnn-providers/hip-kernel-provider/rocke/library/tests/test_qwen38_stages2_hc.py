# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the WP2a HC stage functions (``kernels.gfx950.qwen38_mono.stages2.hc``).

The GPU gates (bitwise vs the FlyDSL stages in situ, must-fail mutants, timing) run in the Qwen3.8 harness
(``hackathon-prep/qwen38/truemono/WP2a/insitu_hc.py``)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace

import pytest

pytest.importorskip("rocke")

from rocke.helpers import mailbox as M  # noqa: E402

from kernels.gfx950.qwen38_mono.stages2 import hc as H  # noqa: E402

FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")
VARIANTS = ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s")
KINDS = ("delta", "delta_ple", "qsa")


def _spec(stage, kind="delta", v="r1", **kw):
    k = H.HcKnobs(**kw)
    if k.pre == "args" and (stage != "a1" or kind == "delta_ple"):
        k = replace(k, pre="none")
    return H.HcSpec(stage, kind, int(v[1]), v.endswith("s"), k)


def _ir(spec):
    return H.lower_hc_function(spec, flavor=FLAVOR).ir


def _atom_layout():
    root = os.environ.get("ATOM_TRUEMONO_ROOT")
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

    try:
        return _import_atom("atom.models.qwen4_exp_mono.kernels.layout", root)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no ATOM checkout ({e})")


def test_lds_layout_matches_atom():
    lay = _atom_layout()
    for rows in (1, 2, 3, 4):
        for seq in ((False, True) if rows > 1 else (False,)):
            L = lay.for_rows(rows, seq)
            mine = H.hc_lds(rows, seq)
            assert {n: L.lds[n] for n in H.HC_LDS} == mine.lds
            assert {n: L.stride[n] for n in H.HC_LDS} == mine.stride
            assert (L.lds_words, L.split) == (mine.words, mine.split)
    assert lay.SPREAD_OFF == H.SPREAD_OFF and lay.INJ_SEG == H.INJ_SEG
    for v in H.SPREAD_VALUES:
        assert lay.spread_mode(v) == H.spread_mode(v)
        for r in (1, 2, 3, 4):
            assert lay.spread_down_tasks(v, r) == H.spread_down_tasks(v, r)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", VARIANTS)
def test_every_narrow_variant_lowers(kind, v):
    for st in H.HC_STAGES:
        spec = _spec(st, kind, v)
        assert H.supported(spec) is None
        fn = H.lower_hc_function(spec, flavor=FLAVOR)
        assert fn.symbol == f"rocke_q38_{st}_{v}"
        assert fn.abi[-1] == ("lds_base", "i32") and fn.lds_bytes == 4 * H.hc_lds(spec.rows, spec.seq).words
        assert "alwaysinline" in fn.ir and "convergent" in fn.ir


def test_ir_deterministic_and_default_spelled_out():
    a = _ir(_spec("a1"))
    assert a == _ir(_spec("a1"))
    explicit = H.HcKnobs(mailbox=M.MailboxKnobs(impl="ir"), ra=0, scale="ldsstep", w_cpol=0, acc=1, spread=0)
    assert a == _ir(H.HcSpec("a1", "delta", 1, False, explicit))


KNOB_STAGE = [
    (dict(ra=1), "a1"), (dict(ra=2), "m1"), (dict(sgb=1), "a1"), (dict(scale="hoist"), "a1"),
    (dict(scale="opsel"), "a2"), (dict(w_cpol=2), "m1"), (dict(wi_cpol=2), "a1"), (dict(nw_cpol=16), "a1"),
    (dict(a2order="h_first"), "a2"), (dict(acc=2), "a1"), (dict(wait="prepoll"), "m1"), (dict(pause_sleep=4), "m2"),
    (dict(numa=1), "a2"), (dict(mailbox=M.MailboxKnobs(impl="asm")), "m2"), (dict(mailbox=M.MailboxKnobs(impl="irf")), "m1"),
    (dict(pre="args"), "a1"), (dict(spread=10), "a1"), (dict(spread=2), "a2"), (dict(stats="halves"), "m1"),
    (dict(hbar=0), "a1"), (dict(sgb=1, ra=1), "m1"), (dict(anchor=0), "a1"), (dict(anchor=0), "m2"),
    (dict(amax="fmax"), "a2"), (dict(amax="fmax"), "a1"), (dict(shfl4="select"), "m1"), (dict(inv="regs"), "a1"),
    (dict(inv="regs"), "m2"),
] + [(dict(mutate=m), st) for m, st in (("eps", "a1"), ("down_part", "m1"), ("pack_scale", "a1"), ("inj_lane", "a1"),
                                         ("up_kit", "a2"), ("mix_avg", "m2"))]  # fmt: skip


@pytest.mark.parametrize("kw,stage", KNOB_STAGE)
def test_every_knob_acts(kw, stage):
    """Every knob value changes the IR of a stage it applies to (no silently ignored knob)."""
    assert _ir(_spec(stage, **kw)) != _ir(_spec(stage))


def test_sgb_with_read_ahead_differs_from_sgb_alone():
    assert _ir(_spec("a1", sgb=1, ra=1)) != _ir(_spec("a1", sgb=1))


def test_quarter_mutant_acts_on_spread10():
    assert _ir(_spec("a1", spread=10, mutate="quarter")) != _ir(_spec("a1", spread=10))


def test_inapplicable_knobs_are_inert():
    """Knobs of another stage change nothing (they never reach that stage's code)."""
    base = _ir(_spec("m2"))
    assert _ir(_spec("m2", a2order="h_first")) == base  # a memory-read HC only
    assert _ir(_spec("m2", wi_cpol=2)) == base  # the injection rows are A1 / M1's
    assert _ir(_spec("a1", stats="halves")) == _ir(_spec("a1"))  # a memory-read A1 computes its sums itself
    assert _ir(_spec("m1", hbar=0)) == _ir(_spec("m1"))  # a polled HC has no such barrier


@pytest.mark.parametrize("kw", [dict(), dict(ra=1, scale="opsel", numa=1), dict(spread=10, numa=1, pre="args")])
def test_knobs_spec_round_trip(kw):
    k = H.HcKnobs(**kw)
    assert H.knobs_from_spec(H.knobs_spec(k)) == k


def test_tolerance_class_flag():
    assert not H.HcKnobs().tolerance and not H.HcKnobs(ra=2, scale="opsel").tolerance
    assert H.HcKnobs(acc=2).tolerance


def test_refusals():
    assert "pre=args" in H.supported(H.HcSpec("a2", "delta", 1, False, H.HcKnobs(pre="args")))
    assert "pre=args" in H.supported(H.HcSpec("a1", "delta_ple", 1, False, H.HcKnobs(pre="args")))
    assert "tasks" in H.supported(H.HcSpec("a1", "delta", 4, True, H.HcKnobs(spread=1)))  # 336 > 256
    assert "rows" in H.supported(H.HcSpec("a1", "delta", 8, False))
    with pytest.raises(ValueError):
        H.HcKnobs(ra=3)
    with pytest.raises(ValueError):
        H.HcKnobs(w_cpol=4)


def test_pre_args_abi():
    fn = H.lower_hc_function(_spec("a1", pre="args"), flavor=FLAVOR)
    names = [n for n, _ in fn.abi]
    assert names[-25:-1] == [f"pre{i}" for i in range(H.PRE_WORDS)]


def test_writer_sidecars(tmp_path):
    specs = [_spec("a1", "qsa", "r1", numa=1, spread=10, pre="args"), _spec("m2", "qsa", "r1", numa=1)]
    paths = H.write_hc_functions(str(tmp_path), specs, flavor=FLAVOR)
    assert [os.path.basename(p) for p in paths] == ["a1_r1.ll", "m2_r1.ll"]
    meta = json.load(open(tmp_path / "qsa" / "a1_r1.json"))
    assert meta["mailboxes"] == "numa1" and meta["pre_args"] == "a1_preload" and meta["hc"]["spread"] == 10
    assert H.knobs_from_spec(meta["stage_knobs"]) == specs[0].knobs
    m2 = json.load(open(tmp_path / "qsa" / "m2_r1.json"))
    assert "pre_args" not in m2 and m2["hc"]["spread"] == 0


@pytest.mark.skipif(not (os.environ.get("ROCKE_LLVM_AS") or shutil.which("llvm-as")), reason="no llvm-as")
@pytest.mark.parametrize("kw", [dict(), dict(spread=1, numa=1), dict(spread=2), dict(spread=4), dict(spread=10, pre="args"),
                                dict(inv="regs", spread=10), dict(inv="regs", stats="halves", wait="prepoll", spread=2),
                                dict(ra=2, sgb=1, scale="opsel", w_cpol=18, acc=2)])  # fmt: skip
def test_ir_is_valid_llvm(tmp_path, kw):
    tool = os.environ.get("ROCKE_LLVM_AS") or shutil.which("llvm-as")
    for kind in KINDS:
        for v in ("r1", "r3s", "r4s"):
            for st in H.HC_STAGES:
                spec = _spec(st, kind, v, **kw)
                if H.supported(spec):
                    continue
                p = tmp_path / "x.ll"
                p.write_text(_ir(spec))
                r = subprocess.run([tool, str(p), "-o", os.devnull], capture_output=True, text=True)
                assert r.returncode == 0, (st, kind, v, kw, r.stderr[:400])


@pytest.mark.parametrize("kind", KINDS)
def test_framework_layer_kernels_build(kind):
    """The HC bodies as framework stages (Path B layers=1 / WP3): rows 1 and r4s, every kind, within one CU's LDS."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.lds_arena import CU_LDS_WORDS_GFX950

    from kernels.gfx950.qwen38_mono import stage_adapt as SA
    from kernels.gfx950.qwen38_mono.framework import LayerShape, build_layer

    for shape in (LayerShape(), LayerShape(rows=4, seq=True)):
        for run in (("a1", "a2"), ("m1", "m2")):
            f = {st: H.framework_factory(st) for st in run}
            k = build_layer(kind, shape, stages=run, factories=f, knobs=SA.adapted_knobs())
            ll = lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
            assert SA.lds_words(ll) <= CU_LDS_WORDS_GFX950, (kind, shape, run)
    with pytest.raises(ValueError):
        H.framework_factory("a1", H.HcKnobs(pre="args"))


def test_stage_fn_registry():
    from kernels.gfx950.qwen38_mono import stage_fn as SF

    for st in H.HC_STAGES:
        assert st in SF.STAGES
