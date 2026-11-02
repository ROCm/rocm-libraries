# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the grouped / wide HC stage functions (``stages2.hc``, true-mono WP2a fix round): the grouped verify
builds ``r<rows>g<S>`` (K = S - 1 at CONC rows / S). Up to 4 rows a grouped build is the seq build's layout and code
under its own symbol; past 4 rows it is ATOM's wide build (``layout_wide.for_wide``, ``hc_wide.a2``, the inverse-RMS
hand-off from 12 rows). The GPU gates run in ``hackathon-prep/qwen38/truemono/WP2a-fix/insitu_hc2.py``."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

pytest.importorskip("rocke")

from kernels.gfx950.qwen38_mono.stages2 import hc as H  # noqa: E402

FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")
GROUPED = ("r4g2", "r6g2", "r8g2", "r6g3", "r9g3", "r12g3", "r8g4", "r12g4", "r16g4")
WIDE = tuple(v for v in GROUPED if H.parse_variant(v)[0] > 4)
KINDS = ("delta", "delta_ple", "qsa")


def _spec(stage, kind="delta", v="r8g4", **kw):
    rows, seq, group = H.parse_variant(v)
    return H.HcSpec(stage, kind, rows, seq, H.HcKnobs(**kw), group=group)


def _ir(spec):
    return H.lower_hc_function(spec, flavor=FLAVOR).ir


def _atom(mod):
    root = os.environ.get("ATOM_TRUEMONO_ROOT")
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

    try:
        return _import_atom(mod, root)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no ATOM checkout ({e})")


def test_parse_variant():
    assert H.parse_variant("r1") == (1, False, 0)
    assert H.parse_variant("r4s") == (4, True, 0)
    assert H.parse_variant("r16g4") == (16, False, 4)
    with pytest.raises(ValueError):
        H.parse_variant("r4x")


def test_wide_lds_layout_matches_atom():
    lw = _atom("atom.models.qwen4_exp_mono.kernels.layout_wide")
    lay = _atom("atom.models.qwen4_exp_mono.kernels.layout")
    assert H.WIDE_ROWS == lw.WIDE_ROWS and H.M4_SLICES_WIDE == lw.M4_SLICES
    for v in GROUPED:
        rows, _, g = H.parse_variant(v)
        L = lw.for_wide(rows, g)
        mine = H.hc_lds_wide(rows, g)
        assert {n: L.lds[n] for n in H.HC_LDS} == mine.lds, v
        assert {n: L.stride[n] for n in H.HC_LDS} == mine.stride, v
        assert L.lds_words == mine.words, v
        # the tasks the extern A1 / A2 assume (the caller passes the task; A1 runs rows x 14, A2 80)
        assert L.tasks["a1"] == L.tasks["m1"] == rows * H.TASKS_HC and L.tasks["a2"] == H.TASKS_UP, v
    assert lay.MAX_ROWS == 4


def test_atom_hc_constants():
    hc = _atom("atom.models.qwen4_exp_mono.kernels.stages.hc")
    cw = _atom("atom.models.qwen4_exp_mono.kernels.stages.common_wide")
    assert (hc.INV_ROWS, hc.INV_TID) == (H.INV_ROWS, H.INV_TID)
    assert cw.PRE_POLL_MAX == H.PRE_POLL_MAX


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", GROUPED)
def test_every_grouped_variant_lowers(kind, v):
    for st in H.HC_STAGES:
        spec = _spec(st, kind, v)
        assert H.supported(spec) is None, (st, kind, v, H.supported(spec))
        fn = H.lower_hc_function(spec, flavor=FLAVOR)
        assert fn.symbol == f"rocke_q38_{st}_{v}"
        assert fn.lds_bytes == 4 * H.spec_lds(spec).words


@pytest.mark.parametrize("kind", KINDS)
def test_grouped_narrow_is_the_seq_code(kind):
    """r4g2: ATOM's grouped layout up to 4 rows is the seq build's, and the HC never reads the group: the function is
    the r4s function under its own symbol, byte for byte."""
    for st in H.HC_STAGES:
        for kw in (dict(), dict(numa=1), dict(sgb=1, ra=1, scale="opsel")):
            g = _ir(_spec(st, kind, "r4g2", **kw))
            s = _ir(H.HcSpec(st, kind, 4, True, H.HcKnobs(**kw)))
            assert g == s.replace(f"rocke_q38_{st}_r4s", f"rocke_q38_{st}_r4g2"), (st, kind, kw)


def test_refusals():
    assert H.supported(_spec("a1", v="r8g4", numa=1))  # ATOM drops numa above 4 rows
    assert H.supported(_spec("a1", v="r8g4", pre="args"))  # no a1_preload in wide builds
    assert H.supported(_spec("a1", v="r8g4", spread=2))  # no spread in grouped builds
    assert H.supported(_spec("a1", v="r4g2", spread=10))
    assert H.supported(H.HcSpec("a1", "delta", 8, False, H.HcKnobs()))  # ungrouped past 4 rows
    assert H.supported(H.HcSpec("a1", "delta", 8, True, H.HcKnobs(), group=4))  # seq + group
    assert H.supported(H.HcSpec("a1", "delta", 6, False, H.HcKnobs(), group=4))  # rows not a multiple
    assert H.supported(H.HcSpec("a1", "delta", 4, False, H.HcKnobs(), group=4))  # one group
    assert H.supported(H.HcSpec("a1", "delta", 20, False, H.HcKnobs(), group=4))  # past WIDE_ROWS
    assert H.supported(_spec("a1", v="r4g2", pre="args")) is None  # up to 4 rows a1_preload exists
    assert H.supported(_spec("a1", v="r4g2", numa=1)) is None


def test_inverse_rms_handoff_from_12_rows():
    """From INV_ROWS rows A1 / M1's task 0 puts the row's inverse RMS (``inv`` / ``minv``) and A2 / M2 poll it instead
    of rebuilding every row's statistics: A2 at 12+ rows loads no norm rows (memory-read HC) and polls no ``hs``."""
    from kernels.gfx950.qwen38_mono import abi as A

    def puts(ir):
        return ir.count("store")

    a1_9, a1_12 = _ir(_spec("a1", "delta", "r9g3")), _ir(_spec("a1", "delta", "r12g3"))
    assert puts(a1_12) > puts(a1_9)
    a2_9, a2_12 = _ir(_spec("a2", "delta", "r9g3")), _ir(_spec("a2", "delta", "r12g3"))
    assert "@llvm.sqrt.f64" in a2_9 and "@llvm.sqrt.f64" not in a2_12  # no inverse RMS rebuilt in A2
    assert A.SC["inv"] != A.SC["minv"]


def test_wide_epilogue_passes():
    """Past WAVES rows A2's epilogue runs in passes of WAVES rows: two passes at 9-16 rows, one at 6-8."""
    def xe_puts(ir):
        return len(re.findall(r"llvm\.amdgcn\.(?:raw\.ptr\.)?buffer\.store", ir))

    one, two = _ir(_spec("a2", "delta", "r8g4")), _ir(_spec("a2", "delta", "r16g4"))
    assert xe_puts(two) > xe_puts(one)


@pytest.mark.parametrize("kw", [dict(ra=2), dict(sgb=1, ra=1), dict(scale="opsel"), dict(scale="hoist"), dict(w_cpol=18),
                                dict(nw_cpol=2), dict(amax="fmax"), dict(shfl4="select"), dict(anchor=0),
                                dict(mailbox=H.M.MailboxKnobs(impl="asm")), dict(pause_sleep=4), dict(acc=2)])  # fmt: skip
@pytest.mark.parametrize("stage", ["a1", "a2", "m1", "m2"])
def test_wide_knobs_act(kw, stage):
    if stage == "a1" and "pause_sleep" in kw:
        pytest.skip("the delta layer's attention A1 reads memory: no poll, no pause")
    base = _ir(_spec(stage, "delta", "r16g4"))
    assert _ir(_spec(stage, "delta", "r16g4", **kw)) != base, (stage, kw)


def test_wide_wait_knob_is_inert():
    """Past 4 rows FlyDSL's payload wait is always the pre-poll (``payload_wait``): the ``wait`` knob has no effect."""
    for st in ("m1", "m2"):
        assert _ir(_spec(st, "delta", "r8g4")) == _ir(_spec(st, "delta", "r8g4", wait="prepoll"))


def test_mutants_act_on_wide():
    for m, st in (("down_part", "a1"), ("pack_scale", "m1"), ("inj_lane", "a1"), ("up_kit", "a2"), ("mix_avg", "m2"),
                  ("eps", "a1")):  # fmt: skip
        assert _ir(_spec(st, "qsa", "r12g4", mutate=m)) != _ir(_spec(st, "qsa", "r12g4")), (m, st)
    # A2's own inverse RMS (and so ``eps`` there) exists below INV_ROWS rows only; from 12 rows it is A1's
    assert _ir(_spec("a2", "qsa", "r9g3", mutate="eps")) != _ir(_spec("a2", "qsa", "r9g3"))
    assert _ir(_spec("a2", "qsa", "r12g4", mutate="eps")) == _ir(_spec("a2", "qsa", "r12g4"))


def test_writer_and_cli(tmp_path):
    H.main(["--out", str(tmp_path), "--kinds", "delta", "--variants", "r4g2,r8g4", "--knobs", "numa=1,pre=args"])
    names = sorted(p.name for p in (tmp_path / "delta").iterdir() if p.suffix == ".ll")
    assert names == sorted(f"{st}_{v}.ll" for st in H.HC_STAGES for v in ("r4g2", "r8g4"))
    import json

    meta = json.load(open(tmp_path / "delta" / "a1_r8g4.json"))
    assert "pre_args" not in meta and "mailboxes" not in meta  # the CLI drops pre / numa above 4 rows
    meta = json.load(open(tmp_path / "delta" / "a1_r4g2.json"))
    assert meta.get("pre_args") == "a1_preload" and meta.get("mailboxes") == "numa1"


@pytest.mark.skipif(not (os.environ.get("ROCKE_LLVM_AS") or shutil.which("llvm-as")), reason="no llvm-as")
@pytest.mark.parametrize("kw", [dict(), dict(ra=2, sgb=1, scale="opsel", w_cpol=18), dict(inv="regs", a2order="h_first"),
                                dict(stats="halves", hbar=0, scale="hoist")])  # fmt: skip
def test_wide_ir_is_valid_llvm(tmp_path, kw):
    tool = os.environ.get("ROCKE_LLVM_AS") or shutil.which("llvm-as")
    for kind in KINDS:
        for v in ("r4g2", "r8g4", "r9g3", "r12g4", "r16g4"):
            for st in H.HC_STAGES:
                spec = _spec(st, kind, v, **kw)
                if H.supported(spec):
                    continue
                p = tmp_path / "x.ll"
                p.write_text(_ir(spec))
                r = subprocess.run([tool, str(p), "-o", os.devnull], capture_output=True, text=True)
                assert r.returncode == 0, (st, kind, v, kw, r.stderr[:400])


@pytest.mark.parametrize("kind", KINDS)
def test_framework_grouped_kernels_build(kind):
    """The grouped HC bodies as framework stages (Path B): r4g2 and r8g4, within one CU's LDS."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.lds_arena import CU_LDS_WORDS_GFX950

    from kernels.gfx950.qwen38_mono import stage_adapt as SA
    from kernels.gfx950.qwen38_mono.framework import LayerShape, build_layer

    for shape in (LayerShape(rows=4, group=2), LayerShape(rows=8, group=4)):
        for run in (("a1", "a2"), ("m1", "m2")):
            f = {st: H.framework_factory(st) for st in run}
            k = build_layer(kind, shape, stages=run, factories=f, knobs=SA.adapted_knobs())
            ll = lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
            assert SA.lds_words(ll) <= CU_LDS_WORDS_GFX950, (kind, shape, run)


def test_narrow_unchanged_spot():
    """The fix round's edit leaves the narrow functions alone (the full identity gate is
    ``WP2a-fix/hc_ident.py``: 2462 narrow functions byte-identical): the narrow spec has group 0 and its old name."""
    s = H.HcSpec("a1", "qsa", 4, True, H.HcKnobs(pre="args"))
    assert s.group == 0 and s.variant == "r4s" and not s.wide
    assert replace(s, group=0) == s
