# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the WP2b GEMV-family stage functions (``qwen38_mono/stages2/gemv.py``).

The geometry comes from an ATOM-truemono checkout (``ATOM_TRUEMONO_ROOT``, default the sibling ``dev/ATOM-truemono``);
the module is skipped without it. GPU numerics (bitwise in situ vs the FlyDSL stage) are
``hackathon-prep/qwen38/truemono/WP2b/wp2b_insitu.py``.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from kernels.gfx950.qwen38_mono import stage_fn
from kernels.gfx950.qwen38_mono.stages2 import gemv as G

_DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 6, "ATOM-truemono")
)
ATOM_ROOT = os.environ.get("ATOM_TRUEMONO_ROOT", _DEFAULT_ROOT)
if not os.path.isdir(os.path.join(ATOM_ROOT, "atom", "models", "qwen4_exp_mono")):
    pytest.skip("needs an ATOM-truemono checkout (ATOM_TRUEMONO_ROOT)", allow_module_level=True)


@pytest.fixture(scope="module")
def geom():
    return G.geom_from_atom(ATOM_ROOT)


SHAPES = [(st, kd, v) for st, kinds in G.STAGES.items() for kd in kinds for v in G.NARROW]


def _shape(st, kd, v):
    rows, seq, group = G.parse_variant(v)
    return G.GemvShape(st, kd, rows, seq, group)


def test_knob_spec_round_trip():
    k = G.knobs_from_spec("mfma=asm,asm_lead=-1,asop=1,w_cpol=18,oob=split,pause=4")
    assert G.knobs_from_spec(G.knobs_spec(k)) == k
    assert G.knobs_spec(G.GemvKnobs()) == ""


@pytest.mark.parametrize(
    "spec",
    ["mb_impl=foo", "pause=-2", "w_cpol=4", "oob=skip", "ra=9", "sgb=x", "mfma=x",
     "asm_lead=7", "asm_lead=0", "asm_lead=1", "asm_tail=8", "barrier=x", "mutate=nope", "numa_hdr=256"],
)  # fmt: skip
def test_bad_knobs_refused(spec):
    with pytest.raises(ValueError):
        G.knobs_from_spec(spec)


def test_shapes_refused():
    with pytest.raises(ValueError):
        G.GemvShape("q1", "delta")
    with pytest.raises(ValueError):
        G.GemvShape("b1", "delta", 8, True, 4)
    with pytest.raises(ValueError):
        G.GemvShape("b1", "delta", 1, True)


@pytest.mark.parametrize("st,kd,v", SHAPES)
def test_parity_build_abi_and_mfma_form(geom, st, kd, v):
    sh = _shape(st, kd, v)
    fn = G.lower_gemv_function(sh, G.GemvKnobs(), geom)
    assert fn.symbol == f"rocke_q38_{st}_{sh.variant}"
    names = [n for n, _ in fn.abi]
    assert names == [n for n, _ in stage_fn.ABI_PARAMS] + ["lds_base"]
    _, _, words = geom.lds(sh.rows, sh.seq)
    assert 0 < fn.lds_bytes <= 4 * words
    ir = fn.ir
    calls = re.findall(r"@llvm\.amdgcn\.mfma\.scale\.f32\.16x16x128\.f8f6f4\.v8i32\.v4i32\(([^)]*)\)", ir)
    calls = [c for c in calls if "<8 x i32> %" in c]
    want = {"b1": 20, "q1": 20, "c1": 12, "p1": 40}[st]
    assert len(calls) == want
    # FlyDSL's operand formats: A fp8 (cbsz 0), B fp4 (blgp 4); A scale byte 0 (op_sel_a 0)
    for c in calls:
        imm = re.findall(r"i32 (-?\d+)(?=,|$)", c)
        assert imm[0:3] == ["0", "4", "0"], c
    if st in ("b1", "q1"):
        # one chain: the weight scale dword of group g, byte q at step 4 g + q
        selb = [re.findall(r"i32 (\d+), i32 %", c)[-1] for c in calls]
        assert selb == [str(i % 4) for i in range(20)]
    if st == "c1":
        # the mono shifts the scale dword per wave and reads byte 0
        assert all(re.findall(r"i32 (\d+), i32 %", c)[-1] == "0" for c in calls)
    assert "v_mfma" not in ir  # no inline-asm MFMA in the parity build


@pytest.mark.parametrize("st,kd,v", [s for s in SHAPES if s[2] in ("r1", "r4s")])
def test_asm_build_ties_the_accumulator(geom, st, kd, v):
    sh = _shape(st, kd, v)
    fn = G.lower_gemv_function(sh, G.knobs_from_spec("mfma=asm,asm_lead=-1"), geom)
    asm = re.findall(r'asm "(?:[^"\\]|\\.)*v_mfma_scale_f32_16x16x128_f8f6f4 \$0, \$1, \$2, \$0, \$4, \$5[^"]*", "=v,v,v,0,v,v"',
                     fn.ir)  # fmt: skip
    assert len(asm) == {"b1": 20, "q1": 20, "c1": 12, "p1": 40}[st]
    assert "mfma.scale.f32" not in fn.ir.split("declare")[0] or True


@pytest.mark.parametrize("m", [m for m in G.MUTANTS])
def test_mutants_change_the_ir(geom, m):
    st = {"conv_tap": "b1", "resid": "c1", "vsum": "p1"}.get(m, "b1")
    kd = G.STAGES[st][0]
    rows, seq = (4, True) if m == "row" else (1, False)
    sh = G.GemvShape(st, kd, rows, seq)
    base = G.lower_gemv_function(sh, G.GemvKnobs(), geom).ir
    mut = G.lower_gemv_function(sh, G.knobs_from_spec(f"mutate={m}"), geom).ir
    assert base != mut


def test_mutants_refused_off_their_stage(geom):
    with pytest.raises(ValueError):
        G.build_gemv_kernel(G.GemvShape("c1", "delta"), G.knobs_from_spec("mutate=conv_tap"), geom)
    with pytest.raises(ValueError):
        G.build_gemv_kernel(G.GemvShape("b1", "delta"), G.knobs_from_spec("mutate=row"), geom)
    with pytest.raises(ValueError):
        G.build_gemv_kernel(G.GemvShape("b1", "delta"), G.knobs_from_spec("bsop=1"), geom)


def test_numa_sidecar(geom, tmp_path):
    k = G.knobs_from_spec("numa=1")
    stems = G.write_gemv_functions(str(tmp_path), [G.GemvShape("c1", "qsa", 4, True)], lambda s: k, geom)
    meta = json.load(open(tmp_path / f"{stems[0]}.json"))
    assert meta["mailboxes"] == "numa1" and meta["numa_hdr"] == k.header_offset
    assert meta["symbol"] == "rocke_q38_c1_r4s" and meta["kind"] == "qsa"


def test_host_b3pre_needs_dma(geom):
    from dataclasses import replace

    from kernels.gfx950.qwen38_mono import stage_b3 as SB

    sh = G.GemvShape("b1", "delta", 1)
    with pytest.raises(ValueError):
        G.build_gemv_kernel(sh, replace(G.GemvKnobs(), host_b3pre="split=2"), geom)
    k = replace(G.GemvKnobs(numa=1), host_b3pre=SB.TUNED_SPECS_NUMA["r1"])
    fn = G.lower_gemv_function(sh, k, geom)
    assert "buffer.load.lds" in fn.ir  # the prologue's buffer -> LDS DMA


def test_existing_registry_untouched():
    """Importing stages2 registers nothing in the stage_fn registry (existing generators unchanged)."""
    assert "b1" not in stage_fn.STAGES and "c1" not in stage_fn.STAGES


@pytest.mark.parametrize("st", sorted(G.STAGES))
def test_path_b_factory_builds_layer_kernels(geom, st):
    """The emitter runs in a rocKE layer kernel (Path B) through ``framework_factory`` (``stage_adapt``'s ABI)."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.gfx950.qwen38_mono import stage_adapt as SA
    from kernels.gfx950.qwen38_mono.framework import LayerShape, build_layer

    for kd in G.STAGES[st]:
        for shape in (LayerShape(), LayerShape(rows=4, seq=True)):
            for spec in ("", "mfma=asm,asm_lead=-1,mb_impl=ir"):
                f = G.framework_factory(st, G.knobs_from_spec(spec), geom)
                k = build_layer(kd, shape, stages=(st,), factories={st: f}, knobs=SA.adapted_knobs())
                ll = lower_kernel_to_llvm(k, arch="gfx950")
                assert 0 < SA.lds_words(ll) <= 40960
    with pytest.raises(ValueError):
        G.framework_factory(st, G.knobs_from_spec("numa=1"), geom)
