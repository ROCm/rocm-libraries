# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.qwen38_mono.qsa2`` (true-mono WP2c: the mono QSA layer's Q3-Q7 as rocKE stage functions).
CPU only: knobs, layout arithmetic, the emitted LLVM text (the ``q38.*`` lowering, the ABI, default stability,
mutants and levers changing the text). The bitwise proofs against the FlyDSL stages run on gfx950 in the q38
container (``hackathon-prep/qwen38/truemono/WP2c/ab_qsa.py``, ``t0_qsa.py``, ``bench_qsa.py``).

The layout comes from an ATOM checkout: ``ATOM_ROOT`` (default ``../../../../../ATOM-truemono`` next to
copy-rocm-lib); the module is skipped without one.
"""

from __future__ import annotations

import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ATOM_ROOT = os.environ.get(
    "ATOM_ROOT", os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "..", "ATOM-truemono"))
)
if not os.path.isdir(os.path.join(ATOM_ROOT, "atom", "models", "qwen4_exp_mono")):
    pytest.skip(f"no ATOM checkout at {ATOM_ROOT}", allow_module_level=True)

from kernels.gfx950.qwen38_mono.qsa2 import build as QB  # noqa: E402
from kernels.gfx950.qwen38_mono.qsa2 import layout as QL  # noqa: E402
from kernels.gfx950.qwen38_mono.qsa2.knobs import MUTANTS, QsaKnobs  # noqa: E402
from kernels.gfx950.qwen38_mono.stage_fn import ABI_PARAMS  # noqa: E402

FLAVOR = "llvm23"


def _lay(rows=1, seq=False):
    return QL.from_atom(rows, seq, ATOM_ROOT)


def _ir(stage, rows=1, seq=False, knobs=QsaKnobs()):  # noqa: B008
    return QB.lower_stage(stage, rows, seq, knobs, lay=_lay(rows, seq), flavor=FLAVOR).ir


def _canon(text):
    """SSA names stripped (the lowerer's counters move when an unrelated op is added)."""
    return re.sub(r"%[\w.$-]+", "%v", text)


# ------------------------------------------------------------------ knobs


def test_knob_defaults_and_validation():
    k = QsaKnobs()
    assert k.diff() == {} and k.suffix() == "" and not k.tolerance
    assert QsaKnobs(qdense=3).tolerance and QsaKnobs(fuse="tile").tolerance
    for bad in ({"pause": "x"}, {"qdense": 4}, {"fuse": "flash"}, {"q4_chunks": 12}, {"mailbox_impl": "dma"},
                {"q3_switch": 3}, {"prepoll": "q4"}, {"mutate": "nope"}):  # fmt: skip
        with pytest.raises(ValueError):
            QsaKnobs(**bad)


def test_knob_parse_roundtrip():
    k = QsaKnobs.parse("numa=1,qdense=3,put_nt=True,prepoll=q5+q7,q4_chunks=16")
    assert (k.numa, k.qdense, k.put_nt, k.prepoll_skip, k.q4_chunks) == (1, 3, True, ("q5", "q7"), 16)
    assert QsaKnobs.parse(k.as_dict()) == k
    assert QsaKnobs.parse({kk: str(v) for kk, v in k.as_dict().items()}) == k


# ------------------------------------------------------------------ layout


def test_layout_matches_atom_literals():
    lay = _lay()
    assert lay.grid == 256 and lay.threads == 512 and lay.waves == 8
    assert (lay.p_ik, lay.p_q, lay.p_k, lay.p_v) == (256, 320, 6464, 6720)
    assert (lay.sel, lay.group, lay.q7_rounds, lay.q4_tasks, lay.chunks) == (512, 12, 17, 9, 8)
    assert lay.tasks["q5"] == 130 and lay.tasks["q7"] == 48 and lay.tasks["q6"] == 24
    assert lay.fsc["fcand"] + 32 * lay.sel * 2 * lay.pair <= lay.sc["isc"]  # q4_chunks=32 fits
    ql = QL.q4_lds(lay.sel, lay.threads)
    assert ql["LDS_WORDS"] <= 5120  # layout's q4_keys entry
    s4 = _lay(4, True)
    assert s4.tasks["q5"] == 520 and s4.task_rounds["q5"] == 3 and s4.tasks["q4"] == 36


# ------------------------------------------------------------------ IR


@pytest.mark.parametrize("stage", QB.STAGES)
@pytest.mark.parametrize("v", ["r1", "r4s"])
def test_every_stage_lowers_with_the_abi(stage, v):
    rows, seq = int(v[1]), v.endswith("s")
    fn = QB.lower_stage(stage, rows, seq, lay=_lay(rows, seq), flavor=FLAVOR)
    assert fn.symbol == f"rocke_q38_{stage}_{v}"
    assert [n for n, _ in fn.abi] == [n for n, _ in ABI_PARAMS] + ["lds_base"]
    assert "alwaysinline" in fn.ir and "convergent" in fn.ir
    assert "@llvm.maxnum" not in fn.ir and "@llvm.fmuladd" not in fn.ir  # FlyDSL's maximum / fma


def test_q38_ops_lowered():
    ir5 = _ir("q5")
    assert "@llvm.fma.f32(" in ir5 and "@llvm.amdgcn.update.dpp.i32(" in ir5
    ir3 = _ir("q3")
    assert "@llvm.fma.f64(" in ir3 and "@llvm.maximum.f32(" in ir3 and "@llvm.sqrt.f64(" in ir3
    ir4 = _ir("q4")
    assert " ashr i32 " in ir4 and "phi i32" in ir4  # q38.bin / q38.ifr
    # the default mailbox: FlyDSL's shape (buffer loads / stores with sc1, the poll's retry block)
    assert "fpoll.retry" in ir5 and "raw.ptr.buffer.store" in ir5 and "global_load_dwordx2" not in ir5


def test_default_text_is_stable_under_off_knobs():
    """Knobs at their defaults give the default text; a lever changes it."""
    for st in QB.STAGES:
        base = _ir(st)
        assert _ir(st, knobs=QsaKnobs.parse({})) == base
    assert _ir("q4", knobs=QsaKnobs(q4_chunks=16)) != _ir("q4")
    assert _ir("q3", knobs=QsaKnobs(q3_switch=0)) != _ir("q3")
    assert _ir("q5", knobs=QsaKnobs(q5_kpre=True)) != _ir("q5")
    assert _ir("q7", knobs=QsaKnobs(prepoll="q7")) != _ir("q7")
    asm = _ir("q6", knobs=QsaKnobs(mailbox_impl="asm"))
    assert "global_load_dwordx2" in asm and "fpoll.retry" not in asm


def test_numa_mailboxes():
    ir = _ir("q7", knobs=QsaKnobs(numa=1))
    assert "HW_REG_XCC_ID" in ir  # the IOD of the CTA picks the local copy
    calls = "call void @llvm.amdgcn.raw.ptr.buffer.store"
    assert ir.count(calls) == 2 * _ir("q7").count(calls)  # both copies of every put


# mutant -> (stage, base knobs)
MUTANT_CELLS = {
    "q5_scale": ("q5", {}), "q5_cur": ("q5", {}), "q6_den": ("q6", {}), "q7_gate": ("q7", {}),
    "q7_tail": ("q7", {}), "q3_relu": ("q3", {}), "q3_key_row": ("q3", {}), "q4_tie": ("q4", {}),
    "q4_rank": ("q4", {}), "qdense_drop": ("q7", {"qdense": 3}), "splitkv_norescale": ("q7", {"fuse": "splitkv"}),
    "splitkv_drop": ("q7", {"fuse": "splitkv"}), "fuse_den": ("q7", {"fuse": "single"}),
    "tile_norescale": ("q7", {"fuse": "tile"}),
}  # fmt: skip


def test_every_mutant_has_a_cell():
    assert set(MUTANT_CELLS) == set(MUTANTS)


@pytest.mark.parametrize("m", sorted(MUTANTS))
def test_every_mutant_changes_its_stage(m):
    stage, kw = MUTANT_CELLS[m]
    a = _ir(stage, knobs=QsaKnobs(**kw))
    b = _ir(stage, knobs=QsaKnobs(**kw, mutate=m))
    assert a != b, m  # (raw text: a mutant may only swap an operand)


def test_fused_q6_idles():
    ir = _ir("q6", knobs=QsaKnobs(fuse="tile"))
    assert "raw.ptr.buffer" not in ir  # no poll, no put
