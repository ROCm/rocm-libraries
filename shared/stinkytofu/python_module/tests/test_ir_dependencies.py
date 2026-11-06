# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""stinkytofu.ir.analyze / check_order: the dependency edges behind the IR checkpoint editor."""

import os
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import stinkytofu

# A gfx1250 F8 kernel's bundle from label_openLoopL on. The prologue before it is left out: its
# SRD set-up prints symbolic constants such as BufferOOB, which the parser does not read back.
DATA = Path(__file__).parent / "data" / "ir_checkpoint_f8"

# One edge of each register kind. The comment after each line is its checkpoint id.
KINDS = """\
st.func @kinds() {
  ^entry:
    v1 = "st.v_mov_b32"(v0) { mod.ckpt = { id = 0 } }
    v2 = "st.v_add_f32"(v1, v0) { mod.ckpt = { id = 1 } }
    v0 = "st.v_mov_b32"(1) { mod.ckpt = { id = 2 } }
    v2 = "st.v_mov_b32"(3) { mod.ckpt = { id = 3 } }
    s1, SCC0 = "st.s_add_u32"(s0, 1) { mod.ckpt = { id = 4 } }
    s2 = "st.s_cselect_b32"(s1, 0, SCC0) { mod.ckpt = { id = 5 } }
    vcc_lo0 = "st.v_cmp_eq_u32"(v0, v1) { mod.ckpt = { id = 6 } }
    v3 = "st.v_cndmask_b32"(v4, v5, vcc_lo0) { mod.ckpt = { id = 7 } }
    LDS0 = "st.tensor_load_to_lds"(s[8:11], s[16:23]) { mod.ckpt = { id = 8 } }
    v[6:7] = "st.ds_load_b64"(v10, LDS0) { mod.ds = { na = 1, offset = 0, gds = false }, mod.ckpt = { id = 9 } }
    exec_lo0 = "st.s_mov_b32"(s5) { mod.ckpt = { id = 10 } }
    v8 = "st.v_mov_b32"(v9) { mod.ckpt = { id = 11 } }
    "st.s_endpgm"() { mod.ckpt = { id = 12 } }
}
"""


def edges_of(result):
    return {
        (e["src"], e["dst"], e["kind"], e["class"]): e["regs"] for e in result["edges"]
    }


def swap(text, a, b):
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if f"id = {a} }}" in l)
    j = next(k for k, l in enumerate(lines) if f"id = {b} }}" in l)
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


def test_register_edge_kinds():
    result = stinkytofu.ir.analyze(KINDS)
    edges = edges_of(result)
    assert result["blocks"] == ["entry"]
    assert [i["id"] for i in result["insts"]] == list(range(13))
    assert edges[(0, 1, "raw", "vgpr")] == ["v1"]
    assert edges[(0, 2, "war", "vgpr")] == ["v0"]
    assert edges[(1, 2, "war", "vgpr")] == ["v0"]
    assert edges[(1, 3, "waw", "vgpr")] == ["v2"]
    assert edges[(4, 5, "raw", "sgpr")] == ["s1"]
    assert edges[(4, 5, "raw", "scc")] == ["scc"]
    assert edges[(6, 7, "raw", "vcc")] == ["vcc"]
    assert edges[(8, 9, "raw", "lds")] == ["LDS0"]


def test_instruction_rows():
    insts = {i["id"]: i for i in stinkytofu.ir.analyze(KINDS)["insts"]}
    assert insts[1]["opcode"] == "v_add_f32"
    assert insts[1]["reads"] == ["v1", "v0"] and insts[1]["writes"] == ["v2"]
    assert insts[9]["reads"] == ["v10", "LDS0"]
    assert insts[1]["issue"] > 0 and insts[9]["latency"] > 0
    assert "mod.ckpt" not in insts[1]["text"]
    assert not insts[1]["pinned"]
    assert insts[10]["pin_reason"] == "writes exec"
    assert insts[12]["pin_reason"] == "terminator"


def test_ids_without_ckpt_are_fresh():
    text = KINDS.replace(" { mod.ckpt = { id = 11 } }", "")
    insts = stinkytofu.ir.analyze(text)["insts"]
    assert [i["id"] for i in insts][-2:] == [13, 12]


def test_parse_error_names_the_line():
    with pytest.raises(ValueError, match="line 4"):
        stinkytofu.ir.analyze(
            KINDS.replace('"st.v_add_f32"(v1, v0)', '"st.v_add_f32"(v1,')
        )


def test_check_order_accepts_a_legal_move():
    # The s_cselect and the v_cmp share no register.
    legal = stinkytofu.ir.check_order(KINDS, swap(KINDS, 5, 6))
    assert legal == {"reversed": [], "problems": [], "ok": True}


def test_check_order_reports_reversed_edges():
    result = stinkytofu.ir.check_order(KINDS, swap(KINDS, 4, 5))
    assert not result["ok"]
    assert {(e["src"], e["dst"], e["class"]) for e in result["reversed"]} == {
        (4, 5, "sgpr"),
        (4, 5, "scc"),
    }


def test_check_order_refuses_crossing_a_pinned_line():
    result = stinkytofu.ir.check_order(KINDS, swap(KINDS, 10, 11))
    assert not result["ok"] and result["reversed"] == []
    assert any("crosses id 10" in p and "writes exec" in p for p in result["problems"])


def test_check_order_reports_unknown_and_duplicate_ids():
    result = stinkytofu.ir.check_order(KINDS, KINDS.replace("id = 7 }", "id = 70 }"))
    assert any("unknown id 70" in p for p in result["problems"])
    result = stinkytofu.ir.check_order(KINDS, KINDS.replace("id = 7 }", "id = 6 }"))
    assert any("id 6 appears twice" in p for p in result["problems"])


def test_f8_region():
    stir = (DATA / "region.stir").read_text()
    result = stinkytofu.ir.analyze(stir, [12, 5, 0])
    assert len(result["insts"]) == 248
    assert [i["id"] for i in result["insts"]] == list(range(248))
    assert "label_LoopBeginL" in result["blocks"]
    kinds = Counter(e["kind"] for e in result["edges"])
    for kind in ("raw", "war", "waw"):
        assert kinds[kind] > 0, kind
    classes = Counter(e["class"] for e in result["edges"])
    for cls in ("vgpr", "sgpr", "scc", "lds"):
        assert classes[cls] > 0, cls

    assert stinkytofu.ir.check_order(stir, stir)["ok"]
    lds = next(e for e in result["edges"] if e["class"] == "lds")
    edited = stinkytofu.ir.check_order(stir, swap(stir, lds["src"], lds["dst"]))
    assert lds in edited["reversed"]
