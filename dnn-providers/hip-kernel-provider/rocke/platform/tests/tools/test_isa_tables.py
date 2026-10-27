# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx11 instruction description tool (`tools/isa_tables.py`): the
checked-in description and generated Rust tables are current, `check` fails
when they are not, `switch` is idempotent and keeps line endings, and
`import` follows the additions other changes make to the Rust sources."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM / "tools"))

import isa_tables

CRATE = PLATFORM / "rust" / "rocke-codeobj"
RIR = [
    "mir.rs",
    "asm.rs",
    "vopd.rs",
    "post.rs",
    "mod.rs",
    "isel.rs",
    "gen_opcodes.rs",
    "gen_isa.rs",
    "gen_burs.rs",
]


def copy_crate(dst: Path) -> Path:
    """The files the tool reads and writes, in a scratch crate."""
    (dst / "src" / "rir").mkdir(parents=True)
    shutil.copytree(CRATE / "isa", dst / "isa")
    for name in RIR:
        src = CRATE / "src" / "rir" / name
        if src.exists():
            shutil.copyfile(src, dst / "src" / "rir" / name)
    return dst


def run(crate: Path, *args: str) -> int:
    return isa_tables.main(["--crate", str(crate), *args])


def test_checked_in_description_and_tables_are_current():
    assert run(CRATE, "check") == 0


def test_description_round_trips():
    tree = isa_tables.Tree(CRATE)
    text = tree.read(isa_tables.DESCRIPTION)
    secs = isa_tables.parse(text, "gfx11.isa")
    assert isa_tables.write_description(secs) == text


def test_gfx950_description_and_table_are_current():
    assert run(CRATE, "check", "--isa", "gfx950") == 0


def test_gfx950_check_fails_when_its_table_is_stale(tmp_path):
    crate = copy_crate(tmp_path / "crate")
    gen = crate / isa_tables.GFX950_GEN_ISA
    gen.parent.mkdir(parents=True)
    shutil.copyfile(CRATE / isa_tables.GFX950_GEN_ISA, gen)
    assert run(crate, "check", "--isa", "gfx950") == 0
    original = gen.read_bytes()
    gen.write_bytes(original + b"// edited\n")
    assert run(crate, "check", "--isa", "gfx950") == 1
    assert run(crate, "generate", "--isa", "gfx950") == 0
    assert gen.read_bytes() == original
    # The gfx950 commands leave the gfx11 description and tables alone.
    assert run(crate, "check") == 0


def test_gfx950_table_covers_only_its_rows():
    text = (CRATE / isa_tables.GFX950_GEN_ISA).read_text()
    assert "use crate::rir::mir::{Fmt, Opc};" in text and "        _ => None," in text
    desc_text = isa_tables.Tree(CRATE).read(isa_tables.GFX950_DESCRIPTION)
    secs = isa_tables.parse(desc_text, "gfx950.isa")
    names = {r.args[0] for r in secs["inst"] if r.kind == "inst"}
    assert "VAddNcU32" in names and not names & {
        "VDual",
        "SAndSaveexecB32",
        "VMovB32Dpp8",
    }
    rows = {r.args[0]: r for r in secs["inst"] if r.kind == "inst"}
    # gfx9's no-carry add, and lane masks as SGPR pairs.
    assert (rows["VAddNcU32"].args[1], rows["VAddNcU32"].args[3]) == (
        "v_add_u32",
        "0x34",
    )
    assert "sdst:s2" in rows["VAddCoU32"].get("d")
    assert all(r.get("op") for r in secs["vcmp"])


def test_gfx9_own_opcodes_are_the_opc_gfx9_family():
    desc_text = isa_tables.Tree(CRATE).read(isa_tables.GFX950_DESCRIPTION)
    secs = isa_tables.parse(desc_text, "gfx950.isa")
    own = [
        r.args[0] for r in secs["inst"] if r.kind == "inst" and r.get("opc") == "gfx9"
    ]
    assert "VPkMulF32" in own and "BufferInv" in own
    gfx9 = (CRATE / isa_tables.GFX950_GEN_ISA).read_text()
    assert "pub enum Gfx9Op {" in gfx9
    for name in own:
        assert f"Opc::Gfx9(Gfx9Op::{name})" in gfx9
    # gfx11 gives the family one arm per table that matches every opcode, and
    # its opcode rows and description name none of them.
    gfx11 = (CRATE / isa_tables.GEN_ISA).read_text()
    assert gfx11.count("Opc::Gfx9(_) =>") == 9
    rows = (CRATE / isa_tables.GEN_OPCODES).read_text()
    desc = isa_tables.Tree(CRATE).read(isa_tables.DESCRIPTION)
    assert not any(f" {name} = (" in rows or f" {name} " in desc for name in own)


@pytest.mark.parametrize(
    "old,new",
    [
        # A gfx9-only row without opc=gfx9: no shared opcode of that name.
        (
            b"VPkMulF32         v_pk_mul_f32              VOP3P   0x31   opc=gfx9 ",
            b"VPkMulF32         v_pk_mul_f32              VOP3P   0x31   ",
        ),
        # A shared opcode's twin marked opc=gfx9.
        (
            b"VAddF32           v_add_f32                 VOP2    0x1    ",
            b"VAddF32           v_add_f32                 VOP2    0x1    opc=gfx9 ",
        ),
    ],
)
def test_gfx950_rows_name_a_shared_opcode_or_say_opc_gfx9(tmp_path, old, new):
    crate = copy_crate(tmp_path / "crate")
    desc = crate / isa_tables.GFX950_DESCRIPTION
    data = desc.read_bytes()
    assert data.count(old) == 1
    desc.write_bytes(data.replace(old, new))
    with pytest.raises(isa_tables.DescError, match="opc=gfx9"):
        isa_tables.run_generate_gfx950(isa_tables.Tree(crate))


def test_check_fails_when_a_generated_file_or_the_description_is_stale(tmp_path):
    crate = copy_crate(tmp_path / "crate")
    assert run(crate, "check") == 0
    gen = crate / isa_tables.GEN_ISA
    original = gen.read_bytes()
    gen.write_bytes(original + b"// edited\n")
    assert run(crate, "check") == 1
    assert run(crate, "generate") == 0
    assert gen.read_bytes() == original
    overlay = crate / isa_tables.OVERLAY
    overlay.write_bytes(
        overlay.read_bytes().replace(
            b'lat salu "scalar ALU"', b'lat salu "scalar ALU, edited"'
        )
    )
    assert run(crate, "check") == 1
    assert run(crate, "import") == 0 and run(crate, "generate") == 0
    assert run(crate, "check") == 0


def test_generated_files_use_the_line_endings_of_mir_rs(tmp_path):
    crate = copy_crate(tmp_path / "crate")
    newline = isa_tables.Tree(crate).newline(isa_tables.MIR)
    for rel in (isa_tables.DESCRIPTION, isa_tables.GEN_ISA, isa_tables.GEN_OPCODES):
        data = (crate / rel).read_bytes()
        assert data.count(b"\r\n") == (
            data.count(b"\n") if newline == "\r\n" else 0
        ), rel


def unswitch_rows(crate: Path) -> None:
    """mir.rs as it was before `switch`: the rows written out in place."""
    mir = crate / isa_tables.MIR
    text = mir.read_bytes().decode()
    rows = (crate / isa_tables.GEN_OPCODES).read_bytes().decode()
    block = rows[rows.index("opcodes! {") :]
    nl = "\r\n" if "\r\n" in text else "\n"
    text = text.replace(
        'include!("gen_opcodes.rs");',
        block.rstrip().replace("\r\n", "\n").replace("\n", nl),
    )
    mir.write_bytes(text.encode())


def test_import_follows_new_rows_encoder_arms_and_vopd_forms(tmp_path):
    crate = copy_crate(tmp_path / "crate")
    unswitch_rows(crate)
    mir = crate / isa_tables.MIR
    text = mir.read_bytes().decode()
    nl = "\r\n" if "\r\n" in text else "\n"
    # Shapes other changes add: a plain row, a row the hand encoder singles
    # out with a new arm, and a row with a flag combination no class has.
    row = '    VFmacF32 = (Vop2, 0x2b, "v_fmac_f32", VALU | F_TIED),'
    added = (
        nl
        + '    VPlainNewB32 = (Vop2, 0x3a, "v_plain_new_b32", VALU),'
        + nl
        + '    VSpecialNewB32 = (Vop2, 0x3b, "v_special_new_b32", VALU),'
        + nl
        + '    VOddNewB32 = (Vop1, 0x3c, "v_odd_new_b32", VALU | F_VMEM_LOAD),'
    )
    assert row in text
    mir.write_bytes(text.replace(row, row + added).encode())
    asm = crate / isa_tables.ASM
    text = asm.read_bytes().decode()
    head = "fn encode_legacy(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {"
    anchor = text.index("match op.fmt() {", text.index(head)) + len("match op.fmt() {")
    text = (
        text[:anchor]
        + nl
        + "        Fmt::Vop2 if op == Opc::VSpecialNewB32 => e.push(0),"
        + text[anchor:]
    )
    asm.write_bytes(text.encode())
    assert run(crate, "import") == 0
    desc = isa_tables.load_desc(isa_tables.Tree(crate))
    insts = {r.args[0]: r for r in desc.insts}
    assert insts["VPlainNewB32"].get("table") is None
    assert insts["VPlainNewB32"].get("lat") == "valu"
    assert insts["VSpecialNewB32"].get("table") == "no"
    assert insts["VOddNewB32"].get("lat").startswith("auto_")
    assert run(crate, "generate") == 0


def test_import_keeps_synthesized_classes_once_switched(tmp_path):
    # After the switch-over post.rs no longer states the classes import
    # synthesized from it; the description carries them.
    crate = copy_crate(tmp_path / "crate")
    path = crate / isa_tables.DESCRIPTION
    text = path.read_bytes().decode()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    fma = next(i for i, line in enumerate(lines) if line.startswith("inst   VFmaF32 "))
    lines[fma] = lines[fma].replace("lat=valu", "lat=auto_6_valu5")
    store = next(
        i for i, line in enumerate(lines) if line.startswith("lat vmem_store ")
    )
    lines.insert(
        store + 1,
        'lat auto_6_valu5 "synthesized by import: no overlay class has this model" sched=6 delay=valu:5',
    )
    path.write_bytes(nl.join(lines).encode())
    assert run(crate, "import") == 0
    assert run(crate, "generate") == 0
    assert run(crate, "check") == 0
    insts = {r.args[0]: r for r in isa_tables.load_desc(isa_tables.Tree(crate)).insts}
    assert insts["VFmaF32"].get("lat") == "auto_6_valu5"


SYNTHETIC = {
    "mod.rs": "pub mod asm;\npub mod fx;\npub mod isel;\n",
    "mir.rs": (
        'opcodes! {\n    // SOP1\n    SMovB32 = (Sop1, 0x00, "s_mov_b32", 0),\n}\n\n'
        'pub fn vopd_name(code: u8) -> &\'static str {\n    match code {\n        3 => "v_dual_mul_f32",\n'
        '        _ => "v_dual_?",\n    }\n}\n'
    ),
    "asm.rs": "fn encode(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {\n    match op.fmt() {}\n}\n",
    "vopd.rs": (
        "pub fn part(inst: &MInst) -> Option<Part> {\n    let (code, y_only) = match inst.op {\n"
        "        Opc::VMulF32 => (3, false),\n        _ => return None,\n    };\n    None\n}\n"
    ),
    "post.rs": (
        "fn result_latency(inst: &MInst) -> u32 {\n    let flags = inst.op.flags();\n    match inst.op.fmt() {\n"
        "        _ if flags & F_WMMA != 0 => 16,\n        _ => 5,\n    }\n}\n\n"
        "fn run() {\n        for inst in insts {\n            let kind = match fmt {\n"
        "                Fmt::Sop1 => AluKind::Salu,\n                _ => AluKind::Other,\n            };\n"
        "            if matches!(fmt, Fmt::Ds) {\n                state.clear();\n            } else if kind != AluKind::Other {\n"
        "            }\n            if kind != AluKind::Other {\n                let latency = match kind {\n"
        "                    AluKind::Trans => 10,\n                    _ => 2,\n                };\n            }\n        }\n}\n\n"
        "fn is_valu(op: Opc) -> bool {\n    matches!(op.fmt(), Fmt::Vop1)\n}\n\n"
        "fn step() {\n    if flags & F_VMEM_LOAD != 0 {\n        state.vm.issue(&inst.defs);\n        state.vm_any = true;\n    }\n"
        "    if flags & (F_LDS | F_SMEM) != 0 {\n        state.lgkm.issue(&inst.defs);\n        state.lgkm_any = true;\n"
        "        if flags & F_SMEM != 0 {\n            state.smem = true;\n        }\n    }\n}\n\n"
        "fn hz() {\n            if inst.op.flags() & F_WMMA != 0 {\n"
        "                let prev = out.iter().rev().find(|i| is_valu(i.op));\n"
        "                let hazard = |p: &MInst| {\n                    p.op.flags() & F_WMMA != 0\n"
        "                };\n            }\n}\n"
    ),
}


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_switch_applies_once_and_keeps_line_endings(tmp_path, newline):
    crate = tmp_path / "crate"
    (crate / "src" / "rir").mkdir(parents=True)
    for name, text in SYNTHETIC.items():
        (crate / "src" / "rir" / name).write_bytes(text.replace("\n", newline).encode())
    assert run(crate, "switch", "--skip", "isel:") == 0
    first = {n: (crate / "src" / "rir" / n).read_bytes() for n in SYNTHETIC}
    for name, data in first.items():
        assert data.count(b"\r\n") == (
            data.count(b"\n") if newline == "\r\n" else 0
        ), name
    assert b'include!("gen_opcodes.rs");' in first["mir.rs"]
    assert b"super::gen_isa::delay_clears(inst.op)" in first["post.rs"]
    assert b"let latency = match kind" not in first["post.rs"]
    assert run(crate, "switch", "--skip", "isel:") == 0
    assert {n: (crate / "src" / "rir" / n).read_bytes() for n in SYNTHETIC} == first
    assert all(
        line.startswith(("applied", "skip"))
        for line in isa_tables.run_status(isa_tables.Tree(crate))
        if "isel" not in line
    )


def test_switch_reports_a_missing_anchor_and_applies_the_rest(tmp_path):
    crate = tmp_path / "crate"
    (crate / "src" / "rir").mkdir(parents=True)
    for name, text in SYNTHETIC.items():
        if name == "post.rs":
            text = text.replace(
                "fn result_latency(inst: &MInst) -> u32",
                "fn latency_of(inst: &MInst) -> u32",
            )
        (crate / "src" / "rir" / name).write_text(text)
    before = (crate / "src" / "rir" / "post.rs").read_bytes()
    assert run(crate, "switch", "--skip", "isel:") == 2
    assert b"latency_of" in (crate / "src" / "rir" / "post.rs").read_bytes()
    assert (
        crate / "src" / "rir" / "post.rs"
    ).read_bytes() != before  # the other post.rs edits applied


def test_condition_evaluator():
    consts = {"F_LDS": 1 << 7, "F_SMEM": 1 << 8, "F_TRANS": 1 << 10}
    f = isa_tables.compile_expr(
        "flags & (F_LDS | F_SMEM) != 0 && !matches!(inst.op, Opc::DsBpermuteB32)",
        consts,
    )
    assert f({"flags": 1 << 7, "fmt": "Fmt::Ds", "op": "Opc::DsLoadB32"})
    assert not f({"flags": 1 << 7, "fmt": "Fmt::Ds", "op": "Opc::DsBpermuteB32"})
    g = isa_tables.compile_expr(
        "matches!(op.fmt(), Fmt::Vop1 | Fmt::Vop2) || op == Opc::VDual", consts
    )
    assert g({"flags": 0, "fmt": "Fmt::Vop2", "op": "Opc::X"}) and g(
        {"flags": 0, "fmt": "Fmt::Vopd", "op": "Opc::VDual"}
    )
    with pytest.raises(isa_tables.DescError):
        isa_tables.compile_expr("flags.count_ones() > 1", consts)
