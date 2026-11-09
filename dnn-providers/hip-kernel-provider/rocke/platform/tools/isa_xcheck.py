#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cross-checks the direct compiler's encoder against llvm-mc, offline.

    python tools/isa_xcheck.py --llvm-bin DIR [--crate DIR] [--out report.md] [--json details.json]

Runs ``cargo run --release --example isa_xcheck`` in the crate, which prints
every table-encoded opcode of the instruction description with a systematic
set of operand, modifier and immediate variants and the words both encoders
give them. Each variant is written in LLVM syntax and assembled by
``llvm-mc -arch=amdgcn -mcpu=gfx1151 -show-encoding``; differing words are
disassembled with ``llvm-mc --disassemble`` to classify the difference:

  match         the same words
  equivalent    different words that disassemble to the same instruction
  literal       a literal where llvm-mc uses an inline constant (same value)
  mismatch      different instructions: a possible encoder bug
  llvm-rejects  llvm-mc rejects the operands; our encoders accept them
  both-reject   llvm-mc and our encoders reject the operands
  ours-rejects  llvm-mc accepts the operands; our encoders reject them
  unprintable   the variant has no LLVM syntax (noted with the reason)

llvm-mc is used for analysis only; it is never part of the compile path.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isa_tables

FLOAT_TYPES = ("_f32", "_f16", "_f64", "_bf16")
E64_ALWAYS = {
    "v_add_co_ci_u32",
    "v_sub_co_ci_u32",
    "v_subrev_co_ci_u32",
    "v_cndmask_b32",
}
SPECIALS = {"vcc_lo", "exec_lo", "m0", "null", "scc"}


class Unprintable(Exception):
    pass


@dataclass
class Variant:
    index: int
    name: str
    code: int | None
    defs: list[str]
    uses: list[str]
    imm: int
    mods: int
    table: str
    hand: str
    asm: str = ""
    why: str = ""
    llvm: str = ""
    llvm_bytes: list[int] | None = None
    llvm_error: str = ""
    ours_text: str = ""
    verdict: str = ""
    extra: dict = field(default_factory=dict)

    def words(self) -> list[int] | None:
        if self.table.startswith("ERR:"):
            return None
        return [int(w, 16) for w in self.table.split(",") if w]

    def our_bytes(self) -> list[int] | None:
        w = self.words()
        if w is None:
            return None
        return [b for x in w for b in x.to_bytes(4, "little")]


def parse_variants(text: str) -> list[Variant]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        out.append(
            Variant(
                int(f[0]),
                f[1],
                None if f[2] == "-" else int(f[2], 16),
                [x for x in f[3].split(",") if x],
                [x for x in f[4].split(",") if x],
                int(f[5]),
                int(f[6], 16),
                f[7],
                f[8],
            )
        )
    return out


# --------------------------------------------------------------------------
# LLVM syntax
# --------------------------------------------------------------------------


F32_INLINE_TEXT = {
    0x3F000000: "0.5",
    0xBF000000: "-0.5",
    0x3F800000: "1.0",
    0xBF800000: "-1.0",
    0x40000000: "2.0",
    0xC0000000: "-2.0",
    0x40800000: "4.0",
    0xC0800000: "-4.0",
    0x3E22F983: "0.15915494",
}


def reg(
    tok: str, half: bool = False, true16: bool = False, float_text: bool = False
) -> str:
    """An operand in LLVM syntax: `half` picks a VGPR's high half, `true16`
    names a 16-bit VGPR half explicitly, `float_text` writes f32 inline
    constants as floats."""
    m = re.fullmatch(r"([vs])(\d+)x(\d+)", tok)
    if m:
        k, r, n = m.group(1), int(m.group(2)), int(m.group(3))
        text = f"{k}{r}" if n == 1 else f"{k}[{r}:{r + n - 1}]"
        if k == "v" and (half or true16):
            text += ".h" if half else ".l"
        return text
    if float_text and tok.startswith("i") and int(tok[1:], 16) in F32_INLINE_TEXT:
        return F32_INLINE_TEXT[int(tok[1:], 16)]
    if tok == "scc":
        return "src_scc"
    if tok in SPECIALS:
        return tok
    if tok.startswith("i"):
        # An inline integer is the value sign-extended to the operand's width
        # (what the compiler means in a 64-bit operand); anything else is the
        # 32-bit pattern.
        k = int(tok[1:], 16)
        s = k - (1 << 32) if k >> 31 else k
        return str(s) if -16 <= s <= 64 else f"0x{k:x}"
    raise Unprintable(f"operand {tok}")


def is_vgpr(tok: str) -> bool:
    return tok.startswith("v") and tok[1:2].isdigit()


def is_imm(tok: str) -> bool:
    return tok.startswith("i")


@dataclass
class Info:
    mnemonic: str
    enc: str
    fmt: str
    form: str
    flags: set[str]
    mods: set[str]
    uses: list[tuple[str, str, int]]
    defs: list[tuple[str, str, int]]


def load_info(crate: Path) -> dict[str, Info]:
    tree = isa_tables.Tree(crate)
    desc = isa_tables.load_desc(tree)
    aliases = {r.args[0]: r.get("flags").split("|") for r in desc.section("alias")}
    out = {}
    for r in desc.insts + [desc.family]:
        flags = set()
        for t in isa_tables._flags_field(
            r.get("flags") if r.get("flags") != "macro" else "VALU"
        ):
            flags.update(aliases.get(t, [t]))
        fmt = "Vopc" if r.kind == "family" else desc.fmt_of(r)
        out[r.args[0]] = Info(
            r.args[1],
            r.args[2],
            fmt,
            r.get("form") or "plain",
            flags,
            set(filter(None, (r.get("m") or "").split(","))),
            isa_tables.parse_sig_list(r.get("u")),
            isa_tables.parse_sig_list(r.get("d")),
        )
    return out


def vcmp_names() -> dict[int, str]:
    return {c: n for c, n, _t in isa_tables.vcmp_table()}


def vopd_names(crate: Path) -> dict[int, str]:
    """Mnemonic of each VOPD component code, from the description."""
    desc = isa_tables.load_desc(isa_tables.Tree(crate))
    return {int(r.args[0]): r.args[1] for r in desc.section("vopd")}


def is_vop3_word(w: int) -> bool:
    return w >> 26 == 0x35


def src_text(
    tok: str,
    neg: bool,
    hi: bool,
    float_op: bool,
    true16: bool = False,
    float_text: bool = False,
) -> str:
    text = reg(tok, half=hi, true16=true16, float_text=float_text)
    if neg:
        if not float_op:
            raise Unprintable("neg on an integer operand")
        text = f"-{text}" if not is_imm(tok) else f"neg({text})"
    return text


def float_source(mn: str, k: int) -> bool:
    """Whether source `k` of `mn` is a float (accepts neg)."""
    if mn.startswith("v_cvt_"):
        return mn.split("_")[-1] in ("f32", "f16", "f64")
    if mn in ("v_ldexp_f32", "v_ldexp_f64", "v_cmp_class_f32") and k == 1:
        return False
    return any(t in mn for t in FLOAT_TYPES)


def dpp16_text(imm: int) -> str:
    ctrl = imm & 0x1FF
    if ctrl <= 0xFF:
        sel = ",".join(str((ctrl >> (2 * k)) & 3) for k in range(4))
        c = f"quad_perm:[{sel}]"
    elif 0x101 <= ctrl <= 0x10F:
        c = f"row_shl:{ctrl & 0xF}"
    elif 0x111 <= ctrl <= 0x11F:
        c = f"row_shr:{ctrl & 0xF}"
    elif 0x121 <= ctrl <= 0x12F:
        c = f"row_ror:{ctrl & 0xF}"
    elif ctrl == 0x140:
        c = "row_mirror"
    elif ctrl == 0x141:
        c = "row_half_mirror"
    elif 0x150 <= ctrl <= 0x15F:
        c = f"row_share:{ctrl & 0xF}"
    elif 0x160 <= ctrl <= 0x16F:
        c = f"row_xmask:{ctrl & 0xF}"
    else:
        raise Unprintable(f"dpp_ctrl {ctrl:#x}")
    parts = [
        c,
        f"row_mask:0x{(imm >> 20) & 0xF:x}",
        f"bank_mask:0x{(imm >> 16) & 0xF:x}",
    ]
    if imm >> 11 & 1:
        parts.append("bound_ctrl:1")
    if imm >> 10 & 1:
        parts.append("fi:1")
    if imm & 0xF000:
        raise Unprintable("DPP source modifiers")
    return " ".join(parts)


def mem_offset(name: str, value: int) -> str:
    return f" {name}:{value}" if value else ""


def render(
    v: Variant, info: dict[str, Info], vcmp: dict[int, str], vopd: dict[int, str]
) -> str:
    if v.name == "VCmp":
        i = info["VCmp"]
        mn = vcmp[v.code]
    else:
        i = info[v.name]
        mn = i.mnemonic
    words = v.words()
    m = v.mods
    d, u = v.defs, v.uses
    neg = lambda k: bool(m >> k & 1)
    fmt = i.fmt
    if v.name in ("SAndSaveexecB32", "SAndSaveexecB64"):
        return f"{mn} {reg(d[0])}, {reg(u[0])}"
    if fmt in ("Sop1", "Sop2"):
        return f"{mn} {reg(d[0])}, " + ", ".join(reg(x) for x in u)
    if fmt == "Sopc":
        return f"{mn} {reg(u[0])}, {reg(u[1])}"
    if fmt == "Sopk":
        return f"{mn} {reg(d[0])}, 0x{v.imm & 0xFFFF:x}"
    if fmt == "Sopp":
        if mn in ("s_endpgm", "s_barrier", "s_code_end"):
            if v.imm:
                raise Unprintable("an operand on a SOPP without one")
            return mn
        return f"{mn} {v.imm & 0xFFFF:#x}"
    if fmt == "Smem":
        off = v.imm if v.imm < 0 or v.imm < 0x100000 else v.imm - 0x200000
        return (
            f"{mn} {reg(d[0])}, {reg(u[0])}, {off:#x}"
            if off >= 0
            else f"{mn} {reg(d[0])}, {reg(u[0])}, -{-off:#x}"
        )
    if fmt == "Vop1":
        if not d:
            return mn
        if i.form == "dpp16":
            return f"v_mov_b32_dpp {reg(d[0])}, {reg(u[0])} {dpp16_text(v.imm)}"
        if i.form == "dpp8":
            sel = ",".join(str((v.imm >> (3 * k)) & 7) for k in range(8))
            return f"v_mov_b32_dpp {reg(d[0])}, {reg(u[0])} dpp8:[{sel}]"
        e64 = words is not None and is_vop3_word(words[0])
        half_dst, half_src = "16BIT_DST" in i.flags, "16BIT_SRC" in i.flags
        if m & ~0xF7:
            raise Unprintable("modifier bits beyond neg and op_sel")
        if not half_src and m & 0x70:
            raise Unprintable("op_sel on a 32-bit source")
        if not half_dst and m & 0x80:
            raise Unprintable("op_sel on a 32-bit destination")
        if half_src and m & 0x10 and not is_vgpr(u[0]):
            raise Unprintable("the high half of a non-VGPR source")
        dst = reg(d[0], half=half_dst and bool(m & 0x80), true16=half_dst)
        src = src_text(
            u[0],
            neg(0),
            half_src and bool(m & 0x10),
            float_source(mn, 0),
            true16=half_src,
        )
        return f"{mn}{'_e64' if e64 else '_e32'} {dst}, {src}"
    if fmt == "Vop2":
        e64 = words is not None and is_vop3_word(words[0])
        if m & ~0x7:
            raise Unprintable("op_sel on a 32-bit VOP2")
        srcs = [
            src_text(x, neg(k), False, float_source(mn, k)) for k, x in enumerate(u)
        ]
        return f"{mn}{'_e64' if e64 else '_e32'} {reg(d[0])}, " + ", ".join(srcs)
    if fmt in ("Vop3", "Vop3b", "Vopc"):
        enc_uses = [x for x, sig in zip(u, i.uses) if sig[0] != "old"]
        sfx = "_e64" if fmt == "Vopc" and v.name == "VCmp" or mn in E64_ALWAYS else ""
        opsel = ""
        if v.name in ("VPermlane16B32", "VPermlanex16B32"):
            if m & ~0x30:
                raise Unprintable("modifiers beyond fi and bound_ctrl")
            opsel = f" op_sel:[{m >> 4 & 1},{m >> 5 & 1}]" if m & 0x30 else ""
            m = 0
        halves = [False] * 4
        if v.name == "VPackB32F16":
            for k in range(2):
                if m >> (4 + k) & 1:
                    if not is_vgpr(u[k]):
                        raise Unprintable("the high half of a non-VGPR source")
                    halves[k] = True
            m &= ~0x30
        if m & 0xF0:
            raise Unprintable("op_sel on a 32-bit VOP3")
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg")
        sixteen = v.name == "VPackB32F16"
        srcs = [
            src_text(x, neg(k), halves[k], float_source(mn, k), true16=sixteen)
            for k, x in enumerate(enc_uses)
        ]
        dsts = [reg(x) for x in d]
        return f"{mn}{sfx} " + ", ".join(dsts + srcs) + opsel
    if fmt == "Vop3p":
        enc_uses = [x for x, sig in zip(u, i.uses) if sig[0] != "old"]
        if i.form == "mix":
            if m & ~0xE77:
                raise Unprintable("modifier bits beyond neg, op_sel and op_sel_hi")
            srcs = [
                src_text(x, neg(k), False, True, float_text=True)
                for k, x in enumerate(enc_uses)
            ]
            op_sel = ",".join(str(m >> (4 + k) & 1) for k in range(3))
            op_sel_hi = ",".join(str(m >> (9 + k) & 1) for k in range(3))
            tail = f" op_sel:[{op_sel}] op_sel_hi:[{op_sel_hi}]" if m & 0xE70 else ""
            return f"{mn} {reg(d[0])}, " + ", ".join(srcs) + tail
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg_lo")
        srcs = [reg(x) for x in enc_uses]
        tail = ""
        if m & 7:
            tail = (
                " neg_lo:[" + ",".join(str(m >> k & 1) for k in range(len(srcs))) + "]"
            )
        return f"{mn} {reg(d[0])}, " + ", ".join(srcs) + tail
    if fmt == "Vopd":
        nx = v.mods
        xs, ys = u[:nx], u[nx:]
        cx, cy = v.imm & 0xFF, v.imm >> 8 & 0xFF
        for c in (cx, cy):
            if c not in vopd:
                raise Unprintable(
                    f"VOPD component {c} has no mnemonic in the description"
                )
        half = lambda nm, dst, srcs: (
            f"{nm} {reg(dst)}, " + ", ".join(reg(x) for x in srcs[:2])
        )
        return half(vopd[cx], d[0], xs) + " :: " + half(vopd[cy], d[1], ys)
    if fmt == "Ds":
        roles = [r for r, _c, _n in i.uses]
        if v.name == "DsStore2addrB32":
            return (
                f"{mn} {reg(u[0])}, {reg(u[1])}, {reg(u[2])}"
                + mem_offset("offset0", v.imm & 0xFF)
                + mem_offset("offset1", v.imm >> 8 & 0xFF)
            )
        ops = [reg(d[0])] if d else []
        ops += [reg(x) for x, r in zip(u, roles) if r != "vdst"]
        return f"{mn} " + ", ".join(ops) + mem_offset("offset", v.imm & 0xFFFF)
    if fmt == "Global" and i.form == "scratch":
        tail = mem_offset("offset", v.imm)
        if d:
            return f"{mn} {reg(d[0])}, off, off{tail}"
        return f"{mn} off, {reg(u[0])}, off{tail}"
    if fmt == "Global":
        saddr = bool(m & (1 << 8))
        roles = [r for r, _c, _n in i.uses]
        addr = reg(u[0])
        base = reg(u[1]) if saddr else "off"
        rest = u[2:] if saddr else u[1:]
        data = [reg(x) for x, r in zip(rest, roles[1:]) if r != "vdst"]
        ops = ([reg(d[0])] if d else []) + [addr] + data + [base]
        tail = mem_offset("offset", v.imm) + (" glc" if m & (1 << 12) else "")
        if m & ~((1 << 8) | (1 << 12)):
            raise Unprintable("modifier bits beyond saddr and glc")
        return f"{mn} " + ", ".join(ops) + tail
    if fmt == "Mubuf":
        if not d and not u:
            return mn
        if d:
            vdata, rest = reg(d[0]), u
        else:
            vdata, rest = reg(u[0]), u[1:]
        return (
            f"{mn} {vdata}, {reg(rest[0])}, {reg(rest[1])}, {reg(rest[2])} offen"
            + mem_offset("offset", v.imm)
        )
    raise Unprintable(f"format {fmt}")


# --------------------------------------------------------------------------
# gfx950 (`--isa gfx950`): the description isa/gfx950.isa and gfx9 syntax
# --------------------------------------------------------------------------

GFX9_E64_ALWAYS = {
    "v_cndmask_b32",
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_addc_co_u32",
    "v_subb_co_u32",
}


def load_info_gfx950(crate: Path) -> tuple[dict[str, Info], dict[int, str]]:
    """Per-variant facts of isa/gfx950.isa, and its compares by shared code."""
    tree = isa_tables.Tree(crate)
    text = tree.read(isa_tables.GFX950_DESCRIPTION)
    desc = isa_tables.Desc(isa_tables.parse(text, isa_tables.GFX950_DESCRIPTION), text)
    aliases = {r.args[0]: r.get("flags").split("|") for r in desc.section("alias")}
    out = {}
    for r in desc.insts + [desc.family]:
        flags = set()
        for t in isa_tables._flags_field(
            r.get("flags") if r.get("flags") != "macro" else "VALU"
        ):
            flags.update(aliases.get(t, [t]))
        fmt = "Vopc" if r.kind == "family" else desc.fmt_of(r)
        out[r.args[0]] = Info(
            r.args[1],
            r.args[2],
            fmt,
            r.get("form") or "plain",
            flags,
            set(filter(None, (r.get("m") or "").split(","))),
            isa_tables.parse_sig_list(r.get("u")),
            isa_tables.parse_sig_list(r.get("d")),
        )
    vcmp = {int(c.args[0], 0): c.args[1] for c in desc.section("vcmp")}
    return out, vcmp


def reg9(tok: str, float_text: bool = False) -> str:
    """`reg` with gfx9's 64-bit lane masks and AGPRs (`a<k>x<n>`)."""
    if tok in ("vcc", "exec"):
        return tok
    m = re.fullmatch(r"a(\d+)x(\d+)", tok)
    if m:
        r, n = int(m.group(1)), int(m.group(2))
        return f"a{r}" if n == 1 else f"a[{r}:{r + n - 1}]"
    return reg(tok, float_text=float_text)


F64_INLINE_HI_TEXT = {
    0x3FE00000: "0.5",
    0xBFE00000: "-0.5",
    0x3FF00000: "1.0",
    0xBFF00000: "-1.0",
    0x40000000: "2.0",
    0xC0000000: "-2.0",
    0x40100000: "4.0",
    0xC0100000: "-4.0",
}


def f64_text9(tok: str) -> str:
    """An f64 operand whose immediate is the constant's high dword (the f64
    forms' convention): an f64 inline constant as its value."""
    if is_imm(tok) and int(tok[1:], 16) in F64_INLINE_HI_TEXT:
        return F64_INLINE_HI_TEXT[int(tok[1:], 16)]
    return reg9(tok)


def src_text9(tok: str, neg: bool, float_op: bool) -> str:
    text = reg9(tok)
    if neg:
        if not float_op:
            raise Unprintable("neg on an integer operand")
        text = f"-{text}" if not is_imm(tok) else f"neg({text})"
    return text


def is_vop3_word9(w: int) -> bool:
    return w >> 26 == 0x34


def dpp_text9(imm: int) -> str:
    """The gfx9 DPP controls; others (gfx10+ row_share, row_xmask) in gfx10
    syntax, which llvm-mc then refuses for gfx950."""
    ctrl = imm & 0x1FF
    named = {
        0x130: "wave_shl:1",
        0x134: "wave_rol:1",
        0x138: "wave_shr:1",
        0x13C: "wave_ror:1",
        0x140: "row_mirror",
        0x141: "row_half_mirror",
        0x142: "row_bcast:15",
        0x143: "row_bcast:31",
    }
    if ctrl <= 0xFF:
        c = "quad_perm:[" + ",".join(str((ctrl >> (2 * k)) & 3) for k in range(4)) + "]"
    elif 0x101 <= ctrl <= 0x10F:
        c = f"row_shl:{ctrl & 0xF}"
    elif 0x111 <= ctrl <= 0x11F:
        c = f"row_shr:{ctrl & 0xF}"
    elif 0x121 <= ctrl <= 0x12F:
        c = f"row_ror:{ctrl & 0xF}"
    elif ctrl in named:
        c = named[ctrl]
    else:
        return dpp16_text(imm)
    if imm & 0xF000:
        raise Unprintable("DPP source modifiers")
    if imm >> 10 & 1:
        raise Unprintable("fi is a gfx10+ DPP bit")
    parts = [
        c,
        f"row_mask:0x{(imm >> 20) & 0xF:x}",
        f"bank_mask:0x{(imm >> 16) & 0xF:x}",
    ]
    if imm >> 11 & 1:
        parts.append("bound_ctrl:1")
    return " ".join(parts)


def render9(v: Variant, info: dict[str, Info], vcmp: dict[int, str]) -> str:
    if v.name == "VCmp":
        i = info["VCmp"]
        mn = vcmp[v.code]
    else:
        i = info[v.name]
        mn = i.mnemonic
    words = v.words()
    m = v.mods
    d, u = v.defs, v.uses
    neg = lambda k: bool(m >> k & 1)
    fmt = i.fmt
    if v.name == "SAndSaveexecB64":
        return f"{mn} {reg9(d[0])}, {reg9(u[0])}"
    if fmt in ("Sop1", "Sop2"):
        return f"{mn} {reg9(d[0])}, " + ", ".join(reg9(x) for x in u)
    if fmt == "Sopc":
        return f"{mn} {reg9(u[0])}, {reg9(u[1])}"
    if fmt == "Sopk":
        return f"{mn} {reg9(d[0])}, 0x{v.imm & 0xFFFF:x}"
    if fmt == "Sopp":
        if mn in ("s_endpgm", "s_barrier"):
            if v.imm:
                raise Unprintable("an operand on a SOPP without one")
            return mn
        return f"{mn} {v.imm & 0xFFFF:#x}"
    if fmt == "Smem" and i.form == "nobase":
        if u or v.imm:
            raise Unprintable("an address or offset on s_memrealtime / s_dcache_inv")
        return f"{mn} {reg9(d[0])}" if d else mn
    if fmt == "Smem":
        off = v.imm
        glc = " glc" if v.mods & 0x1000 else ""
        return f"{mn} {reg9(d[0])}, {reg9(u[0])}, " + (
            f"{off:#x}" if off >= 0 else f"-{-off:#x}"
        ) + glc
    if fmt == "Vop1":
        if not d:
            return mn
        if i.form == "swap":
            # Both registers are read and written; the uses repeat the defs.
            return f"{mn}_e32 {reg9(d[0])}, {reg9(d[1])}"
        if i.form == "acc_mov":
            return f"{mn} {reg9(d[0])}, {reg9(u[0])}"
        if i.form == "dpp16":
            return f"v_mov_b32_dpp {reg9(d[0])}, {reg9(u[0])} {dpp_text9(v.imm)}"
        if i.form == "sdwa_hi" and m & 0x10:
            if m & ~0x11:
                raise Unprintable("SDWA modifiers beyond neg")
            src = src_text9(u[0], neg(0), True)
            # A 64-bit result has no destination select in the syntax (the word holds DWORD).
            dst_sel = "" if i.defs[0][2] > 1 else " dst_sel:DWORD dst_unused:UNUSED_PAD"
            return f"{mn}_sdwa {reg9(d[0])}, {src}{dst_sel} src0_sel:WORD_1"
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg")
        e64 = words is not None and is_vop3_word9(words[0])
        return f"{mn}{'_e64' if e64 else '_e32'} {reg9(d[0])}, {src_text9(u[0], neg(0), float_source(mn, 0))}"
    if fmt == "Vop2" and i.form == "dpp16":
        if m:
            raise Unprintable("modifiers on a DPP VOP2")
        return f"{mn}_dpp {reg9(d[0])}, {reg9(u[0])}, {reg9(u[1])} {dpp_text9(v.imm)}"
    if fmt == "Vop2":
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg")
        e64 = words is not None and is_vop3_word9(words[0])
        srcs = [src_text9(x, neg(k), float_source(mn, k)) for k, x in enumerate(u)]
        return f"{mn}{'_e64' if e64 else '_e32'} {reg9(d[0])}, " + ", ".join(srcs)
    if fmt in ("Vop3", "Vop3b", "Vopc"):
        enc_uses = [x for x, sig in zip(u, i.uses) if sig[0] != "old"]
        sfx = "_e64" if v.name == "VCmp" or mn in GFX9_E64_ALWAYS else ""
        opsel = ""
        if v.name in ("VPackB32F16", "VCvtScalef32PkBf16Fp4"):
            if m & ~0x30:
                raise Unprintable("modifier bits beyond op_sel")
            opsel = f" op_sel:[{m >> 4 & 1},{m >> 5 & 1},0]" if m & 0x30 else ""
            m = 0
        if m == 0x80:
            # op_sel[3], the high half of the destination: after one bit per source.
            opsel = " op_sel:[" + ",".join(["0"] * len(enc_uses) + ["1"]) + "]"
            m = 0
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg")
        srcs = [
            src_text9(x, neg(k), float_source(mn, k)) for k, x in enumerate(enc_uses)
        ]
        if v.name == "VBitop3B32":
            # The truth table (imm), encoded in the neg / abs / omod fields.
            opsel = f" bitop3:{v.imm:#x}" if v.imm else ""
        return f"{mn}{sfx} " + ", ".join([reg9(x) for x in d] + srcs) + opsel
    if fmt == "Vop3p":
        enc_uses = [x for x, sig in zip(u, i.uses) if sig[0] != "old"]
        if i.form in ("acc_read", "acc_write"):
            if m:
                raise Unprintable("modifiers on an AGPR move")
            return f"{mn} {reg9(d[0])}, {src_text9(u[0], False, False)}"
        if i.form in ("mai", "mai_scale"):
            # D, A, B and C (C may be an inline constant, written as llvm-mc
            # reads one); imm holds cbsz (2:0), abid (6:3) and blgp (9:7).
            # The block-scaled form then names its scale pair as two VGPRs.
            if m:
                raise Unprintable("modifiers on an MFMA")
            if v.imm >> 10:
                raise Unprintable("MFMA controls past blgp")
            ctl = [
                ("cbsz", v.imm & 7),
                ("abid", v.imm >> 3 & 15),
                ("blgp", v.imm >> 7 & 7),
            ]
            tail = "".join(f" {k}:{x}" for k, x in ctl if x)
            c = (
                f64_text9(u[2])
                if mn.startswith("v_mfma_f64")
                else reg9(u[2], float_text=True)
            )
            ops = [reg9(d[0]), reg9(u[0]), reg9(u[1]), c]
            if i.form == "mai_scale":
                # The A and B scales: one VGPR each.
                for x in u[3:5]:
                    if not re.fullmatch(r"v\d+x1", x):
                        raise Unprintable(f"a scale {x}")
                ops += [reg9(u[3]), reg9(u[4])]
            return f"{mn} " + ", ".join(ops) + tail
        if i.form == "pk32":
            # Lane selects: op_sel in imm bits 2:0, op_sel_hi cleared in 6:4;
            # neg negates both lanes (neg_lo and neg_hi).
            if m & ~0x7:
                raise Unprintable("modifier bits beyond neg")
            n = len(enc_uses)
            op_sel = [v.imm >> k & 1 for k in range(n)]
            op_sel_hi = [1 - (v.imm >> (4 + k) & 1) for k in range(n)]
            negs = [m >> k & 1 for k in range(n)]
            tail = ""
            if any(op_sel):
                tail += " op_sel:[" + ",".join(map(str, op_sel)) + "]"
            if not all(op_sel_hi):
                tail += " op_sel_hi:[" + ",".join(map(str, op_sel_hi)) + "]"
            if any(negs):
                lst = ",".join(map(str, negs))
                tail += f" neg_lo:[{lst}] neg_hi:[{lst}]"
            return f"{mn} {reg9(d[0])}, " + ", ".join(reg9(x) for x in enc_uses) + tail
        if i.form == "mix":
            if m & ~0xE77:
                raise Unprintable("modifier bits beyond neg, op_sel and op_sel_hi")
            srcs = [
                src_text(x, neg(k), False, True, float_text=True)
                for k, x in enumerate(enc_uses)
            ]
            op_sel = ",".join(str(m >> (4 + k) & 1) for k in range(3))
            op_sel_hi = ",".join(str(m >> (9 + k) & 1) for k in range(3))
            tail = f" op_sel:[{op_sel}] op_sel_hi:[{op_sel_hi}]" if m & 0xE70 else ""
            return f"{mn} {reg9(d[0])}, " + ", ".join(srcs) + tail
        if m & ~0x7:
            raise Unprintable("modifier bits beyond neg_lo")
        srcs = [reg9(x) for x in enc_uses]
        tail = (
            " neg_lo:[" + ",".join(str(m >> k & 1) for k in range(len(srcs))) + "]"
            if m & 7
            else ""
        )
        return f"{mn} {reg9(d[0])}, " + ", ".join(srcs) + tail
    if fmt == "Ds":
        roles = [r for r, _c, _n in i.uses]
        if v.name == "DsStore2addrB32":
            return (
                f"{mn} {reg9(u[0])}, {reg9(u[1])}, {reg9(u[2])}"
                + mem_offset("offset0", v.imm & 0xFF)
                + mem_offset("offset1", v.imm >> 8 & 0xFF)
            )
        ops = [reg9(d[0])] if d else []
        ops += [reg9(x) for x, r in zip(u, roles) if r != "vdst"]
        return f"{mn} " + ", ".join(ops) + mem_offset("offset", v.imm & 0xFFFF)
    if fmt == "Global" and i.form == "scratch":
        tail = mem_offset("offset", v.imm)
        if d:
            return f"{mn} {reg9(d[0])}, off, off{tail}"
        return f"{mn} off, {reg9(u[0])}, off{tail}"
    if fmt == "Global" and i.form == "lds":
        # global_load_lds: a VGPR pair address, saddr off; imm holds the
        # offset and the cache policy sc0, nt, sc1 in bits 12-14.
        if v.imm >> 15:
            raise Unprintable("cache policy bits past sc1")
        policy = [("sc0", 12), ("sc1", 14), ("nt", 13)]
        tail = "".join(f" {name}" for name, bit in policy if v.imm >> bit & 1)
        return f"{mn} {reg9(u[0])}, off" + mem_offset("offset", v.imm & 0xFFF) + tail
    if fmt == "Global":
        saddr = bool(m & (1 << 8))
        roles = [r for r, _c, _n in i.uses]
        base = reg9(u[1]) if saddr else "off"
        rest = u[2:] if saddr else u[1:]
        data = [reg9(x) for x, r in zip(rest, roles[1:]) if r != "vdst"]
        ops = ([reg9(d[0])] if d else []) + [reg9(u[0])] + data + [base]
        if m & ~((1 << 8) | (1 << 12) | (1 << 13) | (1 << 14)):
            raise Unprintable("modifier bits beyond saddr, sc0, sc1 and nt")
        # MInst bits: sc0 (M_GLC) 12, sc1 13, nt 14 (target/gfx9/cpol.rs).
        policy = [("sc0", 12), ("sc1", 13), ("nt", 14)]
        return (
            f"{mn} "
            + ", ".join(ops)
            + mem_offset("offset", v.imm)
            + "".join(f" {name}" for name, bit in policy if m >> bit & 1)
        )
    if fmt == "Mubuf" and i.form == "nodata":
        return (
            mn + (" sc0" if v.imm & 1 else "") + (" sc1" if v.imm & 2 else "")
        ).strip()
    if fmt == "Mubuf" and i.form == "lds":
        # LDS DMA: no data register (M0 is the LDS address); imm holds the
        # offset and the cache policy sc0, nt, sc1 in bits 12-14.
        if v.imm >> 15:
            raise Unprintable("cache policy bits past sc1")
        policy = [("sc0", 12), ("sc1", 14), ("nt", 13)]
        tail = "".join(f" {name}" for name, bit in policy if v.imm >> bit & 1)
        return (
            f"{mn} {reg9(u[0])}, {reg9(u[1])}, {reg9(u[2])} offen"
            + mem_offset("offset", v.imm & 0xFFF)
            + tail
            + " lds"
        )
    if fmt == "Mubuf":
        vdata, rest = (reg9(d[0]), u) if d else (reg9(u[0]), u[1:])
        if m & ~((1 << 12) | (1 << 13) | (1 << 14)):
            raise Unprintable("modifier bits beyond sc0, sc1 and nt")
        # MInst bits: sc0 (M_GLC) 12, sc1 13, nt 14 (target/gfx9/cpol.rs).
        policy = [("sc0", 12), ("sc1", 13), ("nt", 14)]
        return (
            f"{mn} {vdata}, {reg9(rest[0])}, {reg9(rest[1])}, {reg9(rest[2])} offen"
            + mem_offset("offset", v.imm)
            + "".join(f" {name}" for name, bit in policy if m >> bit & 1)
        )
    raise Unprintable(f"format {fmt}")


# --------------------------------------------------------------------------
# llvm-mc
# --------------------------------------------------------------------------


def run_mc(
    llvm_bin: Path, lines: list[str], disassemble: bool, mcpu: str = "gfx1151"
) -> tuple[list[str], dict[int, str]]:
    """Output lines (in order) and errors by 1-based input line."""
    exe = llvm_bin / ("llvm-mc.exe" if sys.platform == "win32" else "llvm-mc")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ("in.txt" if disassemble else "in.s")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        args = [str(exe), "-arch=amdgcn", f"-mcpu={mcpu}"]
        args += ["--disassemble"] if disassemble else ["-show-encoding"]
        proc = subprocess.run(
            args + [str(path)], capture_output=True, text=True, check=False
        )
    errors: dict[int, str] = {}
    for line in proc.stderr.splitlines():
        m = re.match(r".*?:(\d+):\d+: (error|warning): (.*)", line)
        if m:
            errors.setdefault(int(m.group(1)), m.group(3))
    out = [
        ln.strip()
        for ln in proc.stdout.splitlines()
        if ln.strip() and not ln.strip().startswith(".")
    ]
    return out, errors


def disassemble(
    llvm_bin: Path, byte_lists: list[list[int]], mcpu: str = "gfx1151"
) -> list[str]:
    """Disassembly of each byte list. The disassembler reads its input as one
    byte stream (an instruction that expects a literal takes the next line's
    bytes), so each list gets its own process."""

    def one(b: list[int]) -> str:
        out, errors = run_mc(
            llvm_bin, [" ".join(f"0x{x:02x}" for x in b)], disassemble=True, mcpu=mcpu
        )
        text = " ; ".join(line.split(";")[0].strip() for line in out)
        return text if text and not errors else (text + " <invalid encoding>").strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, byte_lists))


def parse_encoding(line: str) -> tuple[str, list[int]]:
    text, _, enc = line.partition("; encoding:")
    return text.strip(), [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", enc)]


def norm(text: str) -> str:
    text = text.split(";")[0]
    return " ".join(text.replace(",", ", ").split()).replace(" ,", ",")


def classify(vs: list[Variant], llvm_bin: Path, mcpu: str = "gfx1151") -> None:
    printable = [v for v in vs if v.asm]
    out, errors = run_mc(
        llvm_bin, [v.asm for v in printable], disassemble=False, mcpu=mcpu
    )
    it = iter(out)
    for n, v in enumerate(printable, 1):
        if n in errors:
            v.llvm_error = errors[n]
            continue
        line = next(it, None)
        if line is None:
            v.llvm_error = "no output"
            continue
        v.llvm, v.llvm_bytes = parse_encoding(line)
    # Disassemble our bytes where they differ from llvm-mc's or llvm-mc rejected the text.
    dis = [
        v
        for v in printable
        if v.our_bytes() is not None and v.our_bytes() != v.llvm_bytes
    ]
    canon = [v for v in dis if v.llvm_bytes]
    for v, text in zip(dis, disassemble(llvm_bin, [v.our_bytes() for v in dis], mcpu)):
        v.ours_text = text
    for v, text in zip(
        canon, disassemble(llvm_bin, [v.llvm_bytes for v in canon], mcpu)
    ):
        v.extra["llvm_dis"] = text
    for v in vs:
        ours = v.our_bytes()
        if not v.asm:
            v.verdict = "unprintable"
        elif v.llvm_error:
            v.verdict = "both-reject" if ours is None else "llvm-rejects"
        elif ours is None:
            v.verdict = "ours-rejects"
        elif ours == v.llvm_bytes:
            v.verdict = "match"
        elif norm(v.ours_text) == norm(v.extra.get("llvm_dis", "")):
            v.verdict = "equivalent"
        elif len(ours) == len(v.llvm_bytes) + 4 and literal_for_inline(v):
            v.verdict = "literal"
        else:
            v.verdict = "mismatch"


def literal_for_inline(v: Variant) -> bool:
    """Our encoding carries a literal whose value llvm-mc encodes inline."""
    ours = v.our_bytes()
    words = [
        int.from_bytes(bytes(ours[k : k + 4]), "little") for k in range(0, len(ours), 4)
    ]
    lit = words[-1]
    return any(is_imm(x) and int(x[1:], 16) == lit for x in v.uses)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report(vs: list[Variant], path: Path, isa: str = "gfx11") -> str:
    counts = collections.Counter(v.verdict for v in vs)
    by_op = collections.defaultdict(collections.Counter)
    for v in vs:
        by_op[v.name if v.name != "VCmp" else f"VCmp({v.code:#04x})"][v.verdict] += 1
    summary = (
        (
            f"Variants: {len(vs)} over {len(by_op)} opcodes (table-encoded opcodes of `isa/gfx11.isa`; "
            "operands by class: VGPRs below and above v127, SGPRs, special registers, inline integer, f32 and "
            "f64 constants, literals; each accepted modifier bit; immediate ranges). Reference: "
            "`llvm-mc -arch=amdgcn -mcpu=gfx1151 -show-encoding`. Table and hand encoders agree on every variant: "
            f"{sum(v.table == v.hand for v in vs)} of {len(vs)}."
        )
        if isa == "gfx11"
        else (
            f"Variants: {len(vs)} over {len(by_op)} opcodes (the rows of `isa/gfx950.isa`; operands by class: "
            "VGPRs below and above v127, SGPRs up to s101, special registers and the 64-bit lane masks vcc and "
            "exec, inline integer, f32 and f64 constants, literals; each accepted modifier bit; immediate "
            "ranges). Reference: `llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding`. One encoder "
            "(`src/target/gfx9/asm.rs`)."
        )
    )
    lines = [
        f"# llvm-mc cross-check of the {isa} encoder",
        "",
        summary,
        "",
        "| verdict | variants |",
        "|---|---|",
    ]
    for k in (
        "match",
        "equivalent",
        "literal",
        "mismatch",
        "llvm-rejects",
        "both-reject",
        "ours-rejects",
        "unprintable",
    ):
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    for verdict, title in (
        ("mismatch", "Mismatches (different instructions)"),
        (
            "llvm-rejects",
            "Operand combinations llvm-mc rejects and our encoders accept",
        ),
        (
            "ours-rejects",
            "Operand combinations our encoders reject and llvm-mc accepts",
        ),
        ("literal", "Literals where llvm-mc uses an inline constant"),
        ("equivalent", "Different but equivalent encodings"),
    ):
        sel = [v for v in vs if v.verdict == verdict]
        if not sel:
            continue
        lines += ["", f"## {title} ({len(sel)})", ""]
        groups = collections.defaultdict(list)
        for v in sel:
            groups[v.name].append(v)
        for name, items in sorted(groups.items()):
            lines.append(f"- `{name}` ({len(items)}):")
            for v in items[:4]:
                detail = (
                    f"llvm-mc: {v.llvm_error}"
                    if verdict == "llvm-rejects"
                    else f"ours: `{v.ours_text or v.table}`"
                )
                if verdict in ("mismatch", "equivalent", "literal"):
                    detail = f"ours `{v.table}` = `{norm(v.ours_text)}`; llvm-mc `{' '.join(f'{b:02x}' for b in v.llvm_bytes)}` = `{norm(v.extra.get('llvm_dis', v.llvm))}`"
                if verdict == "ours-rejects":
                    detail = f"ours: {v.table[4:]}; llvm-mc: `{v.llvm}`"
                lines.append(f"  - `{v.asm}` (mods {v.mods:#x}, imm {v.imm}): {detail}")
    unp = collections.Counter(v.why for v in vs if v.verdict == "unprintable")
    if unp:
        lines += [
            "",
            "## Variants without LLVM syntax",
            "",
            "| reason | variants |",
            "|---|---|",
        ]
        for why, n in unp.most_common():
            lines.append(f"| {why} | {n} |")
    lines += [
        "",
        "## Per opcode",
        "",
        "| opcode | " + " | ".join(k for k in counts) + " |",
        "|---|" + "---|" * len(counts),
    ]
    for name, c in sorted(by_op.items()):
        lines.append(
            f"| `{name}` | " + " | ".join(str(c.get(k, 0)) for k in counts) + " |"
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--crate", type=Path, default=isa_tables.DEFAULT_CRATE)
    ap.add_argument(
        "--variants",
        type=Path,
        help="read the variants from this file instead of running cargo",
    )
    ap.add_argument("--out", type=Path, default=Path("isa_xcheck.md"))
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--isa",
        choices=("gfx11", "gfx950"),
        default="gfx11",
        help="the encoder: gfx11 (rir, llvm-mc -mcpu=gfx1151) or gfx950 (target::gfx9, -mcpu=gfx950)",
    )
    args = ap.parse_args(argv)
    gfx950 = args.isa == "gfx950"
    if args.variants:
        text = args.variants.read_text(encoding="utf-8-sig")
    else:
        proc = subprocess.run(
            ["cargo", "run", "-q", "--release", "--example", "isa_xcheck"]
            + (["--", "--target", "gfx950"] if gfx950 else []),
            cwd=args.crate,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        text = proc.stdout
    vs = parse_variants(text)
    if gfx950:
        info9, vcmp9 = load_info_gfx950(args.crate)
    else:
        info = load_info(args.crate)
        vcmp, vopd = vcmp_names(), vopd_names(args.crate)
    for v in vs:
        try:
            v.asm = render9(v, info9, vcmp9) if gfx950 else render(v, info, vcmp, vopd)
        except Unprintable as e:
            v.why = str(e)
    classify(vs, args.llvm_bin, "gfx950" if gfx950 else "gfx1151")
    text = report(vs, args.out, args.isa)
    if args.json:
        args.json.write_text(
            json.dumps([v.__dict__ for v in vs], indent=1), encoding="utf-8"
        )
    print(text.split("\n## ")[0])
    return 1 if any(v.verdict == "mismatch" for v in vs) else 0


if __name__ == "__main__":
    sys.exit(main())
