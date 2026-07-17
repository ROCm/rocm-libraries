#!/usr/bin/env python3
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
#
# Validate s_wait_* insertion in STIR (StinkyTofu IR) files.
#
# Checks that register (VGPR/SGPR/acc) and LDS (memtoken) data dependencies
# are correctly satisfied by s_wait_dscnt / s_wait_loadcnt / s_wait_tensorcnt
# instructions already present in the input STIR.

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Counter kinds (mirrors WaitDataflow::CounterKind)
# ---------------------------------------------------------------------------


# Enum member order mirrors C++ CounterKind (CK_DS, CK_Buffer, CK_KM,
# CK_Tensor) so iteration order matches the reference dataflow.
class CK(Enum):
    DS = "DS"
    BUFFER = "Buffer"
    KM = "KM"
    TENSOR = "Tensor"


COUNTER_WAIT_OPS: Dict[CK, str] = {
    CK.DS: "s_wait_dscnt",
    CK.BUFFER: "s_wait_loadcnt",
    CK.KM: "s_wait_kmcnt",
    CK.TENSOR: "s_wait_tensorcnt",
}

WAIT_MOD_FIELDS: Dict[str, Tuple[CK, str]] = {
    "s_wait_dscnt": (CK.DS, "dlcnt"),
    "s_wait_loadcnt": (CK.BUFFER, "vlcnt"),
    "s_wait_kmcnt": (CK.KM, "kmcnt"),
    "s_wait_tensorcnt": (CK.TENSOR, "tlcnt"),
}

K_UNUSED = -1
K_MAX_IN_FLIGHT = 64


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class TokKind(Enum):
    EOF = auto()
    NEWLINE = auto()
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACK = auto()
    RBRACK = auto()
    COMMA = auto()
    EQ = auto()
    COLON = auto()
    TRUE = auto()
    FALSE = auto()
    AT = auto()
    CARET = auto()


@dataclass
class Token:
    kind: TokKind
    value: str = ""
    line: int = 1
    col: int = 1


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self, n: int = 0) -> str:
        i = self.pos + n
        return self.text[i] if i < len(self.text) else ""

    def _advance(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pos < len(self.text):
                if self.text[self.pos] == "\n":
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1

    def _skip_ws(self) -> None:
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c in " \t\r\n":
                self._advance()
            elif c == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self._advance()
            elif c == "/" and self._peek(1) == "/":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self._advance()
            elif c == "/" and self._peek(1) == "*":
                self._advance(2)
                while self.pos < len(self.text):
                    if self.text[self.pos] == "*" and self._peek(1) == "/":
                        self._advance(2)
                        break
                    self._advance()
            else:
                break

    def next_token(self) -> Token:
        self._skip_ws()
        line, col = self.line, self.col
        if self.pos >= len(self.text):
            return Token(TokKind.EOF, "", line, col)

        c = self.text[self.pos]

        singles = {
            "(": TokKind.LPAREN,
            ")": TokKind.RPAREN,
            "{": TokKind.LBRACE,
            "}": TokKind.RBRACE,
            "[": TokKind.LBRACK,
            "]": TokKind.RBRACK,
            ",": TokKind.COMMA,
            "=": TokKind.EQ,
            ":": TokKind.COLON,
            "@": TokKind.AT,
            "^": TokKind.CARET,
        }
        if c in singles:
            self._advance()
            return Token(singles[c], c, line, col)

        if c == '"':
            self._advance()
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] != '"':
                if self.text[self.pos] == "\\":
                    self._advance()
                self._advance()
            val = self.text[start : self.pos]
            self._advance()
            return Token(TokKind.STRING, val, line, col)

        if c.isdigit() or (c == "-" and self._peek(1).isdigit()):
            start = self.pos
            if c == "-":
                self._advance()
            while self.pos < len(self.text) and (
                self.text[self.pos].isdigit()
                or self.text[self.pos] in "abcdefABCDEF"
                or (
                    self.text[self.pos] == "x"
                    and "x" in self.text[start : self.pos + 1]
                )
            ):
                self._advance()
            # Fractional part of a decimal float (e.g. 1.000000). Only consume
            # the dot when it is followed by a digit so we don't swallow other
            # punctuation, and never for hex literals.
            if (
                self._peek() == "."
                and self._peek(1).isdigit()
                and "x" not in self.text[start : self.pos]
            ):
                self._advance()
                while self.pos < len(self.text) and self.text[self.pos].isdigit():
                    self._advance()
            raw = self.text[start : self.pos]
            if "." in raw:
                return Token(TokKind.FLOAT, raw, line, col)
            return Token(TokKind.INT, raw, line, col)

        if c.isalpha() or c == "_" or c == ".":
            start = self.pos
            while self.pos < len(self.text) and (
                self.text[self.pos].isalnum() or self.text[self.pos] in "._+"
            ):
                self._advance()
            val = self.text[start : self.pos]
            if val == "true":
                return Token(TokKind.TRUE, val, line, col)
            if val == "false":
                return Token(TokKind.FALSE, val, line, col)
            return Token(TokKind.IDENT, val, line, col)

        raise SyntaxError(f"Unexpected character {c!r} at line {line}, col {col}")


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegUnit:
    cls: str
    index: int

    def __str__(self) -> str:
        return f"{self.cls}{self.index}"


@dataclass
class Instruction:
    opcode: str
    dests: List[RegUnit] = field(default_factory=list)
    srcs: List[Any] = field(default_factory=list)  # RegUnit | int | str
    attrs: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    block: str = ""
    uid: int = 0

    def memtokens(self) -> List[int]:
        mod = self.attrs.get("mod.memtoken")
        if mod and isinstance(mod, dict):
            tokens = mod.get("tokens", [])
            if isinstance(tokens, list):
                return [int(t) for t in tokens]
        return []

    def label(self) -> str:
        return f"{self.opcode} @ line {self.line}"


@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    line: int = 0


@dataclass
class Function:
    name: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry: str = "entry"


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


class Parser:
    _uid = 0

    def __init__(self, text: str) -> None:
        self.lexer = Lexer(text)
        self.tok = self.lexer.next_token()
        self._peek_tok: Optional[Token] = None

    def _peek(self) -> Token:
        return self.tok

    def _eat(self, kind: Optional[TokKind] = None) -> Token:
        t = self.tok
        if kind is not None and t.kind != kind:
            raise SyntaxError(
                f"Expected {kind}, got {t.kind} ({t.value!r}) at line {t.line}"
            )
        self.tok = self.lexer.next_token()
        return t

    def _match(self, kind: TokKind) -> bool:
        if self.tok.kind == kind:
            self._eat()
            return True
        return False

    def parse(self) -> List[Function]:
        funcs: List[Function] = []
        while self.tok.kind != TokKind.EOF:
            if self.tok.kind == TokKind.IDENT and self.tok.value in ("st", "st.func"):
                funcs.append(self._parse_func())
            elif self.tok.kind in (TokKind.IDENT, TokKind.CARET):
                funcs.append(self._parse_flat())
            else:
                self._eat()
        return funcs

    def _parse_func(self) -> Function:
        hdr = self._eat(TokKind.IDENT).value
        if hdr == "st":
            self._eat(TokKind.IDENT)  # func
        self._eat(TokKind.AT)
        name = self._eat(TokKind.IDENT).value
        self._eat(TokKind.LPAREN)
        self._eat(TokKind.RPAREN)
        self._eat(TokKind.LBRACE)
        func = Function(name=name)
        while self.tok.kind != TokKind.RBRACE and self.tok.kind != TokKind.EOF:
            bb = self._parse_block()
            func.blocks[bb.name] = bb
        self._eat(TokKind.RBRACE)
        if func.blocks:
            func.entry = next(iter(func.blocks))
        return func

    def _parse_flat(self) -> Function:
        func = Function(name="flat")
        bb = BasicBlock(name="entry")
        while self.tok.kind != TokKind.EOF:
            if self.tok.kind == TokKind.CARET:
                bb = self._parse_block_header()
                continue
            if self.tok.kind == TokKind.IDENT and self._peek_label():
                self._eat(TokKind.IDENT)
                self._eat(TokKind.COLON)
                continue
            if self.tok.kind == TokKind.STRING or (
                self.tok.kind == TokKind.IDENT and self._peek_assign()
            ):
                bb.instructions.append(self._parse_instruction(bb.name))
                continue
            break
        func.blocks[bb.name] = bb
        func.entry = bb.name
        return func

    def _peek_label(self) -> bool:
        if self.tok.kind != TokKind.IDENT:
            return False
        saved = self.tok
        self.tok = self.lexer.next_token()
        is_label = self.tok.kind == TokKind.COLON
        self.tok = saved
        return is_label

    def _save_lexer_state(self) -> Tuple[Token, int, int, int]:
        return (self.tok, self.lexer.pos, self.lexer.line, self.lexer.col)

    def _restore_lexer_state(self, state: Tuple[Token, int, int, int]) -> None:
        self.tok, self.lexer.pos, self.lexer.line, self.lexer.col = state

    def _peek_assign(self) -> bool:
        saved = self._save_lexer_state()
        try:
            while self.tok.kind not in (TokKind.EOF, TokKind.EQ, TokKind.STRING):
                self._eat()
            return self.tok.kind == TokKind.EQ
        finally:
            self._restore_lexer_state(saved)

    def _parse_block(self) -> BasicBlock:
        if self.tok.kind == TokKind.CARET:
            return self._parse_block_header()
        raise SyntaxError(f"Expected block label at line {self.tok.line}")

    def _parse_block_header(self) -> BasicBlock:
        self._eat(TokKind.CARET)
        name = self._eat(TokKind.IDENT).value
        self._eat(TokKind.COLON)
        bb = BasicBlock(name=name, line=self.tok.line)
        while self.tok.kind not in (TokKind.CARET, TokKind.RBRACE, TokKind.EOF):
            if self.tok.kind == TokKind.IDENT and self.tok.value == "Successors":
                self._parse_successors(bb)
            elif self.tok.kind == TokKind.STRING or (
                self.tok.kind == TokKind.IDENT and self._peek_assign()
            ):
                bb.instructions.append(self._parse_instruction(bb.name))
            else:
                self._eat()
        return bb

    def _parse_successors(self, bb: BasicBlock) -> None:
        self._eat(TokKind.IDENT)
        self._eat(TokKind.COLON)
        while True:
            if self.tok.kind == TokKind.CARET:
                self._eat(TokKind.CARET)
                bb.successors.append(self._eat(TokKind.IDENT).value)
            else:
                break
            if self.tok.kind == TokKind.COMMA:
                self._eat(TokKind.COMMA)
            else:
                break

    def _parse_instruction(self, block: str) -> Instruction:
        dests: List[RegUnit] = []
        if self.tok.kind == TokKind.IDENT:
            dests = self._parse_reg_list()
            self._eat(TokKind.EQ)

        opcode_tok = self._eat(TokKind.STRING)
        opcode = opcode_tok.value
        if opcode.startswith("st."):
            opcode = opcode[3:]

        self._eat(TokKind.LPAREN)
        srcs: List[Any] = []
        if self.tok.kind != TokKind.RPAREN:
            srcs = self._parse_operand_list()
        self._eat(TokKind.RPAREN)

        attrs: Dict[str, Any] = {}
        if self.tok.kind == TokKind.LBRACE:
            attrs = self._parse_attr_block()

        Parser._uid += 1
        return Instruction(
            opcode=opcode,
            dests=dests,
            srcs=srcs,
            attrs=attrs,
            line=opcode_tok.line,
            block=block,
            uid=Parser._uid,
        )

    def _parse_reg_list(self) -> List[RegUnit]:
        regs: List[RegUnit] = []
        while True:
            regs.extend(self._parse_register())
            if self.tok.kind == TokKind.COMMA:
                self._eat(TokKind.COMMA)
            else:
                break
        return regs

    def _parse_operand_list(self) -> List[Any]:
        ops: List[Any] = []
        while True:
            ops.append(self._parse_operand())
            if self.tok.kind == TokKind.COMMA:
                self._eat(TokKind.COMMA)
            else:
                break
        return ops

    def _parse_operand(self) -> Any:
        if self.tok.kind in (TokKind.INT, TokKind.FLOAT):
            raw = self._eat().value
            try:
                if raw.startswith("0x") or raw.startswith("-0x"):
                    return int(raw, 16)
                if "." in raw:
                    return float(raw)
                return int(raw)
            except ValueError:
                return raw
        if self.tok.kind == TokKind.STRING:
            return self._eat(TokKind.STRING).value
        if self.tok.kind == TokKind.TRUE:
            self._eat()
            return True
        if self.tok.kind == TokKind.FALSE:
            self._eat()
            return False
        if self.tok.kind == TokKind.IDENT:
            if self.tok.value == "hwreg":
                return self._parse_hwreg()
            regs = self._parse_register()
            if len(regs) == 1:
                return regs[0]
            return regs
        raise SyntaxError(f"Unexpected operand at line {self.tok.line}")

    def _parse_hwreg(self) -> str:
        self._eat(TokKind.IDENT)
        self._eat(TokKind.LPAREN)
        parts = []
        while self.tok.kind != TokKind.RPAREN:
            parts.append(self._eat().value)
            if self.tok.kind == TokKind.COMMA:
                self._eat(TokKind.COMMA)
        self._eat(TokKind.RPAREN)
        return "hwreg(" + ",".join(parts) + ")"

    def _parse_register(self) -> List[RegUnit]:
        name = self._eat(TokKind.IDENT).value
        cls = _normalize_reg_class(name)
        m = re.match(r"^([A-Za-z_]+)(\d+)$", name)
        if m and self.tok.kind != TokKind.LBRACK:
            return [RegUnit(_normalize_reg_class(m.group(1)), int(m.group(2)))]
        if self.tok.kind == TokKind.LBRACK:
            self._eat(TokKind.LBRACK)
            lo = int(self._eat(TokKind.INT).value)
            if self.tok.kind == TokKind.COLON:
                self._eat(TokKind.COLON)
                hi = int(self._eat(TokKind.INT).value)
                self._eat(TokKind.RBRACK)
                return [RegUnit(cls, i) for i in range(lo, hi + 1)]
            self._eat(TokKind.RBRACK)
            return [RegUnit(cls, lo)]

        return [RegUnit(cls, 0)]

    def _parse_attr_block(self) -> Dict[str, Any]:
        self._eat(TokKind.LBRACE)
        attrs: Dict[str, Any] = {}
        while self.tok.kind != TokKind.RBRACE:
            key = self._eat(TokKind.IDENT).value
            self._eat(TokKind.EQ)
            attrs[key] = self._parse_attr_value()
            if self.tok.kind == TokKind.COMMA:
                self._eat(TokKind.COMMA)
        self._eat(TokKind.RBRACE)
        return attrs

    def _parse_attr_value(self) -> Any:
        if self.tok.kind == TokKind.LBRACE:
            self._eat(TokKind.LBRACE)
            d: Dict[str, Any] = {}
            while self.tok.kind != TokKind.RBRACE:
                k = self._eat(TokKind.IDENT).value
                self._eat(TokKind.EQ)
                d[k] = self._parse_attr_value()
                if self.tok.kind == TokKind.COMMA:
                    self._eat(TokKind.COMMA)
            self._eat(TokKind.RBRACE)
            return d
        if self.tok.kind == TokKind.LBRACK:
            self._eat(TokKind.LBRACK)
            arr: List[Any] = []
            while self.tok.kind != TokKind.RBRACK:
                arr.append(self._parse_attr_value())
                if self.tok.kind == TokKind.COMMA:
                    self._eat(TokKind.COMMA)
            self._eat(TokKind.RBRACK)
            return arr
        if self.tok.kind == TokKind.STRING:
            return self._eat(TokKind.STRING).value
        if self.tok.kind == TokKind.INT:
            return int(self._eat(TokKind.INT).value)
        if self.tok.kind == TokKind.FLOAT:
            return float(self._eat(TokKind.FLOAT).value)
        if self.tok.kind == TokKind.TRUE:
            self._eat()
            return True
        if self.tok.kind == TokKind.FALSE:
            self._eat()
            return False
        if self.tok.kind == TokKind.IDENT:
            return self._eat(TokKind.IDENT).value
        raise SyntaxError(f"Bad attribute value at line {self.tok.line}")


def _normalize_reg_class(name: str) -> str:
    lower = name.lower()
    if lower in ("v", "s", "acc", "a", "agpr", "lds", "scc", "vcc", "exec", "m"):
        if lower in ("a", "agpr"):
            return "acc"
        if lower == "lds":
            return "LDS"
        if lower == "scc":
            return "SCC"
        return lower
    if name in ("SCC", "LDS", "AGPR"):
        return "acc" if name == "AGPR" else name
    if name.startswith("vcc"):
        return name
    if name.startswith("exec"):
        return name
    return name


# ---------------------------------------------------------------------------
# Instruction classification
# ---------------------------------------------------------------------------


# Counter classification mirrors WaitDataflow.cpp's defaultCounterPolicy:
#   CK_DS     : ds_read / ds_write / ds_atomic            -> s_wait_dscnt
#   CK_Buffer : vector buffer/global/flat load+store      -> s_wait_loadcnt
#   CK_KM     : scalar SMEM loads (s_load / s_buffer_load)-> s_wait_kmcnt
#   CK_Tensor : tensor_load_to_lds                        -> s_wait_tensorcnt
def classify_counter(inst: Instruction) -> Optional[CK]:
    op = inst.opcode
    if _is_ds_producer(op):
        return CK.DS
    if _is_km_producer(op):
        return CK.KM
    if _is_buffer_producer(op):
        return CK.BUFFER
    if op == "tensor_load_to_lds":
        return CK.TENSOR
    return None


def _is_ds_producer(op: str) -> bool:
    return (
        op.startswith("ds_read")
        or op.startswith("ds_load")
        or op.startswith("ds_write")
        or op.startswith("ds_store")
        or op.startswith("ds_atomic")
    )


def _is_km_producer(op: str) -> bool:
    # Scalar memory loads complete on the kmcnt counter.
    return op.startswith("s_load") or op.startswith("s_buffer_load")


def _is_buffer_producer(op: str) -> bool:
    prefixes = (
        "buffer_load",
        "buffer_store",
        "global_load",
        "global_store",
        "flat_load",
        "flat_store",
    )
    return any(op.startswith(p) for p in prefixes)


def is_barrier(inst: Instruction) -> bool:
    return inst.opcode.startswith("s_barrier")


def is_ds_read(inst: Instruction) -> bool:
    return inst.opcode.startswith("ds_read") or inst.opcode.startswith("ds_load")


def is_ds_write(inst: Instruction) -> bool:
    return inst.opcode.startswith("ds_write") or inst.opcode.startswith("ds_store")


def is_ds_atomic(inst: Instruction) -> bool:
    return inst.opcode.startswith("ds_atomic")


def is_tensor_anchor(inst: Instruction) -> bool:
    return (
        is_barrier(inst) or is_ds_read(inst) or is_ds_write(inst) or is_ds_atomic(inst)
    )


def is_lds_writer_anchor(inst: Instruction) -> bool:
    return inst.opcode == "tensor_load_to_lds" or is_ds_write(inst)


def is_wait_inst(inst: Instruction) -> bool:
    return inst.opcode in WAIT_MOD_FIELDS


def is_branch(inst: Instruction) -> bool:
    return inst.opcode.startswith("s_cbranch") or inst.opcode.startswith("s_branch")


def get_wait_counts(inst: Instruction) -> Dict[CK, int]:
    """Return counter -> keep-count for each wait instruction present."""
    result: Dict[CK, int] = {}
    if inst.opcode not in WAIT_MOD_FIELDS:
        return result
    ck, field_name = WAIT_MOD_FIELDS[inst.opcode]
    mod_key = "mod.swaitcnt" if ck != CK.TENSOR else "mod.swaittensorcnt"
    mod = inst.attrs.get(mod_key)
    if isinstance(mod, dict) and field_name in mod:
        val = int(mod[field_name])
        if val != K_UNUSED:
            result[ck] = val
            return result
    for src in inst.srcs:
        if isinstance(src, int):
            result[ck] = src
            break
    return result


def has_token_overlap(a: List[int], b: List[int]) -> bool:
    return bool(set(a) & set(b))


def is_on_same_pipeline(a: Instruction, b: Instruction) -> bool:
    return classify_counter(a) == CK.DS and classify_counter(b) == CK.DS


def raw_needs_wait_tensor(inst: Instruction, num_waves: int) -> bool:
    return is_barrier(inst) or num_waves == 1


def raw_needs_wait(inst: Instruction, ck: CK, num_waves: int) -> bool:
    if ck == CK.TENSOR:
        return raw_needs_wait_tensor(inst, num_waves)
    return True


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------


@dataclass
class WaitReport:
    inst: Instruction
    counter: CK
    keep: int
    drained: List[Instruction] = field(default_factory=list)


@dataclass
class Violation:
    kind: str  # MISSING | ERROR
    consumer: Instruction
    producer: Instruction
    counter: CK
    message: str
    detail: str = ""


@dataclass
class PerPredQueue:
    """One queue of in-flight memops on a counter, tagged by the CFG
    predecessor it was seeded from (None == in-block / synthetic).

    Mirrors C++ waitcnt::PerPredQueue. For an op at index I in a queue of
    size N, the wait value to drain it is N - I - 1 (count_from == N - I).
    """

    pred: Optional[str] = None
    ops: List[Instruction] = field(default_factory=list)

    def count_from(self, op: Instruction) -> int:
        for i, o in enumerate(self.ops):
            if o.uid == op.uid:
                return len(self.ops) - i
        return 0


@dataclass
class CounterState:
    # One LIST of PerPredQueue per counter (kept per-pred, never collapsed
    # into a single union), mirroring DataflowState::queues. Per-pred queues
    # let a consumer at a join compute the strictest required wait as the
    # min over paths instead of an over-deep union.
    queues: Dict[CK, List[PerPredQueue]] = field(
        default_factory=lambda: {ck: [] for ck in CK}
    )
    reg_defs: Dict[RegUnit, Instruction] = field(default_factory=dict)
    num_waves: int = 0

    def copy(self) -> "CounterState":
        return CounterState(
            queues={
                ck: [PerPredQueue(q.pred, list(q.ops)) for q in qs]
                for ck, qs in self.queues.items()
            },
            reg_defs=dict(self.reg_defs),
            num_waves=self.num_waves,
        )

    def seed_from_pred(self, pred: str, other: "CounterState") -> None:
        """Seed one PerPredQueue per counter from a predecessor's exit
        queues, retagged as via-`pred`. Identical (pred, ops) queues are
        deduplicated (mirrors mergeFromPredecessors). reg_defs are merged
        first-writer-wins as an SSA approximation."""
        for ck in CK:
            for predQ in other.queues[ck]:
                uids = [o.uid for o in predQ.ops]
                dup = any(
                    q.pred == pred and [o.uid for o in q.ops] == uids
                    for q in self.queues[ck]
                )
                if not dup:
                    self.queues[ck].append(PerPredQueue(pred=pred, ops=list(predQ.ops)))
        for reg, prod in other.reg_defs.items():
            if reg not in self.reg_defs:
                self.reg_defs[reg] = prod

    def iter_ops(self, ck: CK) -> Iterator[Instruction]:
        """Yield each distinct in-flight op across all per-pred queues."""
        seen: Set[int] = set()
        for q in self.queues[ck]:
            for op in q.ops:
                if op.uid not in seen:
                    seen.add(op.uid)
                    yield op

    def in_flight(self, ck: CK) -> bool:
        return any(q.ops for q in self.queues[ck])

    def count_from(self, ck: CK, producer: Instruction) -> int:
        """Strictest residual depth of `producer`: the MIN positive
        count_from across per-pred queues (the closest-to-tail path bounds
        the wait), or 0 if drained on every path."""
        best = 0
        for q in self.queues[ck]:
            n = q.count_from(producer)
            if n > 0 and (best == 0 or n < best):
                best = n
        return best

    def trim_queue(self, ck: CK, keep: int) -> List[Instruction]:
        """Trim every per-pred queue to keep at most `keep` tail ops;
        return the distinct ops drained (for reporting)."""
        drained: List[Instruction] = []
        seen: Set[int] = set()

        def _record(op: Instruction) -> None:
            if op.uid not in seen:
                seen.add(op.uid)
                drained.append(op)

        for q in self.queues[ck]:
            if keep <= 0:
                for op in q.ops:
                    _record(op)
                q.ops.clear()
            elif len(q.ops) > keep:
                n = len(q.ops) - keep
                for op in q.ops[:n]:
                    _record(op)
                del q.ops[:n]
        return drained

    def append_producer(self, inst: Instruction, ck: CK) -> None:
        """Append an in-block memop to every per-pred path (it is in flight
        on all paths through this block). Each queue is capped at
        K_MAX_IN_FLIGHT so an undrained counter cannot grow forever."""
        qs = self.queues[ck]
        if not qs:
            qs.append(PerPredQueue())
        for q in qs:
            q.ops.append(inst)
            while len(q.ops) > K_MAX_IN_FLIGHT:
                q.ops.pop(0)


# ---------------------------------------------------------------------------
# Dependency collection
# ---------------------------------------------------------------------------


def _operand_regs(srcs: List[Any]) -> List[RegUnit]:
    regs: List[RegUnit] = []
    for s in srcs:
        if isinstance(s, RegUnit):
            regs.append(s)
        elif isinstance(s, list):
            regs.extend(r for r in s if isinstance(r, RegUnit))
    return regs


def _lds_pseudo_tokens(regs: List[RegUnit]) -> List[int]:
    return [r.index for r in regs if r.cls == "LDS"]


def collect_register_deps(
    inst: Instruction, state: CounterState
) -> List[Tuple[Instruction, CK]]:
    deps: List[Tuple[Instruction, CK]] = []
    seen: Set[int] = set()
    for reg in _operand_regs(inst.srcs):
        prod = state.reg_defs.get(reg)
        if prod is None:
            continue
        ck = classify_counter(prod)
        if ck is None:
            continue
        if prod.uid in seen:
            continue
        if not raw_needs_wait(inst, ck, state.num_waves):
            continue
        if state.count_from(ck, prod) > 0:
            deps.append((prod, ck))
            seen.add(prod.uid)
    return deps


def is_lds_reader(inst: Instruction) -> bool:
    """Instructions that consume LDS data (RAW on memtoken), not writers."""
    if is_ds_read(inst):
        return True
    if is_lds_writer_anchor(inst):
        return False
    if _lds_pseudo_tokens(_operand_regs(inst.srcs)):
        return True
    if is_tensor_anchor(inst) and inst.memtokens():
        return True
    return False


def collect_lds_raw_deps(
    inst: Instruction, state: CounterState
) -> List[Tuple[Instruction, CK]]:
    if not is_lds_reader(inst):
        return []
    tokens = inst.memtokens() or _lds_pseudo_tokens(_operand_regs(inst.srcs))
    if not tokens:
        return []
    deps: List[Tuple[Instruction, CK]] = []
    seen: Set[int] = set()
    for ck in CK:
        if not raw_needs_wait(inst, ck, state.num_waves):
            continue
        for prod in state.iter_ops(ck):
            pt = prod.memtokens()
            if not pt:
                continue
            if not has_token_overlap(tokens, pt):
                continue
            # Same DS FIFO orders consecutive DS ops; only cross-pipeline
            # memtoken RAW needs an explicit wait.
            if is_on_same_pipeline(inst, prod):
                continue
            if prod.uid in seen:
                continue
            if state.count_from(ck, prod) > 0:
                deps.append((prod, ck))
                seen.add(prod.uid)
    return deps


def collect_war_deps(
    inst: Instruction, state: CounterState
) -> List[Tuple[Instruction, CK]]:
    deps: List[Tuple[Instruction, CK]] = []
    barrier_mode = is_barrier(inst)
    if not is_lds_writer_anchor(inst) and not barrier_mode:
        return deps

    anchor_tokens = inst.memtokens()
    if anchor_tokens is None:
        anchor_tokens = []

    seen: Set[int] = set()
    for op in state.iter_ops(CK.DS):
        if op.uid == inst.uid:
            continue
        if not barrier_mode and not (is_ds_read(op) or is_ds_atomic(op)):
            continue
        if is_on_same_pipeline(inst, op):
            continue
        op_tokens = op.memtokens()
        overlap = (
            op_tokens is None
            or not anchor_tokens
            or has_token_overlap(op_tokens, anchor_tokens)
        )
        if not overlap:
            continue
        if op.uid in seen:
            continue
        if state.count_from(CK.DS, op) > 0:
            deps.append((op, CK.DS))
            seen.add(op.uid)
    return deps


def collect_conservative_deps(
    inst: Instruction, state: CounterState
) -> List[Tuple[Instruction, CK]]:
    deps: List[Tuple[Instruction, CK]] = []
    if is_tensor_anchor(inst) and not inst.memtokens() and state.in_flight(CK.TENSOR):
        for prod in state.iter_ops(CK.TENSOR):
            if state.count_from(CK.TENSOR, prod) > 0:
                deps.append((prod, CK.TENSOR))
    if is_lds_writer_anchor(inst) and not inst.memtokens() and not is_ds_write(inst):
        if state.in_flight(CK.DS):
            for prod in state.iter_ops(CK.DS):
                if state.count_from(CK.DS, prod) > 0:
                    deps.append((prod, CK.DS))
    if is_barrier(inst) and state.in_flight(CK.DS):
        barrier_untagged = not inst.memtokens()
        for prod in state.iter_ops(CK.DS):
            if barrier_untagged or not prod.memtokens():
                if state.count_from(CK.DS, prod) > 0:
                    deps.append((prod, CK.DS))
    if is_tensor_anchor(inst) and inst.memtokens():
        for prod in state.iter_ops(CK.TENSOR):
            if not prod.memtokens() and state.count_from(CK.TENSOR, prod) > 0:
                deps.append((prod, CK.TENSOR))
    return deps


def collect_all_deps(
    inst: Instruction, state: CounterState
) -> List[Tuple[Instruction, CK]]:
    if is_wait_inst(inst):
        return []
    all_deps: List[Tuple[Instruction, CK]] = []
    seen: Set[Tuple[int, CK]] = set()
    for collector in (
        collect_register_deps,
        collect_lds_raw_deps,
        collect_war_deps,
        collect_conservative_deps,
    ):
        for prod, ck in collector(inst, state):
            key = (prod.uid, ck)
            if key not in seen:
                all_deps.append((prod, ck))
                seen.add(key)
    return all_deps


def required_wait_value(
    deps: List[Tuple[Instruction, CK]], state: CounterState
) -> Dict[CK, int]:
    required: Dict[CK, int] = {ck: K_UNUSED for ck in CK}
    for prod, ck in deps:
        n = state.count_from(ck, prod)
        if n <= 0:
            continue
        w = n - 1
        if required[ck] == K_UNUSED or w < required[ck]:
            required[ck] = w
    return required


# ---------------------------------------------------------------------------
# CFG utilities
# ---------------------------------------------------------------------------


def infer_successors(func: Function) -> None:
    block_names = list(func.blocks.keys())
    for i, (name, bb) in enumerate(func.blocks.items()):
        if bb.successors:
            continue
        for inst in reversed(bb.instructions):
            if is_branch(inst):
                for src in inst.srcs:
                    if isinstance(src, str) and not src.startswith("0"):
                        target = src.lstrip("^")
                        if target in func.blocks:
                            bb.successors.append(target)
                break
        if not bb.successors and i + 1 < len(block_names):
            bb.successors.append(block_names[i + 1])


def compute_predecessors(func: Function) -> Dict[str, List[str]]:
    preds: Dict[str, List[str]] = {name: [] for name in func.blocks}
    for name, bb in func.blocks.items():
        for succ in bb.successors:
            if succ in preds:
                preds[succ].append(name)
    return preds


def compute_rpo(func: Function, preds: Dict[str, List[str]]) -> List[str]:
    visited: Set[str] = set()
    order: List[str] = []

    def dfs(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        bb = func.blocks.get(name)
        if bb:
            for succ in bb.successors:
                dfs(succ)
        order.append(name)

    dfs(func.entry)
    for name in func.blocks:
        if name not in visited:
            dfs(name)
    order.reverse()
    return order


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    wait_reports: List[WaitReport] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)


def _dump_violation_detail(
    consumer: Instruction,
    bb_name: str,
    ck: CK,
    undrained: List[Tuple[Instruction, CK]],
    state: CounterState,
) -> str:
    """Render the live per-pred queue snapshot for counter `ck` at the
    consumer, marking the undrained producer(s) and the wait each needs."""
    wait_op = COUNTER_WAIT_OPS[ck]
    bad_uids = {p.uid for p, c in undrained if c == ck}

    lines: List[str] = []
    lines.append(
        f"    Queue dump for {wait_op} at consumer {consumer.label()} in ^{bb_name}:"
    )

    # Required wait: strictest (smallest keep) that still drains every
    # undrained producer on this counter, i.e. min over producers of
    # (count_from - 1).
    required = K_UNUSED
    for prod, c in undrained:
        if c != ck:
            continue
        n = state.count_from(ck, prod)
        if n > 0:
            w = n - 1
            required = w if required == K_UNUSED else min(required, w)
    req_str = "none" if required == K_UNUSED else str(required)
    lines.append(
        f"      required (drains ALL {len(bad_uids)} undrained producer(s) "
        f"this consumer uses): {wait_op} {req_str}"
    )

    queues = state.queues[ck]
    lines.append(f"      live queues ({ck.value}): {len(queues)} per-pred queue(s)")
    if not queues:
        lines.append("        (empty — nothing in flight)")
    for qi, q in enumerate(queues):
        pred = q.pred if q.pred is not None else "<in-block>"
        lines.append(f"        queue[{qi}] pred=^{pred} depth={len(q.ops)}")
        size = len(q.ops)
        for idx, op in enumerate(q.ops):
            drain = size - idx - 1  # keep value that drains this op
            mark = "  <-- undrained producer" if op.uid in bad_uids else ""
            lines.append(
                f"          idx {idx}: {op.opcode} @ line {op.line} "
                f"(drain with {wait_op} {drain}){mark}"
            )
    return "\n".join(lines)


class WaitCntValidator:
    def __init__(self, num_waves: int = 0) -> None:
        self.num_waves = num_waves
        self.result = ValidationResult()
        # Counters that already have a reported violation; once a queue has
        # its first violation we stop reporting further ones for it.
        self._reported_counters: Set[CK] = set()

    def validate_function(self, func: Function) -> ValidationResult:
        infer_successors(func)
        preds = compute_predecessors(func)
        rpo = compute_rpo(func, preds)

        exit_states: Dict[str, CounterState] = {}
        max_iter = min(256, max(K_MAX_IN_FLIGHT + 8, 2 * len(func.blocks)))

        for _ in range(max_iter):
            changed = False
            for bb_name in rpo:
                bb = func.blocks[bb_name]
                entry = self._merge_entry(preds, bb_name, exit_states)
                exit_state = self._process_block(bb, entry, collect_reports=False)
                prev = exit_states.get(bb_name)
                if prev is None or not _states_equal(prev, exit_state):
                    exit_states[bb_name] = exit_state
                    changed = True
            if not changed:
                break

        self._reported_counters.clear()
        for bb_name in rpo:
            bb = func.blocks[bb_name]
            entry = self._merge_entry(preds, bb_name, exit_states)
            self._process_block(bb, entry, collect_reports=True)
            if len(self._reported_counters) == len(CK):
                break

        return self.result

    def _merge_entry(
        self,
        preds: Dict[str, List[str]],
        bb_name: str,
        exit_states: Dict[str, CounterState],
    ) -> CounterState:
        entry = CounterState(num_waves=self.num_waves)
        for p in preds.get(bb_name, []):
            if p in exit_states:
                entry.seed_from_pred(p, exit_states[p])
        return entry

    def _process_block(
        self, bb: BasicBlock, state: CounterState, *, collect_reports: bool
    ) -> CounterState:
        state = state.copy()
        state.num_waves = self.num_waves

        for inst in bb.instructions:
            if is_wait_inst(inst):
                self._process_wait(inst, state, collect_reports=collect_reports)
                continue

            deps = collect_all_deps(inst, state)
            undrained = [(p, ck) for p, ck in deps if state.count_from(ck, p) > 0]
            if undrained and collect_reports:
                for prod, ck in undrained:
                    # Report only the FIRST violation per counter/queue, then
                    # stop tracking that queue (one diagnostic per s_wait_* kind).
                    if ck in self._reported_counters:
                        continue
                    self._reported_counters.add(ck)
                    n = state.count_from(ck, prod)
                    w_needed = n - 1
                    self.result.violations.append(
                        Violation(
                            kind="MISSING",
                            consumer=inst,
                            producer=prod,
                            counter=ck,
                            message=(
                                f"Consumer {inst.label()} in ^{bb.name} uses async "
                                f"producer {prod.label()} ({COUNTER_WAIT_OPS[ck]}) "
                                f"still in-flight (queue depth={n}, needs wait "
                                f"{COUNTER_WAIT_OPS[ck]} {w_needed})"
                            ),
                            detail=_dump_violation_detail(
                                inst, bb.name, ck, undrained, state
                            ),
                        )
                    )

            ck_prod = classify_counter(inst)
            if ck_prod is not None:
                state.append_producer(inst, ck_prod)

            for reg in inst.dests:
                state.reg_defs[reg] = inst

        return state

    def _process_wait(
        self, inst: Instruction, state: CounterState, *, collect_reports: bool
    ) -> None:
        counts = get_wait_counts(inst)
        for ck, keep in counts.items():
            drained = state.trim_queue(ck, keep)
            if collect_reports:
                self.result.wait_reports.append(
                    WaitReport(inst=inst, counter=ck, keep=keep, drained=drained)
                )
                if not drained:
                    self.result.infos.append(
                        f"INFO: {inst.label()} {COUNTER_WAIT_OPS[ck]} {keep} "
                        f"— no in-flight ops drained (possibly redundant)"
                    )


def _states_equal(a: CounterState, b: CounterState) -> bool:
    for ck in CK:
        qa, qb = a.queues[ck], b.queues[ck]
        if len(qa) != len(qb):
            return False
        for x, y in zip(qa, qb):
            if x.pred != y.pred:
                return False
            if [o.uid for o in x.ops] != [o.uid for o in y.ops]:
                return False
    if a.reg_defs.keys() != b.reg_defs.keys():
        return False
    for reg, prod in a.reg_defs.items():
        if prod.uid != b.reg_defs[reg].uid:
            return False
    return True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(
    func: Function, result: ValidationResult, verbose: bool = False
) -> str:
    lines: List[str] = []
    lines.append(f"=== Waitcnt Validation: @{func.name} ===")
    lines.append("")

    if verbose and result.wait_reports:
        lines.append("--- Wait Instructions ---")
        for wr in result.wait_reports:
            op = COUNTER_WAIT_OPS[wr.counter]
            lines.append(
                f"  {op} {wr.keep} @ line {wr.inst.line} " f"(block ^{wr.inst.block})"
            )
            if wr.drained:
                lines.append("    Waiting for:")
                for prod in wr.drained:
                    ck = classify_counter(prod)
                    ck_name = ck.value if ck else "?"
                    lines.append(
                        f"      - [{ck_name}] {prod.opcode} @ line {prod.line} "
                        f"(block ^{prod.block})"
                    )
            else:
                lines.append("    Waiting for: (nothing in-flight)")
        lines.append("")

    if verbose and result.infos:
        lines.append("--- Notes ---")
        for info in result.infos:
            lines.append(f"  {info}")
        lines.append("")

    if result.violations:
        lines.append("--- Violations ---")
        for v in result.violations:
            lines.append(f"  {v.kind}: {v.message}")
            if v.detail:
                lines.append(v.detail)
        lines.append("")
        lines.append(f"RESULT: FAIL ({len(result.violations)} violation(s))")
    else:
        lines.append("RESULT: PASS")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate s_wait_* insertion in STIR files."
    )
    parser.add_argument("stir_file", help="Path to STIR file to validate")
    parser.add_argument(
        "--num-waves",
        type=int,
        default=0,
        help="NumWaves tile config (0=multi-wave: tensor waits only at barriers)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full report (wait instructions + notes) in addition to "
        "violations; default prints only violations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary line",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.stir_file, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        print(f"Error reading {args.stir_file}: {e}", file=sys.stderr)
        return 2

    try:
        funcs = Parser(source).parse()
    except SyntaxError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 2

    if not funcs:
        print("No functions found in STIR file.", file=sys.stderr)
        return 2

    exit_code = 0
    for func in funcs:
        validator = WaitCntValidator(num_waves=args.num_waves)
        result = validator.validate_function(func)
        if args.quiet:
            status = "PASS" if not result.violations else "FAIL"
            print(f"@{func.name}: {status} ({len(result.violations)} violations)")
        else:
            print(format_report(func, result, verbose=args.verbose))
        if result.violations:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
