# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx12 SGPR read hazards: find them in assembly text and insert the waits that cover them.

On gfx1200/gfx1201, once a VALU instruction has read an SGPR pair, later writes to that pair are not
interlocked against later reads:

* after an SALU write, an SALU or VALU read needs ``s_wait_alu depctr_sa_sdst(0)`` first;
* after a VALU write, a VALU read needs ``s_wait_alu depctr_va_sdst(0)`` first
  (``depctr_va_vcc(0)`` when the register is VCC).

Without the wait the read can return the register's old value. LLVM covers this with
AMDGPUWaitSGPRHazards; TensileLite writes assembly directly, so this module applies the same
state machine to the kernel text: SGPR pairs become tracked on their first VALU read, SMEM and
non-FLAT VMEM instructions retire pending writes, and an SALU read of a tracked SGPR retires
pending VALU writes. Control flow comes from labels and branches; indirect jumps
(``s_setpc_b64``, ``s_swappc_b64``) flush every pending write first, and a label whose address
is taken, or that has no known predecessor, is entered with every pair tracked and an SALU write
pending on every SGPR (a call's return address is written just before the callee reads it).

A missed SGPR operand would silently drop waits, so the parser fails closed: an operand,
instruction or directive it cannot classify raises SgprHazardError.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

WA_VALU = 1
WA_SALU = 2
WA_VCC = 4

VCC_LO = 106
VCC_HI = 107
DS_NOP_CULL = 4  # wave32
ALL_PAIRS = (1 << 64) - 1


class SgprHazardError(RuntimeError):
    """The kernel holds something the analysis cannot model, so no safe set of waits is known."""


_SMEM_PREFIXES = ("s_load_", "s_buffer_load_", "s_store_", "s_buffer_store_", "s_atomic",
                  "s_buffer_atomic", "s_scratch_", "s_dcache_", "s_prefetch_", "s_buffer_prefetch",
                  "s_atc_probe", "s_memtime", "s_memrealtime")
_VMEM_PREFIXES = ("buffer_", "tbuffer_", "image_")
# Instruction families with no SGPR hazard semantics (LLVM's pass only looks at SALU and VALU).
_OTHER_PREFIXES = ("ds_", "global_", "flat_", "scratch_", "exp", "tensor_")
_COND_BRANCH = re.compile(r"^s_cbranch_\w+$")
_SGPR = re.compile(r"^s(\d+)$")
_SGPR_INDEXED = re.compile(r"^s\[([^\]]+)\]$")  # s[3:4], s[sgprSrdA:sgprSrdA+3], s[sgprAlpha]
# Anything an SGPR operand could be spelled with; seen anywhere it did not parse, it is an error.
_SGPR_LIKE = re.compile(r"(?<![\w.$])(?:s\[|s\d|vcc)")
_SET = re.compile(r"^\s*\.set\s+([A-Za-z_.$][\w.$]*)\s*,\s*(.+?)\s*$")
_EXPR_OK = re.compile(r"^[\d\s+\-*()x<>a-fA-F]*$")
_IDENT = re.compile(r"[A-Za-z_.$][\w.$]*")
_LABEL_DEF = re.compile(r"^\s*([A-Za-z_.$][\w.$]*):(?!\S)")
_OBJDUMP_LABEL = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:\s*$")
_OBJDUMP_ADDR = re.compile(r"//\s*([0-9A-Fa-f]{12}):\s*([0-9A-Fa-f]{8})?")
_DEPCTR_FIELD = re.compile(r"depctr_(\w+)\((\d+)\)")
# Directives that would make the text something other than one straight-line program per block.
_UNSUPPORTED_DIRECTIVES = re.compile(r"^\.(if\w*|else\w*|endif|rept|irp\w*|endr|byte|short|word|"
                                     r"long|quad|int|[248]byte|inst\w*)$")

# SALU mnemonics whose first operand is a source, not a destination.
_SALU_NO_DST = re.compile(r"^s_(cmp\w*|cmpk\w*|bitcmp\w*|setreg_b32|setpc_b64|rfe_b64|cbranch\w*|"
                          r"branch|nop|wait\w*|barrier\w*|delay_alu|setprio|sleep\w*|endpgm\w*|"
                          r"trap|sendmsg|sendmsghalt|ttrace\w*|icache_inv|set_inst_prefetch_distance|"
                          r"clause|setreg_imm32_b32|singleuse_vdst|version|code_end|setkill|"
                          r"setvskip|round_mode|denorm_mode|incperflevel|decperflevel|wakeup\w*|"
                          r"alloc_vgpr|set_vgpr_msb)$")
# SALU mnemonics whose first operand is a destination.
_SALU_DST = re.compile(r"^s_(mov|cmov|add|sub|mul|and|or|xor|nand|nor|xnor|lshl|lshr|ashr|bfe|bfm|"
                       r"not|wqm|brev|bcnt|ff0|ff1|flbit|ctz|clz|cls|sext|bitset|abs|min|max|"
                       r"cselect|getpc|swappc|getreg|quadmask|pack|cvt|ceil|floor|trunc|rndne|fmac|"
                       r"fmamk|fmaak|get_barrier_state|sendmsg_rtn)\w*$")
# SALU mnemonics that also read their destination.
_SALU_DST_READ = re.compile(r"^s_(cmov\w*|addk\w*|mulk\w*|bitset[01]\w*|fmac\w*)$")
# VALU mnemonics with an SGPR (or VCC) carry/condition output as their second operand.
_VALU_SDST2 = re.compile(r"^v_(add_co|sub_co|subrev_co|addc_co|subb_co|subbrev_co|div_scale|"
                         r"mad_u64_u32|mad_i64_i32|mad_co_u64_u32|mad_co_i64_i32)")
# VALU mnemonics whose first operand is an SGPR destination.
_VALU_SDST1 = re.compile(r"^v_(readfirstlane|readlane|cmp_|cmp_class|s_)")


@dataclass
class Op:
    """One instruction or label of the assembly text."""

    line: int
    kind: str  # "label", "salu", "valu", "clear", "waitalu", "dsnop", "other"
    mnemonic: str = ""
    label: str = ""
    uses: List[Tuple[int, int]] = field(default_factory=list)  # (first SGPR, count)
    defs: List[Tuple[int, int]] = field(default_factory=list)
    target: Optional[str] = None  # direct branch target
    uncond: bool = False  # no fallthrough
    setpc: bool = False  # call, return or indirect branch
    call: bool = False
    waitClears: int = 0  # WA_* bits an existing s_wait_alu retires
    refs: Tuple[str, ...] = ()  # identifiers in operands of non-branch instructions
    addr: Optional[int] = None  # objdump address
    targetAddr: Optional[int] = None  # objdump branch target (labels at one address print one name)
    col: int = 0  # where the instruction starts on its line, after a label on the same line
    raw: str = ""


def _stripComment(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text).split("//", 1)[0]
    if "/*" in text:
        raise SgprHazardError(f"multi-line comment is not supported: {text.strip()!r}")
    return text if text.lstrip().startswith(".") else text.split(";", 1)[0]


def _splitOperands(text: str) -> List[str]:
    ops, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        ops.append("".join(cur).strip())
    return ops


def _firstToken(text: str) -> Tuple[str, str]:
    """Split an operand into its first whitespace-separated token and the modifiers after it."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch.isspace() and depth == 0:
            return text[:i], text[i:]
    return text, ""


def evalExpr(expr: str, symbols: Dict[str, Optional[int]]) -> Optional[int]:
    """Value of an assembler expression over integers and .set symbols; None if unknown."""
    unknown = False

    def sub(m):
        nonlocal unknown
        name = m.group(0)
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", name):
            return name
        value = symbols.get(name)
        if value is None:
            unknown = True
            return "0"
        return str(value)

    text = re.sub(r"0x[0-9a-fA-F]+|[A-Za-z_.$][\w.$]*|\d+", sub, expr)
    if unknown or not _EXPR_OK.match(text):
        return None
    try:
        return int(eval(text, {"__builtins__": {}}, {}))  # digits and operators only
    except Exception:
        return None


def _parseSgpr(token: str, symbols: Dict[str, Optional[int]]) -> Optional[Tuple[int, int]]:
    t = token
    for prefix in ("neg(", "abs(", "sext("):
        if t.startswith(prefix) and t.endswith(")"):
            t = t[len(prefix):-1]
    t = t.lstrip("-").strip("|").strip()
    if t in ("vcc", "vcc_lo"):
        return (VCC_LO, 2 if t == "vcc" else 1)
    if t == "vcc_hi":
        return (VCC_HI, 1)
    m = _SGPR.match(t)
    if m:
        return (int(m.group(1)), 1)
    m = _SGPR_INDEXED.match(t)
    if m:
        lo, _, hi = m.group(1).partition(":")
        first = evalExpr(lo, symbols)
        last = evalExpr(hi, symbols) if hi else first
        if first is None or last is None:
            raise SgprHazardError(f"cannot resolve SGPR operand {token!r}")
        return (first, last - first + 1)
    return None


def _sgprOperand(operand: str, symbols: Dict[str, Optional[int]]) -> Optional[Tuple[int, int]]:
    token, rest = _firstToken(operand.strip())
    reg = _parseSgpr(token, symbols)
    if _SGPR_LIKE.search(rest) or (reg is None and _SGPR_LIKE.search(token)):
        raise SgprHazardError(f"unrecognized SGPR operand {operand!r}")
    return reg


def _waitClears(operands: str) -> int:
    clears = 0
    fields = dict((k, int(v)) for k, v in _DEPCTR_FIELD.findall(operands))
    if fields:
        if fields.get("sa_sdst", 1) == 0:
            clears |= WA_SALU
        if fields.get("va_sdst", 7) == 0:
            clears |= WA_VALU
        if fields.get("va_vcc", 1) == 0:
            clears |= WA_VCC
        return clears
    m = re.match(r"\s*(0x[0-9a-fA-F]+|\d+)\s*$", operands)
    if not m:
        raise SgprHazardError(f"unrecognized s_wait_alu operand {operands.strip()!r}")
    imm = int(m.group(1), 0)
    if (imm & 1) == 0:
        clears |= WA_SALU
    if ((imm >> 9) & 7) == 0:
        clears |= WA_VALU
    if ((imm >> 1) & 1) == 0:
        clears |= WA_VCC
    return clears


def _parseInstruction(lineNo: int, text: str, symbols: Dict[str, Optional[int]],
                      addr: Optional[int] = None, encoding: Optional[int] = None) -> Op:
    if "::" in text and text.startswith("v_dual_"):
        halves = [_parseInstruction(lineNo, h.strip(), symbols) for h in text.split("::")]
        merged = Op(lineNo, "valu", mnemonic=halves[0].mnemonic, addr=addr)
        for h in halves:
            merged.uses += h.uses
            merged.defs += h.defs
        return merged
    parts = text.split(None, 1)
    mnemonic = parts[0].lower()
    operandText = parts[1] if len(parts) > 1 else ""
    op = Op(lineNo, "other", mnemonic=mnemonic, addr=addr)

    if mnemonic == "ds_nop":
        op.kind = "dsnop"
        return op
    if mnemonic.startswith(_VMEM_PREFIXES) or mnemonic.startswith(_SMEM_PREFIXES):
        op.kind = "clear"
        return op
    if mnemonic in ("s_wait_alu", "s_waitcnt_depctr"):
        op.kind = "waitalu"
        op.waitClears = _waitClears(operandText)
        return op
    if mnemonic.startswith(_OTHER_PREFIXES):
        return op
    if not (mnemonic.startswith("s_") or mnemonic.startswith("v_")):
        raise SgprHazardError(f"unknown instruction {text!r}: if it has no SGPR operands, add its "
                              "prefix to _OTHER_PREFIXES in tensilelite/SgprHazards.py")

    operands = _splitOperands(operandText)
    if mnemonic == "s_branch" or _COND_BRANCH.match(mnemonic):
        op.kind = "salu"
        if operands:
            op.target = operands[0]
        if addr is not None and encoding is not None:
            simm16 = encoding & 0xFFFF
            op.targetAddr = addr + 4 + 4 * (simm16 - 0x10000 if simm16 & 0x8000 else simm16)
        op.uncond = mnemonic == "s_branch"
        if mnemonic in ("s_cbranch_vccz", "s_cbranch_vccnz"):
            op.uses.append((VCC_LO, 1))
        return op

    op.refs = tuple(t for o in operands for t in _IDENT.findall(o))
    regs = [_sgprOperand(o, symbols) for o in operands]
    hasSgpr = any(r is not None for r in regs)

    if mnemonic.startswith("s_"):
        op.kind = "salu"
        if mnemonic.startswith("s_movrel"):
            raise SgprHazardError(f"M0-relative SGPR access is not supported: {text!r}")
        if mnemonic in ("s_setpc_b64", "s_rfe_b64"):
            op.setpc, op.uncond = True, True
        elif mnemonic in ("s_swappc_b64", "s_call_b64"):
            op.setpc, op.call = True, True
        elif mnemonic.startswith("s_endpgm"):
            op.uncond = True
        if _SALU_NO_DST.match(mnemonic):
            hasDst = False
        elif _SALU_DST.match(mnemonic) or mnemonic == "s_call_b64":
            hasDst = True
        elif hasSgpr:
            raise SgprHazardError(f"unknown SALU instruction {mnemonic!r} with SGPR operands: add it "
                                  "to _SALU_DST or _SALU_NO_DST in tensilelite/SgprHazards.py")
        else:
            hasDst = False
        for i, r in enumerate(regs):
            if r is None:
                continue
            if i == 0 and hasDst:
                op.defs.append(r)
                if _SALU_DST_READ.match(mnemonic):
                    op.uses.append(r)
            else:
                op.uses.append(r)
        return op

    op.kind = "valu"
    if mnemonic.startswith("v_cmpx"):
        nDst = 0
    elif _VALU_SDST1.match(mnemonic):
        nDst = 1
        if mnemonic.startswith("v_cmp") and (not regs or regs[0] is None):
            op.defs.append((VCC_LO, 1))  # VOPC with implicit VCC
            nDst = 0
    else:
        nDst = 1  # VGPR destination
        if regs and regs[0] is not None:
            raise SgprHazardError(f"unknown VALU instruction {mnemonic!r} writing an SGPR: add it to "
                                  "_VALU_SDST1 or _VALU_SDST2 in tensilelite/SgprHazards.py")
        if _VALU_SDST2.match(mnemonic) and len(regs) > 1 and regs[1] is not None:
            op.defs.append(regs[1])
            nDst = 2
    for i, r in enumerate(regs):
        if r is None:
            continue
        if i < nDst:
            op.defs.append(r)
        else:
            op.uses.append(r)
    if mnemonic.startswith("v_div_fmas") or mnemonic == "v_dual_cndmask_b32":
        op.uses.append((VCC_LO, 1))
    return op


def parseLine(lineNo: int, raw: str, symbols: Optional[Dict[str, Optional[int]]] = None,
              objdump: bool = False) -> List[Op]:
    """The label and instruction on one line of TensileLite (or, with *objdump*, llvm-objdump) text.

    *symbols* holds the .set values in effect at this line, for operands such as s[sgprSrdA+2].
    """
    symbols = symbols if symbols is not None else {}
    if objdump:
        m = _OBJDUMP_LABEL.match(raw)
        if m:
            return [Op(lineNo, "label", label=m.group(2), addr=int(m.group(1), 16))]
        am = _OBJDUMP_ADDR.search(raw)
        if not am:
            return []
        encoding = int(am.group(2), 16) if am.group(2) else None
        return [_parseInstruction(lineNo, _stripComment(raw).strip(), symbols,
                                  int(am.group(1), 16), encoding)]
    text = _stripComment(raw).strip()
    if not text or text.startswith("."):
        return []
    ops = []
    m = _LABEL_DEF.match(text)
    if m:
        ops.append(Op(lineNo, "label", label=m.group(1)))
        text = text[m.end():].strip()
        if not text:
            return ops
    op = _parseInstruction(lineNo, text, symbols)
    if ops:
        colon = raw.index(":", raw.index(ops[0].label) + len(ops[0].label)) + 1
        op.col = colon + len(raw[colon:]) - len(raw[colon:].lstrip())
    return ops + [op]


@dataclass
class State:
    tracked: int = 0
    salu: int = 0  # 128-bit set of SGPRs with uncommitted SALU writes
    valu: int = 0  # 128-bit set of SGPRs with uncommitted VALU writes
    vcc: int = 0  # WA_SALU / WA_VALU source of the pending VCC write

    def copy(self) -> "State":
        return State(self.tracked, self.salu, self.valu, self.vcc)

    def merge(self, o: "State") -> bool:
        n = State(self.tracked | o.tracked, self.salu | o.salu, self.valu | o.valu, self.vcc | o.vcc)
        changed = (n.tracked, n.salu, n.valu, n.vcc) != (self.tracked, self.salu, self.valu, self.vcc)
        self.tracked, self.salu, self.valu, self.vcc = n.tracked, n.salu, n.valu, n.vcc
        return changed


def _isVcc(reg: int) -> bool:
    return reg in (VCC_LO, VCC_HI)


def _regs(operand: Tuple[int, int]):
    first, count = operand
    return range(first, min(first + count, 128))


def _step(s: State, op: Op, dsNops: List[int]) -> int:
    """Advance the state over one op; return the WA_* waits it needs before it."""
    if op.kind == "dsnop":
        dsNops[0] += 1
        if dsNops[0] >= DS_NOP_CULL:
            s.tracked = 0
        return 0
    dsNops[0] = 0
    if op.kind == "clear":
        s.salu = s.valu = s.vcc = 0
        return 0
    if op.kind == "waitalu":
        if op.waitClears & WA_VCC:
            s.vcc &= ~WA_VALU
        if op.waitClears & WA_SALU:
            s.salu = 0
            s.vcc &= ~WA_SALU
        if op.waitClears & WA_VALU:
            s.valu = 0
        return 0
    if op.kind not in ("salu", "valu"):
        return 0
    isSalu = op.kind == "salu"
    wait = 0
    if op.setpc:
        if s.vcc & WA_VALU:
            wait |= WA_VCC
        if s.salu or (s.vcc & WA_SALU):
            wait |= WA_SALU
        if s.valu:
            wait |= WA_VALU
    else:
        seen = set()
        for operand in op.uses:
            for r in _regs(operand):
                if r in seen:
                    continue
                seen.add(r)
                pair = 1 << ((r >> 1) & 0x3F)
                if not s.tracked & pair:
                    if not isSalu:
                        s.tracked |= pair
                    continue
                if isSalu:
                    if _isVcc(r):
                        if s.vcc & WA_VALU:
                            s.vcc = 0
                    else:
                        s.valu = 0
                if (s.salu >> r) & 1:
                    wait |= WA_SALU
                if not isSalu and (s.valu >> r) & 1:
                    wait |= WA_VALU
                if _isVcc(r) and s.vcc:
                    if s.vcc & WA_SALU:
                        wait |= WA_SALU
                    if s.vcc & WA_VALU:
                        wait |= WA_VCC
    if wait & WA_VCC:
        s.vcc &= ~WA_VALU
    if wait & WA_SALU:
        s.salu = 0
        s.vcc &= ~WA_SALU
    if wait & WA_VALU:
        s.valu = 0
    if op.call:
        s.tracked = ALL_PAIRS
    seen = set()
    for operand in op.defs:
        for r in _regs(operand):
            if r in seen:
                continue
            seen.add(r)
            if not s.tracked & (1 << ((r >> 1) & 0x3F)):
                continue
            if _isVcc(r):
                s.vcc = WA_SALU if isSalu else WA_VALU
            elif isSalu:
                s.salu |= 1 << r
            else:
                s.valu |= 1 << r
    return wait


def _objdumpIndirectTargets(ops: List[Op]) -> set:
    """llvm-objdump prints s_getpc/s_add jump offsets as numbers; recover the target addresses."""
    targets = set()
    for i, op in enumerate(ops):
        if op.mnemonic != "s_getpc_b64" or op.addr is None:
            continue
        for nxt in ops[i + 1:i + 4]:
            if nxt.mnemonic in ("s_add_co_i32", "s_add_i32"):
                m = re.search(r",\s*(0x[0-9a-fA-F]+|-?\d+),\s*4\s*$", _stripComment(nxt.raw))
                if m:
                    targets.add(op.addr + 4 + int(m.group(1), 0) + 4)
                break
    return targets


def _parse(lines: List[str]) -> List[Op]:
    objdump = any(_OBJDUMP_ADDR.search(l) for l in lines[:200]) or \
        any(_OBJDUMP_LABEL.match(l) for l in lines[:200])
    ops: List[Op] = []
    symbols: Dict[str, Optional[int]] = {}
    macros = set()
    block = None  # .macro, .amdhsa_kernel or .amdgpu_metadata body being skipped
    ends = {".macro": ".endm", ".amdhsa_kernel": ".end_amdhsa_kernel",
            ".amdgpu_metadata": ".end_amdgpu_metadata"}
    for i, raw in enumerate(lines):
        directive = raw.strip().split(None, 1)[0] if raw.strip() else ""
        if block is not None:
            if directive == ends[block]:
                block = None
            continue
        if directive in ends:
            block = directive
            if directive == ".macro":
                macros.add(raw.split()[1].split(",")[0])
            continue
        if not objdump and directive.startswith("."):
            if directive == ".set":
                m = _SET.match(_stripComment(raw))
                if m:
                    symbols[m.group(1)] = evalExpr(m.group(2), symbols)
            elif _UNSUPPORTED_DIRECTIVES.match(directive):
                raise SgprHazardError(f"directive {directive} is not supported in kernel code")
            continue
        if directive in macros:
            raise SgprHazardError(f"macro invocation {directive} is not supported")
        for op in parseLine(i, raw, symbols, objdump):
            op.raw = raw
            ops.append(op)
    return ops


def analyze(lines: List[str]) -> Tuple[List[Op], Dict[int, int]]:
    """Return the parsed ops and {op index: WA_* waits needed before that op}."""
    ops = _parse(lines)
    labels = {op.label: i for i, op in enumerate(ops) if op.kind == "label"}
    addressTaken = {name for op in ops if op.kind in ("salu", "valu") and op.target is None
                    for name in op.refs if name in labels}
    indirectAddrs = _objdumpIndirectTargets(ops)
    addressTaken |= {op.label for op in ops if op.kind == "label" and op.addr in indirectAddrs}

    # Basic blocks: start at labels and after control transfers.
    starts = [0]
    for i, op in enumerate(ops):
        if op.kind == "label" and i != 0:
            starts.append(i)
        if (op.target is not None or op.setpc or op.uncond) and i + 1 < len(ops):
            starts.append(i + 1)
    starts = sorted(set(starts))
    blockOf = {}
    blocks = []
    for b, st in enumerate(starts):
        end = starts[b + 1] if b + 1 < len(starts) else len(ops)
        blocks.append((st, end))
        for i in range(st, end):
            blockOf[i] = b

    labelAddrs = {op.addr: i for i, op in enumerate(ops) if op.kind == "label" and op.addr is not None}
    succs: List[List[int]] = [[] for _ in blocks]
    preds: List[int] = [0] * len(blocks)
    for b, (st, end) in enumerate(blocks):
        if st == end:
            continue
        last = ops[end - 1]
        if last.targetAddr is not None and last.targetAddr in labelAddrs:
            succs[b].append(blockOf[labelAddrs[last.targetAddr]])
        elif last.target is not None and last.target in labels:
            succs[b].append(blockOf[labels[last.target]])
        if not last.uncond and end < len(ops):
            succs[b].append(blockOf[end])
        for t in succs[b]:
            preds[t] += 1

    entry = [State() for _ in blocks]
    for b, (st, end) in enumerate(blocks):
        first = ops[st] if st < end else None
        isLabel = first is not None and first.kind == "label"
        if b != 0 and (preds[b] == 0 or (isLabel and first.label in addressTaken)):
            entry[b] = State(ALL_PAIRS, (1 << 128) - 1, 0, WA_SALU)

    out = [None] * len(blocks)
    work = list(range(len(blocks)))
    inWork = set(work)
    while work:
        b = work.pop(0)
        inWork.discard(b)
        s = entry[b].copy()
        dsNops = [0]
        for i in range(*blocks[b]):
            _step(s, ops[i], dsNops)
        if out[b] is not None and (s.tracked, s.salu, s.valu, s.vcc) == out[b]:
            continue
        out[b] = (s.tracked, s.salu, s.valu, s.vcc)
        for t in succs[b]:
            if entry[t].merge(s) and t not in inWork:
                work.append(t)
                inWork.add(t)

    waits: Dict[int, int] = {}
    for b in range(len(blocks)):
        s = entry[b].copy()
        dsNops = [0]
        for i in range(*blocks[b]):
            w = _step(s, ops[i], dsNops)
            if w:
                waits[i] = w
    return ops, waits


_FIELDS = ((WA_VALU, "va_sdst"), (WA_VCC, "va_vcc"), (WA_SALU, "sa_sdst"))


def waitText(w: int) -> str:
    return "s_wait_alu " + " ".join(f"depctr_{name}(0)" for bit, name in _FIELDS if w & bit)


def _mergeWait(raw: str, w: int) -> str:
    code, sep, comment = raw.partition("//")
    code = code.rstrip()
    for bit, name in _FIELDS:
        if w & bit:
            field_ = re.compile(rf"depctr_{name}\(\d+\)")
            code = field_.sub(f"depctr_{name}(0)", code) if field_.search(code) else \
                f"{code} depctr_{name}(0)"
    return f"{code} {sep}{comment}" if sep else code


def withSgprHazardWaits(asm: str, archCaps) -> str:
    """*asm* with the waits added when rocisa reports the hazard for the target, else *asm*."""
    if not archCaps["HasVALUReadSGPRHazard"]:
        return asm
    return insertSgprHazardWaits(asm)[0]


def insertSgprHazardWaits(asm: str) -> Tuple[str, int]:
    """Insert the s_wait_alu instructions gfx1200/gfx1201 need for SGPR read hazards.

    A wait goes in front of an s_delay_alu that precedes its instruction, so the delay hint keeps
    its target, and folds into an s_wait_alu right before it. Returns the new assembly and the
    number of hazard sites covered.
    """
    lines = asm.split("\n")
    ops, waits = analyze(lines)
    if not waits:
        return asm, 0
    inserts: Dict[int, int] = {}  # op index -> waits to put in front of it
    merges: Dict[int, int] = {}  # index of an existing s_wait_alu -> fields to add to it
    for i, w in waits.items():
        j = i
        if j > 0 and ops[j - 1].mnemonic == "s_delay_alu":
            j -= 1
        if j > 0 and ops[j - 1].mnemonic == "s_getpc_b64":
            # s_add_i32 sY, label, 4 after s_getpc_b64 assumes it is the instruction at the PC.
            raise SgprHazardError(f"a wait would split the s_getpc_b64 sequence at {ops[j].raw!r}")
        if j > 0 and ops[j - 1].kind == "waitalu" and _DEPCTR_FIELD.search(ops[j - 1].raw):
            merges[j - 1] = merges.get(j - 1, 0) | w
        else:
            inserts[j] = inserts.get(j, 0) | w
    byLine = {ops[j].line: (ops[j].col, w) for j, w in inserts.items()}
    mergeLines = {ops[j].line: w for j, w in merges.items()}
    out = []
    for n, raw in enumerate(lines):
        if n in mergeLines:
            raw = _mergeWait(raw, mergeLines[n])
        if n not in byLine:
            out.append(raw)
            continue
        col, w = byLine[n]
        indent = raw[:len(raw) - len(raw.lstrip())]
        if col:
            out.append(raw[:col].rstrip())
            raw = indent + raw[col:]
        out.append(f"{indent}{waitText(w)} // SGPR read hazard")
        out.append(raw)
    return "\n".join(out), len(waits)
