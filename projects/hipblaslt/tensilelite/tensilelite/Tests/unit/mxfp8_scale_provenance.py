# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which scale bytes each MFMA of a dual-period MXFP8 kernel consumes.

Executes the LDS/GEMM region of an emitted gfx950 subtile kernel (DepthU 128,
direct-to-LDS loads, 256-K scale epoch) for one K and StreamK local start,
from the loop-counter setup to ``label_Summation_End``.  Every branch there is
uniform and depends only on K and the start, so a scalar interpreter resolves
each to one path through preloop, main loop, NGLL/NLL and tail.

Along the path it tracks the SRD offset and LDS write half of every operand,
the read half of every local-read base, and which load last wrote each LDS
range.  For every ``v_mfma_scale`` it takes the data DepthU from the SRD offset
of its A/B operands, and follows each scale operand back through VALU masks and
shifts to its ds_read and the load that wrote those bytes.  The scale is right
iff that load's epoch is DU // 2 and the half selected by op_sel, op_sel_hi and
any shift is DU % 2.

Only the instructions this region uses are modelled; anything else that writes
a tracked register poisons it, which shows up as an unresolved operand.
"""
import re
from collections import defaultdict

DEPTHU = 128
SCALE_EPOCH_BYTES = 256
TENSORS = ("A", "B", "MXSA", "MXSB")
SCALE = ("MXSA", "MXSB")
_A_READ_BYTES = 2048     # 16 rows of one DepthU
_SCALE_READ_BYTES = 256

_COMMENT = re.compile(r"/\*(.*?)\*/")
_SYM = re.compile(r"\b((?:sgpr|vgpr|agpr)\w+)\b")
_REG = re.compile(r"^(s|v|acc|a)\[(-?\d+)(?::(-?\d+))?\]$|^(s|v|acc|a)(\d+)$")
_REG_RANGE = re.compile(r"(s|v|acc|a)\[([^\]]+)\]")


class _Ins:
    __slots__ = ("line", "op", "args", "mods", "label")

    def __init__(self, line, op, args, mods, label=None):
        self.line, self.op, self.args, self.mods, self.label = line, op, args, mods, label

    def mod(self, key, default=None):
        for m in self.mods:
            if m == key:
                return True
            if m.startswith(key + ":"):
                return m.split(":", 1)[1]
        return default


def _eval(expr, syms):
    expr = _SYM.sub(lambda m: str(syms[m.group(1)])
                    if syms.get(m.group(1)) is not None else m.group(1), expr)
    try:
        return int(eval(expr, {"__builtins__": {}}))
    except Exception:
        return None


def _parse(src):
    """Instructions with register symbols resolved, labels, and the symbols in
    force at the main loop (the region's SGPR names)."""
    syms, loop_syms, prog, labels = {}, None, [], {}
    for ln, raw in enumerate(src.splitlines(), 1):
        s = _COMMENT.sub("", raw).split("//", 1)[0].strip()
        if not s:
            continue
        if s.startswith(".set"):
            name, val = [x.strip() for x in s[4:].split(",", 1)]
            syms[name] = _eval(val, syms) if val != "UNDEF" else None
            continue
        if s.startswith("."):
            continue
        m = re.match(r"^(\w+):\s*$", s)
        if m:
            if m.group(1) == "label_LoopBeginL":
                loop_syms = dict(syms)
            labels[m.group(1)] = len(prog)
            prog.append(_Ins(ln, None, [], [], label=m.group(1)))
            continue

        def resolve(tok):
            def rng(mm):
                vals = [_eval(p, syms) for p in mm.group(2).split(":")]
                if any(v is None for v in vals):
                    return mm.group(0)
                return f"{mm.group(1)}[{':'.join(str(v) for v in vals)}]"
            return _REG_RANGE.sub(rng, tok)

        parts = s.replace(", ", ",").split()
        if parts[0] == "s_waitcnt":
            prog.append(_Ins(ln, parts[0], [], parts[1:]))
            continue
        args = [resolve(a) for a in parts[1].split(",")] if len(parts) > 1 else []
        prog.append(_Ins(ln, parts[0], args, parts[2:]))
    return prog, labels, loop_syms


def _regs(tok):
    """'v[20:27]' -> [('v', 20) .. ('v', 27)]; 'acc5' -> [('a', 5)]; else []."""
    if tok == "m0":
        return [("m0", 0)]
    m = _REG.match(tok)
    if not m:
        return []
    if m.group(1):
        f, lo = m.group(1), int(m.group(2))
        hi = int(m.group(3)) if m.group(3) is not None else lo
    else:
        f, lo = m.group(4), int(m.group(5))
        hi = lo
    f = "a" if f in ("acc", "a") else f
    return [(f, i) for i in range(lo, hi + 1)]


def _local_read_bases(src):
    """Local-read base VGPR -> (tensor, swap VGPR), from the allocation comments."""
    lr, sw = {}, {}
    for line in src.splitlines():
        m = re.search(r"Allocating v\[([\d, ]+)\] for (A|B|MXSA|MXSB) LR( Swap)?", line)
        if m:
            vs = [int(x) for x in m.group(1).split(",")]
            (sw if m.group(3) else lr).setdefault(m.group(2), vs)
        if "PRELOOP start" in line:
            break
    return {v: (t, s) for t, vs in lr.items() for v, s in zip(vs, sw[t])}


def _load_strides(prog, syms):
    """Per tensor, the LDS bytes one direct-to-LDS load covers (smallest m0 step)."""
    lw = {syms[f"sgprLocalWriteBaseAddr{t}"]: t for t in TENSORS}
    offs = defaultdict(set)
    for x in prog:
        if x.op == "s_add_u32" and x.args and x.args[0] == "m0":
            r = _regs(x.args[1])
            if r and r[0][1] in lw:
                offs[lw[r[0][1]]].add(int(x.args[2], 0))
    out = {}
    for t in TENSORS:
        s = sorted(offs.get(t, {0}))
        d = [b - a for a, b in zip(s, s[1:]) if b > a]
        out[t] = min(d) if d else None
    return out


class Kernel:
    def __init__(self, src):
        self.prog, self.labels, self.syms = _parse(src)
        assert self.syms is not None, "kernel has no main loop"
        self.lr = _local_read_bases(src)
        self.strides = _load_strides(self.prog, self.syms)

    def run(self, K, start=0, max_steps=400000):
        """The scaled MFMAs on the path for (K, start).

        Returns (checks, unresolved): one dict per scale operand with the
        DepthU it serves ('du') and the (epoch, half) it reads ('got'; epoch
        None when it reads LDS never written on this path), and the count of
        operands whose origin could not be followed."""
        prog, syms = self.prog, self.syms
        lw = {syms[f"sgprLocalWriteBaseAddr{t}"]: t for t in TENSORS}
        swap = {syms[f"sgprSwap{t}"]: t for t in TENSORS}
        srd = {syms[f"sgprSrd{t}"]: t for t in TENSORS}
        nIter = -(-K // DEPTHU)
        S = {syms["sgprStreamKLocalEnd"]: nIter, syms["sgprStreamKLocalStart"]: start,
             syms["sgprItersPerTile"]: nIter, syms["sgprSizesSum"]: K}
        scc = None
        lwPhase = {t: 0 for t in TENSORS}
        lrPhase = {v: 0 for v in self.lr}
        srdOff = {t: 0 for t in TENSORS}
        m0 = None
        mem = defaultdict(list)    # (tensor, half) -> [(lo, hi, identity, seq)]
        vval = {}
        seq = 0
        checks, unresolved = [], 0

        def sval(tok):
            if re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", tok):
                return int(tok, 0)
            r = _regs(tok)
            return S.get(r[0][1]) if len(r) == 1 and r[0][0] == "s" else None

        def sreg(tok):
            r = _regs(tok)
            return r[0][1] if len(r) == 1 and r[0][0] == "s" else None

        loopL, skEnd = syms["sgprLoopCounterL"], syms["sgprStreamKLocalEnd"]
        pc = next(i for i, x in enumerate(prog) if x.op == "s_sub_u32"
                  and x.args[:2] == [f"s[{loopL}]", f"s[{skEnd}]"])
        for _ in range(max_steps):
            ins = prog[pc]
            if ins.label is not None:
                if ins.label.startswith("label_Summation_End"):
                    return checks, unresolved
                pc += 1
                continue
            op, a, npc = ins.op, ins.args, pc + 1
            if op == "s_branch":
                npc = self.labels[a[0]]
            elif op in ("s_cbranch_scc0", "s_cbranch_scc1"):
                assert scc is not None, f"line {ins.line}: branch on unknown SCC"
                if scc == (1 if op.endswith("1") else 0):
                    npc = self.labels[a[0]]
            elif op == "s_cbranch_vccz":
                npc = self.labels[a[0]]    # alpha != 0
            elif op.startswith("s_cbranch") or op == "s_setpc_b64":
                raise AssertionError(f"line {ins.line}: unmodelled branch {op}")
            elif op == "s_endpgm":
                return checks, unresolved
            elif op in ("s_waitcnt", "s_barrier", "s_nop") or op.startswith(
                    ("s_load", "s_buffer_load", "s_store", "s_scratch", "s_atomic", "s_dcache")):
                pass
            elif op.startswith("s_"):
                dst = a[0] if a else None
                if dst == "m0":
                    sr = sreg(a[1]) if len(a) > 1 else None
                    if sr in lw and op in ("s_add_u32", "s_mov_b32"):
                        m0 = (lw[sr], lwPhase[lw[sr]], sval(a[2]) if op == "s_add_u32" else 0)
                    else:
                        m0 = None
                d = sreg(dst) if dst else None
                if d in lw:
                    t = lw[d]
                    if op == "s_xor_b32" and sreg(a[1]) == d and swap.get(sreg(a[2])) == t:
                        lwPhase[t] ^= 1
                    else:
                        lwPhase[t] = None
                if d in srd:
                    t = srd[d]
                    if op in ("s_add_u32", "s_sub_u32") and sreg(a[1]) == d and sval(a[2]) is not None:
                        srdOff[t] += sval(a[2]) if op == "s_add_u32" else -sval(a[2])
                vals = [sval(x) for x in a[1:]]
                res, newScc = None, "keep"
                if op == "s_mov_b32":
                    res = vals[0]
                elif op in ("s_add_u32", "s_sub_u32", "s_and_b32", "s_lshl_b32", "s_lshr_b32",
                            "s_max_i32", "s_min_i32", "s_mul_i32", "s_xor_b32", "s_or_b32",
                            "s_max_u32", "s_min_u32"):
                    x, y = (vals + [None, None])[:2]
                    newScc = "keep" if op == "s_mul_i32" else None
                    if x is not None and y is not None:
                        res = {"s_add_u32": x + y, "s_sub_u32": x - y, "s_and_b32": x & y,
                               "s_lshl_b32": x << y if y < 64 else 0, "s_lshr_b32": x >> y,
                               "s_max_i32": max(x, y), "s_min_i32": min(x, y),
                               "s_max_u32": max(x, y), "s_min_u32": min(x, y),
                               "s_mul_i32": x * y, "s_xor_b32": x ^ y, "s_or_b32": x | y}[op]
                        if op == "s_sub_u32":
                            newScc, res = int(res < 0), res & 0xFFFFFFFF
                        elif op == "s_add_u32":
                            newScc, res = int(res > 0xFFFFFFFF), res & 0xFFFFFFFF
                        elif op in ("s_and_b32", "s_xor_b32", "s_or_b32", "s_lshl_b32", "s_lshr_b32"):
                            newScc = int(res != 0)
                        elif op != "s_mul_i32":
                            newScc = None
                elif op.startswith("s_cmp_"):
                    m = re.match(r"s_cmp_(eq|lg|lt|le|gt|ge)_(u32|i32)", op)
                    x, y = sval(a[0]), sval(a[1])
                    newScc = None
                    if m and x is not None and y is not None:
                        newScc = int({"eq": x == y, "lg": x != y, "lt": x < y, "le": x <= y,
                                      "gt": x > y, "ge": x >= y}[m.group(1)])
                    dst = None
                elif op == "s_bitcmp1_b32":
                    x, y = sval(a[0]), sval(a[1])
                    newScc = int((x >> y) & 1) if x is not None and y is not None else None
                    dst = None
                elif op == "s_cselect_b32":
                    res = (vals[0] if scc == 1 else vals[1]) if scc is not None else None
                elif op == "s_cmov_b32":
                    res = vals[0] if scc == 1 else S.get(sreg(dst)) if scc == 0 else None
                elif re.match(r"s_(addc|subb|and|or|xor|andn2|orn2|nand|nor|xnor|not|lshl|lshr|"
                              r"ashr|bfe|min|max|abs|bcnt|absdiff)", op):
                    newScc = None
                if dst is not None and sreg(dst) is not None:
                    S[sreg(dst)] = res
                if newScc != "keep":
                    scc = newScc
            elif op.startswith("buffer_load") and ins.mod("lds"):
                t = srd.get(_regs(a[1])[0][1]) if _regs(a[1]) else None
                if t is not None and m0 is not None and m0[0] == t and m0[1] is not None:
                    _, half, c = m0
                    lo, hi = (0, 1 << 30) if t in SCALE else (c, c + (self.strides[t] or 1 << 20))
                    mem[(t, half)].append((lo, hi, (t, srdOff[t], c), seq))
                seq += 1
            elif op.split("_")[0] in ("buffer", "global", "flat", "scratch"):
                if "load" in op:
                    for f, i in _regs(a[0]):
                        if f == "v":
                            vval[i] = ("vmem", ins.line)
            elif op.startswith(("ds_bpermute", "ds_permute")):
                for f, i in _regs(a[0]):
                    vval[i] = ("perm", ins.line)
            elif op.startswith("ds_read"):
                dest = [x[1] for x in _regs(a[0])]
                base = _regs(a[1])[0][1]
                off = int(ins.mod("offset", "0"))
                ident = None
                if base in self.lr and lrPhase[base] is not None:
                    t = self.lr[base][0]
                    width = _SCALE_READ_BYTES if t in SCALE else _A_READ_BYTES
                    hits = [h for h in mem[(t, lrPhase[base])] if h[0] < off + width and off < h[1]]
                    cuts = sorted({off, off + width} | {x for h in hits for x in h[:2]
                                                         if off < x < off + width})
                    parts = []
                    for lo, hi in zip(cuts, cuts[1:]):
                        cover = [h for h in hits if h[0] <= lo and hi <= h[1]]
                        if cover:
                            parts.append(max(cover, key=lambda h: h[3]))
                    if hits and len(parts) == len(cuts) - 1:
                        ident = tuple(sorted({p[2] for p in parts}))
                for i, vr in enumerate(dest):
                    vval[vr] = ("lds", ident, i)
                seq += 1
            elif op.startswith("v_mfma"):
                srcs = [_regs(x) for x in a[1:]]
                ids = [tuple(vval.get(i, ("init", i)) if f == "v" else (f, i) for f, i in sr)
                       for sr in srcs]
                if op.startswith("v_mfma_scale"):
                    unresolved += self._check(ids, ins.mods, start, checks, ins.line)
            elif op.startswith("v_"):
                if op.startswith(("v_cmp", "v_readfirstlane", "v_readlane")):
                    dsts = []
                else:
                    dsts = [i for f, i in _regs(a[0]) if f == "v"] if a else []
                srcvals = []
                for x in a[1:]:
                    r = _regs(x)
                    srcvals += [vval.get(i, ("init", i)) if f == "v" else (f, i) for f, i in r] or [x]
                for i in dsts:
                    if i in self.lr:
                        isSwap = op == "v_xor_b32" and a[1] == f"v{i}" \
                            and _regs(a[2]) == [("v", self.lr[i][1])]
                        lrPhase[i] = lrPhase[i] ^ 1 if isSwap and lrPhase[i] is not None else None
                    vval[i] = (op, tuple(srcvals))
            pc = npc
        raise AssertionError(f"K={K} start={start}: step limit")

    @staticmethod
    def _check(ids, mods, start, checks, line):
        def sel(key):
            for m in mods:
                if m.startswith(key + ":["):
                    return [int(x) for x in m[len(key) + 2:-1].split(",")]
            return [0, 0]

        dataOffs = {load[1] for v in ids[0] + ids[1]
                    if isinstance(v, tuple) and v and v[0] == "lds" and v[1]
                    for load in v[1] if load[0] in ("A", "B")}
        if len(dataOffs) != 1:
            return 2
        du = start + dataOffs.pop() // DEPTHU
        opSel, opSelHi = sel("op_sel"), sel("op_sel_hi")
        missed = 0
        for j, vals in enumerate(ids[3:5]):
            roots = {_scale_root(v) for v in vals}
            if len(roots) != 1 or None in roots:
                missed += 1
                continue
            loads, shift = roots.pop()
            if loads is None:
                checks.append(dict(line=line, du=du, operand=j, got=(None, None)))
                continue
            offs = {ld[1] for ld in loads}
            if len(offs) != 1:
                missed += 1
                continue
            epoch = start // 2 + offs.pop() // SCALE_EPOCH_BYTES
            half = (opSel[j] + 2 * opSelHi[j] + shift // 8) // 2
            checks.append(dict(line=line, du=du, operand=j, got=(epoch, half)))
        return missed


def _scale_root(val, shift=0):
    """(loads behind one LDS read, total right shift) for a scale VGPR value."""
    if not isinstance(val, tuple) or not val:
        return None
    if val[0] == "lds":
        return val[1], shift
    if val[0] == "v_mov_b32":
        return _scale_root(val[1][0], shift)
    if val[0] == "v_and_b32":
        hits = [h for h in (_scale_root(x, shift) for x in val[1]) if h]
        return hits[0] if len(hits) == 1 else None
    if val[0] == "v_lshrrev_b32":
        amount, src = val[1][0], val[1][1]
        return _scale_root(src, shift + int(amount, 0)) if isinstance(amount, str) else None
    return None
