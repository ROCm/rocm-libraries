#!/usr/bin/env python3
"""Throwaway probe: parse a capture listing, build the unrolled timeline,
classify reads/writes of ValuA_X0_I0+12..15, and compute latest-writer per read.

Pure text. No kernel build. The listing is already body-ordered in the unrolled
timeline (PRO -> ML_PREV -> ML -> NGL -> NLL), so global file order == unrolled
order. We just need to parse register operands.
"""
import re, sys

TARGET = {12, 13, 14, 15}  # ValuA_X0_I0+k offsets we track
BODY_ORDER = ["PRO", "ML_PREV", "ML", "NGL", "NLL"]

def parse_regs(render):
    """Return (reads:set, writes:set) of ValuA_X0_I0+k offsets in TARGET.

    Operand 0 of an instruction is the dest (write). For mfma with an in/out
    accumulator operand that repeats the dest, it is also a read. ds_read writes
    its dest. v_cvt reads operands 1,2 and writes operand 0.
    """
    # find all ValuA_X0_I0+N (possibly with :+3 range)
    # We classify by instruction mnemonic + operand position.
    reads, writes = set(), set()
    m = re.match(r'\s*(\S+)\s+(.*)', render)
    if not m:
        return reads, writes
    mnem, rest = m.group(1), m.group(2)
    # split top-level operands by comma (ranges use ':' not ',')
    ops = [o.strip() for o in rest.split(',')]

    def offs(op):
        """all X0_I0+k offsets named in this operand string, expanding +k+3 ranges"""
        out = set()
        for mm in re.finditer(r'vgprValuA_X0_I0\+(\d+)(?:\:vgprValuA_X0_I0\+\d+\+(\d+))?', op):
            base = int(mm.group(1))
            if mm.group(2) is not None:
                span = int(mm.group(2))
                for k in range(base, base + span + 1):
                    out.add(k)
            else:
                out.add(base)
        return out & TARGET

    if mnem.startswith('ds_read'):
        # ds_read DEST, ADDR  -> dest write
        if ops:
            writes |= offs(ops[0])
    elif mnem.startswith('v_cvt'):
        # v_cvt DEST, SRC1, SRC2 -> dest write, srcs read
        if ops:
            writes |= offs(ops[0])
        for op in ops[1:]:
            reads |= offs(op)
    elif mnem.startswith('v_mfma'):
        # v_mfma DEST, A, B, C(=acc, in/out)  -> dest write; A,B read; C read+write
        if ops:
            writes |= offs(ops[0])
            reads |= offs(ops[0])  # mfma dest is also accumulator-input here (C==dest)
        for op in ops[1:]:
            reads |= offs(op)
    else:
        # default: op0 dest, rest read
        if ops:
            writes |= offs(ops[0])
        for op in ops[1:]:
            reads |= offs(op)
    return reads, writes

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*\d+\s*\|\s*(PRO|ML_PREV|ML|NGL|NLL)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*\|\s*(.*)', line)
            if not m:
                continue
            body, body_idx, mfma_index, seq, render = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5).rstrip()
            fileline = int(re.match(r'\s*\d+', line).group(0)) if False else None
            # get the leading listing-line number (the first int before first |)
            ln = re.match(r'\s*(\d+)\s*\|', line)
            listing_line = int(ln.group(1)) if ln else None
            reads, writes = parse_regs(render)
            rows.append(dict(line=listing_line, body=body, bidx=body_idx, mfma=mfma_index,
                             seq=seq, render=render, reads=reads, writes=writes))
    return rows

def main(path):
    rows = load(path)
    # unrolled order = file order (already body-ordered). Assign global pos.
    for gpos, r in enumerate(rows):
        r['gpos'] = gpos
    # filter to rows touching target
    touch = [r for r in rows if r['reads'] or r['writes']]
    print(f"### {path}")
    print(f"# rows touching ValuA_X0_I0+12..15: {len(touch)}")
    # for each row, compute latest prior writer per byte it reads
    for r in touch:
        kind = []
        if r['reads']: kind.append("R" + ''.join(str(k) for k in sorted(r['reads'])))
        if r['writes']: kind.append("W" + ''.join(str(k) for k in sorted(r['writes'])))
        mnem = r['render'].split()[0]
        tag = f"L{r['line']} {r['body']}/bidx{r['bidx']}/mfma{r['mfma']} {mnem:24s} {'+'.join(kind)}"
        # latest writer trace for reads
        if r['reads']:
            lw = {}
            for k in sorted(r['reads']):
                # walk back
                prev = None
                for p in touch:
                    if p['gpos'] >= r['gpos']:
                        break
                    if k in p['writes']:
                        prev = p
                if prev is None:
                    lw[k] = "NONE(initial/prior-iter)"
                else:
                    lw[k] = f"L{prev['line']}({prev['body']}/{prev['render'].split()[0]})"
            tag += "   reads<-{" + ", ".join(f"+{k}:{v}" for k,v in lw.items()) + "}"
        print(tag)

if __name__ == "__main__":
    main(sys.argv[1])
