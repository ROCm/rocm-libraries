import re, sys
from collections import Counter
lines = open(sys.argv[1]).read().splitlines()
insts = []  # (addr, mnem, is_branch, target)
for l in lines:
    if not l.startswith("\t"): continue
    mn = l[1:].split(None, 1)[0] if l[1:].strip() else ""
    if not mn: continue
    am = re.search(r'//\s*([0-9A-Fa-f]+):', l)
    if not am: continue
    addr = int(am.group(1), 16)
    tgt = None
    if mn.startswith("s_cbranch") or mn.startswith("s_branch"):
        tm = re.search(r'\+0x([0-9a-f]+)>', l)
        if tm: tgt = int(tm.group(1), 16)
    insts.append((addr, mn, tgt))
loops = [(t, a) for a, mn, t in insts if t is not None and t < a]
def mfma_in(span):
    lo, hi = span
    return sum(1 for a, mn, _ in insts if lo <= a <= hi and "mfma" in mn)
hot = max(loops, key=mfma_in) if loops else (0, 0)
lo, hi = hot
def hist(pred):
    c = Counter()
    for a, mn, _ in insts:
        if pred(a) and mn.startswith("v_"):
            c[re.sub(r'_e(32|64)$', '', mn)] += 1
    return c
inloop = hist(lambda a: lo <= a <= hi)
total = hist(lambda a: True)
# LDS/mem ops too
def memhist(pred):
    c = Counter()
    for a, mn, _ in insts:
        if pred(a) and (mn.startswith("ds_") or mn.startswith("global_") or mn.startswith("buffer_")):
            c[mn] += 1
    return c
mem_in = memhist(lambda a: lo <= a <= hi)
print(f"=== {sys.argv[1].split('/')[-1]}: hot KV loop [{lo:#x},{hi:#x}] mfma_in={mfma_in(hot)} ===")
print(f"{'VALU op':22} {'in-loop':>8} {'total':>8}")
for op, n in total.most_common(12):
    print(f"{op:22} {inloop.get(op,0):8} {n:8}")
print(f"{'VALU-sum':22} {sum(inloop.values()):8} {sum(total.values()):8}")
print(f"{'-- mem in-loop --':22}")
for op, n in mem_in.most_common(8):
    print(f"{op:22} {n:8}")
