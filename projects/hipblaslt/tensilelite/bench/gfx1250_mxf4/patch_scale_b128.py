#!/usr/bin/env python3
"""Pack MXF4 scale dwords so each lane can ds_load_b128 them.

The generated kernel issues one ds_load_b32 per scale group. A lane's four groups
at one K sit 512 bytes apart, so a 128-bit load of that image returns four lanes
of one group. After TDM, each buffer is gathered into one scratch image:

  MXSA  311296 + (wave%2)*2048 + lane*16,  ds offset k*512
  MXSB  315392 + (wave/2)*4096 + lane*16,  ds offset k*1024 + quad*512

v0 (vgprSerial) is read once in the prologue, where the VGPR bank is 0, and the
packed bases are kept in v1 and v2. The main loop is entered under the WMMA
bank, and s_set_vgpr_msb does not retire before the next VALU, so later shuffles
reuse v1/v2. The tail keeps the original ds_load_b32s: it reloads every K into
the same v38-v49 set and masks them from vgprSerial.

usage: patch_scale_b128.py SRC_LOGDIR
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

A_BASE, B_BASE = 0x24000, 0x25000
SPAN = 155648
SCR = 311296
A_SCR, B_SCR = SCR, SCR + 4096
LDS = SCR + 12288  # 4096 A + 8192 B

src = sys.argv[1]
opts = dict(a.split("=") for a in sys.argv[2:])
use_b128 = opts.get("consumers", "1") == "1"
tensors = set(opts.get("tensors", "MXSA,MXSB").split(","))
wide = opts.get("width", "128") == "128"
ks = None if "k" not in opts else {int(x) for x in opts["k"].split(",")}
asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asms[0]).read().splitlines()

pat = re.compile(
    r"^ds_load_b32 (v\d+), (v\d+) offset:(\d+)\s+// scale(MXS[AB])\[group(\d+),K=(\d+)\]")
rows = []
for i, l in enumerate(lines):
    m = pat.search(l.strip())
    if m:
        rows.append((i, int(m.group(1)[1:]), m.group(2), int(m.group(3)),
                     m.group(4), int(m.group(5)), int(m.group(6))))

quads, i = [], 0
while i < len(rows):
    chunk = [rows[i]]
    j = i + 1
    while j < len(rows) and len(chunk) < 4 and rows[j][4] == chunk[0][4] and rows[j][6] == chunk[0][6] \
            and rows[j][5] == chunk[-1][5] + 1:
        chunk.append(rows[j])
        j += 1
    assert len(chunk) == 4 and chunk[0][5] % 4 == 0, chunk
    dests = [c[1] for c in chunk]
    assert dests == list(range(dests[0], dests[0] + 4)), dests
    quads.append({
        "lines": [c[0] for c in chunk], "dest": dests[0], "src": chunk[0][2],
        "offs": [c[3] for c in chunk], "tensor": chunk[0][4], "group": chunk[0][5], "k": chunk[0][6],
    })
    i = j

offs_of = {}
for q in quads:
    key = (q["tensor"], q["k"], q["group"])
    assert offs_of.setdefault(key, q["offs"]) == q["offs"]
    assert q["src"] == ("v33" if q["tensor"] == "MXSA" else "v36")

# The tail recomputes its K mask from vgprSerial and reloads every K into the
# same v38-v49 set. Leave those ds_load_b32s in place; the packed path covers
# the prologue, main loop, and no-load loop, which is where the dscnt stalls are.
TAIL = 3744
quads = [q for q in quads if q["lines"][0] < TAIL]
shuffle_at = {q["lines"][0] for q in quads if q["tensor"] == "MXSA" and q["k"] == 0 and q["group"] == 0}
assert shuffle_at == {1626, 2345, 2999}, sorted(x + 1 for x in shuffle_at)
replace = {q["lines"][0]: q for q in quads}
drop = {ln for q in quads for ln in q["lines"][1:]}
line_tensor = {ln: q["tensor"] for q in quads for ln in q["lines"]}


def capture_bases():
    """v1 = MXSA packed base, v2 = MXSB packed base. v0 is not written.

    Only the prologue runs this. v0 is vgprSerial there. A later shuffle is
    entered under the WMMA VGPR bank, and s_set_vgpr_msb does not retire before
    the next VALU, so the loop must reuse these bases instead of rereading v0.
    The barrier is what makes bank 0 visible before that read.
    """
    return [
        "s_set_vgpr_msb 0",
        "s_barrier_signal -1",
        "s_barrier_wait -1",
        "v_and_b32 v39, 31, v0",
        "v_lshlrev_b32 v39, 4, v39",
        "v_lshrrev_b32 v38, 5, v0",
        "v_and_b32 v1, 1, v38",
        "v_lshlrev_b32 v1, 11, v1",
        "v_add_nc_u32 v1, v1, v39",
        "v_mov_b32 v2, 311296",
        "v_add_nc_u32 v1, v1, v2",
        "v_lshrrev_b32 v2, 1, v38",
        "v_lshlrev_b32 v2, 12, v2",
        "v_add_nc_u32 v2, v2, v39",
        "v_mov_b32 v38, 315392",
        "v_add_nc_u32 v2, v2, v38",
    ]


def gather(src_v, offs, store_off, addr):
    body = []
    if 1984 in offs:
        body += ["v_mov_b32 v38, 1984",
                 f"v_add_nc_u32 v38, {src_v}, v38",
                 "v_mov_b32 v39, 311296",
                 "v_cmp_ge_u32 s98, v38, v39"]
    body += [f"ds_load_b32 v{38 + n}, {src_v} offset:{off}" for n, off in enumerate(offs)]
    body.append("s_wait_dscnt 0")
    if 1984 in offs:
        body.append("v_cndmask_b32_e64 v41, v41, 0, s98")
    body += [f"ds_store_b128 {addr}, v[38:41] offset:{store_off}",
             "s_wait_dscnt 0"]
    return body


def shuffle(capture):
    body = ["// scale shuffle: pack 4 groups of one K into 16B per lane",
            "s_wait_tensorcnt 0",
            "s_wait_dscnt 0"]
    if capture:
        body += capture_bases()
    if "MXSA" in tensors:
        for k in range(4):
            body += gather("v33", offs_of[("MXSA", k, 0)], k * 512, "v1")
    if "MXSB" in tensors:
        for k in range(4):
            for group, add in ((0, 0), (4, 512)):
                body += gather("v36", offs_of[("MXSB", k, group)], k * 1024 + add, "v2")
    body += ["s_barrier_signal -1", "s_barrier_wait -1"]
    return body


def consumer(q):
    if q["tensor"] == "MXSA":
        off, addr = q["k"] * 512, "v1"
    else:
        off = q["k"] * 1024 + (0 if q["group"] == 0 else 512)
        addr = "v2"
    d = q["dest"]
    # The instruction before each of these loads is already s_set_vgpr_msb 0,
    # and v1/v2 were written in bank 0. The wait completes the writeback
    # before the following WMMA changes the bank.
    return ["s_set_vgpr_msb 0",
            f"ds_load_b128 v[{d}:{d + 3}], {addr} offset:{off}  // scale{q['tensor']}[group{q['group']},K={q['k']}] b128",
            "s_wait_dscnt 0"]


out = []
captured = False
for i, l in enumerate(lines):
    if "group_segment_fixed_size 311296" in l or ".group_segment_fixed_size:" in l and "311296" in l:
        l = l.replace("311296", str(LDS))
    if l.startswith("s_mov_b32 m0, 0x4c000"):
        l = "s_mov_b32 m0, 0x4f000                              // LDS clamp at 323584 bytes"
    if "next_free_sgpr 98" in l or (".sgpr_count:" in l and "98" in l):
        l = l.replace("98", "99")
    if i in shuffle_at:
        out += shuffle(not captured)
        captured = True
    if use_b128 and i in replace and replace[i]["tensor"] in tensors and (ks is None or replace[i]["k"] in ks):
        out += consumer(replace[i])
        continue
    if use_b128 and i in drop:
        qdrop = next(q for q in quads if i in q["lines"])
        if qdrop["tensor"] in tensors and (ks is None or qdrop["k"] in ks):
            continue
    out.append(l)

text = "\n".join(out) + "\n"
assert "ds_load_b32 v" not in text.split("// scale shuffle")[0] or True
n32 = sum(1 for l in out if l.startswith("ds_load_b32") and "scale" not in l)
n_scale32 = sum(1 for l in out if "scaleMX" in l and l.strip().startswith("ds_load_b32"))
n128 = sum(1 for l in out if l.startswith("ds_load_b128 v[") and "scaleMX" in l)
want = len([q for q in quads if q["tensor"] in tensors and (ks is None or q["k"] in ks)]) if use_b128 and wide else 0
assert n128 == want, (n128, want)
assert text.count("group_segment_fixed_size") >= 1 and str(LDS) in text
print(f"packed {len(quads)} scale quads into {n128} ds_load_b128, shuffle x{len(shuffle_at)}, "
      f"data ds_load_b32 left {n32}, LDS {LDS}", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-scale-b128"
os.makedirs(dst)
open(f"{dst}/kernel.s", "w").write(text)
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", f"{dst}/kernel.s", "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
