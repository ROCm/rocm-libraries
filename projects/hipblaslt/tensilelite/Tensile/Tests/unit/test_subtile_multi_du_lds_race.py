# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile MXFP8 (gfx950) main loops must be free of LDS races after scheduling.

Emits kernels and replays the preloop -> main loop -> NLL path for a range of
loop counts, modelling one wave's vmcnt / lgkmcnt queues and s_barrier. All
waves run the same program, so "retired before barrier b" in this wave means
retired before b in every wave. Per LDS region (tensor, buffer, 1 KB unit):

  RAW  a ds_read needs its producing buffer_load..lds retired by s_waitcnt
       vmcnt, followed by an s_barrier, before the read.
  WAR  a buffer_load..lds needs every earlier ds_read of the region retired
       (lgkmcnt), followed by an s_barrier, before the load.
  GEN  writes alternate with reads, and successive writes fetch K blocks at a
       constant stride with a fixed buffer parity (wrong offset12 / m0 / SRD
       increments show up here).

Covers the multi-DU PGR=1 per-uid offset GRs + exact vmcnt schedule
(SchedulerConfig.grUidOffset) and the SUBTILE_NO_UID_OFFSET fallback.
"""

from collections import defaultdict, deque
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_UNIT = 1024
_MFMA_OPERANDS = ["A", "B", "MXSA", "MXSB"]  # srcA, srcB, scaleA, scaleB
_LOOP_COUNTS = range(0, 7)                   # covers skip / NLL-only / both loop exits


def _vregs(tok):
    m = re.match(r"v\[(\d+):(\d+)\]$", tok) or re.match(r"v(\d+)()$", tok)
    if not m:
        return set()
    lo = int(m[1])
    return set(range(lo, int(m[2] or lo) + 1))


class _Kernel:
    def __init__(self, assembly):
        self.ins = [l.split("//")[0].strip() for l in assembly.splitlines()]
        self.ins = [l for l in self.ins if l]
        self.labels = {l[:-1]: i for i, l in enumerate(self.ins) if re.match(r"^\w+:$", l)}
        name = next(l.split()[1] for l in self.ins if l.startswith(".amdhsa_kernel "))
        mt0, mt1, du = map(int, re.search(r"_MT(\d+)x(\d+)x(\d+)_", name).groups())
        # MXFP8 subtile: data K per LDS buffer is DU/2 when multi-DU (Solution.py).
        self.dataDU = du // 2 if max(mt0, mt1) > du // 2 else du
        self.lrTensor = self._lr_tensors()
        swaps = {int(m[1]) for l in self.ins
                 if (m := re.match(r"s_add_u32 s\[sgprSwap\w+\], s\[sgprLocalWriteBaseAddr\w+\], (\d+)", l))}
        assert len(swaps) == 1, swaps
        self.swap = swaps.pop()
        self.grEnd = self._gr_extents()
        self.lrSpan = self._lr_spans()

    def _lr_tensors(self):
        """LR address vgpr -> tensor, from the MFMA operand its data feeds."""
        mfmas = []
        for i, l in enumerate(self.ins):
            if l.startswith("v_mfma_scale"):
                ops = [o.strip().split()[0] for o in l.split(None, 1)[1].split(",")]
                mfmas.append((i, [_vregs(o) for o in (ops[1], ops[2], ops[4], ops[5])]))
        out = {}
        for i, l in enumerate(self.ins):
            m = re.match(r"ds_read_b\d+ (\S+), (v\d+)", l)
            if not m:
                continue
            dst = _vregs(m[1])
            use = next((ops for j, ops in mfmas if j > i and any(dst & r for r in ops)), None)
            assert use is not None, f"LR result never used by an MFMA: {l}"
            t = _MFMA_OPERANDS[next(k for k, r in enumerate(use) if dst & r)]
            assert out.setdefault(m[2], t) == t, f"{m[2]} feeds {out[m[2]]} and {t}"
        return out

    def _gr_extents(self):
        """(tensor, lds const) -> end; a GR covers up to the next per-load step."""
        consts, m0 = defaultdict(set), None
        for l in self.ins:
            m = re.match(r"s_(?:add_u32|mov_b32) m0, s\[sgprLocalWriteBaseAddr(\w+)\](?:, (-?\d+))?$", l)
            if m:
                m0 = (m[1], int(m[2] or 0))
            elif l.startswith("buffer_load") and l.endswith(" lds") and m0:
                off = re.search(r"offset:(\d+)", l)
                consts[m0[0]].add((m0[1] + int(off[1] if off else 0)) % self.swap)
        ends = {}
        for t, cs in consts.items():
            cs = sorted(cs)
            step = min((b - a for a, b in zip(cs, cs[1:])), default=self.swap // 2)
            for c in cs:
                ends[(t, c)] = c + step
        return ends

    def _lr_spans(self):
        offs = defaultdict(set)
        for l in self.ins:
            m = re.match(r"ds_read_b\d+ \S+, (v\d+)(?: offset:(\d+))?", l)
            if m:
                offs[m[1]].add(int(m[2] or 0) % self.swap)
        return {a: min((b - x for x, b in zip(sorted(o), sorted(o)[1:])), default=16)
                for a, o in offs.items()}

    def trace(self, loopCount):
        """Instructions on the preloop -> loop -> NLL path for a loop count."""
        ins, labels = self.ins, self.labels
        pc = next(i for i, l in enumerate(ins) if l.startswith("s_mov_b32 s[sgprOrigLoopCounter]")) + 1
        cnt, scc, out = loopCount, None, []
        while pc != labels["label_SkipToEnd"]:
            l = ins[pc]
            assert len(out) < 100000, "runaway trace"
            if m := re.match(r"s_cmp_(eq|le|lg|lt|gt|ge)_u32 s\[sgprLoopCounterL\], (\d+)", l):
                k = int(m[2])
                scc = {"eq": cnt == k, "le": cnt <= k, "lg": cnt != k,
                       "lt": cnt < k, "gt": cnt > k, "ge": cnt >= k}[m[1]]
            elif l == "s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1":
                cnt -= 1
            elif l.startswith("s_cmp"):
                scc = None
            if m := re.match(r"s_cbranch_scc([01]) (\w+)", l):
                assert scc is not None, f"branch on unmodelled scc: {l}"
                taken, scc = scc == (m[1] == "1"), None
                if taken:
                    pc = labels[m[2]]
                    continue
            elif m := re.match(r"s_branch (\w+)", l):
                pc = labels[m[1]]
                continue
            else:
                assert not l.startswith("s_cbranch"), f"unmodelled branch: {l}"
            out.append((pc, l))
            pc += 1
        return out


def _units(lo, hi):
    return range(lo // _UNIT, max(lo // _UNIT + 1, -(-hi // _UNIT)))


def find_lds_races(assembly, loopCounts=_LOOP_COUNTS):
    """Return a list of race / generation errors (empty == clean)."""
    k = _Kernel(assembly)
    errs = []
    for n in loopCounts:
        errs += [f"loops={n}: {e}" for e in _replay(k, n)]
    return errs


def _replay(k, loopCount):
    errs = []
    m0 = None
    wbFlip, lrFlip, srd = defaultdict(int), defaultdict(int), defaultdict(int)
    vmq, lgq = deque(), deque()
    hist = defaultdict(list)  # region -> [("W"|"R", event)]
    gens = defaultdict(list)  # region -> K block index per write
    bar = -1
    for t, (pc, l) in enumerate(k.trace(loopCount)):
        if l.startswith("s_barrier"):
            bar = t
        elif m := re.match(r"s_waitcnt (.*)", l):
            for kind, n in re.findall(r"(vmcnt|lgkmcnt)\((\d+)\)", m[1]):
                q = vmq if kind == "vmcnt" else lgq
                while len(q) > int(n):
                    q.popleft()["ret"] = t
        elif m := re.match(r"s_(?:add_u32|mov_b32) m0, s\[sgprLocalWriteBaseAddr(\w+)\](?:, (-?\d+))?$", l):
            m0 = (m[1], int(m[2] or 0))
        elif l.startswith(("s_mov_b32 m0", "s_add_u32 m0")):
            m0 = None
        elif m := re.match(r"s_xor_b32 s\[sgprLocalWriteBaseAddr(\w+)\], \S+, s\[sgprSwap", l):
            wbFlip[m[1]] ^= 1
        elif (m := re.match(r"v_xor_b32 (v\d+), \1, v\d+", l)) and m[1] in k.lrTensor:
            lrFlip[m[1]] ^= 1
        elif m := re.match(r"s_(add|sub)_u32 s\[sgprSrd(\w+)\], s\[sgprSrd\w+\], (\d+)", l):
            srd[m[2]] += int(m[3]) if m[1] == "add" else -int(m[3])
        elif l.startswith(("buffer_", "global_")):
            ev = {"pc": pc, "ret": None}
            vmq.append(ev)
            if not l.endswith(" lds"):
                continue
            ten = re.search(r"s\[sgprSrd(\w+):", l)[1]
            if m0 is None or m0[0] != ten:
                errs.append(f"GR {ten} @{pc} with m0 {m0}")
                continue
            off = re.search(r"offset:(\d+)", l)
            off = int(off[1]) if off else 0
            rel = m0[1] + off
            buf, c = (rel // k.swap) ^ wbFlip[ten], rel % k.swap
            unit = k.dataDU if ten in ("A", "B") else 256
            kblk = (srd[ten] + (off if ten in ("A", "B") else 0)) // unit
            for u in _units(c, k.grEnd[(ten, c)]):
                region = (ten, buf, u)
                for kind, r in hist[region]:
                    if kind == "R" and (r["ret"] is None or r["ret"] > bar):
                        errs.append(f"WAR {region}: GR @{pc} overwrites before LR @{r['pc']} "
                                    f"is retired+barriered")
                hist[region].append(("W", ev))
                gens[region].append(kblk)
        elif m := re.match(r"ds_read_b\d+ \S+, (v\d+)(?: offset:(\d+))?", l):
            ev = {"pc": pc, "ret": None}
            lgq.append(ev)
            off = int(m[2] or 0)
            buf, o = lrFlip[m[1]] ^ (off // k.swap), off % k.swap
            for u in _units(o, o + k.lrSpan[m[1]]):
                region = (k.lrTensor[m[1]], buf, u)
                w = next((e for kind, e in reversed(hist[region]) if kind == "W"), None)
                if w is None:
                    errs.append(f"RAW {region}: LR @{pc} before any GR")
                elif w["ret"] is None or w["ret"] >= bar:
                    errs.append(f"RAW {region}: LR @{pc} reads GR @{w['pc']} "
                                f"before it is retired+barriered")
                hist[region].append(("R", ev))
        elif l.startswith(("ds_", "s_load", "s_buffer_load")):
            lgq.append({"pc": pc, "ret": None})

    stride, parity = defaultdict(set), defaultdict(set)
    for region, h in hist.items():
        seq = "".join(kind for kind, _ in h)
        if "R" in seq and "WW" in seq.rstrip("W"):
            errs.append(f"GEN {region}: write overwritten before it was read ({seq[:24]}...)")
        ks = gens[region]
        stride[region[0]].update(b - a for a, b in zip(ks, ks[1:]))
        if "R" in seq:
            parity[region[:2]].update(x % 2 for x in ks)
    for ten, st in stride.items():
        if len(st) > 1 or any(s <= 0 for s in st):
            errs.append(f"GEN {ten}: per-buffer K stride not constant {sorted(st)}")
    for (ten, buf), p in parity.items():
        if len(p) > 1:
            errs.append(f"GEN {ten} buffer {buf}: holds both odd and even K blocks")
    return errs


########################################
# Kernels
########################################

# MI [16,16,128,1,1, MIWT0,MIWT1, WG0,WG1] -> MT (16*MIWT0*WG0)x(16*MIWT1*WG1);
# DepthU=256 gives dataDU=128 (multi-DU) once max(MT) > 128.
_TILES = {
    "MT256x256": [8, 8, 2, 2],    # multi-DU, multi-partition
    "MT192x256": [6, 8, 2, 2],    # partitioned in N
    "MT320x256": [10, 8, 2, 2],   # partitioned in M
    "MT224x128": [14, 2, 1, 4],   # WG 1x4
    "MT256x192": [8, 6, 2, 2],
    "MT128x128": [4, 4, 2, 2],    # single-DU control
}


def _config(tiles):
    return {
        "GlobalParameters": {"MXScaleFormat": 1, "CpuThreads": 1, "PrintLevel": 0},
        "BenchmarkProblems": [[
            {"OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
             "ComputeDataType": "s", "HighPrecisionAccumulate": True,
             "MXBlockA": 32, "MXBlockB": 32, "TransposeA": True, "TransposeB": False,
             "UseBeta": True, "Batched": True, "ActivationFuncCall": True},
            {"InitialSolutionParameters": None,
             "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
             "ForkParameters": [
                 {"MatrixInstruction": [[16, 16, 128, 1, 1, *_TILES[t]] for t in tiles]},
                 {"PrefetchGlobalRead": [1]}, {"PrefetchLocalRead": [1]},
                 {"DepthU": [256]}, {"ScheduleIterAlg": [3]}, {"DirectToLds": [1]},
                 {"StreamK": [3]}, {"StaggerU": [0]}, {"UseSubtileImpl": [True]}],
             "BenchmarkJoinParameters": None,
             "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [960, 960, 1, 1024]}]}]},
        ]],
    }


def _emit(tmp_path, tiles):
    path = tmp_path / "mxfp8_multi_du.yaml"
    path.write_text(yaml.safe_dump(_config(tiles)))
    results = emit_kernels_from_config(path, limit=len(tiles), arch="gfx950")
    assert len(results) == len(tiles)
    out = {}
    for _, assembly, error in results:
        assert error == 0
        mt = re.search(r"_(MT\d+x\d+)x\d+_", assembly)[1]
        out[mt] = assembly
    assert sorted(out) == sorted(tiles)
    return out


def _uses_uid_offset(assembly):
    """uid1 A/B GRs land in the other LDS buffer through m0 (base + swap - offset12)
    rather than through an SRD step + LocalWriteBaseAddr swap."""
    swap = int(re.search(r"s_add_u32 s\[sgprSwapA\], s\[sgprLocalWriteBaseAddrA\], (\d+)", assembly)[1])
    consts = {int(c) for c in re.findall(r"m0, s\[sgprLocalWriteBaseAddr[AB]\], (\d+)", assembly)}
    return any(swap - 4096 < c < swap for c in consts)


def _line_of(lines, label):
    return next(i for i, l in enumerate(lines) if l.strip() == label)


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    return _emit(tmp_path_factory.mktemp("uid"), list(_TILES))


@pytest.fixture(scope="module")
def kernels_no_uid_offset(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("SUBTILE_NO_UID_OFFSET", "1")
    try:
        return _emit(tmp_path_factory.mktemp("nouid"), list(_TILES))
    finally:
        mp.undo()


@pytest.mark.parametrize("mt", list(_TILES))
def test_mainloop_lds_race_free(kernels, mt):
    assembly = kernels[mt]
    assert _uses_uid_offset(assembly) == (mt != "MT128x128")
    assert find_lds_races(assembly) == []


@pytest.mark.parametrize("mt", list(_TILES))
def test_mainloop_lds_race_free_without_uid_offset(kernels_no_uid_offset, mt):
    assembly = kernels_no_uid_offset[mt]
    assert not _uses_uid_offset(assembly)
    assert find_lds_races(assembly) == []


def test_mainloop_vmcnt_is_exact(kernels):
    """Every main-loop vmcnt is as loose as it can be: +1 on any of them races."""
    assembly = kernels["MT256x256"]
    lines = assembly.splitlines()
    begin = _line_of(lines, "label_LoopBeginL:")
    end = _line_of(lines, "label_SkipToNLL:")
    waits = [i for i in range(begin, end) if re.match(r"s_waitcnt vmcnt\(\d+\)", lines[i].strip())]
    assert waits and any("vmcnt(0)" not in lines[i] for i in waits)
    for i in waits:
        n = int(re.search(r"vmcnt\((\d+)\)", lines[i])[1])
        mutated = lines[:i] + [lines[i].replace(f"vmcnt({n})", f"vmcnt({n + 1})")] + lines[i + 1:]
        assert any(e.split(": ", 1)[1].startswith("RAW") for e in find_lds_races("\n".join(mutated))), \
            f"loosening line {i} ({lines[i].strip()}) was not detected"


@pytest.mark.parametrize("mutation", ["drop_barriers", "uid1_to_buf0", "uid1_same_k"])
def test_checker_detects_injected_races(kernels, mutation):
    lines = kernels["MT256x256"].splitlines()
    begin = _line_of(lines, "label_LoopBeginL:")
    if mutation == "drop_barriers":
        # remove the barriers that follow the second main-loop vmcnt wait
        w = [i for i in range(begin, len(lines)) if re.match(r"\s*s_waitcnt vmcnt", lines[i])][1]
        drop = {i for i in range(w, w + 4) if lines[i].strip().startswith("s_barrier")}
        lines = [l for i, l in enumerate(lines) if i not in drop]
    else:
        # first uid1 A GR in the loop: m0 = base + swap - 128, offset:128
        i = next(i for i in range(begin, len(lines))
                 if re.search(r"sgprSrdA:.* offset:128 lds", lines[i]))
        j = max(x for x in range(begin, i) if "m0, s[sgprLocalWriteBaseAddrA]" in lines[x])
        if mutation == "uid1_to_buf0":
            lines[j] = re.sub(r", \d+(\s*(//.*)?)$", r", 0\1", lines[j])
        else:
            lines[i] = lines[i].replace("offset:128 lds", "offset:0 lds")
    assert find_lds_races("\n".join(lines)), mutation
