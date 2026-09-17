# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the StreamK SK3 slab decode (``StreamK.skSlabDecode``).

The decode replaces the row-major ``tileID -> (wg0, wg1)`` step of
``skIndexToWG`` so that the tiles an XCD executes form the FlyDSL walk: a
contiguous slab of the partitioned dimension crossed with the whole shared
dimension, swept as 4-deep sub-bands. See ``skSlabDecodeRef``'s docstring for
the derivation.

Everything here is CPU-only. Two things make the tests worth trusting:

1. **The assembly is simulated, not transcribed.** ``_runAsm`` is a small SALU
   interpreter that executes the *rendered* rocisa output of the emitters, so
   ``skSlabDecodeRef`` is checked against the instructions that will actually
   run, and the ``_bitSwizzleWide`` reference is the committed emitter rather
   than a copy of its formula that could drift.

2. **The panel-traffic model is calibrated before it is used.**
   ``test_panel_model_is_calibrated`` reproduces the four published reference
   values (WGM_XCD_MAPPING_ANALYSIS.md section 5) exactly. If that test fails,
   no other number in this file means anything.

Gate map (plan section 6): G1 bijectivity, G3 identity with ``_bitSwizzleWide``
on 16x128, G4 generation partition at skGrid 256, G7 no regression on 32x103,
G10 one divide on the main path.
"""

import itertools

import pytest

# Import KernelWriter first: importing Tensile.Components.<mod> directly trips
# the Components package's own `from .Components import *` cycle. Same idiom as
# test_streamk_dponly_sgpr_reduction.py.
import Tensile.KernelWriter  # noqa: F401
from Tensile.Components.StreamK import (
    StreamKTwoTileDPFirst,
    skSlabDecodeRef,
    skSlabDecodeReason,
)
from Tensile.Components.WorkGroupMappingAlgos import _bitSwizzleTall, _bitSwizzleWide

pytestmark = pytest.mark.unit

NXCD = 8
CU_PER_XCD = 32  # workgroups resident on one XCD at CUOccupancy 1
G = 4


# ---------------------------------------------------------------------------
# Mock writer: enough surface to render skIndexToWG / the bit swizzles.
# ---------------------------------------------------------------------------
class _Labels:
    def __init__(self):
        self._n = 0

    def getNameInc(self, name):
        self._n += 1
        return "%s_%u" % (name, self._n)


class _VgprPool:
    def checkOut(self, size, tag="", *a, **k):
        return 200

    def checkOutAligned(self, size, align, tag="", *a, **k):
        return 204

    def checkIn(self, idx):
        pass


class _Writer:
    class _States:
        archCaps = {"NumXCD": 8}

    states = _States()

    """Minimal writer for the decode emitters. ``tmpBase`` is where the
    temporary SGPR block starts; ``tmpHigh`` records how far it reached, which
    is what the SGPR-budget test reads."""

    def __init__(self, tmpBase=90):
        self.labels = _Labels()
        self.vgprPool = _VgprPool()
        self._base = tmpBase
        self._next = tmpBase
        self.tmpHigh = tmpBase

    def allocTmpSgpr(self, size, alignment=1, tag=""):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            base = self._next
            self._next += size + alignment
            self.tmpHigh = max(self.tmpHigh, self._next)
            try:
                yield _TmpInfo(base, size)
            finally:
                self._next = base

        return _ctx()


class _TmpInfo:
    def __init__(self, idx, size):
        self.idx = idx
        self.size = size


def _kernel(**overrides):
    kernel = {
        "StreamK": 3,
        "StreamKForceDPOnly": 1,
        "StreamKAtomic": 0,
        "StreamKXCCMapping": 0,
        "SpaceFillingAlgo": [],
        "WGMBitSwizzle": False,
        "WorkGroupMapping": 1,
        "WorkGroupMappingXCC": 1,
        "WavefrontSize": 64,
        # Cijk_Alik_Bljk: one batch index, so the batchCount == 1 guard is
        # emitted as a runtime compare rather than folded away.
        "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2},
    }
    kernel.update(overrides)
    return kernel


_TILE_ID_SGPR = 12  # the sTmp skIndexToWG is called with


def _renderSkIndexToWG(kernel=None):
    writer = _Writer()
    module = StreamKTwoTileDPFirst().skIndexToWG(writer, kernel or _kernel(), _TILE_ID_SGPR)
    return str(module), writer


def _renderBitSwizzle(fn):
    return str(fn(20, 21, 22))


# ---------------------------------------------------------------------------
# A small SALU interpreter over rendered rocisa output.
# ---------------------------------------------------------------------------
_MASK32 = 0xFFFFFFFF


def _parseOperand(tok, regs):
    tok = tok.strip()
    if tok.startswith("s[") and tok.endswith("]"):
        return regs[tok[2:-1]]           # s[sgprFoo]
    if tok.startswith("s") and tok[1:].isdigit():
        return regs["s%s" % tok[1:]]
    if tok.startswith("0x"):
        return int(tok, 16)
    return int(tok, 0)


def _parseDest(tok):
    tok = tok.strip()
    if tok.startswith("s[") and tok.endswith("]"):
        return tok[2:-1]
    return tok


class _AsmStop(Exception):
    """Raised when execution reaches a label the caller wants to stop at."""

    def __init__(self, label):
        self.label = label


def _runAsm(text, regs, stopLabels=()):
    """Execute the SALU subset of ``text`` against the ``regs`` dict.

    ``scalarUInt24DivideAndRemainder``'s VALU block is recognised by its shape
    (two ``v_cvt_f64_u32`` sources, then two ``v_readfirstlane_b32``
    destinations) and applied as one divmod. Returns on falling off the end, or
    raises ``_AsmStop`` on reaching a label in ``stopLabels``.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith("/*"):
            continue
        lines.append(line)

    labels = {}
    for i, line in enumerate(lines):
        if line.endswith(":") or (":" in line and line.split(":")[0].startswith("label_")):
            labels[line.split(":")[0]] = i

    scc = False
    pc = 0
    steps = 0
    while pc < len(lines):
        steps += 1
        assert steps < 100000, "asm interpreter did not terminate"
        line = lines[pc]
        if line.split(":")[0] in labels and (line.endswith(":") or ":" in line):
            name = line.split(":")[0]
            if name in stopLabels:
                raise _AsmStop(name)
            pc += 1
            continue
        op, _, rest = line.partition(" ")
        args = [a for a in rest.split(",")] if rest else []

        if op == "s_branch":
            pc = labels[args[0].strip()]
            continue
        if op == "s_cbranch_scc0":
            if not scc:
                pc = labels[args[0].strip()]
                continue
            pc += 1
            continue
        if op == "s_cbranch_scc1":
            if scc:
                pc = labels[args[0].strip()]
                continue
            pc += 1
            continue

        if op.startswith("v_cvt_f64_u32"):
            # start of the uint24 divide block
            divisor = _parseOperand(lines[pc].split(",")[1], regs)
            dividend = None
            q = r = None
            seen = 0
            j = pc
            while j < len(lines):
                cur = lines[j]
                if cur.startswith("v_cvt_f64_u32") and j != pc:
                    dividend = _parseOperand(cur.split(",")[1], regs)
                if cur.startswith("v_readfirstlane_b32"):
                    dst = _parseDest(cur.split(" ", 1)[1].split(",")[0])
                    if seen == 0:
                        q = dst
                    else:
                        r = dst
                    seen += 1
                    if seen == 2:
                        break
                j += 1
            assert dividend is not None and q is not None and r is not None
            regs[q], regs[r] = divmod(dividend, divisor)
            pc = j + 1
            continue

        if op.startswith("v_") or op.startswith("s_mov_b64") or "exec" in line:
            pc += 1
            continue

        if op == "s_cmp_eq_u32":
            scc = _parseOperand(args[0], regs) == _parseOperand(args[1], regs)
        elif op == "s_cmp_ge_u32":
            scc = _parseOperand(args[0], regs) >= _parseOperand(args[1], regs)
        elif op == "s_cmp_lt_u32":
            scc = _parseOperand(args[0], regs) < _parseOperand(args[1], regs)
        elif op == "s_cmp_gt_u32":
            scc = _parseOperand(args[0], regs) > _parseOperand(args[1], regs)
        elif op == "s_mov_b32":
            regs[_parseDest(args[0])] = _parseOperand(args[1], regs)
        elif op == "s_cselect_b32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs) if scc
                                         else _parseOperand(args[2], regs))
        elif op == "s_mul_i32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         * _parseOperand(args[2], regs)) & _MASK32
        elif op == "s_add_u32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         + _parseOperand(args[2], regs)) & _MASK32
        elif op == "s_sub_u32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         - _parseOperand(args[2], regs)) & _MASK32
        elif op == "s_and_b32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         & _parseOperand(args[2], regs))
        elif op == "s_or_b32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         | _parseOperand(args[2], regs))
        elif op == "s_lshl_b32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         << _parseOperand(args[2], regs)) & _MASK32
        elif op == "s_lshr_b32":
            regs[_parseDest(args[0])] = (_parseOperand(args[1], regs)
                                         >> _parseOperand(args[2], regs))
        else:
            raise AssertionError("interpreter does not model %r" % line)
        pc += 1
    return regs


# ---------------------------------------------------------------------------
# Emitter drivers
# ---------------------------------------------------------------------------
_SK_TEXT_CACHE = {}


def skDecodeEmitted(tileID, nwg0, nwg1):
    """Run the emitted skIndexToWG. Returns (wg0, wg1), or None when the
    runtime guards send it to the retained row-major body."""
    text = _SK_TEXT_CACHE.get("text")
    if text is None:
        text, _ = _renderSkIndexToWG()
        _SK_TEXT_CACHE["text"] = text
    regs = {
        "sgprNumWorkGroups0": nwg0,
        "sgprNumWorkGroups1": nwg1,
        "sgprSizesFree+2": 1,
        "s%u" % _TILE_ID_SGPR: tileID,
        "sgprWorkGroup0": -1,
        "sgprWorkGroup1": -1,
        "sgprWorkGroup2": -1,
    }
    legacy = [l.split(":")[0] for l in text.splitlines() if l.startswith("label_SKSlabLegacy")]
    try:
        _runAsm(text, regs, stopLabels=set(legacy))
    except _AsmStop:
        return None
    return regs["sgprWorkGroup0"], regs["sgprWorkGroup1"]


_BIT_TEXT_CACHE = {}


def bitSwizzleEmitted(which, g, nwg0):
    text = _BIT_TEXT_CACHE.get(which)
    if text is None:
        text = _renderBitSwizzle(_bitSwizzleWide if which == "wide" else _bitSwizzleTall)
        _BIT_TEXT_CACHE[which] = text
    regs = {"sgprWorkGroup0": g % nwg0, "sgprWorkGroup1": g // nwg0}
    _runAsm(text, regs)
    return regs["sgprWorkGroup0"], regs["sgprWorkGroup1"]


# ---------------------------------------------------------------------------
# Panel-traffic model (WGM_XCD_MAPPING_ANALYSIS.md section 5)
# ---------------------------------------------------------------------------
def panelsPerWG(seqs, total, win=CU_PER_XCD):
    """seqs[k] = the ordered tile coordinates XCD k executes. A generation is
    ``win`` consecutive entries; its cost is the distinct tiles it touches on
    each axis (= distinct operand panels)."""
    touches = 0
    for seq in seqs.values():
        for s in range(0, len(seq), win):
            w = seq[s:s + win]
            touches += len({t[0] for t in w}) + len({t[1] for t in w})
    return touches / total


def generations(seq, win=CU_PER_XCD):
    return [len({t[0] for t in seq[s:s + win]}) + len({t[1] for t in seq[s:s + win]})
            for s in range(0, len(seq), win)]


def nonSkSeqs(dec, total):
    """Dispatch order without StreamK: XCD k runs launch ids k, k+8, k+16, ..."""
    return {k: [dec(j * NXCD + k) for j in range((total - 1 - k) // NXCD + 1)]
            for k in range(NXCD)}


def _identity(g, nwg0):
    return (g % nwg0, g // nwg0)


def _slabWalkGeneralRef(g, nwg0, nwg1):
    """_slabWalkGeneral as committed (axis = first of M, N with L % 32 == 0)."""
    k, t = g % NXCD, g // NXCD
    if nwg0 % (NXCD * G) == 0:
        L, Sh, axisM = nwg0, nwg1, True
    elif nwg1 % (NXCD * G) == 0:
        L, Sh, axisM = nwg1, nwg0, False
    else:
        return None
    block, r = divmod(t, G * Sh)
    lng = k * (L // NXCD) + (r % G) + G * block
    shrt = r // G
    return (lng, shrt) if axisM else (shrt, lng)


# ---------------------------------------------------------------------------
# Persistent StreamK schedule
# ---------------------------------------------------------------------------
def _perXcdWgLists(total, skGrid):
    """Per XCD, the tile list each of its persistent workgroups owns.
    WG w runs tiles w, w+skGrid, ...; with skGrid % 8 == 0, w % 8 is the XCD."""
    assert skGrid % NXCD == 0
    out = {}
    for k in range(NXCD):
        lists = [[w + i * skGrid for i in range((total - 1 - w) // skGrid + 1)]
                 for w in range(k, skGrid, NXCD) if w < total]
        out[k] = [l for l in lists if l]
    return out


def skGenerationPartition(total, skGrid, cu=CU_PER_XCD):
    """Per XCD, the tile ids co-resident in each generation. The first ``cu``
    workgroups are resident; the rest backfill as those retire."""
    out = {}
    for k, lists in _perXcdWgLists(total, skGrid).items():
        pending = list(range(len(lists)))
        resident, pending = pending[:cu], pending[cu:]
        pos = {b: 0 for b in resident}
        gens = []
        while resident:
            gens.append(sorted(lists[b][pos[b]] for b in resident))
            nxt = []
            for b in resident:
                pos[b] += 1
                if pos[b] < len(lists[b]):
                    nxt.append(b)
                elif pending:
                    nb = pending.pop(0)
                    pos[nb] = 0
                    nxt.append(nb)
            resident = nxt
        out[k] = gens
    return out


def nonSkGenerationPartition(total, cu=CU_PER_XCD):
    """What the hardware produces without StreamK: a sliding 32-wide window over
    XCD k's launch ids k, k+8, k+16, ..."""
    out = {}
    for k in range(NXCD):
        ids = [j * NXCD + k for j in range((total - 1 - k) // NXCD + 1)]
        out[k] = [ids[s:s + cu] for s in range(0, len(ids), cu)]
    return out


# ---------------------------------------------------------------------------
# 0. The model must reproduce the published numbers before anything else counts.
# ---------------------------------------------------------------------------
def test_panel_model_is_calibrated():
    tall = nonSkSeqs(lambda g: bitSwizzleEmitted("tall", g, 128), 2048)
    assert round(panelsPerWG(tall, 2048), 4) == 0.3750

    wide = nonSkSeqs(lambda g: bitSwizzleEmitted("wide", g, 16), 2048)
    assert round(panelsPerWG(wide, 2048), 4) == 0.3750

    gen = nonSkSeqs(lambda g: _slabWalkGeneralRef(g, 13, 128), 13 * 128)
    assert round(panelsPerWG(gen, 13 * 128), 4) == 0.4423
    assert generations(gen[0]) == [12, 16, 12, 16, 16, 12, 8]

    ident = nonSkSeqs(lambda g: _identity(g, 16), 16 * 103)
    assert round(panelsPerWG(ident, 16 * 103), 4) == 0.5680


# ---------------------------------------------------------------------------
# 1. The emitted assembly computes skSlabDecodeRef, tile for tile.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nwg0,nwg1", [(16, 128), (16, 103), (32, 103), (128, 16),
                                       (16, 104), (64, 16)])
def test_emitted_asm_matches_reference(nwg0, nwg1):
    for tileID in range(nwg0 * nwg1):
        assert skDecodeEmitted(tileID, nwg0, nwg1) == skSlabDecodeRef(tileID, nwg0, nwg1), \
            "tileID %u on %ux%u" % (tileID, nwg0, nwg1)


@pytest.mark.parametrize("nwg0,nwg1", [(13, 128), (16, 104), (12, 103), (5, 7)])
def test_unadmitted_grids_fall_through_to_row_major(nwg0, nwg1):
    """Grids outside the guards must reach the retained row-major body, and the
    reference must agree that they are outside.

    Note 16x104 -- Option A's padded grid -- is one of them: rowsQ = 13 is not a
    multiple of G, so the last sub-band would be partial. Option A needs the
    non-StreamK path, not this decode."""
    assert skSlabDecodeRef(0, nwg0, nwg1) is None
    assert skDecodeEmitted(0, nwg0, nwg1) is None


# ---------------------------------------------------------------------------
# 2. G1 -- bijectivity.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nwg0,nwg1", [(16, 128), (16, 103), (32, 103), (128, 16), (16, 100), (64, 16)])
def test_bijective(nwg0, nwg1):
    total = nwg0 * nwg1
    coords = [skSlabDecodeRef(t, nwg0, nwg1) for t in range(total)]
    assert None not in coords
    assert len(set(coords)) == total
    assert set(coords) == set(itertools.product(range(nwg0), range(nwg1)))


def test_each_xcd_gets_equal_tile_count():
    """The round-robin forces T/8 tiles per XCD; the decode cannot change that,
    which is why 16x103 needs a ragged tail rather than a 13-row rectangle."""
    for nwg0, nwg1 in [(16, 128), (16, 103)]:
        total = nwg0 * nwg1
        counts = {k: sum(1 for t in range(total) if t % NXCD == k) for k in range(NXCD)}
        assert set(counts.values()) == {total // NXCD}


# ---------------------------------------------------------------------------
# 3. G3 -- the load-bearing gate: identical to _bitSwizzleWide on 16x128.
# ---------------------------------------------------------------------------
def test_identity_with_bit_swizzle_wide_on_16x128():
    mismatches = [t for t in range(2048)
                  if skDecodeEmitted(t, 16, 128) != bitSwizzleEmitted("wide", t, 16)]
    assert mismatches == []


def test_identity_with_bit_swizzle_tall_on_128x16():
    mismatches = [t for t in range(2048)
                  if skDecodeEmitted(t, 128, 16) != bitSwizzleEmitted("tall", t, 128)]
    assert mismatches == []


def test_matches_slab_walk_general_where_that_path_is_defined():
    for nwg0, nwg1 in [(16, 128), (128, 16), (64, 16), (16, 64)]:
        for g in range(nwg0 * nwg1):
            assert skSlabDecodeRef(g, nwg0, nwg1) == _slabWalkGeneralRef(g, nwg0, nwg1), \
                "%ux%u g=%u" % (nwg0, nwg1, g)


# ---------------------------------------------------------------------------
# 4. G4 -- at skGrid 256 the SK generation partition equals the non-SK one, and
#    256 is the only usable value.
# ---------------------------------------------------------------------------
def test_generation_partition_matches_non_sk_at_skgrid_256():
    ref = nonSkGenerationPartition(2048)
    got = skGenerationPartition(2048, 256)
    assert all(got[k] == ref[k] for k in range(NXCD))
    # and the mapping applied to it is the FlyDSL floor
    seqs = {k: [skSlabDecodeRef(t, 16, 128) for gen in got[k] for t in gen]
            for k in range(NXCD)}
    assert round(panelsPerWG(seqs, 2048), 4) == 0.3750
    assert all(g == 12 for g in generations(seqs[0]))


@pytest.mark.parametrize("skGrid", [8, 64, 128, 320, 512, 1024])
def test_skgrid_256_is_the_unique_usable_value(skGrid):
    """Only skGrid 256 reproduces the non-SK generation partition (2048, one
    tile per WG, is the degenerate exception and gives up persistence)."""
    ref = nonSkGenerationPartition(2048)
    got = skGenerationPartition(2048, skGrid)
    assert not all(got[k] == ref[k] for k in range(NXCD))


def test_skgrid_must_be_a_multiple_of_eight():
    """tileID % 8 == XCD only holds when the stride is a multiple of 8; that is
    the premise the whole decode rests on."""
    total, skGrid = 1648, 260
    perWg = [[w + i * skGrid for i in range((total - 1 - w) // skGrid + 1)]
             for w in range(skGrid) if w < total]
    assert any(len({t % NXCD for t in tiles}) > 1 for tiles in perWg)


def test_skgrid_256_makespan_is_optimal_on_16x103():
    """1648 tiles on 256 CUs needs ceil(6.4375) = 7 tile-times; skGrid 256
    attains it, 824 (the doc's withdrawn recommendation) does not."""
    assert max(len(g) for g in skGenerationPartition(1648, 256).values()) == 7
    assert max(len(g) for g in skGenerationPartition(1648, 824).values()) == 8


# ---------------------------------------------------------------------------
# 5. Panel traffic: the target grid, and no regression elsewhere.
# ---------------------------------------------------------------------------
def test_panels_per_wg_on_16x103():
    total = 16 * 103
    seqs = nonSkSeqs(lambda t: skSlabDecodeRef(t, 16, 103), total)
    assert round(panelsPerWG(seqs, total), 4) == 0.3932
    # six clean 4-long x 8-short windows, then a 7-long x 2-short tail
    assert generations(seqs[0]) == [12, 12, 12, 12, 12, 12, 9]
    # against what runs today
    ident = nonSkSeqs(lambda g: _identity(g, 16), total)
    assert round(panelsPerWG(ident, total), 4) == 0.5680


def test_no_regression_on_32x103():
    """G7. 32x103 is already well served by the row-major dispatch order; the
    decode admits it, so it must not make it worse."""
    total = 32 * 103
    baseline = panelsPerWG(nonSkSeqs(lambda g: _identity(g, 32), total), total)
    got = panelsPerWG(nonSkSeqs(lambda t: skSlabDecodeRef(t, 32, 103), total), total)
    assert round(baseline, 4) == 0.3762
    assert round(got, 4) <= 0.3762


def test_16x128_is_at_the_flydsl_floor():
    seqs = nonSkSeqs(lambda t: skSlabDecodeRef(t, 16, 128), 2048)
    assert round(panelsPerWG(seqs, 2048), 4) == 0.3750


# ---------------------------------------------------------------------------
# 6. Negative test: the sketch in WGM_XCD_MAPPING_ANALYSIS.md section 7 is
#    broken. Pinned so it cannot be reintroduced as "the documented version".
# ---------------------------------------------------------------------------
def _docSection7Sketch(tileID, nwg0, nwg1):
    """Verbatim from WGM_XCD_MAPPING_ANALYSIS.md section 7, Option B."""
    L, Sh = nwg1, nwg0
    k, j = tileID % NXCD, tileID // NXCD
    q, r = divmod(L, NXCD)
    slab = q + (1 if k < r else 0)
    base = k * q + min(k, r)
    rows = slab // G
    band, u = divmod(j, G * Sh)
    if band < rows:
        win, v = divmod(u, CU_PER_XCD)
        return (CU_PER_XCD // G * win + v // G, base + G * band + (v % G))
    d = slab - G * rows                      # zero on XCD 7 when L == 103
    return (u // d, base + G * rows + (u % d))


def test_doc_section7_sketch_divides_by_zero_and_loses_cells():
    computed, zeroDiv = {}, 0
    for t in range(16 * 103):
        try:
            computed[_docSection7Sketch(t, 16, 103)] = True
        except ZeroDivisionError:
            zeroDiv += 1
    assert zeroDiv == 14, "XCD 7 has slab 12, rows 3, d = 12 - 4*3 = 0"
    missing = set(itertools.product(range(16), range(103))) - set(computed)
    assert len(missing) == 14
    assert (14, 12) in missing and (14, 25) in missing


def test_doc_section7_sketch_is_only_right_where_no_tail_exists():
    """It looks correct on 16x128 (and 13x128), which is why it survived review;
    the tail is where it fails."""
    assert all(_docSection7Sketch(t, 16, 128) == skSlabDecodeRef(t, 16, 128)
               for t in range(2048))


# ---------------------------------------------------------------------------
# 7. Emission gates and code shape.
# ---------------------------------------------------------------------------
def test_emission_is_gated_on_the_safe_parameter_set():
    assert skSlabDecodeReason(_kernel()) is None
    for override, fragment in [
        ({"StreamK": 2}, "StreamK != 3"),
        ({"StreamK": 5}, "StreamK != 3"),
        ({"StreamKForceDPOnly": 0}, "StreamKForceDPOnly"),
        ({"StreamKAtomic": 1}, "StreamKAtomic"),
        ({"StreamKXCCMapping": 1}, "StreamKXCCMapping"),
        ({"SpaceFillingAlgo": ["hilbert"]}, "SpaceFillingAlgo"),
        ({"WGMBitSwizzle": True}, "WGMBitSwizzle"),
        # the landmine: WGM 8 still validates but scrambles the slab layout
        ({"WorkGroupMapping": 8}, "WorkGroupMapping"),
        ({"WorkGroupMapping": 0}, "WorkGroupMapping"),
        ({"WorkGroupMappingXCC": -1}, "WorkGroupMappingXCC"),
    ]:
        reason = skSlabDecodeReason(_kernel(**override))
        assert reason is not None and fragment in reason, override


def test_blocked_kernels_emit_no_slab_code():
    text, _ = _renderSkIndexToWG(_kernel(StreamK=2))
    assert "SKSlab" not in text
    assert "slab decode not emitted" in text
    # and the row-major body is still there, unchanged
    assert text.count("v_rcp_iflag_f32") == 2


def test_one_divide_on_the_main_path():
    """G10. The row-major body it replaces runs two divides per tile; the slab
    decode's rectangular path (93%+ of tiles) runs one."""
    text, _ = _renderSkIndexToWG()
    lines = [l for l in text.splitlines() if l.strip()]

    def at(prefix):
        return next(i for i, l in enumerate(lines) if l.startswith(prefix))

    iTail = at("label_SKSlabTail")
    iWb = at("label_SKSlabWriteback")
    iLegacy = at("label_SKSlabLegacy")
    iEnd = at("label_SKSlabEnd")

    def rcp(chunk):
        return sum(l.count("v_rcp") for l in chunk)

    assert rcp(lines[:iTail]) == 1                     # rectangular path
    assert rcp(lines[iWb:iLegacy]) == 0                # writeback
    assert rcp(lines[iTail:iWb]) == 1                  # tail path
    assert rcp(lines[iLegacy:iEnd]) == 2               # retained row-major body
    assert iTail < iWb < iLegacy < iEnd


def test_temporary_sgpr_budget():
    """The decode must stay a temporary-SGPR consumer: no new persistent SGPR,
    and a small enough block that it is unlikely to move
    .amdhsa_next_free_sgpr. Six unaligned temporaries is the current shape; the
    row-major body it replaces asks for none, so this is the whole marginal
    cost. Pinned because SGPR pressure is the one build-time failure mode of
    this change (G9: .amdhsa_next_free_sgpr <= 96)."""
    _, writer = _renderSkIndexToWG()
    assert writer.tmpHigh - 90 == 7      # 6 temporaries + 1 for the default alignment
