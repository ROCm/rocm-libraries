# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# The f16-accumulation reference model without hardware: compiles
# f16_accumulation_reference/driver.cpp against client/include/F16AccumulationStep.hpp,
# the header the client's reference runs, and checks
#  - the WMMA step against results recorded on gfx1250 silicon, bit for bit;
#  - the K order of every split-K slice against the kernel writer's stagger and
#    split-K code, transcribed below;
#  - the MultipleBuffer reduction and the rounding helpers;
#  - the StreamK split of each tile into chunks against the kernel's partition, transcribed below, and the
#    StreamK tree fixup.
################################################################################

import itertools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_DATA = Path(__file__).parent / "f16_accumulation_reference"
_INCLUDE = Path(__file__).resolve().parents[3] / "client" / "include"


def _compiler():
    for cxx in (os.environ.get("CXX"), "amdclang++", "/opt/rocm/llvm/bin/amdclang++", "clang++", "g++"):
        if cxx and shutil.which(cxx):
            return shutil.which(cxx)
    return None


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    cxx = _compiler()
    if cxx is None:
        if os.environ.get("CI"):
            pytest.fail("no C++ compiler for the f16-accumulation reference test")
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("f16acc") / "driver"
    subprocess.run([cxx, "-std=c++17", "-O2", "-I", str(_INCLUDE), str(_DATA / "driver.cpp"), "-o", str(exe)],
                   check=True)
    return exe


def _run(driver, *args):
    return subprocess.run([str(driver), *map(str, args)], check=True, capture_output=True, text=True).stdout


def test_wmma_step_matches_silicon_bit_for_bit(driver):
    """wmma_golden.txt holds one D element per line, `x64|x128 C A B D`: the f16 accumulator in, the fp8 (OCP e4m3)
    bytes of one row of A and one column of B over the instruction's K, and the f16 result that
    v_wmma_f16_16x16x64_fp8_fp8 or v_wmma_f16_16x16x128_fp8_fp8 wrote on gfx1250 silicon of ASIC revision 1, each
    instruction run once by a capture kernel on chosen operands."""
    records, differ, differShort = map(int, _run(driver, "golden", _DATA / "wmma_golden.txt").split())
    assert records == 288
    assert differ == 0
    assert differShort == 0


def test_wmma_step_overflows_like_silicon(driver):
    """wmma_overflow_golden.txt, same record format: v_wmma_f16_16x16x128_fp8_fp8 on gfx1250 silicon of ASIC revision 1
    with MODE.FP16_OVFL clear, which the kernels never set. A 128 x 1024 sum, 448 x 448 products, overflow through C
    and C = +inf give +inf; so does 448 x 448 with B = +448 on K 0..63 and -448 on K 64..127 (exact result 0, or 1000
    with C = 1000), because the first 64-K step overflows and an infinite C passes through the second.
    Inside one step, +-448 x 448 products cancel without overflow, whether they alternate within a group or fill whole
    groups, and a small product beside a cancelling pair is truncated to the group's resolution toward -inf (252 -> 248,
    -252 -> -256, 0.875 -> 0) while in its own group it stays exact; C = 2^-14 beside them counts as 0."""
    records, differ, differShort = map(int, _run(driver, "golden", _DATA / "wmma_overflow_golden.txt").split())
    assert (records, differ, differShort) == (21, 0, 0)


def test_flipped_golden_result_is_counted(driver, tmp_path):
    lines = (_DATA / "wmma_golden.txt").read_text().splitlines()
    kind, c, a, b, d = lines[7].split()
    lines[7] = " ".join((kind, c, a, b, "%04x" % (int(d, 16) ^ 1)))
    flipped = tmp_path / "flipped.txt"
    flipped.write_text("\n".join(lines) + "\n")
    records, differ, _ = map(int, _run(driver, "golden", flipped).split())
    assert (records, differ) == (288, 1)


def test_rounding_helpers_at_their_boundaries(driver):
    assert _run(driver, "round").split() == ["0"]


def _kernel_order(depthU, gsu, gsuc, staggerU, shift, mapping, pgr, K, wg0, wg1, numTiles0, slice):
    """64-K step starts of one split-K slice of tile (wg0, wg1), transcribed from the kernel writer."""
    # Components/GSU.py: calculateLoopNumIterGsu, computeLoadSrd (GSUC offset), graIncrements, tailLoopNumIter.
    numIter = K // depthU
    q, r = divmod(numIter, gsu)
    n = q + (1 if slice < r else 0)
    if gsu == 1:
        first, stride = 0, 1
    elif gsuc:
        first, stride = ((q + 1) * slice if slice < r else q * slice + r), 1
    else:
        first, stride = slice, gsu
    tailSlice = 0 if gsu == 1 else (r if not gsuc else (r if q == 0 else gsu - 1))
    # KernelWriterAssembly.declareStaggerParms, in 32-bit scalar arithmetic.
    cur = staggerU & 0xff
    if pgr >= 3 and n <= pgr - 1:
        cur = 0
    while n < ((cur << shift) & 0xffffffff):
        cur >>= 1
    mask = cur - 1 if cur >= 1 else 0
    inputs = {0: wg0, 1: wg1, 2: 0xffffffff, 3: (cur + numTiles0 * wg1 + wg0) & 0xffffffff, 4: 0xffffffff}
    start = ((mask & inputs.get(mapping, cur)) << shift) & 0xffffffff
    steps = []
    for t in range(n):
        it = first + ((start + t) % n) * stride
        steps += range(it * depthU, (it + 1) * depthU, 64)
    if K % depthU and slice == tailSlice:
        steps += range(numIter * depthU, K, 64)
    return steps


def _orders(driver, tmp_path, cases):
    path = tmp_path / "order.txt"
    path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    return [list(map(int, line.split())) for line in _run(driver, "order", path).splitlines()]


# (depthU, gsu, gsuc, staggerU, shift, mapping, pgr, K, wg0, wg1, numTiles0, slice) -> expected steps
_HAND_CHECKED = [
    # 8 iterations: StaggerU 32 shrinks to 8, start = wg0 & 7 = 5.
    ((128, 1, 0, 32, 0, 0, 2, 1024, 5, 0, 8, 0), [640, 704, 768, 832, 896, 960] + list(range(0, 640, 64))),
    # Mapping 2 on a GEMM: every tile starts at the largest offset.
    ((128, 1, 0, 32, 0, 2, 2, 1024, 5, 3, 8, 0), [896, 960] + list(range(0, 896, 64))),
    # Mapping 3: start = (8 + numTiles0 * wg1 + wg0) & 7 = (8 + 4 * 2 + 1) & 7 = 1.
    ((128, 1, 0, 32, 0, 3, 2, 1024, 1, 2, 4, 0), list(range(128, 1024, 64)) + [0, 64]),
    # Stride shift 1: two-iteration units, StaggerU 4 shrinks to 4 (4 << 1 <= 8), start = (wg1 & 3) << 1 = 6.
    ((128, 1, 0, 4, 1, 1, 2, 1024, 0, 3, 8, 0), [768, 832, 896, 960] + list(range(0, 768, 64))),
    # PrefetchGlobalRead 3 with 2 iterations: no stagger.
    ((128, 1, 0, 32, 0, 0, 3, 256, 1, 0, 8, 0), [0, 64, 128, 192]),
    # A tail runs last, after the rotated loop.
    ((128, 1, 0, 2, 0, 0, 2, 320, 1, 0, 8, 0), [128, 192, 0, 64, 256]),
    # Split-K 3, interleaved: slice 2 runs iterations 2 and 5, then the tail (numIter % GSU == 2).
    ((128, 3, 0, 0, 0, 0, 2, 1088, 0, 0, 8, 2), [256, 320, 640, 704, 1024]),
    ((128, 3, 0, 0, 0, 0, 2, 1088, 0, 0, 8, 0), [0, 64, 384, 448, 768, 832]),
    # Split-K 3, coalesced: slice 1 runs iterations 3..5; the last slice takes the tail.
    ((128, 3, 1, 0, 0, 0, 2, 1088, 0, 0, 8, 1), list(range(384, 768, 64))),
    ((128, 3, 1, 0, 0, 0, 2, 1088, 0, 0, 8, 2), list(range(768, 1088, 64))),
    # Split-K 2 with stagger: each slice shrinks StaggerU to its own loop count (4 iterations -> 4).
    ((128, 2, 0, 32, 0, 0, 2, 1024, 3, 0, 8, 1), [896, 960, 128, 192, 384, 448, 640, 704]),
]


def test_slice_order_matches_hand_checked_cases(driver, tmp_path):
    cases = [c for c, _ in _HAND_CHECKED]
    assert _orders(driver, tmp_path, cases) == [e for _, e in _HAND_CHECKED]
    assert [_kernel_order(*c) for c in cases] == [e for _, e in _HAND_CHECKED]


def test_slice_order_matches_the_kernel_writer(driver, tmp_path):
    cases = []
    for depthU, gsu, gsuc, staggerU, shift, mapping, pgr, K in itertools.product(
            (128, 256), (1, 2, 3, 4), (0, 1), (0, 2, 32, 48), (0, 1), range(6), (2, 3), (0, 64, 960, 4288, 8320)):
        if gsu == 1 and gsuc:
            continue
        for slice in range(gsu):
            cases.append((depthU, gsu, gsuc, staggerU, shift, mapping, pgr, K, 3, 5, 9, slice))
    got = _orders(driver, tmp_path, cases)
    assert len(got) == len(cases)
    for case, steps in zip(cases, got):
        assert steps == _kernel_order(*case), case
    # Every K step is accumulated by exactly one slice.
    byLaunch = {}
    for case, steps in zip(cases, got):
        byLaunch.setdefault(case[:-1], []).extend(steps)
    for case, steps in byLaunch.items():
        assert sorted(steps) == list(range(0, case[7], 64)), case


def test_reduction_adds_slices_in_f32_in_order(driver, tmp_path):
    rng = np.random.default_rng(7)
    cases = [[0x6800, 0x0001, 0xe800], [0x6800, 0xe800, 0x0001], [0x8000, 0x8000], [0x7bff, 0x7bff, 0xfbff],
             [0x7c00, 0x3c00], [0x0001, 0x8001, 0x0001]]
    for gsu in (2, 3, 4, 5, 7):
        for _ in range(40):
            h = rng.integers(0, 0x10000, gsu, dtype=np.uint32)
            h = h[(h & 0x7c00) != 0x7c00][:gsu]
            if len(h) == gsu:
                cases.append([int(x) for x in h])
    path = tmp_path / "reduce.txt"
    path.write_text("".join(" ".join("%04x" % x for x in c) + "\n" for c in cases))
    got = [int(x, 16) for x in _run(driver, "reduce", path).split()]
    assert got[:2] == [0x0000, 0x0001]
    for c, g in zip(cases, got):
        acc = np.float32(np.array([c[0]], np.uint16).view(np.float16)[0])
        for x in c[1:]:
            acc = np.float32(acc + np.float32(np.array([x], np.uint16).view(np.float16)[0]))
        want = int(np.array([np.float16(acc)]).view(np.uint16)[0])
        assert g == want or (np.isnan(acc) and g & 0x7fff == 0x7e00), (c, hex(g), hex(want))


def test_reduction_of_overflowed_slices(driver, tmp_path):
    # Opposite infinities, in either order and with a finite slice between them, give NaN; same-sign infinities and a
    # finite opposite pair at the f16 limit stay exact in the f32 sum.
    cases = [[0x7c00, 0xfc00], [0xfc00, 0x7c00], [0x7c00, 0x3c00, 0xfc00],
             [0x7c00, 0x7c00, 0x7c00], [0xfc00, 0xfc00], [0x7bff, 0xfbff], [0xfbff, 0xfbff]]
    path = tmp_path / "reduce_inf.txt"
    path.write_text("".join(" ".join("%04x" % x for x in c) + "\n" for c in cases))
    got = [int(x, 16) for x in _run(driver, "reduce", path).split()]
    assert all(g & 0x7c00 == 0x7c00 and g & 0x3ff for g in got[:3]), [hex(g) for g in got]
    assert got[3:] == [0x7c00, 0xfc00, 0x0000, 0xfc00], [hex(g) for g in got]


_KEY_FIELDS = ("depthU", "gsu", "gsuc", "staggerU", "shift", "mapping", "pgr", "mt0", "mt1")


def test_order_key_changes_exactly_with_the_order_inputs(driver, tmp_path):
    base = dict(depthU=256, gsu=1, gsuc=0, staggerU=32, shift=2, mapping=0, pgr=2, mt0=128, mt1=64)
    pairs = []
    for mapping in range(8):
        on = {**base, "mapping": mapping}
        pairs.append((on, {**on, "mt0": 256}, mapping in (0, 3)))
        pairs.append((on, {**on, "mt1": 128}, mapping in (1, 3)))
    split = {**base, "gsu": 2, "staggerU": 0}
    pairs += [
        (base, {**base, "depthU": 128}, True),
        (base, {**base, "gsu": 2}, True),
        (split, {**split, "gsuc": 1}, True),
        (base, {**base, "gsuc": 1}, False),
        (split, {**split, "mt0": 256, "mt1": 128, "mapping": 3}, False),
        (base, {**base, "staggerU": 16}, True),
        (base, {**base, "shift": 3}, True),
        (base, {**base, "mapping": 1}, True),
        (base, {**base, "pgr": 3}, True),
        ({**base, "pgr": 3}, {**base, "pgr": 4}, True),
        ({**base, "pgr": 1}, base, False),
        ({**base, "staggerU": 0}, {**base, "staggerU": 0, "depthU": 128, "mapping": 3, "mt0": 256}, False),
    ]
    path = tmp_path / "key.txt"
    path.write_text("".join(" ".join(str(l[f]) for f in _KEY_FIELDS) + "\n" for p in pairs for l in p[:2]))
    keys = _run(driver, "key", path).split("\n")
    for i, (a, b, differs) in enumerate(pairs):
        assert (keys[2 * i] != keys[2 * i + 1]) == differs, (a, b)


_DP, _STATIC, _PARALLEL, _DYNAMIC = 1, 2, 3, 4


def _static_split(tiles, I, grid, skFullTiles):
    # streamKStaticSplit (ContractionSolution.cpp): skTiles, SKItersPerWG and the global extra iterations.
    skTiles = grid
    if tiles % grid:
        skTiles = min(grid * skFullTiles + tiles % grid if tiles > grid else tiles, tiles)
    W = skTiles * I // grid
    return skTiles, W, skTiles * I - W * grid


def _kernel_workgroup_ranges(kind, tiles, I, grid, skTiles, W, E, split, WI, perTile):
    # The iteration ranges the workgroups or work items run, in their order, transcribed from StreamK.py:
    # StreamKTwoTileDPFirst.initializePartition / nextStaticCursor / skAssignIters, and StreamKDynamic.activateWorkItem.
    if kind == _PARALLEL:
        for w in range(grid):
            tile, p = divmod(w, split)
            extra = I - split * W
            start = p * (W + 1) if p < extra else p * W + extra
            yield tile * I + start, tile * I + start + W + (p < extra)
    elif kind == _STATIC:
        dpSection = (tiles - skTiles) * I
        yield from ((t * I, t * I + I) for t in range(tiles - skTiles))
        for w in range(grid):
            if perTile and skTiles and grid % skTiles == 0:
                F = grid // skTiles
                q, s = divmod(w, F)
                remI = I - F * W
                start = q * I + s * W + min(s, remI)
                end = start + W + (s < remI)
            elif w < E:
                start, end = w * (W + 1), w * (W + 1) + W + 1
            else:
                start, end = w * W + E, w * W + E + W
            yield dpSection + start, min(dpSection + end, tiles * I)
    elif kind == _DYNAMIC:
        full = tiles - skTiles
        for wi in range(full + skTiles * split):
            if wi < full:
                yield wi * I, wi * I + I
            else:
                t, p = divmod(wi - full, split)
                yield (t + full) * I + p * WI, (t + full) * I + min(p * WI + WI, I)
    else:
        yield from ((t * I, t * I + I) for t in range(tiles))


def _kernel_chunks(*case):
    # Walks each range tile by tile as the persistent loop does (skTileIndex: local start and end per tile).
    tiles, I = case[1], case[2]
    chunks = [[] for _ in range(tiles)]
    for start, end in _kernel_workgroup_ranges(*case):
        while start < end:
            t = start // I
            stop = min(end, t * I + I)
            chunks[t].append((start - t * I, stop - t * I))
            start = stop
    return chunks


def _chunk_cases():
    for tiles, I, grid in itertools.product((1, 3, 7, 12), (1, 2, 5, 8, 13), (1, 2, 3, 4, 6, 8, 13, 24)):
        for skFullTiles, perTile in itertools.product((0, 1), (0, 1)):
            skTiles, W, E = _static_split(tiles, I, grid, skFullTiles)
            if perTile and skTiles and grid % skTiles == 0 and grid // skTiles > I:
                continue
            yield (_STATIC, tiles, I, grid, skTiles, W, E, 0, 0, perTile)
        if grid % tiles == 0 and 2 <= grid // tiles <= I:
            F = grid // tiles
            yield (_PARALLEL, tiles, I, grid, F, I // F, 0, F, 0, 0)
        for skTiles, split in itertools.product(sorted({0, 1, tiles // 2, tiles}), (2, 3, 4)):
            WI = -(-I // split)
            yield (_DYNAMIC, tiles, I, grid, skTiles, 0, 0, -(-I // WI), WI, 0)
        yield (_DP, tiles, I, grid, 0, 0, 0, 0, 0, 0)


def _driver_chunks(driver, tmp_path, cases):
    lines = []
    for kind, tiles, I, grid, skTiles, W, E, split, WI, perTile in cases:
        for t in range(tiles):
            lines.append(" ".join(map(str, (kind, 0, perTile, tiles, I, grid, skTiles, W, E, split, WI, t))))
    path = tmp_path / "chunks.txt"
    path.write_text("\n".join(lines) + "\n")
    out = iter(_run(driver, "chunks", path).splitlines())
    result = []
    for case in cases:
        tiles = case[1]
        rows = [next(out).split() for _ in range(tiles)]
        result.append([row if row == ["error"] else list(zip(map(int, row[::2]), map(int, row[1::2]))) for row in rows])
    return result


def test_streamk_chunks_match_the_kernel_partition(driver, tmp_path):
    cases = list(_chunk_cases())
    for case, got in zip(cases, _driver_chunks(driver, tmp_path, cases)):
        assert got == _kernel_chunks(*case), case


def test_streamk_chunks_hand_checked(driver, tmp_path):
    # Two-tile StreamK, 7 tiles of 5 iterations on 3 workgroups with no full stream-k tiles: tiles 0-5 are data
    # parallel and tile 6 is split 2 + 2 + 1 (global first-E: E = 2). The same with per-tile extras is unchanged
    # (3 % 1 == 0, W = 1, 5 - 3 = 2 extras). Parallel with F = 3 over I = 8 splits every tile 3 + 3 + 2; the dynamic
    # queue splits the last tile into ceil(8 / 3) = 3 pieces of up to 3.
    cases = [(_STATIC, 7, 5, 3, 1, 1, 2, 0, 0, 0), (_STATIC, 7, 5, 3, 1, 1, 2, 0, 0, 1),
             (_PARALLEL, 2, 8, 6, 3, 2, 0, 3, 0, 0), (_DYNAMIC, 2, 8, 8, 1, 0, 0, 3, 3, 0)]
    got = _driver_chunks(driver, tmp_path, cases)
    assert got[0] == [[(0, 5)]] * 6 + [[(0, 2), (2, 4), (4, 5)]]
    assert got[1] == got[0]
    assert got[2] == [[(0, 3), (3, 6), (6, 8)]] * 2
    assert got[3] == [[(0, 8)], [(0, 3), (3, 6), (6, 8)]]


def test_streamk_chunk_without_iterations_is_an_error(driver, tmp_path):
    # Per-tile extras with more workgroups per tile than iterations leave a workgroup nothing to run.
    assert _driver_chunks(driver, tmp_path, [(_STATIC, 2, 2, 6, 2, 0, 0, 0, 0, 1)]) == [[["error"]] * 2]


def test_streamk_uneven_static_split_stays_inside_each_tile(driver, tmp_path):
    # The host also selects per-tile extras for a static tree-fixup launch whose workgroups divide evenly among the
    # tiles while a tile's iterations do not divide among its workgroups: 4 tiles of 65 iterations on 8 workgroups give
    # every tile 33 + 32 and no workgroup crosses a tile. Global first-E on the same launch runs workgroups across
    # tiles, so the K order differs and the reference must follow the bit the host packs.
    skTiles, W, E = _static_split(4, 65, 8, 0)
    perTile = (_STATIC, 4, 65, 8, skTiles, W, E, 0, 0, 1)
    globalE = perTile[:-1] + (0,)
    got = _driver_chunks(driver, tmp_path, [perTile, globalE])
    assert got[0] == _kernel_chunks(*perTile) == [[(0, 33), (33, 65)]] * 4
    assert all(start // 65 == (end - 1) // 65 for start, end in _kernel_workgroup_ranges(*perTile))
    assert got[1] == _kernel_chunks(*globalE) != got[0]


def test_streamk_chunk_steps(driver, tmp_path):
    # depthU 128 and K = 600: four full iterations and a tail of 88. A whole tile follows the unsplit loop with its
    # stagger (StaggerU 2 on WorkGroup0 1 starts at iteration 1); other chunks run in order without one, and only the
    # chunk that ends the tile runs the tail.
    cases = [(128, 2, 0, 0, 2, 600, 1, 0, 1, 5, 0, 5), (128, 2, 0, 0, 2, 600, 1, 0, 1, 5, 1, 3),
             (128, 2, 0, 0, 2, 600, 1, 0, 1, 5, 3, 5), (128, 2, 0, 0, 2, 512, 1, 0, 1, 4, 2, 4)]
    path = tmp_path / "chunksteps.txt"
    path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    got = [list(map(int, line.split())) for line in _run(driver, "chunksteps", path).splitlines()]
    assert got[0] == [128, 192, 256, 320, 384, 448, 0, 64, 512, 576]
    assert got[1] == [128, 192, 256, 320]
    assert got[2] == [384, 448, 512, 576]
    assert got[3] == [256, 320, 384, 448]


def _kernel_tree(partials):
    # The tree fixup loop of StreamK.storeBranchesCommon in f32: a workgroup with partialIdx bit L set stores its sum
    # at level L; otherwise it adds the sum of workgroup i + 2^L, unless that is past the tile.
    acc = [np.float32(np.array([p], np.uint16).view(np.float16)[0]) for p in partials]
    for i in reversed(range(len(partials))):
        offset = 1
        while not (i // offset) & 1 and i + offset < len(partials):
            acc[i] = np.float32(acc[i] + acc[i + offset])
            offset *= 2
    return int(np.array([np.float16(acc[0])]).view(np.uint16)[0])


def _default_wgm(wgm, n0, n1, wg0, wg1):
    """DefaultWGM step by step as WorkGroupMappingAlgos.DefaultWGM emits it: WGM > 1 remaps (WorkGroup0, WorkGroup1),
    WGM < 0 does the same with the two swapped, 0 and 1 leave them."""
    if 0 <= wgm <= 1:
        return wg0, wg1
    a = abs(wgm)
    first, second, nFirst, nSecond = (wg1, wg0, n1, n0) if wgm < 0 else (wg0, wg1, n0, n1)
    blockId = second // a
    wgSerial = (second - blockId * a) * nFirst + first
    numFullBlocks = nSecond // a
    rem = nSecond - a * numFullBlocks
    divisor = (rem if rem else a) if blockId >= numFullBlocks else a
    first, second = wgSerial // divisor, wgSerial - wgSerial // divisor * divisor + blockId * a
    return (second, first) if wgm < 0 else (first, second)


def test_streamk_tile_order_follows_default_wgm(driver, tmp_path):
    # tileIndexToWorkGroup puts linear tile t at (t % n0, t // n0 % n1) of batch t // (n0 * n1); DefaultWGM moves it.
    cases = [(w, n0, n1, b) for w in (-5, -3, -2, -1, 0, 1, 2, 3, 4, 8) for n0 in (1, 2, 3, 5, 7)
             for n1 in (1, 2, 3, 6) for b in (1, 2)]
    path = tmp_path / "tileorder.txt"
    path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    got = [list(map(int, line.split())) for line in _run(driver, "tileorder", path).splitlines()]
    assert len(got) == len(cases)
    for (w, n0, n1, b), tileOf in zip(cases, got):
        want = [None] * (n0 * n1 * b)
        for t in range(n0 * n1 * b):
            wg0, wg1 = _default_wgm(w, n0, n1, t % (n0 * n1) % n0, t % (n0 * n1) // n0)
            want[wg0 + wg1 * n0 + t // (n0 * n1) * n0 * n1] = t
        assert tileOf == want, (w, n0, n1, b)
    # WGM 2 on 3 x 3: tiles 0-5 walk WorkGroup1 0-1 two at a time, the last column follows alone.
    assert got[cases.index((2, 3, 3, 1))] == [0, 2, 4, 1, 3, 5, 6, 7, 8]


def _space_filling(algo, wgm, n0, n1, index):
    """SpaceFillingCurveWalk step by step as WorkGroupMappingAlgos emits it: chiplet_transform over 8 XCCs, the
    multi-level block and three edge blocks for more than one level, TransformNLevels, ColRowMajor (both orders divide
    by the block's M) and SpaceFillCurveSimpleImpl with its direction tables."""
    levels = len(algo)
    pos, xcc = divmod(index, 8)
    minPer, extra = divmod(n0 * n1, 8)
    index = xcc * minPer + (xcc if xcc < extra else extra) + pos

    def simple(orderID, M, N, li):
        if orderID == 2:
            ncc, n, scc, s, ecc, e, wcc, w = range(8)
            blocksOf = [[0, 1, 3, 2], [2, 3, 1, 0], [3, 2, 0, 1], [1, 0, 2, 3],
                        [2, 0, 1, 3], [3, 1, 0, 2], [0, 1, 3, 2], [1, 3, 2, 0]]
            newDir = [[w, ncc, ncc, e], [ecc, n, n, wcc], [wcc, scc, scc, ecc], [e, s, s, w],
                      [n, ecc, ecc, s], [scc, e, e, ncc], [s, wcc, wcc, n], [ncc, w, w, scc]]
        else:
            blocksOf, newDir = [{3: [0, 2, 1, 3], 4: [0, 1, 2, 3], 5: [0, 1, 3, 2]}[orderID]], [[0] * 4]
        half = lambda L: L // 2 if 1 << (L.bit_length() - 1) == L else 1 << (L.bit_length() - 1)
        y = x = d = 0
        while M * N > 1:
            M1, N1 = half(M), half(N)
            M2, N2 = M - M1, N - N1
            blocks = [(M1, N1, 0, 0), (M2, N1, M1, 0), (M1, N2, 0, N1), (M2, N2, M1, N1)]
            for j, b in enumerate(blocksOf[d]):
                bm, bn, oy, ox = blocks[b]
                if j < 3 and li >= bm * bn:
                    li -= bm * bn
                    continue
                M, N, y, x, d = bm, bn, y + oy, x + ox, newDir[d][j]
                break
        return y, x

    def transform(idx, M, N, numLevels, gy, gx):
        numLevels = min(numLevels, levels)
        cumDen = [1]

        def order(lvl, bm, bn):
            li = (idx if lvl == 0 else idx // cumDen[0]) % (bm * bn)
            cumDen[0] *= bm * bn
            return (li % bm, li // bm) if algo[lvl] in (0, 1) else simple(algo[lvl], bm, bn, li)

        if numLevels == 1:
            ly, lx = order(0, M, N)
            return gy + ly, gx + lx
        cumX = cumY = 1
        temp = wgm
        for lvl in range(numLevels):
            if lvl != numLevels - 1:
                bm, bn = temp & 0xff, (temp >> 8) & 0xff
            else:
                bm, bn = wgm & 0xff, (wgm >> 8) & 0xff
                if numLevels == 3:
                    bm, bn = bm * ((wgm >> 16) & 0xff), bn * ((wgm >> 24) & 0xff)
                bm, bn = M // bm, N // bn
            ly, lx = order(lvl, bm, bn)
            gx, gy = gx + lx * cumX, gy + ly * cumY
            if lvl != numLevels - 1:
                cumX, cumY, temp = cumX * bn, cumY * bm, temp >> 16
        return gy, gx

    if levels == 1:
        return transform(index, n0, n1, 1, 0, 0)
    M1 = (wgm & 0xff) * ((wgm >> 16) & 0xff if levels == 3 else 1)
    N1 = ((wgm >> 8) & 0xff) * ((wgm >> 24) & 0xff if levels == 3 else 1)
    M2, N2 = n0 % M1, n1 % N1
    M1, N1 = n0 - M2, n1 - N2
    for size, M, N, gy, gx, lv in ((M1 * N1, M1, N1, 0, 0, levels), (M2 * N1, M2, N1, M1, 0, 1),
                                   (M2 * N2, M2, N2, M1, N1, 1), (M1 * N2, M1, N2, 0, N1, 1)):
        if index < size:
            return transform(index, M, N, lv, gy, gx)
        index -= size


def test_streamk_tile_order_follows_space_filling_walk(driver, tmp_path):
    # The persistent tile id of SpaceFillingCurveWalk is the in-batch tile t % (n0 * n1).
    algos = [(0,), (1,), (2,), (3,), (4,), (5,), (2, 2), (3, 0), (2, 5), (2, 3, 1)]
    words = {1: [0], 2: [0x0202, 0x0302, 0x0104], 3: [0x02020202, 0x01020302]}
    cases = [(a, w, n0, n1, b) for a in algos for w in words[len(a)] for n0 in (1, 2, 3, 4, 5, 7, 8, 9)
             for n1 in (1, 2, 4, 5, 6) for b in (1, 2)]
    pack = lambda a: sum(o << (4 * i) for i, o in enumerate(a)) | len(a) << 28
    path = tmp_path / "tileorder.txt"
    path.write_text("".join("1 %u %u %u %u %u\n" % (n0, n1, b, pack(a), w) for a, w, n0, n1, b in cases))
    got = [list(map(int, line.split())) for line in _run(driver, "tileorder", path).splitlines()]
    assert len(got) == len(cases)
    for (a, w, n0, n1, b), tileOf in zip(cases, got):
        want = [None] * (n0 * n1 * b)
        for t in range(n0 * n1 * b):
            wg0, wg1 = _space_filling(a, w, n0, n1, t % (n0 * n1))
            want[wg0 + wg1 * n0 + t // (n0 * n1) * n0 * n1] = t
        assert tileOf == want, (a, hex(w), n0, n1, b)
    # Hilbert on 2 x 2 walks a U: (0,0), (1,0), (1,1), (0,1).
    assert got[cases.index(((2,), 0, 2, 2, 1))] == [0, 1, 3, 2]


def test_tree_reduction_follows_the_kernel_tree(driver, tmp_path):
    rng = np.random.default_rng(11)
    cases = [[0x7800, 0x0001, 0xf800, 0x0001]]
    for n in range(2, 12):
        for _ in range(30):
            h = rng.integers(0, 0x10000, 2 * n, dtype=np.uint32)
            h = h[(h & 0x7c00) != 0x7c00][:n]
            if len(h) == n:
                cases.append([int(x) for x in h])
    path = tmp_path / "tree.txt"
    path.write_text("".join(" ".join("%04x" % x for x in c) + "\n" for c in cases))
    tree = [int(x, 16) for x in _run(driver, "tree", path).split()]
    linear = [int(x, 16) for x in _run(driver, "reduce", path).split()]
    assert (tree[0], linear[0]) == (0x0000, 0x0001)
    for c, g in zip(cases, tree):
        assert g == _kernel_tree(c), (c, hex(g))
    assert sum(t != l for t, l in zip(tree, linear)) > 0
