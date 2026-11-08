# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU tests of the pack / GEMM overlap (``kernels/gfx950/mxfp6_gemm/overlap.py``); skipped elsewhere.

* the co-resident pack kernel (:func:`overlap.build_pack_lite`) is byte-identical to the host packers on every
  output byte (two launches over different garbage), fits next to a resident GEMM CTA (no LDS, <= 48 VGPRs), and
  its mutants differ;
* the multi-stream schedules (M-chunk strict / pipelined, next-call, serial) run on real streams with events and
  ALTERNATING inputs (A, A rotated by 256 rows, ...) into one C buffer per call: every C is bitwise equal to the
  single pre-packed GEMM of that input; a GEMM that does not wait for its (deliberately slow) pack and a chunk GEMM
  reading the wrong chunk both fail.
GEMMs here are the LLVM-safe defaults at a small shape (the schedule / offsets are under test, not the kernel).
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import overlap as O
from kernels.gfx950.mxfp6_gemm import pack as P


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU")
_MODS: dict = {}
GEMM_FREE_VGPRS = 48  # 512 - the pre-packed GEMM's 464 allocated registers per lane (460 rounded to 8)


@pytest.fixture(scope="module")
def runtime():
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    yield rt
    for m in _MODS.values():
        m.unload()
    _MODS.clear()


def _hsaco(kfn):
    from rocke.helpers.compile import compile_kernel

    return compile_kernel(kfn, arch="gfx950", backend="python").hsaco


def _fn(rt, kfn):
    if kfn.name not in _MODS:
        _MODS[kfn.name] = rt.load_module(_hsaco(kfn))
    return _MODS[kfn.name].get_function(kfn.name)


def _u8(a):
    a = np.ascontiguousarray(a)
    return (ctypes.c_ubyte * a.nbytes).from_buffer(a)


def _put(rt, a):
    a = np.ascontiguousarray(a)
    p = rt.alloc(a.nbytes)
    rt.memcpy_h2d(p, _u8(a), a.nbytes)
    return p


def _get(rt, p, n):
    out = np.empty(n, np.uint8)
    rt.memcpy_d2h(_u8(out), p, n)
    return out


def _operands(rows, k, seed=11):
    gen = L.generate(rows, k, L.DIST_GAUSS, seed=seed, tag=1)
    rng = np.random.default_rng(seed + 1)  # arbitrary bytes incl. NaN (0xFF) scales
    rnd = L.Operand(rows, k, rng.integers(0, 256, size=gen.codes.shape, dtype=np.uint8),
                    rng.integers(0, 256, size=gen.scales.shape, dtype=np.uint8))
    return gen, rnd


# ----------------------------------------------------------------- co-resident pack kernel

_LITE = [(256, 512, {}), (512, 4096, {"order": "row"}), (768, 1024, {"n_wg": 7}), (1024, 4096, {"nt_store": True}),
         (512, 4096, {"n_wg": 1}), (2048, 8192, {}), (512, 4096, {"scales": False}), (256, 1024, {"block_rows": 128}),
         (512, 4096, {"stage": "lds16"}), (1024, 8192, {"stage": "lds16", "n_wg": 7}),
         (256, 2048, {"stage": "lds16", "nt_store": True}), (512, 4096, {"stage": "lds16", "block_rows": 128}),
         (512, 4096, {"stage": "lds16", "sc1_store": True, "nt_store": True}),
         (512, 4096, {"nt_load": True}), (512, 4096, {"stage": "lds16", "nt_load": True, "nt_store": True})]


@pytest.mark.parametrize("rows,k,kw", _LITE)
def test_pack_lite_matches_host_packers(runtime, rows, k, kw):
    rt = runtime
    kf = O.build_pack_lite(rows, k, **kw)
    fn = _fn(rt, kf)
    for op in _operands(rows, k):
        img, sc = P.pack_ref(op, kw.get("block_rows", 256))
        src_c, src_s = _put(rt, op.codes), _put(rt, op.scales)
        dc, ds = rt.alloc(img.nbytes), rt.alloc(sc.nbytes)
        for fill in (0xAB, 0x54):
            rt.memset(dc, fill, img.nbytes)
            rt.memset(ds, fill, sc.nbytes)
            rt.launch(fn, (O.pack_lite_grid(kw.get("n_wg", 256)), 1, 1),
                      (O.pack_lite_block(kw.get("stage", "none")), 1, 1), struct.pack("<QQQQ", src_c, src_s, dc, ds))
            rt.wait_stream(0)
            gi = _get(rt, dc, img.nbytes)
            gs = _get(rt, ds, sc.nbytes)
            assert np.array_equal(gi, img.view(np.uint8).reshape(-1)), f"{kf.name} codes fill {fill:#x}"
            if kw.get("scales") is False:
                assert (gs == fill).all(), "scales=False must leave the scale buffer untouched"
            else:
                assert np.array_equal(gs, sc.view(np.uint8).reshape(-1)), f"{kf.name} scales fill {fill:#x}"
        for p in (src_c, src_s, dc, ds):
            rt.free(p)


@pytest.mark.parametrize("stage", ["none", "lds16"])
@pytest.mark.parametrize("mutant", O.LITE_MUTANTS[1:])
def test_pack_lite_mutants_fail(runtime, mutant, stage):
    rt = runtime
    rows, k = 512, 4096
    op = _operands(rows, k)[0]
    img, sc = P.pack_ref(op)
    fn = _fn(rt, O.build_pack_lite(rows, k, mutant=mutant, stage=stage))
    src_c, src_s, dc, ds = _put(rt, op.codes), _put(rt, op.scales), rt.alloc(img.nbytes), rt.alloc(sc.nbytes)
    rt.launch(fn, (256, 1, 1), (O.pack_lite_block(stage), 1, 1), struct.pack("<QQQQ", src_c, src_s, dc, ds))
    rt.wait_stream(0)
    same = np.array_equal(_get(rt, dc, img.nbytes), img.view(np.uint8).reshape(-1)) and np.array_equal(
        _get(rt, ds, sc.nbytes), sc.view(np.uint8).reshape(-1))
    assert not same


@pytest.mark.parametrize("kw", [{}, {"nt_store": True}, {"order": "row"}, {"n_wg": 512}, {"stage": "lds16"},
                                {"stage": "lds16", "flag_tiles": 16, "release": "sc1", "wait_prev": True,
                                 "nt_store": True}])
def test_pack_lite_fits_next_to_a_resident_gemm_cta(kw):
    from benchmarks.gfx950.mxfp6_gemm.bench import code_object_notes

    notes = code_object_notes(_hsaco(O.build_pack_lite(4096, 65536, **kw)))
    lds_free = 160 * 1024 - 153600  # what a 150 KiB GEMM CTA leaves
    assert notes.get("group_segment_fixed_size", 0) <= (lds_free if kw.get("stage") == "lds16" else 0), notes
    assert notes.get("vgpr_count", 999) <= GEMM_FREE_VGPRS, notes
    assert notes.get("private_segment_fixed_size", 0) == 0, notes


# ----------------------------------------------------------------- multi-stream schedules

M = N = 1024
K = 4096


def _gemm_over(rows):
    return {"m": rows, "n": N, "k": K, "xcd_block": "none"}


@pytest.fixture(scope="module")
def setup(runtime):
    from benchmarks.gfx950.mxfp6_gemm.overlap_bench import Streams, _hip
    from kernels.gfx950.mxfp6_gemm.gemm import Mxfp6GemmSpec, build_mxfp6_gemm, grid_of

    rt = runtime
    hip = _hip()
    a_op = L.generate(M, K, L.DIST_GAUSS, seed=5, tag=1)
    b_op = L.generate(N, K, L.DIST_GAUSS, seed=5, tag=2)
    a_rot = L.Operand(M, K, np.roll(a_op.codes, -256, axis=0), np.roll(a_op.scales, -256, axis=0))
    gemms = {}
    for rows in (M, M // 2, M // 4):
        spec = replace(Mxfp6GemmSpec(), **_gemm_over(rows))
        kf = build_mxfp6_gemm(spec)
        gemms[rows] = (_fn(rt, kf), grid_of(spec))
    d = {"can": [_put(rt, a_op.codes), _put(rt, a_rot.codes)], "raw": [_put(rt, a_op.scales), _put(rt, a_rot.scales)],
         "pre": [_put(rt, L.pack_operand(x, "c0c1", 256)) for x in (a_op, a_rot)],
         "pre_sc": [_put(rt, L.pack_scales_tile(x.scales, "16x16x128")) for x in (a_op, a_rot)],
         "b": _put(rt, L.pack_operand(b_op, "c0c1", 256)), "b_sc": _put(rt, L.pack_scales_tile(b_op.scales, "16x16x128")),
         "img": [rt.alloc(M * K * 3 // 4) for _ in range(2)], "sc": [rt.alloc(M * K // 32) for _ in range(2)],
         "C": [rt.alloc(M * N * 2) for _ in range(4)], "kf": {}}
    for sl in range(2):  # K-flag arrival counters / ready flags per slot (uncached)
        for what in ("cnt", "flg"):
            h = ctypes.c_void_p()
            assert hip.hipExtMallocWithFlags(ctypes.byref(h), 4 * 64, 3) == 0
            d["kf"][(what, sl)] = int(h.value)
    # reference C per input: the single pre-packed GEMM
    fn, (grid, block) = gemms[M]
    want = []
    for v in (0, 1):
        rt.launch(fn, grid, block, struct.pack("<QQQQQ", d["pre"][v], d["b"], d["pre_sc"][v], d["b_sc"], d["C"][v]))
        rt.wait_stream(0)
        want.append(_get(rt, d["C"][v], M * N * 2))
    ref_c, ref_s = L.reference_f64(a_op, b_op)
    chk = L.check_against_reference(want[0].view(np.uint16).reshape(M, N), ref_c, ref_s)
    assert chk["pass"], chk
    assert np.array_equal(np.roll(want[0].view(np.uint16).reshape(M, N), -256, axis=0),
                          want[1].view(np.uint16).reshape(M, N))
    return {"rt": rt, "hip": hip, "streams": Streams(hip), "gemms": gemms, "d": d, "want": want}


def _variant(call):
    return (call // 2) % 2  # inputs A, A, A_rot, A_rot: a buffer slot sees both, so a missing WAR wait shows


def _delay_kernel(iters: int = 160):
    """One wave sleeping ~iters x 8K cycles: launched before every GEMM it makes the GEMMs lag the packs, so a pack
    that does not wait for its buffer's reader overwrites it deterministically."""
    from rocke.core.ir import IRBuilder

    b = IRBuilder(f"rocke_overlap_test_delay_{iters}")
    b.kernel.attrs["max_workgroup_size"] = 64
    c = b.const_i32
    with b.scf_for(c(0), c(iters), c(1), iv_name="dl"):
        b.inline_asm("s_sleep 127", "", [], None)
    return b.kernel


def _run(st, sched, pack_kw, ncalls=4, chunk_mutant=False, kflag=None, delay=False):
    """Run setup + ``ncalls`` logical calls of ``sched``; -> C bytes per call. ``kflag``: (K-flag GEMM, flag-mode
    pack kwargs) for :func:`overlap.schedule_kflag` schedules. ``delay``: a sleeping kernel before every GEMM."""
    from benchmarks.gfx950.mxfp6_gemm.overlap_bench import _ck

    rt, hip, d = st["rt"], st["hip"], st["d"]
    K = st.get("K", globals()["K"])
    for p in d["img"]:
        rt.memset(p, 0xAB, M * K * 3 // 4)
    for p in d["sc"]:
        rt.memset(p, 0xAB, M * K // 32)
    for p in d["C"]:
        rt.memset(p, 0xFF, M * N * 2)
    for p in d["kf"].values():
        rt.memset(p, 0, 4 * 64)
    if "dsig" in d:
        ctypes.c_uint32.from_address(d["dsig"]).value = O.kflag_parity(-1, 1)
        rt.memcpy_h2d(d["dsig_dev"], _u8(np.full(2, O.kflag_parity(-1, 1), np.int32)), 8)
        for p in (d["dcnt"], d["gtmo"], d["ptmo"]):
            rt.memset(p, 0, 64)
    rt.sync()
    dfn = _fn(rt, _delay_kernel()) if delay else None
    chunks = sched.meta.get("chunks", 1)
    rows = M // chunks
    per = sched.meta.get("calls_per_list", 1)
    slots = sched.meta.get("slots", 1)
    packs = {}
    evs = {}
    ops = [(op, 0) for op in sched.setup]
    for j in range((ncalls + per - 1) // per):
        ops += [(op, j * per) for op in sched.call]
    for op, call0 in ops:
        s = st["streams"].get("t" + op.stream)
        if op.kind == "record":
            evs.setdefault(op.what, rt.event()).record(stream=s)
            continue
        if op.kind == "wait":
            if op.what in evs:
                _ck(hip.hipStreamWaitEvent(s, evs[op.what].handle, 0), "wait")
            continue
        if op.kind in ("signal", "kwait"):  # the kernels signal / wait themselves
            continue
        if op.kind == "waitvalue":
            if call0 or op not in sched.setup:
                _ck(hip.hipStreamWaitValue32(s, d["dsig"], O.kflag_parity(call0 - 1, slots), 1, 0xFFFFFFFF), "wv")
            continue
        if dfn is not None and op.what == "gemm":
            rt.launch(dfn, (1, 1, 1), (64, 1, 1), b"", stream=s)
        multi = per > 1
        call = call0 + (op.arg[1] if multi else 0)
        var = _variant(call)
        if kflag is not None:
            slot = op.arg[0]
            par = O.kflag_parity(call, slots)
            if op.what == "pack":
                key = ("kf", tuple(sorted(kflag[1].items())))
                if key not in packs:
                    kf = O.build_pack_lite(M, K, **kflag[1])
                    packs[key] = (_fn(rt, kf), O.pack_lite_grid(kflag[1].get("n_wg", 256)))
                fn, grid = packs[key]
                rt.launch(fn, (grid, 1, 1), (O.pack_lite_block(kflag[1].get("stage", "none")), 1, 1),
                          O.pack_lite_flag_args(
                    d["can"][var], d["raw"][var], d["img"][slot], d["sc"][slot], d["kf"][("cnt", slot)],
                    d["kf"][("flg", slot)], par, d["dsig_dev"], d["ptmo"]), stream=s)
            else:
                fn, (grid, block) = kflag[0]
                sig = d["dsig"] if sched.meta.get("dep") == "value" else d["dsig_dev"]
                rt.launch(fn, grid, block, O.gemm_kflag_args(d["img"][slot], d["b"], d["sc"][slot], d["b_sc"],
                                                             d["C"][call % 4], d["kf"][("flg", slot)], par,
                                                             d["dcnt"], sig, d["gtmo"]), stream=s)
            continue
        if op.what == "pack":
            if multi:  # next-call: (slot, call offset)
                slot = op.arg[0]
                r0, prow, img, scb, sco = 0, M, d["img"][slot], d["sc"][slot], 0
            else:
                c = 0 if op.arg[0] == "all" else op.arg[0]
                prow = M if op.arg[0] == "all" else rows
                r0, img, scb, sco = c * prow, d["img"][0], d["sc"][0], c * prow * K // 32
            key = (prow, tuple(sorted(pack_kw.items())))
            if key not in packs:
                kf = O.build_pack_lite(prow, K, **pack_kw)
                packs[key] = (_fn(rt, kf), O.pack_lite_grid(pack_kw.get("n_wg", 256)))
            fn, grid = packs[key]
            rt.launch(fn, (grid, 1, 1), (O.pack_lite_block(pack_kw.get("stage", "none")), 1, 1),
                      struct.pack("<QQQQ", d["can"][var] + r0 * K * 3 // 4, d["raw"][var] + r0 * K // 32,
                                  img + r0 * K * 3 // 4, scb + sco), stream=s)
            continue
        if multi:
            fn, (grid, block) = st["gemms"][M]
            ka = struct.pack("<QQQQQ", d["img"][op.arg[0]], d["b"], d["sc"][op.arg[0]], d["b_sc"], d["C"][call % 4])
        elif op.arg[0] == "all":
            fn, (grid, block) = st["gemms"][M]
            ka = struct.pack("<QQQQQ", d["img"][0], d["b"], d["sc"][0], d["b_sc"], d["C"][call % 4])
        else:
            c = op.arg[0]
            fn, (grid, block) = st["gemms"][rows]
            src = (c + 1) % chunks if chunk_mutant else c
            ka = struct.pack("<QQQQQ", d["img"][0] + src * rows * K * 3 // 4, d["b"], d["sc"][0] + c * rows * K // 32,
                             d["b_sc"], d["C"][call % 4] + c * rows * N * 2)
        rt.launch(fn, grid, block, ka, stream=s)
    rt.sync()
    # C buffer = call % 4; calls 4j..4j+3 have the inputs of calls 0..3 (_variant has period 4)
    assert ncalls <= 4 or ncalls % 4 == 0
    return [_get(rt, d["C"][i], M * N * 2) for i in range(min(ncalls, 4))]


def _good(st, outs):
    return [np.array_equal(o, st["want"][_variant(i)]) for i, o in enumerate(outs)]


SCHEDS = [O.schedule_serial(), O.schedule_chunked(2), O.schedule_chunked(4), O.schedule_chunked(4, mode="pipelined"),
          O.schedule_next_call(), O.schedule_next_call(gemm_first=False)]


@pytest.mark.parametrize("sched", SCHEDS, ids=lambda s: s.label)
@pytest.mark.parametrize("pack_kw", [{}, {"n_wg": 16, "nt_store": True}])
def test_schedules_reproduce_the_prepacked_gemm(setup, sched, pack_kw):
    assert all(_good(setup, _run(setup, sched, pack_kw))), sched.label


def test_gemm_without_its_wait_fails(setup):
    """The GEMMs launched without waiting for a single-work-group (slow) pack read the poisoned image."""
    assert not all(_good(setup, _run(setup, O.schedule_chunked(4, mutant="nowait"), {"n_wg": 1})))


_WAR = [O.schedule_chunked(4), O.schedule_next_call()]


@pytest.mark.parametrize("i", range(len(_WAR)))
def test_war_waits_hold_with_lagging_gemms(setup, i):
    """Positive control of the WAR mutants: with delayed GEMMs the correct schedules still reproduce the GEMM."""
    assert all(_good(setup, _run(setup, _WAR[i], {}, delay=True)))


@pytest.mark.parametrize("sched", [O.schedule_chunked(4, mutant="nowar"), O.schedule_next_call(mutant="nowar")],
                         ids=lambda s: s.label)
def test_pack_without_its_war_wait_fails(setup, sched):
    """Packs that do not wait for the (delayed) GEMM still reading their buffer overwrite it with the next input."""
    assert not all(_good(setup, _run(setup, sched, {}, delay=True)))


def test_chunk_gemm_reading_the_wrong_chunk_fails(setup):
    assert not any(_good(setup, _run(setup, O.schedule_chunked(4), {}, chunk_mutant=True)))


# ----------------------------------------------------------------- fused K-flag GEMM + flag-mode pack

KF_TILES = 4


KF_K = 8192  # the v6 K-flag tests: a 12-tile loop body needs a few bodies of K
_RUST_MODS: dict = {}


def _v6_spec():
    from kernels.gfx950.mxfp6_gemm.gemm import PRESETS, Mxfp6GemmSpec

    return replace(Mxfp6GemmSpec(), **{**PRESETS["best_v6"], "m": M, "n": N, "k": KF_K, "xcd_block": "none"})


def _rust_hsaco(kfn):
    """Direct-compiler build with the v6 lever set (ROCKE_RUST_COMPILE must point at target_compile)."""
    import os
    import tempfile
    from pathlib import Path

    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust
    from kernels.gfx950.mxfp6_gemm.gemm import PRESET_LEVERS

    old = os.environ.get("ROCKE_RUST_LEVERS")
    os.environ["ROCKE_RUST_LEVERS"] = PRESET_LEVERS["best_v6"]
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / f"{kfn.name}.hsaco"
            _compile_rust(kfn, out)
            return out.read_bytes()
    finally:
        if old is None:
            os.environ.pop("ROCKE_RUST_LEVERS", None)
        else:
            os.environ["ROCKE_RUST_LEVERS"] = old


def _rust_fn(rt, kfn):
    if kfn.name not in _RUST_MODS:
        _RUST_MODS[kfn.name] = rt.load_module(_rust_hsaco(kfn))
    return _RUST_MODS[kfn.name].get_function(kfn.name)


@pytest.fixture(scope="module")
def kf_setup(runtime):
    """Operands at K = KF_K, the pre-packed v6 C of both inputs, and the K-flag v6 builds (plain, done signal and
    the mutants)."""
    import os

    from benchmarks.gfx950.mxfp6_gemm.overlap_bench import Streams, _hip
    from kernels.gfx950.mxfp6_gemm.gemm import grid_of

    if not os.environ.get("ROCKE_RUST_COMPILE"):
        pytest.skip("the K-flag v6 GEMM needs the rocKE direct compiler (ROCKE_RUST_COMPILE)")
    rt = runtime
    hip = _hip()
    k = KF_K
    a_op = L.generate(M, k, L.DIST_GAUSS, seed=7, tag=1)
    b_op = L.generate(N, k, L.DIST_GAUSS, seed=7, tag=2)
    a_rot = L.Operand(M, k, np.roll(a_op.codes, -256, axis=0), np.roll(a_op.scales, -256, axis=0))
    spec = _v6_spec()
    grid = grid_of(spec)
    from kernels.gfx950.mxfp6_gemm.gemm import build_mxfp6_gemm

    plain = _rust_fn(rt, build_mxfp6_gemm(spec))
    gemms = {mut: (_rust_fn(rt, O.build_gemm_kflag(spec, KF_TILES, mutant=mut)), grid) for mut in O.KFLAG_MUTANTS}
    gemms["sig"] = (_rust_fn(rt, O.build_gemm_kflag(spec, KF_TILES, done_signal=True)), grid)
    d = {"can": [_put(rt, a_op.codes), _put(rt, a_rot.codes)], "raw": [_put(rt, a_op.scales), _put(rt, a_rot.scales)],
         "b": _put(rt, L.pack_operand(b_op, "c0c1", 256)), "b_sc": _put(rt, L.pack_scales_tile(b_op.scales, "16x16x128")),
         "img": [rt.alloc(M * k * 3 // 4) for _ in range(2)], "sc": [rt.alloc(M * k // 32) for _ in range(2)],
         "C": [rt.alloc(M * N * 2) for _ in range(4)], "kf": {}}
    for sl in range(2):
        for what in ("cnt", "flg"):
            h = ctypes.c_void_p()
            assert hip.hipExtMallocWithFlags(ctypes.byref(h), 4 * 64, 3) == 0
            d["kf"][(what, sl)] = int(h.value)
    h = ctypes.c_void_p()
    assert hip.hipExtMallocWithFlags(ctypes.byref(h), 4 * 64, 3) == 0
    d["dcnt"] = int(h.value)
    h = ctypes.c_void_p()
    assert hip.hipExtMallocWithFlags(ctypes.byref(h), 8, 2) == 0  # hipMallocSignalMemory
    d["dsig"] = int(h.value)
    for what in ("dsig_dev", "gtmo", "ptmo"):  # done word the pack polls; hang-guard counters
        h = ctypes.c_void_p()
        assert hip.hipExtMallocWithFlags(ctypes.byref(h), 64, 3) == 0
        d[what] = int(h.value)
    want = []
    for x in (a_op, a_rot):
        pa, ps = _put(rt, L.pack_operand(x, "c0c1", 256)), _put(rt, L.pack_scales_tile(x.scales, "16x16x128"))
        rt.launch(plain, *grid, struct.pack("<QQQQQ", pa, d["b"], ps, d["b_sc"], d["C"][0]))
        rt.wait_stream(0)
        want.append(_get(rt, d["C"][0], M * N * 2))
    ref_c, ref_s = L.reference_f64(a_op, b_op)
    assert L.check_against_reference(want[0].view(np.uint16).reshape(M, N), ref_c, ref_s)["pass"]
    return {"rt": rt, "hip": hip, "streams": Streams(hip), "d": d, "want": want, "gemms": gemms, "K": k}


def _timeouts(st):
    out = []
    for what in ("gtmo", "ptmo"):
        out.append(int(_get(st["rt"], st["d"][what], 4).view(np.int32)[0]))
    return out


@pytest.mark.parametrize("mode,dep", [("strict", "event"), ("strict", "value"), ("strict", "spin"),
                                      ("next", "event")])
@pytest.mark.parametrize("pack_kw", [{}, {"release": "sc1"}, {"n_wg": 3, "release": "sc1"},
                                     {"release": "sc1", "nt_store": True},
                                     {"release": "sc1", "nt_store": True, "stage": "lds16"},
                                     {"release": "sc1", "stage": "lds16", "n_wg": 5}])
def test_kflag_overlap_reproduces_the_prepacked_gemm(kf_setup, mode, dep, pack_kw):
    st = kf_setup
    sched = O.schedule_kflag(mode=mode, dep=dep)
    g = st["gemms"]["none" if dep == "event" else "sig"]
    kw = {"flag_tiles": KF_TILES, **pack_kw, **({"wait_prev": True} if dep == "spin" else {})}
    outs = _run(st, sched, {}, ncalls=8, kflag=(g, kw))
    assert all(_good(st, outs)), [int((o != st["want"][_variant(i)]).sum()) for i, o in enumerate(outs)]
    assert _timeouts(st) == [0, 0]


def test_kflag_spin_without_its_wait_fails(kf_setup):
    """dep=spin: the WAR order lives in the pack kernel; a pack built without wait_prev overwrites the image the
    (delayed) previous GEMM still reads."""
    st = kf_setup
    outs = _run(st, O.schedule_kflag(dep="spin"), {}, ncalls=8, delay=True,
                kflag=(st["gemms"]["sig"], {"flag_tiles": KF_TILES, "release": "sc1"}))
    assert not all(_good(st, outs))


def test_kflag_hang_guard_counts(kf_setup):
    """A GEMM whose flags never arrive (no pack) gives up after the poll limit instead of hanging, and counts it."""
    st = kf_setup
    rt, d = st["rt"], st["d"]
    for p in list(d["kf"].values()) + [d["gtmo"]]:
        rt.memset(p, 0, 64)
    fn, (grid, block) = st["gemms"]["none"]
    rt.launch(fn, grid, block, O.gemm_kflag_args(d["img"][0], d["b"], d["sc"][0], d["b_sc"], d["C"][0],
                                                 d["kf"][("flg", 0)], 1, d["dcnt"], d["dsig_dev"], d["gtmo"]))
    rt.sync()
    assert _timeouts(st)[0] > 0
    rt.memset(d["gtmo"], 0, 64)


@pytest.mark.parametrize("gemm_mut,pack_kw,sched_mut", [
    ("noflag", {"n_wg": 1}, "none"),  # the GEMM never waits: reads the poisoned image of a slow pack
    ("parity", {"n_wg": 1}, "none"),  # waits for the previous call flag value: does not wait
    ("none", {"n_wg": 1, "mutant": "early"}, "none"),  # flags published before the stores
    ("none", {}, "nowar"),  # the pack overwrites the image the previous GEMM still reads
])
def test_kflag_mutants_fail(kf_setup, gemm_mut, pack_kw, sched_mut):
    st = kf_setup
    sched = O.schedule_kflag(mode="strict", mutant=sched_mut)
    outs = _run(st, sched, {}, kflag=(st["gemms"][gemm_mut], {"flag_tiles": KF_TILES, "release": "sc1", **pack_kw}),
                delay=sched_mut == "nowar")
    assert not all(_good(st, outs))


@pytest.mark.parametrize("sig,pack_kw", [(False, {}), (True, {}), (True, {"stage": "lds16", "nt_store": True}),
                                         (True, {"stage": "lds16", "nt_store": True, "wait_prev": True}),
                                         (True, {"stage": "lds16", "nt_store": True, "n_wg": 32})])
def test_kflag_gemm_fits_next_to_the_flag_pack(sig, pack_kw):
    """Deadlock freedom of the fused path: one pack wave fits next to the GEMM wave on every SIMD."""
    from benchmarks.gfx950.mxfp6_gemm.bench import code_object_notes
    from kernels.gfx950.mxfp6_gemm.gemm import PRESETS, Mxfp6GemmSpec

    import os

    if not os.environ.get("ROCKE_RUST_COMPILE"):
        pytest.skip("needs the rocKE direct compiler")
    spec = replace(Mxfp6GemmSpec(), **PRESETS["best_v6"])  # the full-shape build
    g = code_object_notes(_rust_hsaco(O.build_gemm_kflag(spec, 16, done_signal=sig)))
    p = code_object_notes(_hsaco(O.build_pack_lite(4096, 65536, flag_tiles=16, release="sc1", **pack_kw)))
    regs = -(-g["vgpr_count"] // 8) * 8 + -(-p["vgpr_count"] // 8) * 8
    assert regs <= 512 and g["group_segment_fixed_size"] + p.get("group_segment_fixed_size", 0) <= 160 * 1024
