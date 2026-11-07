# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the gfx950 MXFP6 GEMM: host layout, LDS maps, spec, lowering.

The on-GPU numerics (fp64 reference, exact-integer set, must-fail mutants) are
in ``test_mxfp6_gemm_numeric.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm.gemm import (
    MUTANTS,
    Mxfp6GemmSpec,
    _Builder,
    build_mxfp6_gemm,
    descriptor,
    grid_of,
    kernel_name,
    lds_bytes,
    plan,
    tile_of_wg,
    validate_spec,
)
from rocke.core.arch import mfma_scale_mx as mx
from rocke.core.ir import uses_python_only_ops

# ----------------------------------------------------------------- data format


def test_e2m3_grid_and_rne():
    assert L.E2M3_MAG[8] == 1.0 and L.E2M3_MAG[7] == 0.875 and L.E2M3_MAG[31] == 7.5
    assert np.all(np.diff(L.E2M3_MAG) > 0)
    # exact values round-trip, sign included (code 32 = -0)
    codes = np.array([c for c in range(64) if c != 32], dtype=np.uint8)
    assert np.array_equal(L.e2m3_encode(L.E2M3_VALUES[codes].astype(np.float32)), codes)
    # ties go to the even mantissa code; saturation at 7.5
    got = L.e2m3_encode(np.array([1.0625, 1.1875, 100.0, -0.0, -100.0], dtype=np.float32))
    assert got.tolist() == [8, 10, 31, 32, 63]


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 64, size=(5, 96), dtype=np.uint8)
    packed = L.pack_codes(codes)
    assert packed.shape == (5, 72)
    assert np.array_equal(L.unpack_codes(packed, 96), codes)
    # value k at bits [6k, 6k + 6) of the little-endian row stream
    row = int.from_bytes(packed[2].tobytes(), "little")
    assert [(row >> (6 * k)) & 63 for k in range(96)] == codes[2].tolist()


@pytest.mark.parametrize("dist", [L.DIST_GAUSS, L.DIST_UNIFORM_CODES])
def test_generator_deterministic_and_sane(dist):
    a = L.generate(32, 1024, dist, seed=42, tag=1)
    a2 = L.generate(32, 1024, dist, seed=42, tag=1)
    b = L.generate(32, 1024, dist, seed=42, tag=2)
    assert np.array_equal(a.codes, a2.codes) and np.array_equal(a.scales, a2.scales)
    assert not np.array_equal(a.codes, b.codes)
    codes = L.unpack_codes(a.codes, 1024)
    assert len(np.unique(codes)) > 20
    if dist == L.DIST_UNIFORM_CODES:
        assert a.scales.min() >= 124 and a.scales.max() <= 130
    else:
        # OCP rule: the block maximum lands in [4, 8) after scaling -> top codes used
        assert (codes & 31).max() >= 24
        v = L.decode_operand(a)
        assert 0.8 < v.std() < 1.2


def test_generator_matches_cxx_hashing():
    """Pin the splitmix64 stream (a regression guard for the C++ identity proof)."""
    assert int(L.splitmix64(0)) == 0xE220A8397B1DCDAF
    a = L.generate(2, 64, L.DIST_UNIFORM_CODES, seed=1, tag=1)
    base = int(L.splitmix64(np.uint64(1) ^ (np.uint64(1) << np.uint64(56))))
    assert a.scales[0, 0] == 124 + int(L.splitmix64(base)) % 7
    assert L.unpack_codes(a.codes, 64)[0, 0] == int(L.splitmix64(base + 1)) & 63


def test_save_load_roundtrip(tmp_path):
    a = L.generate(8, 256, L.DIST_GAUSS, seed=7, tag=1)
    p = tmp_path / "a.bin"
    L.save_operand(a, p)
    raw = p.read_bytes()
    assert int.from_bytes(raw[:8], "little") == 8 and int.from_bytes(raw[8:16], "little") == 256
    assert len(raw) == 16 + 8 * 256 * 6 // 8 + 8 * 256 // 32
    b = L.load_operand(p)
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.scales, b.scales)


def test_int_fixture_is_exact():
    a = L.int_operand(16, 512, seed=1, tag=1)
    b = L.int_operand(16, 512, seed=1, tag=2)
    va, vb = L.decode_operand(a), L.decode_operand(b)
    assert np.array_equal(va, np.round(va)) and np.abs(va).max() <= 4
    c, s = L.reference_f64(a, b)
    assert np.array_equal(c, va @ vb.T)
    bits = L.f32_to_bf16_bits(c.astype(np.float32))
    assert L.check_against_reference(bits, c, s, exact=True)["pass"]


def test_reference_check_tolerance_and_mutation():
    a = L.generate(16, 1024, L.DIST_GAUSS, seed=1, tag=1)
    b = L.generate(16, 1024, L.DIST_GAUSS, seed=1, tag=2)
    c, s = L.reference_f64(a, b)
    good = L.f32_to_bf16_bits(c.astype(np.float32))
    assert L.check_against_reference(good, c, s)["pass"]
    bad = L.f32_to_bf16_bits((c + np.ldexp(s, -6)).astype(np.float32))
    assert not L.check_against_reference(bad, c, s)["pass"]


# ----------------------------------------------------------------- scales


@pytest.mark.parametrize("atom", ["16x16x128", "32x32x64"])
def test_pack_scales_matches_mfma_lane_maps(atom):
    geo = L.atom_geometry(atom)
    rows, kb = 8 * geo.rows, 32
    rng = np.random.default_rng(5)
    sc = rng.integers(0, 255, size=(rows, kb), dtype=np.uint8)
    packed = L.pack_scales(sc, atom).view(np.uint8)
    ar = mx.HostArith
    nq = rows // (4 * geo.rows)
    ksteps_per_u = 4
    for u in range(kb // (4 * geo.groups)):
        for q in range(nq):
            for lane in range(64):
                for i in range(4):
                    for j in range(ksteps_per_u):
                        r_in_tile, kblk = mx.scale_a_coord(ar, atom, lane)
                        row = 4 * geo.rows * q + geo.rows * i + r_in_tile
                        t = 4 * u + j  # K step
                        k_block = t * geo.groups + kblk
                        byte = packed[(((u * nq + q) * 64 + lane) * 4 + i) * 4 + j]
                        assert byte == sc[row, k_block]


# ----------------------------------------------------------------- LDS maps


def _stage_from_global(rows: int, swizzle: str, glob: np.ndarray) -> np.ndarray:
    """Model the DTL fill: LDS unit c of row r <- global unit dtl_source_unit(r, c)."""
    lds = np.zeros((rows, L.ROW_BYTES), dtype=np.uint8)
    for r in range(rows):
        for c in range(L.UNITS):
            g = L.dtl_source_unit(r, c, swizzle)
            lds[r, 16 * c : 16 * c + 16] = glob[r, 16 * g : 16 * g + 16]
    return lds


@pytest.mark.parametrize("swizzle", L.SWIZZLES)
def test_lds_fragment_reads_return_global_chunks(swizzle):
    rng = np.random.default_rng(1)
    glob = rng.integers(0, 256, size=(64, L.ROW_BYTES), dtype=np.uint8)
    lds = _stage_from_global(64, swizzle, glob)
    for r in range(64):
        for j in range(4):
            off = L.lds_chunk_slot(r, j, swizzle)
            assert off % 8 == 0 and off + 24 <= L.ROW_BYTES
            assert np.array_equal(lds[r, off : off + 24], glob[r, 24 * j : 24 * j + 24])


def test_kernel_dtl_lane_math_matches_layout_map():
    """The kernel's per-lane DTL voffset formula (``_setup``) == ``dtl_source_unit``."""
    row_bytes = 49152
    for swz in L.SWIZZLES:
        for q in range(3):
            for lane in range(64):
                u = lane + 64 * q
                r, c = u // 6, u % 6
                rot = (r >> 3) & 1 if swz == "rot48" else 0
                voff = r * row_bytes + 16 * ((c + 3 * rot) % 6)
                assert voff == r * row_bytes + 16 * L.dtl_source_unit(r, c, swz)


def test_fragment_reads_conflict_free_16x16():
    from rocke.analysis.lds import LdsAccess, predict_lds_conflicts

    for part in range(3):
        acc = []
        for lane in range(64):
            r, g = lane % 16, lane // 16
            acc.append(LdsAccess(lane, lane, r * 96 + L.lds_chunk_slot(r, g, "rot48") + 8 * part, 8))
        res = predict_lds_conflicts(target="gfx950", opcode="ds_read_b64", wave_size=64, accesses=tuple(acc))
        assert res.summary.conflict_group_count == 0


# ----------------------------------------------------------------- spec / grid


def test_spec_defaults_and_validation():
    s = Mxfp6GemmSpec()
    assert validate_spec(s) == (True, "")
    assert lds_bytes(s) <= 160 * 1024
    assert validate_spec(replace(s, stages=4))[0] is False  # 4 x 48 KiB LDS
    assert validate_spec(replace(s, k=65536 + 128))[0] is False
    assert validate_spec(replace(s, swizzle="xor"))[0] is False
    assert validate_spec(replace(s, atom="32x32x64"))[0] is True
    assert validate_spec(replace(s, mutant="nope"))[0] is False


def test_kernel_name_and_descriptor():
    s = Mxfp6GemmSpec()
    name = kernel_name(s)
    assert name.startswith("rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128")
    assert kernel_name(replace(s, serpentine=False)) != name
    assert "MUT_scale_sel" in kernel_name(replace(s, mutant="scale_sel"))
    d = json.loads(json.dumps(descriptor(s)))
    assert d["spec"] == asdict(s) and d["grid"] == [256, 1, 1] and d["block"] == [256, 1, 1]


@pytest.mark.parametrize("xcd", ["none", "4x8", "8x4", "2x16", "16x2"])
def test_tile_order_is_a_permutation_and_xcd_local(xcd):
    s = replace(Mxfp6GemmSpec(), xcd_block=xcd)
    (nwg, _, _), _ = grid_of(s)
    tiles = [tile_of_wg(s, w) for w in range(nwg)]
    assert sorted(tiles) == [(i, j) for i in range(16) for j in range(16)]
    if xcd != "none":
        bm, bn = (int(x) for x in xcd.split("x"))
        for x in range(8):
            mine = [tiles[w] for w in range(nwg) if w % 8 == x]
            assert len({t[0] for t in mine}) == bm and len({t[1] for t in mine}) == bn


def test_plan_slots_in_range():
    for s in (Mxfp6GemmSpec(), replace(Mxfp6GemmSpec(), atom="32x32x64", read_every=1)):
        p = plan(s)
        assert all(0 <= x < p.n_mfma for x in p.read_slots + p.dtl_slots + [p.scale_slot])
        assert len(p.dtl_slots) == 12 + (2 if s.scale_src == "lds" else 0)


@pytest.mark.parametrize("stages", [2, 3])
def test_vmcnt_model(stages):
    b = _Builder(replace(Mxfp6GemmSpec(), stages=stages, frag_mode="prefetch", scale_src="vgpr"))
    counts = b._acquire_counts()
    # prefetch: tile 0 is acquired in the prologue after the fill of tiles 0..S-1
    assert counts[0] == (stages - 1) * 12
    jb = _Builder(replace(Mxfp6GemmSpec(), stages=stages, frag_mode="jit"))
    j = jb._acquire_counts()
    # jit: tile 0 is acquired at its own top, after the fill of tiles 0..S-2 (scales ride the fill DTLs)
    assert j[0] == (stages - 2) * len(jb.plan.dtl_slots)
    assert all(0 <= v < 64 for v in counts.values())  # vmcnt field is 6 bits


# ----------------------------------------------------------------- lowering


def _lower(spec):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    k = build_mxfp6_gemm(spec, arch="gfx950")
    return k, _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"atom": "32x32x64"},
        {"stages": 2, "frag_mode": "prefetch", "operand_layout": "canonical", "scale_layout": "kstep", "scale_src": "vgpr"},
        {"peel_first": True},
        {"serpentine": False, "m_outer": False},
        {"tail_mode": "clamp"},
    ],
    ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()) or "default",
)
def test_lowering(over):
    spec = replace(Mxfp6GemmSpec(), **over)
    k, ll = _lower(spec)
    assert uses_python_only_ops(k) == ["tile.mfma_scale_mx"]
    shape = spec.atom
    calls = re.findall(rf"call <(\d+) x float> @llvm\.amdgcn\.mfma\.scale\.f32\.{shape}\.f8f6f4\.v6i32\.v6i32\((.*)\)", ll)
    assert calls, "no fp6 x fp6 scaled MFMA"
    # every MFMA is fp6 x fp6 (cbsz = blgp = 2) and uses all four op_sel bytes
    sels = set()
    for _, args in calls:
        m = re.search(r"i32 2, i32 2, i32 (\d), i32 %\S+, i32 (\d), i32 %\S+$", args)
        assert m, args
        sels.add(int(m.group(1)))
        if spec.scale_layout == "kstep":  # op_sel = K step: the same byte for A and B
            assert m.group(1) == m.group(2)
    assert sels == {0, 1, 2, 3}
    assert "raw.ptr.buffer.load.lds" in ll and "amdgpu-agpr-alloc" in ll
    assert k.attrs.get("mfma_agpr_form") is True and k.attrs.get("lds_alias_scopes") is True


def test_mfma_count_per_kernel():
    spec = Mxfp6GemmSpec()
    k, ll = _lower(spec)
    n_tiles = spec.k // 128
    # static code: loop body + tail; MFMAs per tile = 64 (16x16x128, 8 x 8 per wave)
    n = len(re.findall(r"call <4 x float> @llvm\.amdgcn\.mfma\.scale\.f32\.16x16x128\.f8f6f4\.v6i32\.v6i32\(", ll))
    assert n % 64 == 0 and 64 * spec.stages <= n < 64 * n_tiles


def test_mutants_change_the_kernel():
    base = _lower(Mxfp6GemmSpec())[1]
    for mu in MUTANTS[1:]:
        assert _lower(replace(Mxfp6GemmSpec(), mutant=mu))[1] != base


def test_builder_rejects_other_arch():
    with pytest.raises(ValueError, match="gfx950"):
        build_mxfp6_gemm(Mxfp6GemmSpec(), arch="gfx942")


# ----------------------------------------------------------------- pre-tiled operands / per-tile scales


@pytest.mark.parametrize("atom", ["16x16x128", "32x32x64"])
def test_c0c1_fragment_reads_return_canonical_chunks(atom):
    """Model the kernel's c0c1 reads (``_setup_c0c1`` / ``_frag_load``) on the packed image."""
    R, k = 64, 256
    op = L.generate(R, k, L.DIST_UNIFORM_CODES, seed=3, tag=1)
    img = L.pack_operand_c0c1(op, R).reshape(1, k // 128, R * 96)
    geo = L.atom_geometry(atom)
    T = geo.rows
    for s in range(k // 128):
        blk = img[0, s]
        for f in range(R // T):
            for t in range(128 // geo.k):
                for lane in range(64):
                    if T == 16:
                        o = 64 * f + lane
                    else:
                        o = 64 * (2 * f + (lane % 32) // 16) + 16 * (lane // 32) + lane % 16 + 32 * t
                    frag = np.concatenate([blk[16 * o : 16 * o + 16], blk[64 * R + 8 * o : 64 * R + 8 * o + 8]])
                    row = f * T + lane % T
                    kk = 128 * s + mx.a_coord(mx.HostArith, atom, "fp6", lane, 0)[1] + t * geo.k
                    want = op.codes[row, kk * 3 // 4 : kk * 3 // 4 + 24]
                    assert np.array_equal(frag, want), (s, f, t, lane)


def test_c0c1p_paired_reads_return_canonical_chunks():
    """``c0c1p``: fragments 2j / 2j+1 = three b128 plane reads (48 consecutive bytes per lane)."""
    R, k = 64, 256
    op = L.generate(R, k, L.DIST_UNIFORM_CODES, seed=5, tag=2)
    img = L.pack_operand_c0c1(op, R, pair=True).reshape(1, k // 128, R * 96)
    assert np.array_equal(np.sort(img.reshape(-1)), np.sort(L.pack_operand_c0c1(op, R).reshape(-1)))
    for s in range(k // 128):
        blk = img[0, s]
        for f in range(R // 16):
            for lane in range(64):
                j = f // 2
                regs = np.concatenate([blk[3072 * j + 1024 * p + 16 * lane:][:16] for p in range(3)])
                frag = regs[24 * (f % 2) : 24 * (f % 2) + 24]
                kk = 128 * s + mx.a_coord(mx.HostArith, "16x16x128", "fp6", lane, 0)[1]
                want = op.codes[f * 16 + lane % 16, kk * 3 // 4 : kk * 3 // 4 + 24]
                assert np.array_equal(frag, want), (s, f, lane)
    assert L.pack_operand(op, "c0c1p", R).tobytes() == img.tobytes()
    assert not validate_spec(replace(Mxfp6GemmSpec(), operand_layout="c0c1p", atom="32x32x64"))[0]
    k_, ll = _lower(replace(Mxfp6GemmSpec(), operand_layout="c0c1p"))
    assert ll.count("load <2 x i32>") < _lower(Mxfp6GemmSpec())[1].count("load <2 x i32>")


def test_c0c1p_reads_conflict_free():
    from rocke.analysis.lds import LdsAccess, predict_lds_conflicts

    for p in range(3):  # each plane read: 64 lanes x 16 B contiguous
        acc = tuple(LdsAccess(l, l, 3072 + 1024 * p + 16 * l, 16) for l in range(64))
        res = predict_lds_conflicts(target="gfx950", opcode="ds_read_b128", wave_size=64, accesses=acc)
        assert res.summary.conflict_group_count == 0


def test_c0c1_reads_conflict_free():
    from rocke.analysis.lds import LdsAccess, predict_lds_conflicts

    for op_, width, addr in (("ds_read_b128", 16, lambda l: 16 * l), ("ds_read_b64", 8, lambda l: 16384 + 8 * l)):
        acc = tuple(LdsAccess(l, l, addr(l), width) for l in range(64))
        res = predict_lds_conflicts(target="gfx950", opcode=op_, wave_size=64, accesses=acc)
        assert res.summary.conflict_group_count == 0


@pytest.mark.parametrize("atom", ["16x16x128", "32x32x64"])
def test_pack_scales_tile_matches_mfma_lane_maps(atom):
    geo = L.atom_geometry(atom)
    rows, kb = 256, 16
    rng = np.random.default_rng(9)
    sc = rng.integers(0, 255, size=(rows, kb), dtype=np.uint8)
    packed = L.pack_scales_tile(sc, atom).view(np.uint8)
    ar = mx.HostArith
    nq2 = rows // 128
    for s in range(kb // 4):
        for q2 in range(nq2):
            for lane in range(64):
                r_in, kblk = mx.scale_a_coord(ar, atom, lane)
                for d in range(2):
                    for i in range(4):  # op_sel = MFMA tile within its group of four
                        byte = packed[((((s * nq2 + q2) * 64 + lane) * 2 + d) * 4) + i]
                        if geo.rows == 16:  # d = 64-row group, tile f = 4 d + i
                            row, k_block = 128 * q2 + 16 * (4 * d + i) + r_in, 4 * s + kblk
                        else:  # d = 64-K step, tile i
                            row, k_block = 128 * q2 + 32 * i + r_in, 4 * s + 2 * d + kblk
                        assert byte == sc[row, k_block]


@pytest.mark.parametrize(
    "over",
    [
        {"frag_mode": "jit", "stages": 3, "read_lead": 8, "scale_layout": "tile"},
        {"frag_mode": "jit", "stages": 3, "read_lead": 64, "scale_layout": "tile", "operand_layout": "c0c1"},
        {"stages": 3, "scale_layout": "tile", "operand_layout": "c0c1", "atom": "32x32x64"},
        {"probe": "mfma_only", "scale_src": "vgpr"},
        {"probe": "fill_only", "frag_mode": "jit", "stages": 3},
        {"scale_src": "lds"},
    ],
    ids=["jit-tile", "aiter-like", "c0c1-32x32", "mfma_only", "fill_only", "scale-lds"],
)
def test_lowering_new_knobs(over):
    spec = replace(Mxfp6GemmSpec(), **over)
    assert validate_spec(spec) == (True, "")
    k, ll = _lower(spec)
    n = len(re.findall(r"call <\d+ x float> @llvm\.amdgcn\.mfma\.scale", ll))
    assert (n == 0) == (spec.probe == "fill_only")
    if spec.scale_layout == "tile" and spec.probe == "none":
        sels = set(re.findall(r"i32 2, i32 2, i32 (\d), i32 %\S+, i32 (\d), i32 %\S+\)", ll))
        assert {a for a, _ in sels} == {"0", "1", "2", "3"} and any(a != b for a, b in sels)


def test_knobs_table_covers_spec():
    from kernels.gfx950.mxfp6_gemm.gemm import KNOBS

    names = {f for f in asdict(Mxfp6GemmSpec())}
    assert set(KNOBS) <= names
    d = asdict(Mxfp6GemmSpec())
    for k_, legal in KNOBS.items():
        assert d[k_] == legal[0], k_  # first legal value = default (parity)


def test_vm_issue_order_is_fenced():
    """Regression (search bug 2/4): every VM issue is followed by a VMEM-only sched_barrier, so
    the machine scheduler cannot reorder DTLs / scale loads behind each other and break the
    static vmcnt model (the configuration below raced before the fence)."""
    from kernels.gfx950.mxfp6_gemm.gemm import VM_FENCE_MASK

    spec = replace(
        Mxfp6GemmSpec(), frag_mode="prefetch", stages=2, operand_layout="canonical", scale_layout="kstep",
        scale_src="vgpr", atom="32x32x64", dtl_spread=5, peel_first=True, read_delay=8, scale_slot=56, setprio=3,
        xcd_block="16x2",
    )
    _, ll = _lower(spec)
    lines = ll.splitlines()
    vm = [i for i, x in enumerate(lines) if "raw.ptr.buffer.load" in x and "call" in x and "declare" not in x]
    assert vm
    for i in vm:
        nxt = next(x for x in lines[i + 1 :] if "call" in x)
        assert f"sched.barrier(i32 {VM_FENCE_MASK})" in nxt, (lines[i], nxt)


def test_dtl_imm_shares_m0_and_soffset():
    """``dtl_imm``: blocks of four 1 KiB stage DTLs carry offsets 0 / 1024 / 2048 / 3072 in the instruction."""
    spec = replace(Mxfp6GemmSpec(), dtl_imm=True)
    k, ll = _lower(spec)
    assert "tile.async_buffer_load_lds_imm" in uses_python_only_ops(k)
    offs = re.findall(r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\(.*, i32 16, i32 %\S+, i32 %\S+, i32 (\d+), i32 \d+\)", ll)
    assert set(offs) == {"0", "1024", "2048", "3072"}


def test_carry_moves_the_acquire():
    """``carry``: one barrier per tile, issued at ``carry_slot`` (not at the tile top), and the vmcnt
    model acquires tile j inside tile j - 1; own-tile reads after the moved barrier are rejected."""
    spec = replace(Mxfp6GemmSpec(), carry=9)
    ok, _ = validate_spec(spec)
    assert ok
    bad = replace(spec, carry_slot=40)  # the A7 row read (first use 56, lead 4) would follow the barrier
    assert not validate_spec(bad)[0]
    assert not validate_spec(replace(spec, scale_src="vgpr", scale_slot=2))[0]
    k, ll = _lower(spec)
    assert ll.count("s.barrier()") >= 1
    from kernels.gfx950.mxfp6_gemm.gemm import _Builder

    bd = _Builder(spec)
    bd.build()
    assert bd.acq_off == 1


@pytest.mark.parametrize("over", [{}, {"scale_layout": "kstep", "scale_src": "vgpr", "stages": 2,
                                       "frag_mode": "prefetch", "operand_layout": "canonical"},
                                  {"operand_layout": "c0c1p"}])
def test_descriptor_names_the_packers_the_bench_uses(over):
    """The JSON contract names exactly the host packers bench.py / the GPU test call for this spec."""
    import inspect

    from benchmarks.gfx950.mxfp6_gemm import bench

    spec = replace(Mxfp6GemmSpec(), **over)
    assert validate_spec(spec)[0]
    c = descriptor(spec)["contract"]
    src = inspect.getsource(bench)
    assert "L.pack_scales_for(a.scales, spec.atom, spec.scale_layout)" in src
    assert 'L.pack_operand(a, la, spec.tile_m)' in src and 'la, lb = operand_layout_of(spec, "A"), operand_layout_of(spec, "B")' in src
    want = {"tile": "layout.pack_scales_tile(", "kstep": "layout.pack_scales("}[spec.scale_layout]
    assert want in c["scales"] and f"'{spec.scale_layout}'" in c["scales"]
    other = {"tile": "layout.pack_scales(", "kstep": "layout.pack_scales_tile("}[spec.scale_layout]
    assert other not in c["scales"]
    assert f"layout.pack_operand, '{spec.operand_layout}'" in c["a"]
    # and the named packer is the one pack_scales_for dispatches to
    sc = np.random.default_rng(1).integers(0, 255, size=(256, 64), dtype=np.uint8)
    fn = L.pack_scales_tile if spec.scale_layout == "tile" else L.pack_scales
    assert np.array_equal(L.pack_scales_for(sc, spec.atom, spec.scale_layout), fn(sc, spec.atom))


#: b_direct combos the search found faulting (tail clamp: direct-B loads past K) or racing (a direct-B load and a
#: DTL in the same MFMA slot were issued in a different order than the static vmcnt model assumed)
_BDIR_REGRESSIONS = [
    {"b_direct": True},
    {"b_direct": True, "acc_pin": True, "dtl_aux": 3, "dtl_delay": 4, "dtl_spread": 2, "lds_alias_scopes": False,
     "m_outer": False, "read_lead": 64, "scale_src": "vgpr", "serpentine": False, "setprio": 3, "stages": 2,
     "tile_lgkm0": True},
    {"b_direct": True, "dtl_spread": 2, "m_outer": False, "peel_first": True, "release_slot": 24, "serpentine": False,
     "setprio": 3, "tail_mode": "clamp", "tile_lgkm0": True, "xcd_block": "2x16"},
    {"b_direct": True, "dtl_delay": 4, "b_slot": 0, "b_every": 2, "b_depth": 2, "scale_src": "vgpr", "scale_slot": 2},
]


@pytest.mark.parametrize("over", _BDIR_REGRESSIONS + [{}, {"scale_src": "vgpr", "scale_slot": 2},
                                                      {"frag_mode": "prefetch", "acq_slot": 8, "read_every": 3},
                                                      {"scale_dtl": "x4w"},
                                                      {"scale_dtl": "x4w", "frag_mode": "prefetch", "acq_slot": 4,
                                                       "read_every": 3, "unroll": 3, "rsrc_once": True}])
def test_vm_issue_order_matches_the_vmcnt_model(over):
    """Regression (search: b_direct faults / races): every tile's emitted VM issue kinds, in order, equal the
    static model's (``_vm_events``) for that tile; the prologue too."""
    from kernels.gfx950.mxfp6_gemm.gemm import _Builder

    spec = replace(Mxfp6GemmSpec(), **over)
    assert validate_spec(spec)[0], validate_spec(spec)
    bd = _Builder(spec)
    bd.build()
    ev = bd._vm_events()
    model = {}
    for tile, _slot, kind, _tgt in ev:
        model.setdefault(tile, []).append(kind)
    got = {}
    for tag, kind in bd.vm_trace:
        if kind == "sdtl":  # x4w scale DTLs: issued by one wave each, first in the refill; the model is the
            continue  # other waves' (no scale op), the strictest count
        got.setdefault(tag, []).append(kind)
    assert got.get(None, []) == model.get(-1, [])  # prologue
    T = bd.n_tiles
    for tag, kinds in got.items():
        if tag is None:
            continue
        if isinstance(tag, int):
            assert kinds == model.get(tag, []), (tag, kinds, model.get(tag))
        else:  # loop body tile at position p: an interior tile with the same position
            p = tag[1]
            interior = [i for i in range(T) if i % max(1, len([t for t in got if not isinstance(t, (int, type(None)))])) == p
                        and 0 < i < T - 8]
            assert any(kinds == model.get(i, []) for i in interior), (tag, kinds)
    if spec.b_direct and spec.tail_mode == "clamp":
        # direct-B targets past K are clamped in the loop (one more smin per loop tile than the DTL clamps alone)
        n_b = _lower(spec)[1].count("smin")
        n_0 = _lower(replace(spec, b_direct=False))[1].count("smin")
        assert n_b > n_0, (n_b, n_0)


# ----------------------------------------------------------------- E3M2 element format


def test_e3m2_grid():
    """1/3/2 bits, bias 3: subnormal m/16, normal 2^(e-3)(1+m/4), max 28; values strictly increasing."""
    want = [m / 16 for m in range(4)] + [2.0 ** (e - 3) * (1 + m / 4) for e in range(1, 8) for m in range(4)]
    assert np.array_equal(L.E3M2_VALUES[:32], np.array(want))
    assert np.array_equal(L.E3M2_VALUES[32:], -np.array(want))
    assert L.E3M2_VALUES[31] == 28.0 and np.all(np.diff(L.E3M2_VALUES[:32]) > 0)
    assert L.FMT_EMAX == {"e2m3": 2, "e3m2": 4}


def test_e3m2_rne_ties_and_saturation():
    enc = L.e3m2_encode
    v = np.array([1.125, 1.375, 1.625, 1.875, 1 / 32, 3 / 32, 26.0, 22.0, 27.99, 28.0, 30.0, np.inf, -40.0, -0.0, 0.0,
                  1.2, 1.12], dtype=np.float32)
    got = enc(v).tolist()
    # ties to the even code: 1.125 -> 12 (1.0), 1.375 -> 14 (1.5), 1.625 -> 14, 1.875 -> 16 (2.0);
    # 1/32 -> 0, 3/32 -> 2 (0.125); 26 = (24 + 28) / 2 and 22 = (20 + 24) / 2 both -> even code 30 (24)
    assert got[:4] == [12, 14, 14, 16]
    assert got[4:6] == [0, 2]
    assert got[6] == 30 and got[7] == 30
    assert got[8:12] == [31, 31, 31, 31]  # saturation at 28
    assert got[12] == 63 and got[13] == 32 and got[14] == 0
    assert got[15] == 13 and got[16] == 12
    # every code round-trips (except -0 -> 32 which is itself)
    codes = np.arange(64, dtype=np.uint8)
    assert np.array_equal(enc(L.E3M2_VALUES[codes].astype(np.float32)), codes)


def test_e2m3_unchanged_by_the_format_switch():
    rng = np.random.default_rng(3)
    v = (rng.standard_normal(4096) * 4).astype(np.float32)
    assert np.array_equal(L.e2m3_encode(v), L.fmt_encode(v, "e2m3"))
    a = L.generate(16, 512, L.DIST_GAUSS, seed=1, tag=1)
    b = L.generate(16, 512, L.DIST_GAUSS, seed=1, tag=1, fmt="e2m3")
    assert a.codes.tobytes() == b.codes.tobytes() and a.scales.tobytes() == b.scales.tobytes()
    assert a.fmt == "e2m3"


def test_e3m2_generator_rules():
    g = L.generate(32, 1024, L.DIST_GAUSS, seed=1, tag=2, fmt="e3m2")
    vals = L.decode_operand(g)
    # OCP rule: shared exponent floor(log2 amax) - 4, so every block's largest |code value| sits in [16, 28]
    codes = L.unpack_codes(g.codes, g.k).reshape(32, -1, 32)
    blkmax = np.abs(L.E3M2_VALUES[codes]).max(axis=-1)
    assert np.all((blkmax >= 16) & (blkmax <= 28))
    assert np.isfinite(vals).all()
    u2 = L.generate(32, 1024, L.DIST_UNIFORM_CODES, seed=1, tag=2)
    u3 = L.generate(32, 1024, L.DIST_UNIFORM_CODES, seed=1, tag=2, fmt="e3m2")
    assert u2.codes.tobytes() == u3.codes.tobytes() and u2.scales.tobytes() == u3.scales.tobytes()
    # integer fixture: -4..4 exact in E3M2, scale 2^0
    i3 = L.int_operand(8, 256, fmt="e3m2")
    iv = L.decode_operand(i3)
    assert np.array_equal(iv, np.round(iv)) and np.abs(iv).max() <= 4
    i2 = L.int_operand(8, 256)
    assert np.array_equal(iv, L.decode_operand(i2))  # same integers, different codes
    assert i3.codes.tobytes() != i2.codes.tobytes()


def test_e3m2_kernel_name_descriptor_and_mfma_format():
    s3 = replace(Mxfp6GemmSpec(), elem_fmt="e3m2")
    assert validate_spec(s3)[0] and not validate_spec(replace(s3, elem_fmt="e4m3"))[0]
    assert kernel_name(s3).startswith("rocke_gemm_mx_e3m2_e3m2_e8m0b32_")
    assert kernel_name(s3) == kernel_name(Mxfp6GemmSpec()).replace("e2m3_e2m3", "e3m2_e3m2")
    d = descriptor(s3)
    assert d["elem_fmt"] == "e3m2" and d["contract"]["a"].startswith("e3m2 ")
    for fmt, code in (("e2m3", 2), ("e3m2", 3)):
        for mut, c in (("none", code), ("wrong_fmt", 5 - code)):
            _, ll = _lower(replace(Mxfp6GemmSpec(), elem_fmt=fmt, mutant=mut))
            calls = [x for x in ll.splitlines() if "mfma.scale" in x and "call" in x]
            assert calls and all(f"<4 x float> %" in x and f", i32 {c}, i32 {c}, i32 " in x for x in calls), (fmt, mut)


#: sha256 of the shared C++ generator's ``save`` files for E3M2 (``mxfp6::generate(64, 4096, dist, seed 1, tag,
#: FMT_E3M2)``, hipcc build, identity job 68308599 -- the full 4096 x 65536 files matched byte for byte too).
_CPP_E3M2_SHA256 = {
    (0, 1): "22d06151229dc0aa48b5b354b5b856cc008d607345d3b01a77805dc21e8fc0cd",
    (0, 2): "cd80985bf22cd5f8ab342969fed8633e2156b69b967189fdbf67a8e07b9f4a18",
    (1, 1): "fac0a1d4bca626adefb1b6e18d3a14353bbd9b6c136b09462e442edf62488cb8",
    (1, 2): "c9bcebeacbeb51363054c6e847814df234d08259afa97868cd16d1d8ec0aa0a9",
}


@pytest.mark.parametrize("dist_tag", sorted(_CPP_E3M2_SHA256))
def test_e3m2_generator_matches_the_cpp_generator(dist_tag, tmp_path):
    import hashlib

    dist, tag = dist_tag
    p = tmp_path / "op.bin"
    L.save_operand(L.generate(64, 4096, dist, seed=1, tag=tag, fmt="e3m2"), p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == _CPP_E3M2_SHA256[dist_tag]


def test_rsrc_once_builds_one_descriptor_per_operand():
    """``rsrc_once``: the operand DTLs share one buffer descriptor per operand (tile offset in soffset)."""
    n0 = _lower(Mxfp6GemmSpec())[1].count("buffer.rsrc")
    n1 = _lower(replace(Mxfp6GemmSpec(), rsrc_once=True))[1].count("buffer.rsrc")
    assert n1 < n0
    assert not validate_spec(replace(Mxfp6GemmSpec(), rsrc_once=True, operand_layout="canonical"))[0]


def test_c0c1p_pair_address_is_hoisted():
    """``c0c1p``: the pair base (768 (wrb / 2) + 4 lane) is computed once; reads add immediates only."""
    from kernels.gfx950.mxfp6_gemm.gemm import _Builder

    bd = _Builder(replace(Mxfp6GemmSpec(), operand_layout="c0c1p"))
    bd.build()
    assert set(bd.c0c1p_base) == {"A", "B"}


def test_presets_and_required_compiler():
    """``PRESETS['best']`` needs the direct (Rust) compiler: LLVM builds are refused; the defaults stay LLVM-safe."""
    from kernels.gfx950.mxfp6_gemm.gemm import PRESETS, check_compiler, preset

    assert preset("default") == Mxfp6GemmSpec() and Mxfp6GemmSpec().required_compiler == "any"
    best = preset("best")
    assert validate_spec(best)[0] and best.required_compiler == "rust" and best.frag_mode == "prefetch"
    check_compiler(best, "rust")
    with pytest.raises(ValueError, match="direct"):
        check_compiler(best, "llvm")
    check_compiler(Mxfp6GemmSpec(), "llvm")
    assert descriptor(best)["required_compiler"] == "rust"
    # required_compiler does not change the code (same IR -> same name)
    assert kernel_name(best) == kernel_name(replace(best, required_compiler="any"))
    assert preset("best", elem_fmt="e3m2").elem_fmt == "e3m2"
    with pytest.raises(ValueError):
        preset("nope")
    assert set(PRESETS) >= {"default", "best", "best_v4", "best_v3", "best_v2", "best_c0c1"}
    assert best.swap_ab is False and preset("best_v4").swap_ab is True
    assert validate_spec(preset("best_v2"))[0] and preset("best_v2").required_compiler == "rust"
    assert best.operand_layout == "c0c1" and best.unroll == 4 and validate_spec(preset("best_c0c1"))[0]
    assert preset("best_v3").operand_layout == "c0c1p" and validate_spec(preset("best_v3"))[0]
    assert preset("best_c0c1").required_compiler == "rust"


def test_scale_dtl_x4w_one_dwordx4_per_operand_first_in_refill():
    """``scale_dtl="x4w"``: per tile two 1 KiB dwordx4 scale DTLs (waves 0 / 2), each before the operand DTLs."""
    from kernels.gfx950.mxfp6_gemm.gemm import _Builder

    spec = replace(Mxfp6GemmSpec(), scale_dtl="x4w")
    bd = _Builder(spec)
    bd.build()
    per_tile = {}
    for tag, kind in bd.vm_trace:
        per_tile.setdefault(tag, []).append(kind)
    loop_tiles = [k for k in per_tile if isinstance(k, tuple)]
    assert loop_tiles
    for k in loop_tiles:
        kinds = per_tile[k]
        assert kinds[:2] == ["sdtl", "sdtl"] and "sdtl" not in kinds[2:], kinds
    _, ll = _lower(spec)
    _, l0 = _lower(Mxfp6GemmSpec())

    def lds_dma(t):
        return [x for x in t.splitlines() if "buffer.load.lds" in x and "call" in x and "declare" not in x]

    # the dword scale DTLs (4-byte units) are gone: every DMA moves 16 bytes per lane
    assert all(", i32 16, " in x for x in lds_dma(ll)) and any(", i32 4, " in x for x in lds_dma(l0))
    assert not validate_spec(replace(spec, scale_src="vgpr", scale_slot=2))[0]


@pytest.mark.parametrize("probe", ["dtl_only", "lds_only", "fill_only"])
def test_probes_build_on_the_prefetch_presets(probe):
    """Regression (search): ``probe="dtl_only"`` on a prefetch config raised KeyError (loop-carried fragments)."""
    from kernels.gfx950.mxfp6_gemm.gemm import PRESETS

    for over in (PRESETS["best"], PRESETS["best_c0c1"], {"frag_mode": "prefetch", "acq_slot": 8, "read_every": 3}):
        spec = replace(Mxfp6GemmSpec(), **{k: v for k, v in over.items() if k != "required_compiler"}, probe=probe)
        assert validate_spec(spec)[0]
        build_mxfp6_gemm(spec, arch="gfx950")


@pytest.mark.parametrize("order", ["rows", "pairs", "blk2"])
def test_mma_orders_are_permutations(order):
    from kernels.gfx950.mxfp6_gemm.gemm import PRESETS, mfma_order

    spec = replace(Mxfp6GemmSpec(), mma_order=order)
    o = mfma_order(spec)
    assert sorted(o) == sorted((0, m, n) for m in range(8) for n in range(8)) and len(set(o)) == 64
    if order == "pairs":  # every B fragment feeds two MFMAs back to back
        assert all(o[2 * i][2] == o[2 * i + 1][2] for i in range(32))
    best = replace(Mxfp6GemmSpec(), **{k: v for k, v in PRESETS["best"].items() if k != "required_compiler"},
                   mma_order=order)
    assert validate_spec(best)[0]
    build_mxfp6_gemm(best, arch="gfx950")


def test_k_rot_rotates_the_refill_source_per_xcd():
    spec = replace(Mxfp6GemmSpec(), k_rot=1)
    assert validate_spec(spec)[0] and not validate_spec(replace(spec, xcd_block="none"))[0]
    ll1, ll0 = _lower(spec)[1], _lower(Mxfp6GemmSpec())[1]
    assert ll1.count("select") > ll0.count("select")


def test_swap_ab_false_store_nt_and_stamps_build():
    """``swap_ab=False`` (A as src0: per-element bf16 stores), ``store_nt`` (nt C stores), ``probe="stamps"``."""
    _, l0 = _lower(Mxfp6GemmSpec())
    _, l1 = _lower(replace(Mxfp6GemmSpec(), swap_ab=False))
    assert l1.count("buffer.store") > l0.count("buffer.store")
    calls = [x for x in l1.splitlines() if "mfma.scale" in x and "call" in x]
    assert calls  # same MFMA count, operands swapped
    _, lnt = _lower(replace(Mxfp6GemmSpec(), store_nt=True))
    assert lnt != l0
    _, ls = _lower(replace(Mxfp6GemmSpec(), probe="stamps"))
    assert ls.count("s_memrealtime") == 4 and ls.count("s_memtime") == 2  # start, after prologue, loop end, end


def test_scale_bias_is_exact_and_range_guarded():
    """apply_scale_bias(A, +d) / (B, -d) keeps every per-block scale product: the fp64 reference is unchanged."""
    a = L.generate(32, 512, L.DIST_GAUSS, seed=3, tag=1)
    b = L.generate(32, 512, L.DIST_GAUSS, seed=3, tag=2)
    ref, s = L.reference_f64(a, b)
    for d in (2, 3, 4):
        r2, s2 = L.reference_f64(L.apply_scale_bias(a, d), L.apply_scale_bias(b, -d))
        assert np.array_equal(ref, r2) and np.array_equal(s, s2)
    # D = dA + dB != 0 scales C by exactly 2^D (the kernel's out_exp2 = -D undoes it)
    r3, _ = L.reference_f64(L.apply_scale_bias(a, 3), L.apply_scale_bias(b, 3))
    assert np.array_equal(r3, ref * 64.0)
    with pytest.raises(ValueError):
        L.apply_scale_bias(a, 200)
    u = L.generate(8, 256, L.DIST_UNIFORM_CODES, seed=1, tag=1)
    assert L.apply_scale_bias(u, 4).scales.max() <= 254
    spec = replace(Mxfp6GemmSpec(), out_exp2=-6)
    assert validate_spec(spec)[0] and _lower(spec)[1].count("fmul") >= 256


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_edge_case_reference_convention(fmt):
    """The edge-case suite's reference: NaN rows / columns from 0xFF scales, single-signed overflows only, and the
    checker accepts the reference's own bf16 rounding and rejects a flipped inf / a lost NaN / a finite error."""
    from kernels.gfx950.mxfp6_gemm import edge_cases as E

    es = E.edge_operands(fmt=fmt)
    ref, s = E.reference_ieee(es.a, es.b)
    assert np.isnan(ref[2]).all() and np.isnan(ref[:, 510]).all()
    assert np.isinf(ref[4, 4]) and ref[4, 4] > 0 and np.isfinite(ref[4, 3])
    assert np.isinf(ref).sum() <= 3  # only the designed same-signed overflows
    good = L.f32_to_bf16_bits(ref)
    assert E.check_ieee(good, ref, s, es.cases)["bad"] == 0
    for r, c, bits in ((4, 4, 0xFF80), (2, 0, 0x3F80), (0, 1, 0x4000)):  # -inf for +inf, 1.0 for NaN, a wrong value
        bad = good.copy()
        bad[r, c] = bits
        assert E.check_ieee(bad, ref, s, es.cases)["bad"] == 1
    # the "keep" convention is enforced by check_ieee itself: a flushed (+-0) subnormal output fails by default
    flushed1 = good.copy()
    flushed1[9, 9] = 0x8000  # -0.0 where the reference is a bf16 subnormal
    assert E.check_ieee(flushed1, ref, s, es.cases)["bad"] == 1
    assert E.check_ieee(flushed1, ref, s, es.cases)["fp32_subnormal"] == 1
    assert E.check_ieee(flushed1, ref, s, es.cases, flush_ok=True)["bad"] == 0
    # subnormal_behaviour classifies a whole region
    rows, cols = es.cases["fp32_subnormal"]
    assert E.subnormal_behaviour(good, ref, rows, cols) == "keep"
    flushed = good.copy()
    flushed[np.ix_(rows, cols)] = 0
    assert E.subnormal_behaviour(flushed, ref, rows, cols) == "flush"
    flushed[9, 9] = good[9, 9]
    assert E.subnormal_behaviour(flushed, ref, rows, cols) == "mixed"
    # 0x00 x 0x00 is a ~2^-254 block (adds 0), 0xFE x 0x01 a 2^1 block: both checked per case
    assert {"scale_00_00", "scale_fe_01", "near_max_finite", "subnormal", "neg_zero_row"} <= set(es.cases)


def test_scale_bias_general_rule():
    """Arbitrary tensors: d is clamped into scale_bias_range; NaN scales (0xFF) are preserved and still give NaN."""
    rng = np.random.default_rng(5)
    a = L.generate(16, 256, L.DIST_GAUSS, seed=2, tag=1)
    b = L.generate(16, 256, L.DIST_GAUSS, seed=2, tag=2)
    a.scales[0, 0], b.scales[3, 2] = 255, 0  # a NaN block in A, a 2^-127 block in B
    a.scales[1, 1] = 254
    lo, hi = L.scale_bias_range(a.scales, b.scales)
    assert lo <= 0 <= hi and hi == 0  # A already reaches 254 and B 0: no positive shift is legal
    ref, _ = L.reference_f64(a, b)
    a2 = L.apply_scale_bias(a, lo)
    assert a2.scales[0, 0] == 255
    r2, _ = L.reference_f64(a2, L.apply_scale_bias(b, -lo))
    np.testing.assert_array_equal(np.isnan(ref), np.isnan(r2))
    with pytest.raises(ValueError):
        L.apply_scale_bias(a, hi + 1)
    u = L.generate(8, 256, L.DIST_UNIFORM_CODES, seed=1, tag=1)
    lo2, hi2 = L.scale_bias_range(u.scales, u.scales)
    assert lo2 <= 0 <= hi2


def test_aiter_packers_match_the_harness_definition():
    """bench.aiter_pack_big / aiter_pack_scale equal the literal per-byte definitions of the aiter harness."""
    from benchmarks.gfx950.mxfp6_gemm.bench import aiter_pack_big, aiter_pack_scale

    op = L.generate(512, 512, L.DIST_UNIFORM_CODES, seed=4, tag=1)
    nt, nk, nb = 2, 4, 16
    big = aiter_pack_big(op).reshape(nt, nk + 2, 24576)
    sc = aiter_pack_scale(op).reshape(nt, nk + 2, 1024)
    rowb = op.codes
    for t in range(nt):
        for s_ in range(nk):
            for off in (0, 17, 63, 64, 511, 1023):
                rb, ln = off // 64, off % 64
                row, col = t * 256 + rb * 16 + ln % 16, s_ * 128 + (ln // 16) * 32
                src = rowb[row, col * 6 // 8: col * 6 // 8 + 24]
                assert bytes(big[t, s_, off * 16: off * 16 + 16]) == bytes(src[:16])
                assert bytes(big[t, s_, 16384 + off * 8: 16384 + off * 8 + 8]) == bytes(src[16:])
                su, kg, r16, sub = off // 512, (off % 512) // 128, (off % 128) // 8, off % 8
                assert sc[t, s_, off] == op.scales[t * 256 + su * 128 + sub * 16 + r16, s_ * 4 + kg]
        assert not big[t, nk:].any() and (sc[t, nk:] == 127).all()
    assert nb == op.k // 32


def test_mixed_layout_a_canonical_b_c0c1():
    """``a_layout="canonical"``: A read from the dense stream (3 x b64 per fragment), B from the c0c1 image."""
    from kernels.gfx950.mxfp6_gemm.gemm import operand_layout_of

    spec = replace(Mxfp6GemmSpec(), a_layout="canonical")
    assert validate_spec(spec)[0] and operand_layout_of(spec, "A") == "canonical" and operand_layout_of(spec, "B") == "c0c1"
    assert "_Acanonical" in kernel_name(spec) and descriptor(spec)["contract"]["a"].startswith("e2m3 [M,K] row-major")
    assert "c0c1 pre-tiled" in descriptor(spec)["contract"]["b"]
    _, l0 = _lower(Mxfp6GemmSpec())
    _, lm = _lower(spec)
    _, lc = _lower(replace(Mxfp6GemmSpec(), operand_layout="canonical"))
    n = lambda t: t.count("load <2 x i32>")  # noqa: E731 - the canonical fragment reads are b64
    assert n(l0) < n(lm) < n(lc)
    assert not validate_spec(replace(spec, operand_layout="canonical", a_layout="c0c1"))[0]


def test_host_quantizer_rules():
    """layout.quantize_bf16: OCP shared exponent (exact), RNE codes, zero / NaN blocks, decode within half a step."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((16, 256)) * 3).astype(np.float32)
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
    xb[0, :32] = 0
    xb[1, :32] = 0x7FC0
    xb[2, :32] = 0x3F90  # 1.125 -> blocks of ties
    q = L.quantize_bf16(xb)
    assert q.scales[0, 0] == 0 and q.scales[1, 0] == 255
    codes = L.unpack_codes(q.codes, 256)
    assert (codes[1, :32] == 0).all() and (codes[0, :32] == 0).all()
    xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    v = L.decode_operand(q)
    ok = np.isfinite(xf) & (q.scales.repeat(32, axis=1) != 255)
    sc = np.ldexp(1.0, q.scales.astype(np.int32) - 127).repeat(32, axis=1)
    assert (np.abs(v - xf)[ok] <= (sc * 0.0625 * 2.0 + 1e-300)[ok] * 4).all()  # within the format's step
    amax = np.abs(xf[3:, :]).reshape(13, 8, 32).max(-1)
    assert np.array_equal(q.scales[3:].astype(np.int32) - 127, np.floor(np.log2(amax)).astype(np.int32) - 2)
