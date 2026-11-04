# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-lane hardware probe of ``ds_read_tr16_b64`` (gfx950 wave64).

The backward reads transposed operands (dS^T, Q^T, dO^T, K^T) from a K-outer
LDS tile (``smem[k][mn]``) with ``ds_read_tr16_b64``. A wrong lane map gives a
byte-identical but transposed operand, so the per-lane distribution is probed
on the device before any loader uses it.

Probe kernel (one wave of 64 lanes): a 64 x 64 16-bit LDS tile holds a unique
code per element (``code = row * 64 + col``, raw 16-bit storage, never used
arithmetically). Every lane reads its own ``(row, col)`` addresses from a
global table, issues two ``ds_read_tr16_b64`` and stores the 4 + 4 returned
elements. The host decodes each returned code back to ``(row, col)``.

Model under test (one 16-lane group ``G``, lanes ``j`` in ``0..15``; lane
``16 G + i`` supplies the address of 4 consecutive elements ``H[i][0..3]``)::

    out[16 G + j][r] = H[16 G + 4 r + j // 4][j % 4]

Checks:

* CPU: with the K-outer fragment address map (lane ``l`` addresses row
  ``k0 + (l // 16) * kpl + (l % 16) // 4 + 4 t``, column ``mn0 + (l % 4) * 4``
  for read ``t``, ``kpl`` = elements per lane of the atom), the model delivers
  exactly the catalog operand coordinates of ``_attention_bwd_frag`` for every
  gfx950 16x16 atom, both roles and both dtypes; a lane-uniform address (all
  lanes on the tile origin) does not.
* CPU (comgr + ``llvm-objdump``): the probe kernel compiles for gfx950 and
  holds exactly two ``ds_read_b64_tr_b16`` and no 128-bit transpose read.
* Device (gfx950 only): for the fragment map at K = 16 and K = 32, for
  random distinct per-lane addresses and for lane-uniform addresses, the device
  output equals the model; for the fragment maps it equals the catalog operand
  coordinates. The random table also infers the source lane and element of
  every output slot independently of the model.
* The stacked plan of the K-permuted 16x16x32 B (two 16x16x16 B reads at K
  offsets 0 and 16, ``tr16_read_plan``): its addresses through the model
  (CPU) and on the device, and the backward's loader on that map (device),
  deliver the permuted coordinates.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from rocke.core.ir import BF16, F16, I32, IRBuilder, PtrType

from kernels.common._attention_bwd_caps import bwd_atoms
from kernels.common._attention_bwd_frag import (
    kperm_b_coords,
    operand_coords,
    plan_c_to_a,
    tr16_plan_address,
    tr16_read_plan,
)

ROWS = 64
COLS = 64
WAVE = 64
READS = 2
DTYPES = ("fp16", "bf16")

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

_on_gfx950 = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 device; device is {_ARCH}"
)


# ---------------------------------------------------------------------------
# host model
# ---------------------------------------------------------------------------


def code(row: int, col: int) -> int:
    return row * COLS + col


def decode(c: int) -> tuple[int, int]:
    return divmod(int(c), COLS)


def model_read(addr: np.ndarray) -> np.ndarray:
    """Model output codes ``[lane][4]`` of one read for per-lane ``addr[lane] = (row, col)``."""
    out = np.zeros((WAVE, 4), np.int64)
    for lane in range(WAVE):
        g, j = divmod(lane, 16)
        for r in range(4):
            row, col = addr[16 * g + 4 * r + j // 4]
            out[lane, r] = code(row, col + j % 4)
    return out


def model(table: np.ndarray) -> np.ndarray:
    """Model output codes ``[lane][4 * READS]`` for ``table[lane][read] = (row, col)``."""
    return np.concatenate([model_read(table[:, t]) for t in range(READS)], axis=1)


def frag_table(kpl: int, k0: int, mn0: int) -> np.ndarray:
    """K-outer fragment address map (``kpl`` elements per lane, 4 per read)."""
    t = np.zeros((WAVE, READS, 2), np.int32)
    for lane in range(WAVE):
        for r in range(READS):
            row = k0 + (lane // 16) * kpl + (lane % 16) // 4 + 4 * r
            t[lane, r] = (row, mn0 + (lane % 4) * 4)
    return t


def uniform_table(origins) -> np.ndarray:
    t = np.zeros((WAVE, READS, 2), np.int32)
    for r, o in enumerate(origins):
        t[:, r] = o
    return t


def random_table(seed: int) -> np.ndarray:
    """Distinct, 8-byte aligned addresses for all 64 lanes and both reads."""
    rng = np.random.default_rng(seed)
    slots = rng.permutation(ROWS * (COLS // 4))[: WAVE * READS]
    t = np.zeros((WAVE, READS, 2), np.int32)
    for i, s in enumerate(slots):
        row, chunk = divmod(int(s), COLS // 4)
        t[i // READS, i % READS] = (row, chunk * 4)
    return t


def stacked_table(k0: int, mn0: int) -> np.ndarray:
    """Address map of the stacked plan (the K-permuted 16x16x32 B): stride 4
    per 16-lane group, reads at K offsets 0 and 16 (:func:`tr16_plan_address`)."""
    plan = tr16_read_plan(kperm_coords("fp16"))
    t = np.zeros((WAVE, READS, 2), np.int32)
    for lane in range(WAVE):
        for r in range(READS):
            k, mn = tr16_plan_address(lane, r, plan)
            t[lane, r] = (k0 + k, mn0 + mn)
    return t


def kperm_coords(dtype: str):
    """The K-permuted B coordinates of a 16x16x32 consumer of a 16x16x16 C."""
    ops = {op.k: op for op in bwd_atoms("gfx950", dtype)}
    plan = plan_c_to_a(ops[16], ops[32])
    return kperm_b_coords(plan, ops[16], ops[32])


def coords_codes(fc, k0: int, mn0: int) -> np.ndarray:
    """Codes a lane must hold for operand coordinates ``fc`` of a K-outer tile."""
    return np.array(
        [
            [code(k0 + k, mn0 + mn) for mn, k in fc.int_coords(lane)]
            for lane in range(WAVE)
        ],
        np.int64,
    )


def catalog_codes(op, role: str, k0: int, mn0: int) -> np.ndarray:
    """Codes a lane must hold for ``op``'s ``role`` operand of a K-outer tile."""
    fc = operand_coords(op, role)
    return np.array(
        [
            [code(k0 + k, mn0 + mn) for mn, k in fc.int_coords(lane)]
            for lane in range(WAVE)
        ],
        np.int64,
    )


def infer_sources(table: np.ndarray, got: np.ndarray) -> dict:
    """For every output slot, the (source lane, element) whose address covers its code."""
    owner = {}
    for lane in range(WAVE):
        for t in range(READS):
            row, col = (int(x) for x in table[lane, t])
            for e in range(4):
                owner[(t, code(row, col + e))] = (lane, e)
    src = {}
    for lane in range(WAVE):
        for s in range(4 * READS):
            t, r = divmod(s, 4)
            key = (t, int(got[lane, s]))
            if key not in owner:
                raise AssertionError(
                    f"lane {lane} slot {s}: code {got[lane, s]} was not addressed"
                )
            src[(lane, t, r)] = owner[key]
    return src


def _atoms():
    return [(dt, op) for dt in DTYPES for op in bwd_atoms("gfx950", dt)]


def _atom_id(p):
    return p[1].op_id


# ---------------------------------------------------------------------------
# CPU: model vs catalog operand coordinates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("atom", _atoms(), ids=_atom_id)
def test_fragment_map_through_model_gives_catalog_coords(atom, role):
    _, op = atom
    kpl = op.b_frag_len if role == "b" else op.a_frag_len
    assert kpl in (4, 8)
    k0, mn0 = 8, 20
    table = frag_table(kpl, k0, mn0)
    got = model(table)[:, :kpl]
    np.testing.assert_array_equal(got, catalog_codes(op, role, k0, mn0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_stacked_map_through_model_gives_the_kperm_coords(dtype):
    """The stacked plan's addresses through the model deliver the K-permuted
    B coordinates; the catalog 16x16x32 addresses do not."""
    k0, mn0 = 8, 20
    want = coords_codes(kperm_coords(dtype), k0, mn0)
    assert np.array_equal(model(stacked_table(k0, mn0)), want)
    assert not np.array_equal(model(frag_table(8, k0, mn0)), want)


def test_lane_uniform_address_is_not_a_transposed_tile():
    """The tile-origin-for-every-lane reading gives row 0 to every lane."""
    got = model(uniform_table([(0, 0), (0, 0)]))[:, :4]
    want = np.array(
        [
            [code(4 * (lane // 16) + s, lane % 16) for s in range(4)]
            for lane in range(WAVE)
        ]
    )
    assert not np.array_equal(got, want)
    assert {decode(c)[0] for c in got.ravel()} == {0}


def test_model_is_a_permutation_within_16_lane_groups():
    table = random_table(1)
    src = infer_sources(table, model(table))
    assert sorted(src.values()) == sorted(
        (lane, e) for lane in range(WAVE) for e in range(4) for _ in range(READS)
    )
    for (lane, _t, _r), (sl, _e) in src.items():
        assert lane // 16 == sl // 16


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


def _elem(dtype):
    return F16 if dtype == "fp16" else BF16


def make_probe(dtype: str):
    elem = _elem(dtype)
    b = IRBuilder(f"bwd_tr16_b64_probe_{dtype}")
    psrc = b.param("SRC", PtrType(elem, "global"), readonly=True)
    paddr = b.param("ADDR", PtrType(I32, "global"), readonly=True)
    pout = b.param("OUT", PtrType(elem, "global"))
    b.kernel.attrs["max_workgroup_size"] = WAVE
    tid = b.thread_id_x()
    smem = b.smem_alloc(elem, (ROWS, COLS), "tile")
    vec = 8
    for it in range(ROWS * COLS // (WAVE * vec)):
        idx = b.add(b.mul(tid, b.const_i32(vec)), b.const_i32(it * WAVE * vec))
        v = b.global_load_vN(psrc, idx, elem, vec)
        r = b.div(idx, b.const_i32(COLS))
        c = b.mod(idx, b.const_i32(COLS))
        b.smem_store_vN(smem, [r, c], v, vec)
    b.sync()
    for t in range(READS):
        a0 = b.add(b.mul(tid, b.const_i32(2 * READS)), b.const_i32(2 * t))
        row = b.global_load(paddr, a0, I32, align=4)
        col = b.global_load(paddr, b.add(a0, b.const_i32(1)), I32, align=4)
        frag = b.ds_read_tr16_b64(smem, row, col, dtype=elem)
        for s in range(4):
            o = b.add(b.mul(tid, b.const_i32(4 * READS)), b.const_i32(4 * t + s))
            b.global_store(pout, o, b.vec_extract(frag, s), align=2)
    b.ret()
    return b.kernel


def _compile(dtype: str, arch: str = "gfx950"):
    from rocke.helpers.compile import compile_kernel

    return compile_kernel(
        make_probe(dtype), arch=arch, capture_ir_text=True, backend="python"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_probe_ir_uses_the_b64_transpose_read(dtype):
    pytest.importorskip("rocke.runtime.comgr")
    art = _compile(dtype)
    ir = art.llvm_text
    assert ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(") == READS
    assert "tr16.b128" not in ir
    from ._attention_bwd_resources import disassemble

    try:
        text = disassemble(art.hsaco, arch="gfx950")
    except Exception as exc:  # noqa: BLE001 - disassembler missing or too old
        pytest.skip(f"no usable disassembler: {exc}")
    if "s_endpgm" not in text:
        pytest.skip("disassembler does not know gfx950")
    assert text.count("ds_read_b64_tr_b16") == READS
    assert "ds_read_b128_tr" not in text


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

_KERNELS: dict = {}


def run_probe(dtype: str, table: np.ndarray) -> np.ndarray:
    """Launch the probe; returns the device codes ``[lane][4 * READS]``."""
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    if dtype not in _KERNELS:
        art = _compile(dtype)
        mod = rt.load_module(art.hsaco)
        _KERNELS[dtype] = (mod, mod.get_function(art.kernel_name))
    _, fn = _KERNELS[dtype]
    src = np.arange(ROWS * COLS, dtype=np.uint16)
    addr = np.ascontiguousarray(table.astype(np.int32).reshape(-1))
    out = np.zeros(WAVE * 4 * READS, np.uint16)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)

    dev = []
    for a in (src, addr, out):
        p = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(p, u8(a), a.nbytes)
        dev.append(p)
    rt.launch(fn, (1, 1, 1), (WAVE, 1, 1), b"".join(struct.pack("<Q", p) for p in dev))
    rt.sync()
    rt.memcpy_d2h(u8(out), dev[2], out.nbytes)
    for p in dev:
        rt.free(p)
    return out.reshape(WAVE, 4 * READS).astype(np.int64)


def _first_mismatch(got, want):
    bad = np.argwhere(got != want)
    lane, s = (int(x) for x in bad[0])
    return (
        f"{len(bad)} slots differ; first lane {lane} slot {s}: "
        f"got {decode(got[lane, s])}, want {decode(want[lane, s])}"
    )


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("atom", _atoms(), ids=_atom_id)
def test_device_fragment_map_gives_catalog_coords(atom, role):
    dtype, op = atom
    kpl = op.b_frag_len if role == "b" else op.a_frag_len
    k0, mn0 = 8, 20
    table = frag_table(kpl, k0, mn0)
    got = run_probe(dtype, table)
    want_model = model(table)
    assert np.array_equal(got, want_model), _first_mismatch(got, want_model)
    want = catalog_codes(op, role, k0, mn0)
    assert np.array_equal(got[:, :kpl], want), _first_mismatch(got[:, :kpl], want)


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
def test_device_random_addresses_follow_the_model(dtype, seed):
    table = random_table(seed)
    got = run_probe(dtype, table)
    src = infer_sources(table, got)  # independent of the model
    for (lane, t, r), (sl, e) in src.items():
        g, j = divmod(lane, 16)
        assert (sl, e) == (
            16 * g + 4 * r + j // 4,
            j % 4,
        ), f"lane {lane} read {t} slot {r}: source lane {sl} element {e}"
    want = model(table)
    assert np.array_equal(got, want), _first_mismatch(got, want)


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_device_lane_uniform_address_follows_the_model(dtype):
    table = uniform_table([(0, 0), (5, 8)])
    got = run_probe(dtype, table)
    want = model(table)
    assert np.array_equal(got, want), _first_mismatch(got, want)
    assert {decode(c)[0] for c in got[:, :4].ravel()} == {0}


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_device_stacked_addresses_give_the_kperm_coords(dtype):
    """The stacked plan's per-lane addresses (K offsets 0 and 16) deliver the
    K-permuted 16x16x32 B coordinates on the device."""
    k0, mn0 = 8, 20
    table = stacked_table(k0, mn0)
    got = run_probe(dtype, table)
    want_model = model(table)
    assert np.array_equal(got, want_model), _first_mismatch(got, want_model)
    want = coords_codes(kperm_coords(dtype), k0, mn0)
    assert np.array_equal(got, want), _first_mismatch(got, want)


# ---------------------------------------------------------------------------
# the operand loader of the backward body (re-pointed probe)
# ---------------------------------------------------------------------------


def make_loader_probe(dtype: str, op, role: str, *, k0: int, mn0: int, coords=None):
    """One wave loads one fragment with ``load_frag_lds_tr16`` from a flat view
    (``coords``: operand coordinates other than the catalog ones of ``op``'s
    ``role``, e.g. the K-permuted B)."""
    from kernels.common._attention_bwd_frag import LdsView, load_frag_lds_tr16

    elem = _elem(dtype)
    tag = role if coords is None else "kperm"
    b = IRBuilder(f"bwd_tr16_loader_probe_{dtype}_k{op.k}_{tag}")
    psrc = b.param("SRC", PtrType(elem, "global"), readonly=True)
    pout = b.param("OUT", PtrType(elem, "global"))
    b.kernel.attrs["max_workgroup_size"] = WAVE
    tid = b.thread_id_x()
    base = 64  # a non-zero window origin inside the flat allocation
    smem = b.smem_alloc(elem, (base + ROWS * COLS,), "flat")
    view = LdsView(smem, base, COLS, ROWS)
    vec = 8
    for it in range(ROWS * COLS // (WAVE * vec)):
        idx = b.add(b.mul(tid, b.const_i32(vec)), b.const_i32(it * WAVE * vec))
        v = b.global_load_vN(psrc, idx, elem, vec)
        view.store(
            b, b.div(idx, b.const_i32(COLS)), b.mod(idx, b.const_i32(COLS)), v, vec
        )
    b.sync()
    fc = operand_coords(op, role) if coords is None else coords
    frag = load_frag_lds_tr16(
        b, fc, view, tid, dtype=dtype, arch="gfx950", mn0=mn0, k0=k0
    )
    n = frag.type.count
    for s in range(n):
        o = b.add(b.mul(tid, b.const_i32(n)), b.const_i32(s))
        b.global_store(pout, o, b.vec_extract(frag, s), align=2)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("atom", _atoms(), ids=_atom_id)
def test_loader_probe_compiles_to_b64_reads(atom, role):
    dtype, op = atom
    ir = __import__(
        "rocke.core.lower_llvm", fromlist=["_lower_kernel_to_llvm_python"]
    )._lower_kernel_to_llvm_python(
        make_loader_probe(dtype, op, role, k0=8, mn0=20), arch="gfx950"
    )
    kpl = op.b_frag_len if role == "b" else op.a_frag_len
    assert ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(") == kpl // 4
    assert "tr16.b128" not in ir


def _run_loader(kernel, kpl: int) -> np.ndarray:
    """Compile and launch a loader probe; the device codes ``[lane][kpl]``."""
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    art = compile_kernel(kernel, arch="gfx950", capture_ir_text=False, backend="python")
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    src = np.arange(ROWS * COLS, dtype=np.uint16)
    out = np.zeros(WAVE * kpl, np.uint16)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)

    dev = []
    for a in (src, out):
        p = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(p, u8(a), a.nbytes)
        dev.append(p)
    rt.launch(fn, (1, 1, 1), (WAVE, 1, 1), b"".join(struct.pack("<Q", p) for p in dev))
    rt.sync()
    rt.memcpy_d2h(u8(out), dev[1], out.nbytes)
    for p in dev:
        rt.free(p)
    mod.unload()
    return out.reshape(WAVE, kpl).astype(np.int64)


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("atom", _atoms(), ids=_atom_id)
def test_device_loader_gives_catalog_coords(atom, role):
    dtype, op = atom
    k0, mn0 = 8, 20
    kpl = op.b_frag_len if role == "b" else op.a_frag_len
    got = _run_loader(make_loader_probe(dtype, op, role, k0=k0, mn0=mn0), kpl)
    want = catalog_codes(op, role, k0, mn0)
    assert np.array_equal(got, want), _first_mismatch(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_kperm_loader_probe_compiles_to_two_b64_reads(dtype):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    op = [o for o in bwd_atoms("gfx950", dtype) if o.k == 32][0]
    kernel = make_loader_probe(dtype, op, "b", k0=8, mn0=20, coords=kperm_coords(dtype))
    ir = lower(kernel, arch="gfx950")
    assert ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(") == 2
    assert "tr16.b128" not in ir


@_on_gfx950
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_device_loader_gives_the_kperm_coords(dtype):
    """The backward's loader on the K-permuted 16x16x32 B (the stacked plan)
    delivers the permuted coordinates on every lane and slot."""
    op = [o for o in bwd_atoms("gfx950", dtype) if o.k == 32][0]
    k0, mn0 = 8, 20
    fc = kperm_coords(dtype)
    got = _run_loader(make_loader_probe(dtype, op, "b", k0=k0, mn0=mn0, coords=fc), 8)
    want = coords_codes(fc, k0, mn0)
    assert np.array_equal(got, want), _first_mismatch(got, want)
