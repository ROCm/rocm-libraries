# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Block-id guard of the paged address generator (``_attention_bwd_paged``).

A block-table entry is device data: an entry outside ``[0, num_blocks)``
must never reach an address. The tests drive :class:`PagedKv` through a small
probe kernel that uses exactly the unified main kernel's paths: the clamped
K / V row loads (``PagedSlab.load_vec``), the direct-mode dK / dV scatter
(``store_rule`` + ``store_vec``) and the atomic-mode page workspace
(``workspace_row`` + ``atomic_add_row``), in the ``"multi"`` and ``"sub"``
page classes.

* CPU: the guard emits its checks only when :class:`PagedKv` gets
  ``page_rows`` (no op otherwise), one range check per block-table read.
* device (gfx942 / gfx950 / gfx1151 / gfx1201): with out-of-range entries
  (the first blocks past the cache, ``-1``, ``2^31 - 1``) the loads of a bad
  block read exact zeros, the gradient cache and the workspace are written
  only at the slots of valid blocks, and the canary blocks allocated past
  ``num_blocks`` keep their sentinel. A negative control (guard off, bad
  entries inside the canary blocks only, so nothing faults) fails the same
  checks, so they are live.
* the wiring of the guard in the unified main kernel: its call site passes
  ``page_rows``.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from rocke.core.ir import BF16, F32, I32, IRBuilder, PtrType

from kernels.common._attention_bwd_addr import decode_batched
from kernels.common._attention_bwd_paged import PagedKv

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

DEVICE_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
KN = 64  # rows of a probe tile (one per thread)
D = 8  # elements per row (one 16-byte vector)
CANARY_BLOCKS = 4  # allocated past the cache (ids nb .. nb + 3 land here)
SENTINEL = -7.0
CANARY_VALUE = 99.0
_PARAMS = (
    ("BT", "Q"), ("LENS", "Q"), ("KC", "Q"), ("DST", "Q"), ("OUT", "Q"),
    ("WS", "Q"), ("page", "i"), ("bt_stride", "i"), ("page_rows", "i"),
    ("s_max", "i"), ("out_rows", "i"),
)  # fmt: skip


def _probe(sub_page: int | None, guard: bool):
    """The probe kernel: grid (kv tiles, 1, sequences), ``KN`` threads."""
    tag = f"s{sub_page}" if sub_page else "m"
    b = IRBuilder(f"rocke_test_paged_probe_{tag}_{'g' if guard else 'n'}")
    a = {}
    for name, fmt in _PARAMS:
        if fmt == "i":
            a[name] = b.param(name, I32)
        elif name in ("BT", "LENS"):
            a[name] = b.param(name, PtrType(I32, "global"), readonly=True)
        elif name == "WS":
            a[name] = b.param(name, PtrType(F32, "global"))
        else:
            a[name] = b.param(name, PtrType(BF16, "global"))
    b.kernel.attrs["max_workgroup_size"] = KN
    tid, kt, s = b.thread_id_x(), b.block_id_x(), b.block_id_z()
    seq = decode_batched(b, s, s_max=a["s_max"], seq_ptr=a["LENS"], has_len=1)
    k0 = b.mul(kt, b.const_i32(KN))
    page = a["page"] if sub_page is None else sub_page
    blk = b.mul(a["page"], b.const_i32(D)) if sub_page is None else sub_page * D
    paged = PagedKv(
        b, bt_ptr=a["BT"], bt_stride=a["bt_stride"], seq_idx=s, seq=seq, k_n0=KN,
        page=page, sub_page=sub_page, page_rows=a["page_rows"] if guard else None,
    )  # fmt: skip
    k = b.add(k0, tid)
    # a CTA whose tile starts past the sequence does nothing (the main
    # kernel's live rule): every row of a live tile clamps into the sequence
    with b.scf_if(b.cmp_lt(k0, seq.length)):
        kc = paged.cache(a["KC"], dtype=BF16, strides=(blk, D, D), stage_vec=8)
        v = kc.slab(0, k0).load_vec(k, 0)
        out_row = b.add(b.mul(s, a["out_rows"]), k)
        b.global_store_vN(a["OUT"], b.mul(out_row, b.const_i32(D)), v, D, align=16)
        tile = paged.tile(k0)
        store, row_live = tile.store_rule(k)
        dst = paged.cache(a["DST"], dtype=BF16, strides=(blk, D, D), stage_vec=8)
        with b.scf_if(store):
            dst.slab(0, k0, clamp=False).store_vec(
                k, 0, b.select(row_live, v, b.zero_vec(BF16, D))
            )
        row_slab = tile.workspace_row(
            a["WS"], page_rows=a["page_rows"], width=D, head=0
        )
        slab, rr = row_slab(tid)
        items = [(c, b.const_f32(1.0)) for c in range(D)]
        slab.atomic_add_row(rr, items, b.cmp_lt(k, seq.length))
    b.ret()
    return b.kernel


def _count_ops(kernel, name, pred=None):
    n = 0
    stack = [kernel.body]
    while stack:
        region = stack.pop()
        for op in region.ops:
            if op.name == name and (pred is None or op.attrs.get("pred") == pred):
                n += 1
            stack.extend(op.regions)
    return n


@pytest.mark.parametrize("sub_page", [None, 16, 32])
def test_guard_ops_only_with_page_rows(sub_page):
    """No guard op without ``page_rows``; with it a range check of every
    block-table read (one read per piece: 1 in the multi class, kN0 / page in
    the sub class: ``0 <= id < num_blocks``, two compares) plus the ``id >=
    0`` checks of the accesses."""
    plain, guarded = _probe(sub_page, False), _probe(sub_page, True)
    pieces = 1 if sub_page is None else KN // sub_page
    assert _count_ops(plain, "arith.cmp", "ge") == 0
    assert _count_ops(guarded, "arith.cmp", "ge") > pieces
    assert _count_ops(guarded, "arith.smax") > _count_ops(plain, "arith.smax")
    assert _count_ops(guarded, "arith.select") > _count_ops(plain, "arith.select")
    # the block-table reads themselves are the same
    loads = "memref.global_load_typed"
    assert _count_ops(guarded, loads) == _count_ops(plain, loads)


def test_unified_main_wires_the_guard(monkeypatch):
    """The unified main kernel passes its ``page_rows`` kernarg to
    :class:`PagedKv`, so the guard is in every unified main kernel."""
    import kernels.common.attention_bwd_unified as U

    seen = []
    orig = U.PagedKv.__init__

    def record(self, b, **kw):
        seen.append(kw.get("page_rows"))
        orig(self, b, **kw)

    monkeypatch.setattr(U.PagedKv, "__init__", record)
    U.build_attn_bwd_unified_main(U.AttnBwdUnifiedSpec(head_size=64), arch="gfx950")
    assert seen and all(x is not None for x in seen), (
        "the unified main kernel does not pass page_rows to PagedKv: its block "
        "ids are unguarded"
    )


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def _bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _launch(kernel, grid, bufs, scalars):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    art = compile_kernel(kernel, arch=_ARCH, capture_ir_text=False, backend="python")
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    dev = {}
    for k, arr in bufs.items():
        dev[k] = rt.alloc(max(arr.nbytes, 16))
        rt.memcpy_h2d(
            dev[k], (ctypes.c_uint8 * arr.nbytes).from_buffer(arr), arr.nbytes
        )
    packed = b""
    for name, fmt in _PARAMS:
        if fmt == "Q":
            packed += struct.pack("<Q", dev[name])
        else:
            packed += struct.pack("<i", scalars[name])
    rt.launch(fn, grid, (KN, 1, 1), packed)
    rt.sync()
    for k, arr in bufs.items():
        rt.memcpy_d2h(
            (ctypes.c_uint8 * arr.nbytes).from_buffer(arr), dev[k], arr.nbytes
        )
        rt.free(dev[k])
    mod.unload()


def _case(page: int, bad):
    """Lengths, cache size and block table of one probe run.

    Three sequences over a cache of ``nb`` blocks (two more than the live
    blocks); every live entry is a distinct valid block (block 0 included, so
    a bad entry redirected to block 0 would show up there), then ``bad(nb)``
    replaces live entries at spread-out slots (never the first block of
    sequence 0).
    """
    lens = np.array([5 * KN - 3, 2 * KN + 7, 3 * KN], np.int32)
    n_live = [-(-int(n) // page) for n in lens]
    bt_stride = max(n_live) + 2
    slots = [(s, j) for s in range(3) for j in range(n_live[s])]
    nb = len(slots) + 2
    valid = np.random.default_rng(page).permutation(nb)
    table = np.full((3, bt_stride), 0x7FFF0000, np.int64)  # never read
    for i, (s, j) in enumerate(slots):
        table[s, j] = valid[i]
    entries = bad(nb)
    picks = [slots[(3 + 5 * i) % len(slots)] for i in range(len(entries))]
    for (s, j), v in zip(picks, entries):
        table[s, j] = v
    bt = table.astype(np.int32).reshape(-1)
    return lens, table, bt, bt_stride, n_live, nb


def _run_probe(sub_page, guard, bad):
    page = sub_page or 128
    lens, table, bt, bt_stride, n_live, nb = _case(page, bad)
    rows = page * (nb + CANARY_BLOCKS)
    rng = np.random.default_rng(7)
    kc_val = rng.uniform(1.0, 2.0, size=(rows, D)).astype(np.float32)
    kc_val[nb * page :] = CANARY_VALUE
    kc = _bf16(kc_val).reshape(-1)
    dst = np.full(rows * D, _bf16(np.float32(SENTINEL)), np.uint16)
    n_tiles = -(-int(lens.max()) // KN)
    out_rows = n_tiles * KN
    out = np.full(3 * out_rows * D, _bf16(np.float32(SENTINEL)), np.uint16)
    ws = np.zeros(rows * D, np.float32)
    _launch(
        _probe(sub_page, guard), (n_tiles, 1, 3),
        {"BT": bt, "LENS": lens, "KC": kc, "DST": dst, "OUT": out, "WS": ws},
        {"page": page, "bt_stride": bt_stride, "page_rows": nb * page,
         "s_max": int(lens.max()), "out_rows": out_rows},
    )  # fmt: skip
    kcv = _f32(kc).reshape(rows, D)
    want_out = np.full((3, out_rows, D), SENTINEL, np.float32)
    want_dst = np.full((rows, D), SENTINEL, np.float32)
    want_ws = np.zeros((rows, D), np.float32)
    for s in range(3):
        n = int(lens[s])
        for k in range(out_rows):
            if (k // KN) * KN >= n:
                continue  # dead tile: nothing written
            r = min(k, n - 1)
            pid = int(table[s, r // page])
            want_out[s, k] = kcv[pid * page + r % page] if 0 <= pid < nb else 0.0
            j = k // page
            if j < n_live[s]:
                pid_k = int(table[s, j])
                if 0 <= pid_k < nb:
                    slot = pid_k * page + k % page
                    want_dst[slot] = kcv[slot] if k < n else 0.0
                    if k < n:
                        want_ws[slot] += 1.0
    got_out = _f32(out).reshape(3, out_rows, D)
    got_dst = _f32(dst).reshape(rows, D)
    got_ws = ws.reshape(rows, D)
    problems = []
    if not np.array_equal(got_out, want_out):
        problems.append("loads")
    if not np.array_equal(got_dst, want_dst):
        problems.append("direct stores")
    if not np.all(got_dst[nb * page :] == SENTINEL):
        problems.append("canary blocks written")
    if not np.array_equal(got_ws, want_ws):
        problems.append("workspace adds")
    return problems


_device = pytest.mark.skipif(
    _ARCH not in DEVICE_ARCHS,
    reason=f"needs a {DEVICE_ARCHS} device; device is {_ARCH}",
)


@_device
@pytest.mark.gpu
@pytest.mark.parametrize("sub_page", [None, 16, 32])
@pytest.mark.parametrize(
    "bad",
    [
        lambda nb: [nb],
        lambda nb: [nb + 1, nb + 3],
        lambda nb: [-1],
        lambda nb: [2**31 - 1, -(2**31)],
        lambda nb: [nb, -5, 2**30],
    ],
    ids=["past_end", "canary", "minus_one", "extremes", "mixed"],
)
def test_out_of_range_block_ids_are_never_dereferenced(sub_page, bad):
    assert _run_probe(sub_page, True, bad) == []


@_device
@pytest.mark.gpu
@pytest.mark.parametrize("sub_page", [None, 16])
def test_valid_tables_are_unchanged_by_the_guard(sub_page):
    assert _run_probe(sub_page, True, lambda nb: []) == []
    assert _run_probe(sub_page, False, lambda nb: []) == []


@_device
@pytest.mark.gpu
@pytest.mark.parametrize("sub_page", [None, 16])
def test_negative_control_without_the_guard(sub_page):
    """Guard off, bad entries inside the allocated canary blocks only (no
    fault possible): the unguarded kernel reads and writes them, so the
    checks above fail."""
    problems = _run_probe(sub_page, False, lambda nb: [nb, nb + 2])
    assert "loads" in problems and "canary blocks written" in problems, problems
