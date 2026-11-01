# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The P4 / P5 probe kernels rebuilt on the platform primitives (WP1 gate).

WP1's hardware gate is that the platform primitives reproduce the probes'
verdicts. This module builds every P4 / P5 kernel with the probe's own
builder but with each primitive the probe defines replaced by the platform's
(:mod:`.sync`, :mod:`.kv`), and the KV builder rewritten on :mod:`.kv`'s marked
protocol helpers. Two consequences, both tested:

* CPU: the LLVM IR of every parity kernel is byte-identical to the probe's
  (the primitives are drop-in; the ``rocke.kv`` markers are invisible to the
  lowering), so the probes' measured verdicts are the platform's;
* hardware: ``hackathon-prep/qwen38/truemono/WP1/wp1_parity.py`` runs the
  probes' own benches (``p4_bench.py``, ``p5_bench.py``) with these builders
  and the platform-only arms (the ``irf`` poll, per-CTA arrivals, the grid
  sync with the launch-fatal word), and the static KV check is run on the P5
  kernels: it flags exactly the arms P5 measured wrong on hardware.

Nothing here is imported by a served build.
"""

from __future__ import annotations

import contextlib
import importlib

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from . import kv as KV
from . import sync as PS

__all__ = [
    "P4_PRIMITIVES",
    "P5_PRIMITIVES",
    "build_kv_kernel",
    "build_scalar_kernel",
    "p4_module",
    "p5_module",
    "patched",
    "platform_p4",
    "platform_p5",
]

# probe module attribute -> platform primitive (the probe's own name for it)
P4_PRIMITIVES = {
    "_lds_write": PS.lds_write,
    "_lds_read": PS.lds_read,
    "_pad_lds": PS.pad_lds,
    "_delay": PS.delay,
    "_st_sc1": PS.st_sc1,
    "_ld_sc1": PS.ld_sc1,
    "_stamp": PS.stamp,
    "poll_counter": PS.poll_counter,
    "poll_pair_deadline": PS.poll_pair_deadline,
    "_poll_rank_sum": PS.poll_rank_sum,
    "_ticket_xcd": PS.ticket_xcd,
}


def _scalar_read_platform(orig):
    def read(b, mode, addr, nbytes, rs, p):
        m = {"reg": "reg", "smem": "plain", "smem_dinv": "dinv", "smem_dinv_inv": "dinv_inv"}.get(mode)
        if m is None:
            return orig(b, mode, addr, nbytes, rs, p)
        return KV.scalar_reread(b, m, addr, p)

    return read


def _p5_primitives(p5):
    return {
        "_lds_write": PS.lds_write,
        "_lds_read": PS.lds_read,
        "_pad_lds": PS.pad_lds,
        "_stamp": PS.stamp,
        "poll_counter": PS.poll_counter,
        "poll_pair_deadline": PS.poll_pair_deadline,
        "_acquire": KV.acquire,
        "_scalar_read": _scalar_read_platform(p5._scalar_read),
    }


P5_PRIMITIVES = ("_lds_write", "_lds_read", "_pad_lds", "_stamp", "poll_counter",
                 "poll_pair_deadline", "_acquire", "_scalar_read")  # fmt: skip


def p4_module():
    return importlib.import_module("kernels.gfx950.qwen38_mono.probes.p4_sync")


def p5_module():
    return importlib.import_module("kernels.gfx950.qwen38_mono.probes.p5_coherence")


@contextlib.contextmanager
def patched(module, mapping):
    """Temporarily replace ``module``'s attributes by ``mapping``."""
    old = {k: getattr(module, k) for k in mapping}
    for k in mapping:
        if not hasattr(module, k):
            raise AttributeError(f"{module.__name__} has no {k}")
    try:
        for k, v in mapping.items():
            setattr(module, k, v)
        yield module
    finally:
        for k, v in old.items():
            setattr(module, k, v)


@contextlib.contextmanager
def platform_p4():
    """``p4_sync`` with its primitives replaced by the platform's."""
    with patched(p4_module(), P4_PRIMITIVES) as m:
        yield m


@contextlib.contextmanager
def platform_p5():
    """``p5_coherence`` with its primitives replaced by the platform's and its
    KV builder by :func:`build_kv_kernel` (marked protocol helpers)."""
    p5 = p5_module()
    mp = dict(_p5_primitives(p5))
    with patched(p5, mp):
        with patched(p5, {"build_kv_kernel": build_kv_kernel}):
            yield p5


# --------------------------------------------------- the KV kernel on .kv


def build_kv_kernel(spec, arch: str = "gfx950") -> KernelDef:
    """``p5_coherence.build_kv_kernel`` with the writer's stores, the
    publication, the reader's poll / acquire and every KV load emitted by
    :mod:`.kv` (marked for :func:`..lint.check_kv_acquire`). Same params,
    same LLVM IR (tested)."""
    p5 = p5_module()
    p5._check_arch(arch)
    if spec.wmode not in p5.KV_WMODES or spec.rmode not in p5.KV_RMODES or spec.mutant not in p5.KV_MUTANTS:
        raise ValueError((spec.wmode, spec.rmode, spec.mutant))
    if not 1 <= spec.tail <= 8 or not 1 <= spec.slice_rows <= 64:
        raise ValueError("tail 1..8 rows, slice_rows 1..64")
    proto = {"wbl2": "bulk", "sc1": "sc1", "plain": "plain"}[spec.wmode]
    G, T, W, S = spec.grid, spec.threads, spec.tail, spec.slice_rows
    payload_word = PS_payload_word
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    kv = b.param("kv", gp)
    rbox = b.param("rbox", gp)
    sync = b.param("sync", gp)
    tab = b.param("tab", gp)
    dtab = b.param("dtab", gp)
    wtab = b.param("wtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    sink = b.param("sink", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    base = b.readfirstlane(b.param("base", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    ncons = b.readfirstlane(b.param("ncons", I32))
    nstable = b.readfirstlane(b.param("nstable", I32))
    cold = b.readfirstlane(b.param("cold", I32))
    kvbytes = b.readfirstlane(b.param("kvbytes", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    PS.pad_lds(b)
    p5._info_begin(b, info, bid, G)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    sink_a = b.readfirstlane(b.global_ptr_to_i64(sink))
    poff, pcnt, pbias, aoff = p5._counter_params(b, tab, bid)
    rs = b.buffer_rsrc(kv, kvbytes)
    rmb = M.mailbox_from_ptr(b, rbox, b.const_i32(0x7FFFFFF0), p5.KN)
    tid0 = D.is_thread0(b)
    first_cons = b.sub(b.const_i32(G), ncons)
    is_cons = b.cmp_ge(bid, first_cons)
    ci = b.sub(bid, first_cons)
    salt0 = b.const_i32(p5.SALT0)
    m1 = b.const_i32(-1)
    four = b.const_i32(4)
    rw = b.const_i32(p5.ROW_WORDS)
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        row = b.mul(b.add(eg, bid), b.const_i32(p5.OUT_COLS))
        tag = b.add(base, b.add(ph, b.const_i32(1)))
        writer = b.readfirstlane(b.global_load_i32(wtab, ph))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        # ---- the writer rewrites the tail rows
        with b.scf_if(b.cmp_eq(bid, writer)):
            p5._delay(b, d)
            if spec.stamps:
                with b.scf_if(tid0):
                    PS.stamp(b, out, row, 0)
            tf = D.fresh_v(b, tid)
            for j in range(W):
                w = b.add(b.mul(b.add(nstable, b.const_i32(j)), rw), tf)
                KV.kv_store(b, payload_word(b, salt, ph, w), rs, b.mul(w, four), proto)
            KV.publish(b, proto, rmb, ph, tag, p5.KN, tid0=tid0)
        # ---- readers: acquire, read the tail rows and a slice of stable rows, check
        with b.scf_if(is_cons):
            p5._delay(b, d)
            with b.scf_if(tid0):
                _v, it = KV.poll_ready(b, rmb, ph, tag, maxit)
                with b.scf_if(b.cmp_ge(it, maxit)):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * PS.ERR_TIMEOUT), b.const_i32(1))
                if spec.rmode in ("inv", "inv0"):
                    KV.acquire(b, spec.rmode)
            D.compiler_fence(b)
            b.s_barrier_bare(fence="compiler")
            tf = D.fresh_v(b, tid)
            hot0 = b.mul(ci, b.const_i32(S))
            cold0 = b.mod(b.mul(b.add(b.mul(ph, ncons), ci), b.const_i32(S)), nstable)
            sb = b.readfirstlane(b.select(b.cmp_ne(cold, b.const_i32(0)), cold0, hot0))
            tw, tv = [], []
            for j in range(W):
                w = b.add(b.mul(b.add(nstable, b.const_i32(j)), rw), tf)
                tw.append(w)
                tv.append(KV.kv_load(b, rs, b.mul(w, four), launch=True, sc1=spec.rmode == "sc1"))
            sw, sv = [], []
            for k in range(S):
                w = b.add(b.mul(b.add(sb, b.const_i32(k)), rw), tf)
                sw.append(w)
                sv.append(KV.kv_load(b, rs, b.mul(w, four), launch=False, sc1=False))
            bad = b.const_i32(0)
            stale = b.const_i32(0)
            prev = b.sub(ph, b.const_i32(1))
            has_prev = b.cmp_gt(ph, b.const_i32(0))
            for w, v in zip(tw, tv):
                ne = b.cmp_ne(v, payload_word(b, salt, ph, w))
                bad = b.add(bad, p5._b2i(b, ne))
                old = b.land(has_prev, b.cmp_eq(v, payload_word(b, salt, prev, w)))
                stale = b.add(stale, p5._b2i(b, b.land(ne, old)))
            for w, v in zip(sw, sv):
                bad = b.add(bad, p5._b2i(b, b.cmp_ne(v, payload_word(b, salt0, m1, w))))
            srow = b.mul(b.add(b.mul(ph, ncons), ci), b.const_i32(4 * p5.SINK_COLS))
            with b.scf_if(b.cmp_gt(bad, b.const_i32(0))):
                D.atomic_add_scoped(b, sink_a, srow, bad)
                D.atomic_add_scoped(b, err_a, b.const_i32(4 * PS.ERR_MISMATCH), bad)
            with b.scf_if(b.cmp_gt(stale, b.const_i32(0))):
                D.atomic_add_scoped(b, sink_a, b.add(srow, four), stale)
            b.s_barrier_bare(fence="compiler")
            if spec.stamps:
                with b.scf_if(tid0):
                    PS.stamp(b, out, row, 1)
        # ---- closing sync: orders the next epoch's rewrite after these reads
        wait = p5._true(b) if spec.mutant == "none" else is_cons
        p5._close_sync(b, sync_a, aoff, poff, pcnt, pbias, b.add(ph, b.const_i32(1)), maxit, err_a, wait)
    p5._info_finish(b, info, bid, G)
    b.ret()
    return b.kernel


def build_scalar_kernel(spec, arch: str = "gfx950") -> KernelDef:
    """``p5_coherence.build_scalar_kernel`` with the platform's primitives
    (``kv.scalar_reread`` for the ``reg`` / ``smem*`` modes, marked)."""
    with platform_p5() as p5:
        return p5.build_scalar_kernel(spec, arch)


def PS_payload_word(b, salt, e, w):
    """P4's payload word (the probes' host twin): imported lazily so this
    module does not import the probe at import time."""
    return p4_module().payload_word(b, salt, e, w)
