# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""In-launch coherence protocols of the true-mono platform (P5, D0.8).

``experiments/qwen38-true-mono/probes/P5.md``: inside one persistent launch,

* a KV row rewritten on one XCD after other XCDs read it is stale for 100% of
  the readers on other XCDs unless the writer writes back its L2 (or stores
  ``sc1``), and readers see stale vL1D lines unless they acquire (``buffer_inv
  sc1``) or load ``sc1``: no reader is "safe by eviction";
* a word rewritten in the launch and read through the scalar path returns the
  first value the scalar cache saw, for the whole launch (C-B5): carry it in the
  mailbox payload, or ``s_dcache_inv`` (+ the L2 acquire) before every re-read.

Protocols (``PersistKnobs.kv_bulk``):

* ``sc1`` (default for KV rows written in the launch): ``sc1`` stores,
  ``vmcnt(0)``, a CTA barrier, the ready pair; readers ``sc1``-load the rows
  this launch wrote (positions ``>= L``, which the reader knows) and
  plain-load the rest, with no invalidate (0.62-0.82 of the bulk rule's hop).
* ``bulk`` (fallback for bulk data whose fresh part the reader cannot name):
  plain stores, ``vmcnt(0)``, ``buffer_wbl2 sc1`` and the ready pair
  (``mailbox.release_bulk``); readers poll, ``buffer_inv sc1``, plain loads.

Markers for the static check (:func:`..lint.check_kv_acquire`; P5 verifier:
a missed acquire almost never fails a numerics gate, so the check is static):
every op these helpers emit carries ``attrs["rocke.kv"]``:

* ``"store"``: a store of a launch-written KV row;
* ``"publish"``: the ready-pair publication (and its write-back / wait);
* ``"poll"``: the reader's poll of the ready pair;
* ``"acquire"``: ``buffer_inv`` (agent or system scope);
* ``"launch"``: a load of a launch-written row; ``"stable"``: a load of a row
  written before the launch (no acquire needed);
* ``"sread"``: a scalar re-read of a launch-written word.

The attributes are invisible to the LLVM lowering (tests: the LLVM IR of a
marked build is byte-identical to the unmarked probe's).
"""

from __future__ import annotations

from contextlib import contextmanager

from rocke.core.ir import CPOL_SC0, CPOL_SC1, I32, IRBuilder, Value
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from .sync import poll_pair_deadline

__all__ = [
    "ACQ_ASM",
    "KV_ATTR",
    "KV_PROTOCOLS",
    "SREAD_MODES",
    "acquire",
    "kv_load",
    "kv_store",
    "marked",
    "poll_ready",
    "publish",
    "scalar_reread",
]

KV_ATTR = "rocke.kv"
KV_PROTOCOLS = ("sc1", "bulk", "plain")  # plain = the negative control (no release)
ACQ_ASM = {
    "inv": "s_waitcnt vmcnt(0)\nbuffer_inv sc1",
    "inv0": "s_waitcnt vmcnt(0)\nbuffer_inv sc0",
    "invsys": "s_waitcnt vmcnt(0)\nbuffer_inv sc0 sc1",
}
# scalar re-read of a launch-written word: reg (carried in the ready pair's
# payload: the design default), dinv_inv (s_dcache_inv + buffer_inv sc1 +
# s_load: the memory-model-safe re-read), dinv (s_dcache_inv + s_load: clean
# on hardware, relies on L2 read coherence), plain (s_load: the C-B5 hazard)
SREAD_MODES = ("reg", "dinv_inv", "dinv", "plain")


@contextmanager
def marked(b: IRBuilder, kind: str, attr: str = KV_ATTR, constants: bool = False):
    """Every op ``b`` creates inside the block (constants excepted, unless
    ``constants``) carries ``attrs[attr] = kind`` (default the ``rocke.kv``
    marker). The LLVM lowering ignores the attribute. Nests (the innermost
    marker wins)."""
    had = "_op" in b.__dict__
    prev = b.__dict__.get("_op")
    orig = prev if had else type(b)._op.__get__(b)

    def _op(*a, **kw):
        op = orig(*a, **kw)
        if constants or op.name != "arith.constant":
            op.attrs[attr] = kind
        return op

    b._op = _op
    try:
        yield
    finally:
        if had:
            b._op = prev
        else:
            del b.__dict__["_op"]


def acquire(b: IRBuilder, kind: str = "inv") -> None:
    """``s_waitcnt vmcnt(0)`` + ``buffer_inv`` (``inv`` = agent scope, the
    rule; ``inv0`` = workgroup scope, a no-op that fails like no invalidate;
    ``invsys`` = system scope, for host-rewritten buffers in ``loop`` mode)."""
    with marked(b, "acquire"):
        b.inline_asm(ACQ_ASM[kind], "~{memory}", [], None)


def kv_store(b: IRBuilder, v: Value, rs: Value, voff: Value, proto: str) -> None:
    """Store one dword of a launch-written KV row (``sc1`` for the ``sc1``
    protocol, plain otherwise)."""
    if proto not in KV_PROTOCOLS:
        raise ValueError(f"proto must be one of {KV_PROTOCOLS}")
    with marked(b, "store"):
        b.buffer_store_cp(v, rs, voff, b.const_i32(0), aux=CPOL_SC1 if proto == "sc1" else 0)


def kv_load(b: IRBuilder, rs: Value, voff: Value, *, launch: bool, sc1: bool) -> Value:
    """Load one dword of a KV row: ``launch`` = written in this launch (the
    static check requires ``sc1`` or a preceding acquire), else a stable row
    (plain). ``sc1``: load with ``sc1`` (the ``sc1`` protocol's reader)."""
    with marked(b, "launch" if launch else "stable"):
        return b.buffer_load_cp(rs, voff, b.const_i32(0), dwords=1, aux=CPOL_SC1 if sc1 else 0)


def publish(b: IRBuilder, proto: str, rmb, i, tag: Value, kn: M.MailboxKnobs, value=None, *, tid0=None) -> None:
    """Publish the ready pair after the CTA's KV stores (call from every thread
    of the writer CTA).

    ``bulk``: ``mailbox.release_bulk`` (``vmcnt(0)``, barrier, thread 0
    ``buffer_wbl2 sc1`` + the pair). ``sc1`` / ``plain``: ``vmcnt(0)``, barrier,
    thread 0 puts the pair (``plain`` = the negative control: no write-back of
    plain stores). ``tid0``: the caller's "thread 0" predicate, if it has one."""
    with marked(b, "publish"):
        if proto == "bulk":
            M.release_bulk(b, rmb, i, tag, kn)
        else:
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare(fence="compiler")
            with b.scf_if(D.is_thread0(b) if tid0 is None else tid0):
                M.put(b, rmb, i, b.const_i32(0) if value is None else value, tag, kn)


def poll_ready(b: IRBuilder, rmb, i, tag: Value, maxit: Value, pause: str = "s_nop"):
    """The reader's bounded poll of ready pair ``i``: returns ``(payload,
    iterations)`` (``iterations >= maxit``: timed out)."""
    with marked(b, "poll"):
        return poll_pair_deadline(b, M.pair_offset(b, i), rmb.addr, tag, maxit, pause)


def scalar_reread(b: IRBuilder, mode: str, addr: Value, payload: Value | None = None) -> Value:
    """A launch-written word as a uniform i32 (C-B5). ``reg``: the polled
    ready pair's payload (``payload``); ``dinv_inv`` / ``dinv`` / ``plain``:
    ``s_load_dword`` after the named invalidates (``plain`` = the hazard)."""
    if mode not in SREAD_MODES:
        raise ValueError(f"mode must be one of {SREAD_MODES}")
    if mode == "reg":
        if payload is None:
            raise ValueError("mode 'reg' needs the polled payload")
        with marked(b, "sread"):
            return b.readfirstlane(payload)
    pre = []
    if mode == "dinv_inv":
        pre += ["s_waitcnt vmcnt(0)", "buffer_inv sc1", "s_waitcnt vmcnt(0)"]
    if mode in ("dinv_inv", "dinv"):
        pre += ["s_dcache_inv", "s_waitcnt lgkmcnt(0)"]
    with marked(b, "sread"):
        return b.inline_asm(
            "\n".join([*pre, "s_load_dword $0, $1, 0x0", "s_waitcnt lgkmcnt(0)"]),
            "=s,s,~{memory}",
            [addr],
            I32,
            result_name_hint="p5s",
        )


# P5's vector-cache-policy bits, re-exported for the parity builders
SC0 = CPOL_SC0
SC1 = CPOL_SC1
