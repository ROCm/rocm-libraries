# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tagged-pair mailboxes for persistent (resident-grid) kernels.

Every value one CTA hands to another is an aligned 8-byte ``(u32 payload,
u32 tag)`` pair written by ONE 8- or 16-byte store. ``tag = step *
LAYER_SLOTS + layer + 1`` is unique per launch, so the data is its own ready
flag: no separate flag, no fence, no clearing between launches. A consumer
re-loads its whole batch of pairs until every tag matches, so a hand-off
costs one memory round trip after the last producer's store lands.

Protocol (the same as the public FlyDSL GLM monokernel's mailboxes):

* ``put`` / ``put2`` / ``put_bf``: one ``sc1`` (device-coherent) store of one
  or two pairs; ``scope="system"`` uses ``sc0 sc1``.
* ``poll``: batches of at most ``POLL_MAX`` specs, each one or two pairs,
  re-loaded with ``sc1`` loads until every tag matches. Never poll with plain
  loads: they can hit a stale vL1D line forever.
* ``pre_poll``: wave 0 spins on one hint pair per producer, then the CTA
  barriers. A throttle: with every thread of a full grid spinning on many
  pairs, the sc1 load flood can starve the producers' stores.
* ``release_bulk`` / ``acquire_bulk``: plain-store bulk data (rows, lists)
  published by one ready pair; thread 0 alone fences (``buffer_wbl2 sc1`` /
  ``buffer_inv sc1``).

Implementations, chosen by ``MailboxKnobs.impl`` (a registered
:class:`MailboxImpl` name):

* ``"asm"`` (default, the only one this module ships): inline asm only
  (``global_store_dwordx{2,4} ... sc1`` and one self-contained asm poll loop
  with ``${:uid}`` labels). Both engines lower it, so the kernel keeps cpp
  parity. The loop is opaque to the scheduler and waits ``vmcnt(0)`` per
  retry.
* Python-engine-only implementations (opcodes in
  ``rocke.core.ir.PYTHON_ONLY_OPCODES``) do not live in ``rocke.helpers``
  (``platform/AGENTS.md``: no helper without a C++ port path). They live in
  library families and register themselves with :func:`register_impl`; e.g.
  ``"ir"`` (``buffer_store_cp`` + the ``tile.tagged_poll`` leaf op, which the
  compiler schedules) is ``kernels.gfx950.qwen38_mono.mailbox_ir`` in the
  rocKE library: import it before building ``MailboxKnobs(impl="ir")``.

Tag wrap: tags are u32, so they repeat every ``2**32 // LAYER_SLOTS`` steps,
and the tag of ``step = 2**26 - 1, layer = LAYER_SLOTS - 1`` is 0, which
matches never-written (zeroed) memory. :func:`check_layer` keeps a layer
below ``LAYER_SLOTS - 1`` so no live tag is ever 0. :func:`launch_tag` checks
a static (Python int) layer itself; a runtime (``Value``) layer is NOT
checked on the device, so the host launcher must call :func:`check_layer` on
every layer id it passes (the drafter's 48 included).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from rocke.core.ir import (
    CPOL_SC0,
    CPOL_SC1,
    F32,
    I16,
    I32,
    I64,
    IRBuilder,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as dops

__all__ = [
    "ASM_IMPL",
    "DEFAULT_KNOBS",
    "LAYER_SLOTS",
    "NUMA_COPY_BYTES",
    "PAIR_BYTES",
    "PAYLOAD_WAIT_MAX_ROWS",
    "POLL_MAX",
    "Mailbox",
    "MailboxImpl",
    "MailboxKnobs",
    "acquire_bulk",
    "asm_poll_template",
    "bf2_f32",
    "bf16_pair",
    "check_layer",
    "get",
    "impl_for",
    "iod_id",
    "launch_tag",
    "local_copy",
    "mailbox_from_i64",
    "mailbox_from_ptr",
    "pair_offset",
    "payload_wait",
    "poll",
    "pre_poll",
    "put",
    "put2",
    "put_bf",
    "register_impl",
    "registered_impls",
    "release_bulk",
]

LAYER_SLOTS = 64
POLL_MAX = 12
PAIR_BYTES = 8
# payload_wait: builds of at most this many request rows only barrier (ATOM
# ``layout.MAX_ROWS``)
PAYLOAD_WAIT_MAX_ROWS = 4

IntOrValue = int | Value


class MailboxImpl:
    """How mailbox pairs are stored and polled (the ``MailboxKnobs.impl``).

    This module ships :data:`ASM_IMPL` only. A family that needs an
    implementation built on Python-engine-only opcodes subclasses this in its
    own module (outside ``rocke.helpers``) and calls :func:`register_impl`;
    its builds then compile with ``backend="python"``, carry the knob suffix
    ``mb<name>`` in their kernel names and are excluded from cpp parity.
    """

    name: str = ""

    def rsrc_from_ptr(self, b: IRBuilder, ptr: Value, num_bytes: Value) -> Value | None:
        """Extra per-mailbox state (e.g. a buffer resource) over a pointer."""
        return None

    def rsrc_from_i64(
        self, b: IRBuilder, addr: Value, num_bytes: Value
    ) -> Value | None:
        """The same over a uniform i64 address."""
        return None

    def store(self, b: IRBuilder, mb: Mailbox, voff: Value, data: Value, knobs) -> None:
        """One store of ``data`` (``<2|4 x i32>``) at byte offset ``voff``."""
        raise NotImplementedError

    def poll_batch(self, b: IRBuilder, specs, tag: Value, knobs) -> list[list[Value]]:
        """One poll batch (at most ``knobs.poll_max`` specs); see :func:`poll`."""
        raise NotImplementedError


_IMPLS: dict[str, MailboxImpl] = {}


def register_impl(impl: MailboxImpl) -> MailboxImpl:
    """Make ``impl`` selectable as ``MailboxKnobs(impl=impl.name)``.
    Re-registering the same name with an instance of the same class is a
    no-op; a different class under a taken name raises."""
    name = impl.name
    if not name or not name.isidentifier():
        raise ValueError(f"mailbox impl name must be an identifier, got {name!r}")
    old = _IMPLS.get(name)
    if old is not None and type(old) is not type(impl):
        raise ValueError(f"mailbox impl {name!r} is already registered by {old!r}")
    _IMPLS.setdefault(name, impl)
    return _IMPLS[name]


def registered_impls() -> tuple[str, ...]:
    return tuple(_IMPLS)


def impl_for(name: str) -> MailboxImpl:
    """The registered implementation called ``name``."""
    try:
        return _IMPLS[name]
    except KeyError:
        hint = (
            " ('ir' is the Python-engine-only implementation in the rocKE "
            "library: import kernels.gfx950.qwen38_mono.mailbox_ir first)"
            if name == "ir"
            else ""
        )
        raise ValueError(
            f"mailbox impl {name!r} is not registered; registered: "
            f"{registered_impls()}{hint}"
        ) from None


class _AsmImpl(MailboxImpl):
    """Inline asm only: both engines lower it (the default)."""

    name = dops.ASM

    def store(self, b, mb, voff, data, knobs) -> None:
        n = data.type.count
        nt = " nt" if knobs.put_nt else ""
        b.inline_asm(
            f"global_store_dwordx{n} $0, $1, $2 {knobs.cpol_asm}{nt}",
            "v,v,s",
            [voff, data, mb.addr],
            None,
        )

    def poll_batch(self, b, specs, tag, knobs):
        return _poll_batch_asm(b, specs, tag, knobs)


ASM_IMPL = register_impl(_AsmImpl())


@dataclass(frozen=True)
class MailboxKnobs:
    """Mailbox build knobs (all default to the shipped protocol).

    impl: a registered :class:`MailboxImpl` name: ``"asm"`` (both engines,
      the only one in ``rocke.helpers``) or a family's Python-only one
      (``"ir"``, see the module docstring).
    scope: ``"agent"`` (sc1) | ``"system"`` (sc0 sc1, e.g. peer-GPU memory).
    pause: what a poll retry executes: ``"s_nop"`` | ``"s_sleep1"`` | ``"none"``.
    poll_max: specs per poll batch (bounds live registers).
    fence_impl: ``"asm"`` (``buffer_wbl2`` / ``buffer_inv``) | ``"ir"`` (LLVM
      ``fence``; both engines) for :func:`release_bulk` / :func:`acquire_bulk`.
      ``"ir"`` relies on backend behaviour, not on the LLVM memory model: the
      ready pair is a non-atomic store / poll, and a fence only synchronizes
      through atomic accesses (see :func:`device_ops.release_l2`).
    put_nt: also set the ``nt`` (streaming) bit on mailbox stores.
    numa_copies: 1, or 2 = every put also stores to ``+NUMA_COPY_BYTES`` and
      consumers poll the copy homed at their own IOD (:func:`local_copy`).
    """

    impl: str = dops.ASM
    scope: str = "agent"
    pause: str = "s_nop"
    poll_max: int = POLL_MAX
    fence_impl: str = dops.ASM
    put_nt: bool = False
    numa_copies: int = 1

    def __post_init__(self) -> None:
        impl_for(self.impl)  # raises for an unregistered name
        if self.fence_impl not in dops.IMPLS:
            raise ValueError(f"fence_impl must be one of {dops.IMPLS}")
        if self.scope not in ("agent", "system"):
            raise ValueError("scope must be 'agent' or 'system'")
        if self.pause not in dops.PAUSES:
            raise ValueError(f"pause must be one of {tuple(dops.PAUSES)}")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max must be in 1..16")
        if self.numa_copies not in (1, 2):
            raise ValueError("numa_copies must be 1 or 2")

    @property
    def cpol(self) -> int:
        return CPOL_SC1 | (CPOL_SC0 if self.scope == "system" else 0)

    @property
    def cpol_asm(self) -> str:
        return "sc0 sc1" if self.scope == "system" else "sc1"

    def suffix(self) -> str:
        """Kernel-name suffix naming only the knobs that deviate from default."""
        d = MailboxKnobs()
        parts = []
        if self.impl != d.impl:
            parts.append(f"mb{self.impl}")
        if self.scope != d.scope:
            parts.append(f"mb{self.scope}")
        if self.pause != d.pause:
            parts.append(f"mb{self.pause}")
        if self.poll_max != d.poll_max:
            parts.append(f"mbpm{self.poll_max}")
        if self.fence_impl != d.fence_impl:
            parts.append(f"mbf{self.fence_impl}")
        if self.put_nt:
            parts.append("mbnt")
        if self.numa_copies != 1:
            parts.append(f"mbnuma{self.numa_copies}")
        return "_".join(parts)


DEFAULT_KNOBS = MailboxKnobs()


@dataclass(frozen=True)
class Mailbox:
    """A region of ``(payload, tag)`` pairs.

    ``addr`` is its uniform i64 base address (the asm path stores and loads
    through ``global_* v_off, s[base:base+1]``); ``rsrc`` is the extra state
    the selected :class:`MailboxImpl` keeps (e.g. a buffer resource), ``None``
    for ``asm``.
    """

    addr: Value
    rsrc: Value | None = None
    copy_off: Value | None = None  # byte offset of the polled NUMA copy


def mailbox_from_ptr(
    b: IRBuilder, ptr: Value, num_bytes: Value, knobs: MailboxKnobs = DEFAULT_KNOBS
) -> Mailbox:
    """A mailbox over a global pointer param (``num_bytes`` i32 bounds an
    implementation's buffer resource, if it keeps one)."""
    rsrc = impl_for(knobs.impl).rsrc_from_ptr(b, ptr, num_bytes)
    return Mailbox(addr=b.global_ptr_to_i64(ptr), rsrc=rsrc)


def mailbox_from_i64(
    b: IRBuilder,
    addr: Value,
    num_bytes: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> Mailbox:
    """A mailbox over a runtime 64-bit address (e.g. ``scratch + offset``)."""
    addr = b.readfirstlane(addr)
    rsrc = impl_for(knobs.impl).rsrc_from_i64(b, addr, num_bytes)
    return Mailbox(addr=addr, rsrc=rsrc)


def check_layer(layer: int, slots: int = LAYER_SLOTS) -> int:
    """Validate a layer id: ``0 <= layer < slots - 1`` so that no tag
    ``step * slots + layer + 1`` is ever 0 (see the module docstring). Host
    launchers call this for every runtime layer argument."""
    if not 0 <= int(layer) < slots - 1:
        raise ValueError(
            f"layer {layer} must be in 0..{slots - 2}: layer {slots - 1} makes "
            "the tag of step 2**26 - 1 wrap to 0, which zeroed memory matches"
        )
    return int(layer)


def _i32(b: IRBuilder, v: IntOrValue) -> Value:
    return b.const_i32(int(v)) if isinstance(v, int) else v


def launch_tag(
    b: IRBuilder, step: Value, layer: IntOrValue, *, slots: int = LAYER_SLOTS
) -> Value:
    """``readfirstlane(step) * slots + layer + 1`` as a uniform i32.

    ``step`` is the loaded i32 device step counter; a static ``layer`` is
    range-checked by :func:`check_layer`. A ``Value`` layer is unchecked: the
    launcher must :func:`check_layer` the value it passes."""
    if isinstance(layer, int):
        check_layer(layer, slots)
    s = b.readfirstlane(step)
    tag = b.add(b.mul(s, b.const_i32(slots)), b.add(_i32(b, layer), b.const_i32(1)))
    return b.readfirstlane(tag)


def pair_offset(b: IRBuilder, i: IntOrValue) -> Value:
    """Byte offset of pair ``i``."""
    if isinstance(i, int):
        return b.const_i32(i * PAIR_BYTES)
    return b.mul(i, b.const_i32(PAIR_BYTES))


def _poll_offset(b: IRBuilder, mb: Mailbox, i: IntOrValue) -> Value:
    off = pair_offset(b, i)
    return off if mb.copy_off is None else b.add(off, mb.copy_off)


# ------------------------------------------------------------ NUMA copies
#
# gfx950 homes every 8 KiB block of memory at one of the two IODs (XCDs 0-3
# or 4-7): address bit 13 XOR a constant per 2 MiB page. A hand-off whose
# mailbox is homed at the consumer's IOD is markedly faster (see
# examples/gfx950/persistent_mailbox/mailbox_rtt_case_study.md). With
# ``numa_copies=2`` a mailbox region is laid out as two copies 8 KiB apart
# inside a 16 KiB-aligned block (so the copies are homed at different IODs),
# every put writes both and each consumer polls its local one. ``flip`` is the
# host-calibrated per-2-MiB-page constant (0 when copy 0 is homed at the IOD
# of XCDs 0-3).

NUMA_COPY_BYTES = 8192


def iod_id(b: IRBuilder, xcc_id=None) -> Value:
    """Uniform i32: 0 on XCDs 0-3, 1 on XCDs 4-7. ``xcc_id(b)`` reads the
    XCD id (default ``device_ops.xcc_id``)."""
    xcc = (xcc_id or dops.xcc_id)(b)
    return b.select(b.cmp_lt(xcc, b.const_i32(4)), b.const_i32(0), b.const_i32(1))


def local_copy(b: IRBuilder, mb: Mailbox, flip: IntOrValue = 0, xcc_id=None) -> Mailbox:
    """The view of a ``numa_copies=2`` mailbox a consumer polls: copy
    ``iod_id ^ flip`` (``flip`` uniform i32 or int; ``xcc_id`` as in
    :func:`iod_id`)."""
    idx = b.xor(iod_id(b, xcc_id), _i32(b, flip))
    return replace(mb, copy_off=b.mul(idx, b.const_i32(NUMA_COPY_BYTES)))


def _bits(b: IRBuilder, v: Value) -> Value:
    if v.type == F32:
        return b.bitcast(v, I32)
    if v.type != I32:
        raise TypeError(f"mailbox payload must be f32 or i32, got {v.type}")
    return v


def _store(b: IRBuilder, mb: Mailbox, voff: Value, data: Value, knobs) -> None:
    _store_one(b, mb, voff, data, knobs)
    if knobs.numa_copies == 2:
        _store_one(b, mb, b.add(voff, b.const_i32(NUMA_COPY_BYTES)), data, knobs)


def _store_one(b: IRBuilder, mb: Mailbox, voff: Value, data: Value, knobs) -> None:
    impl_for(knobs.impl).store(b, mb, voff, data, knobs)


def put(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    v: Value,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Pair ``i`` := ``(v, tag)`` in one 8-byte store; ``v`` f32 or i32."""
    data = b.vec_pack([_bits(b, v), tag], I32)
    _store(b, mb, pair_offset(b, i), data, knobs)


def put2(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    v0: Value,
    v1: Value,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Pairs ``i`` and ``i + 1`` (``i`` even) in one 16-byte store."""
    data = b.vec_pack([_bits(b, v0), tag, _bits(b, v1), tag], I32)
    _store(b, mb, pair_offset(b, i), data, knobs)


def bf16_pair(b: IRBuilder, lo: Value, hi: Value) -> Value:
    """Two f32 -> one i32 holding ``bf16(lo) | bf16(hi) << 16`` (RNE)."""
    words = []
    for x in (lo, hi):
        h = b.bitcast(b.trunc_f32_to_bf16(x), I16)
        words.append(b.zext(h, I32))
    return b.lor(words[0], b.shl(words[1], b.const_i32(16)))


def bf2_f32(b: IRBuilder, w: Value) -> tuple[Value, Value]:
    """Packed bf16-pair word -> (f32 low, f32 high)."""
    lo = b.bitcast(b.shl(w, b.const_i32(16)), F32)
    hi = b.bitcast(b.land(w, b.const_i32(-65536)), F32)
    return lo, hi


def put_bf(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    vs: Sequence[Value],
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Elements ``i .. i + len(vs)`` (2 or 4, ``i`` aligned to ``len(vs)``) as
    packed bf16 pairs: pair ``i // 2 + j`` := ``(bf16(vs[2j]) | bf16(vs[2j+1])
    << 16, tag)``, one 8- or 16-byte store."""
    if len(vs) not in (2, 4):
        raise ValueError("put_bf takes 2 or 4 values")
    words: list[Value] = []
    for j in range(len(vs) // 2):
        words += [bf16_pair(b, vs[2 * j], vs[2 * j + 1]), tag]
    if isinstance(i, int):
        voff = b.const_i32((i // 2) * PAIR_BYTES)
    else:
        voff = b.mul(i, b.const_i32(PAIR_BYTES // 2))
    _store(b, mb, voff, b.vec_pack(words, I32), knobs)


# --------------------------------------------------------------- polling

Spec = tuple[Mailbox, IntOrValue, int]


def asm_poll_template(
    npairs: Sequence[int], nbases: int, base_of: Sequence[int], cpol: str, pause: str
) -> tuple[str, str]:
    """(template, constraints) of the asm poll loop.

    Operands: outputs ``$0..$P-1`` one ``<2 x i32>`` per pair, ``$P`` an i64
    VGPR temp, ``$P+1`` an i64 SGPR mismatch mask; inputs: one i32 VGPR byte
    offset per spec, ``nbases`` i64 SGPR bases, then the i64 SGPR tag (zero
    extended). ``base_of[k]`` names spec ``k``'s base. Exposed for tests.
    """
    p_total = sum(npairs)
    tmp, mask = p_total, p_total + 1
    first_in = p_total + 2
    voff0 = first_in
    base0 = voff0 + len(npairs)
    tag = base0 + nbases
    top = "${:private}poll${:uid}_top"
    done = "${:private}poll${:uid}_done"
    lines = [f"{top}:"]
    p = 0
    for k, n in enumerate(npairs):
        for q in range(n):
            off = f" offset:{q * PAIR_BYTES}" if q else ""
            lines.append(
                f"global_load_dwordx2 ${p}, ${voff0 + k}, ${base0 + base_of[k]}"
                f"{off} {cpol}"
            )
            p += 1
    lines.append("s_waitcnt vmcnt(0)")
    for q in range(p_total):
        lines.append(f"v_lshrrev_b64 ${tmp}, 32, ${q}")
        dst = f"${mask}" if q == 0 else "vcc"
        lines.append(f"v_cmp_ne_u64_e64 {dst}, ${tmp}, ${tag}")
        if q:
            lines.append(f"s_or_b64 ${mask}, ${mask}, vcc")
    lines.append(f"s_and_b64 ${mask}, ${mask}, exec")
    lines.append(f"s_cbranch_scc0 {done}")
    if pause:
        lines.append(pause)
    lines.append(f"s_branch {top}")
    lines.append(f"{done}:")
    cons = ["=&v"] * p_total + ["=&v", "=&s"]
    cons += ["v"] * len(npairs) + ["s"] * nbases + ["s"]
    cons += ["~{vcc}", "~{scc}"]
    return "\n".join(lines), ",".join(cons)


def _poll_batch_asm(
    b: IRBuilder, specs: Sequence[Spec], tag: Value, knobs: MailboxKnobs
) -> list[list[Value]]:
    bases: list[Value] = []
    base_of: list[int] = []
    for mb, _, _ in specs:
        for j, x in enumerate(bases):
            if x is mb.addr:
                base_of.append(j)
                break
        else:
            base_of.append(len(bases))
            bases.append(mb.addr)
    npairs = [n for _, _, n in specs]
    template, cons = asm_poll_template(
        npairs, len(bases), base_of, knobs.cpol_asm, dops.PAUSES[knobs.pause]
    )
    voffs = [_poll_offset(b, mb, i) for mb, i, _ in specs]
    tag64 = b.zext(tag, I64)
    p_total = sum(npairs)
    outs = b.inline_asm_multi(
        template,
        cons,
        voffs + bases + [tag64],
        result_types=[VectorType(I32, 2)] * p_total + [I64, I64],
        result_name_hint="apoll",
    )
    res, p = [], 0
    for n in npairs:
        res.append([b.vec_extract(outs[p + q], 0) for q in range(n)])
        p += n
    return res


def poll(
    b: IRBuilder,
    specs: Sequence[Spec],
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> list[list[Value]]:
    """Spin until every pair in ``specs`` = ``[(mailbox, pair index, npairs in
    {1, 2})]`` carries ``tag``; return one list of i32 payload words per spec.

    Batches of more than ``knobs.poll_max`` specs are split, each costing one
    round trip after its last producer. Issue independent loads (weights)
    BEFORE the poll so they overlap the wait."""
    for _, _, n in specs:
        if n not in (1, 2):
            raise ValueError(f"npairs must be 1 or 2, got {n}")
    out: list[list[Value]] = []
    fn = impl_for(knobs.impl).poll_batch
    for first in range(0, len(specs), knobs.poll_max):
        out += fn(b, specs[first : first + knobs.poll_max], tag, knobs)
    return out


def get(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> Value:
    """The payload of pair ``i`` once it carries ``tag``."""
    return poll(b, [(mb, i, 1)], tag, knobs)[0][0]


def pre_poll(
    b: IRBuilder,
    hint: Callable[[Value], tuple[Mailbox, Value]],
    n: int,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Wave 0 spins on one hint pair per producer (lane ``l`` on producers
    ``l, l + 64, ...``; ``hint(m) -> (mailbox, pair index)``), then the CTA
    barriers (``s_barrier`` with a compiler fence, no ``vmcnt`` drain). Every
    thread must call this. Callers still poll every payload pair."""
    if not 0 < n <= 64 * knobs.poll_max:
        raise ValueError(f"pre_poll n must be in 1..{64 * knobs.poll_max}")
    with b.scf_if(dops.is_wave0(b)):
        lane = b.lane_id()
        specs: list[Spec] = []
        for r in range((n + 63) // 64):
            m = b.add(lane, b.const_i32(64 * r))
            last = b.const_i32(n - 1)
            m = b.select(b.cmp_lt(m, last), m, last)
            mb, i = hint(m)
            specs.append((mb, i, 1))
        poll(b, specs, tag, knobs)
    b.s_barrier_bare(fence="compiler")


def payload_wait(
    b: IRBuilder,
    hint: Callable[[Value], tuple[Mailbox, Value]],
    n: int,
    tag: Value,
    rows: int,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
    *,
    max_rows: int = PAYLOAD_WAIT_MAX_ROWS,
) -> None:
    """The wait ahead of a payload poll whose producers end with the hint
    pairs ``hint`` (``n`` of them, as :func:`pre_poll`); ATOM's
    ``stages.common.payload_wait``.

    Builds of at most ``max_rows`` request rows only barrier
    (``s_barrier_bare(fence="compiler")``, no ``vmcnt`` drain): the payload
    poll spins itself, one round trip after the producers' stores instead of
    the hint's two. Wider builds keep the :func:`pre_poll` throttle (their
    payload polls hold more pairs per thread, on more CTAs). ``rows`` is a
    build-time int. Every thread must call this."""
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    if rows <= max_rows:
        b.s_barrier_bare(fence="compiler")
    else:
        pre_poll(b, hint, n, tag, knobs)


def release_bulk(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Publish this CTA's prior plain stores with one ready pair ``i``.

    Every thread: ``s_waitcnt vmcnt(0)`` and a barrier; thread 0: release at
    the knob scope, then ``put(i, 0, tag)``. Every thread must call this."""
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare(fence="compiler")
    with b.scf_if(dops.is_thread0(b)):
        dops.release_l2(b, knobs.scope, impl=knobs.fence_impl)
        put(b, mb, i, b.const_i32(0), tag, knobs)


def acquire_bulk(
    b: IRBuilder,
    mb: Mailbox,
    i: IntOrValue,
    tag: Value,
    knobs: MailboxKnobs = DEFAULT_KNOBS,
) -> None:
    """Wait for ready pair ``i`` (thread 0 polls, then acquires), then a
    barrier; afterwards every thread may plain-load the bulk data. Every
    thread must call this."""
    with b.scf_if(dops.is_thread0(b)):
        poll(b, [(mb, i, 1)], tag, knobs)
        dops.acquire_l2(b, knobs.scope, impl=knobs.fence_impl)
    b.s_barrier_bare(fence="compiler")
