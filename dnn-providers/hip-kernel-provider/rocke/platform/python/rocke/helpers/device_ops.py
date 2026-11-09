# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device ops for persistent (resident-grid) kernels on gfx94x / gfx950.

Small emitters a resident-grid kernel needs around its cross-CTA mailboxes:
runtime pointers from 64-bit addresses, wave-uniform values, cache
write-back / invalidate, sleeps, clocks and hardware ids.

Every helper here emits only ops that BOTH engines lower (``tile.inline_asm``,
``tile.readfirstlane``, ``tile.readlane``, ``tile.vec_pack``, ``memref.fence``,
plain arith), so a kernel built from them keeps cpp parity
(``platform/AGENTS.md``: no helper without a C++ port path). The
compiler-visible Python-engine-only alternatives (``IRBuilder.int_to_ptr``,
``buffer_load_cp`` / ``buffer_store_cp``, ``tagged_poll``; see
``rocke.core.ir.PYTHON_ONLY_OPCODES``) are used from library families only,
e.g. ``kernels.gfx950.qwen38_mono.mailbox_ir`` in the rocKE library.

Cache-policy facts these helpers rely on (gfx94x / gfx950 memory model):

* an ``sc1`` load skips the per-CU vL1D; a plain or ``sc0`` load can hit a
  stale vL1D line forever, so every poll uses ``sc1`` (device scope) or
  ``sc0 sc1`` (system scope);
* releasing plain stores to another CU: ``buffer_wbl2 sc1`` then
  ``s_waitcnt vmcnt(0)`` before the ready flag; acquiring on the consumer:
  ``buffer_inv sc1`` after the flag was observed. One wave per CTA fences.
"""

from __future__ import annotations

from rocke.core.ir import I32, I64, IRBuilder, PtrType, Value

__all__ = [
    "ASM",
    "ATOMIC_SCOPES",
    "IMPLS",
    "IR",
    "PAUSES",
    "RSRC_WORD3_GFX9",
    "acquire_l2",
    "atomic_add_scoped",
    "buffer_inv",
    "buffer_wbl2",
    "compiler_fence",
    "fresh",
    "fresh_s",
    "fresh_v",
    "hw_id",
    "is_thread0",
    "is_wave0",
    "memrealtime",
    "memtime",
    "ptr_from_i64",
    "release_l2",
    "rsrc_vec",
    "s_nop",
    "s_sleep",
    "spin_pause",
    "uniform",
    "vec2_i32",
    "wave_id",
    "wave_umax_dpp",
    "xcc_id",
]

ASM = "asm"
IR = "ir"
IMPLS = (ASM, IR)

# DWORD3 of a raw (stride 0, bounds-checked, 32-bit uint) gfx9 buffer
# descriptor; the value ``tile.buffer_rsrc`` passes to make.buffer.rsrc on
# CDNA targets.
RSRC_WORD3_GFX9 = 0x00027000


def _check_impl(impl: str) -> str:
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
    return impl


def ptr_from_i64(
    b: IRBuilder,
    addr: Value,
    *,
    pointee=I32,
    uniform: bool = True,
) -> Value:
    """A global pointer Value from a runtime 64-bit address.

    Moves the address through an ``s_mov_b64`` (or ``v_mov_b64`` when
    ``uniform=False``) whose result LLVM types as ``ptr addrspace(1)``; both
    engines lower it. ``uniform=True`` pins the pointer in SGPRs: pass only
    wave-uniform addresses (LLVM would otherwise insert a ``readfirstlane``).

    Cost: the asm is not ``sideeffect``, but the lowered call carries no
    ``memory(none)`` attribute, so LLVM IR passes treat it as possibly
    accessing memory: an unused pointer is not necessarily removed, and IR
    passes do not move memory operations across it (the machine scheduler is
    not constrained by it). The Python-engine-only alternative
    ``IRBuilder.int_to_ptr`` (used from library families, e.g.
    ``kernels.gfx950.qwen38_mono.mailbox_ir.ptr_from_i64_ir``) has neither
    cost.
    """
    if addr.type != I64:
        raise TypeError(f"ptr_from_i64 expects an i64 address, got {addr.type}")
    ptr_t = PtrType(pointee, "global")
    if uniform:
        return b.inline_asm(
            "s_mov_b64 $0, $1",
            "=s,s",
            [addr],
            ptr_t,
            sideeffect=False,
            result_name_hint="aptr",
        )
    return b.inline_asm(
        "v_mov_b64 $0, $1",
        "=v,v",
        [addr],
        ptr_t,
        sideeffect=False,
        result_name_hint="aptr",
    )


def rsrc_vec(
    b: IRBuilder,
    addr: Value,
    num_bytes: Value,
    *,
    word3: int = RSRC_WORD3_GFX9,
) -> Value:
    """A raw gfx9 buffer descriptor as a ``<4 x i32>`` value.

    ``{addr.lo, addr.hi & 0xffff (stride 0), num_records, word3}``: the operand
    an inline-asm ``buffer_load`` / ``buffer_store`` takes through an ``s``
    constraint. ``addr`` is i64, ``num_bytes`` i32; both wave-uniform. (The
    ``tile.buffer_*`` ops take ``IRBuilder.buffer_rsrc``'s
    ``ptr addrspace(8)`` instead; the two are not interchangeable.)
    """
    if addr.type != I64 or num_bytes.type != I32:
        raise TypeError("rsrc_vec expects (i64 addr, i32 num_bytes)")
    lo = b.trunc(addr, I32)
    hi = b.trunc(b.lshr(addr, b.const_i64(32)), I32)
    hi = b.land(hi, b.const_i32(0xFFFF))
    return b.vec_pack([lo, hi, num_bytes, b.const_i32(int(word3))], I32)


def uniform(b: IRBuilder, v: Value) -> Value:
    """``readfirstlane``: make a wave-uniform value an SGPR value."""
    return b.readfirstlane(v)


def wave_id(b: IRBuilder, wave_size: int = 64) -> Value:
    """This wave's index in the workgroup, as a uniform i32."""
    shift = {64: 6, 32: 5}[wave_size]
    return b.readfirstlane(b.lshr(b.thread_id_x(), b.const_i32(shift)))


def is_wave0(b: IRBuilder, wave_size: int = 64) -> Value:
    """Uniform i1: this is wave 0 of the workgroup (branch on it without
    divergence)."""
    return b.cmp_eq(wave_id(b, wave_size), b.const_i32(0))


def is_thread0(b: IRBuilder) -> Value:
    """i1: this is thread 0 of the workgroup."""
    return b.cmp_eq(b.thread_id_x(), b.const_i32(0))


def s_nop(b: IRBuilder, n: int = 0) -> None:
    """``s_nop n`` (n in 0..15) as a side-effecting asm statement."""
    if not 0 <= n <= 15:
        raise ValueError(f"s_nop takes 0..15, got {n}")
    b.inline_asm(f"s_nop {n}", "", [], None)


def s_sleep(b: IRBuilder, units: int) -> None:
    """Sleep ``units`` x 64 clocks: ``s_sleep`` chained in steps of <= 127."""
    if units < 0:
        raise ValueError(f"s_sleep units must be >= 0, got {units}")
    steps = []
    while units > 0:
        steps.append(min(units, 127))
        units -= steps[-1]
    if steps:
        b.inline_asm("\n".join(f"s_sleep {s}" for s in steps), "", [], None)


PAUSES = {"s_nop": "s_nop 0", "s_sleep1": "s_sleep 1", "none": ""}


def spin_pause(b: IRBuilder, pause: str = "s_nop") -> None:
    """The side-effecting statement a poll loop executes between retries.

    It keeps the polled loads inside the loop; ``"s_sleep1"`` also throttles
    the retry rate (fewer sc1 loads competing with the producers' stores).
    """
    text = PAUSES[pause]
    b.inline_asm(text, "", [], None)


def memrealtime(b: IRBuilder) -> Value:
    """``s_memrealtime``: the device-wide constant-rate (100 MHz) clock as a
    uniform i64. Comparable across CUs and XCDs."""
    return b.inline_asm(
        "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
        "=s",
        [],
        I64,
        result_name_hint="rtc",
    )


def memtime(b: IRBuilder) -> Value:
    """``s_memtime``: this XCD's shader clock counter as a uniform i64 (not
    comparable across XCDs)."""
    return b.inline_asm(
        "s_memtime $0\ns_waitcnt lgkmcnt(0)", "=s", [], I64, result_name_hint="clk"
    )


def xcc_id(b: IRBuilder) -> Value:
    """``HW_REG_XCC_ID`` low 4 bits: the XCD this wave runs on (gfx94x/gfx950)."""
    raw = b.inline_asm(
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
        "=s",
        [],
        I32,
        result_name_hint="xcc",
    )
    return b.land(raw, b.const_i32(0xF))


def hw_id(b: IRBuilder) -> Value:
    """Raw ``HW_REG_HW_ID`` (wave / SIMD / CU / SH / SE ids, gfx9 layout)."""
    return b.inline_asm(
        "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s", [], I32, result_name_hint="hwid"
    )


def _scope_bits(scope: str) -> str:
    if scope == "agent":
        return "sc1"
    if scope == "system":
        return "sc0 sc1"
    raise ValueError(f"scope must be 'agent' or 'system', got {scope!r}")


def buffer_wbl2(b: IRBuilder, scope: str = "agent") -> None:
    """``buffer_wbl2 sc1`` (agent) / ``sc0 sc1`` (system): write back dirty L2
    lines so another XCD sees them. Does not wait; see :func:`release_l2`."""
    b.inline_asm(f"buffer_wbl2 {_scope_bits(scope)}", "~{memory}", [], None)


def buffer_inv(b: IRBuilder, scope: str = "agent") -> None:
    """``buffer_inv sc1`` (agent) / ``sc0 sc1`` (system): invalidate the vL1D
    and non-coherent L2 lines so later plain loads refetch."""
    b.inline_asm(f"buffer_inv {_scope_bits(scope)}", "~{memory}", [], None)


def release_l2(b: IRBuilder, scope: str = "agent", *, impl: str = ASM) -> None:
    """Release this wave's prior global stores at ``scope``.

    ``impl="asm"`` (default): ``s_waitcnt vmcnt(0); buffer_wbl2;
    s_waitcnt vmcnt(0)`` with a memory clobber.
    ``impl="ir"``: an LLVM ``fence syncscope(scope) release`` (both engines
    lower it; the backend emits the same write-back plus its own waits).
    Formally weaker: under the LLVM memory model a fence synchronizes only
    through an atomic access, and the mailbox ready store that follows is a
    non-atomic store (an asm ``global_store``, or a family's
    ``buffer_store_cp``), so the compiler may legally move it above the fence.
    It works because the backend does not, which is behaviour, not a
    guarantee; keep ``"asm"`` unless measuring.
    Call from one wave only; every wave fencing multiplies the write-back.
    """
    if _check_impl(impl) == IR:
        b.fence(scope=scope, ordering="release")
        return
    bits = _scope_bits(scope)
    b.inline_asm(
        f"s_waitcnt vmcnt(0)\nbuffer_wbl2 {bits}\ns_waitcnt vmcnt(0)",
        "~{memory}",
        [],
        None,
    )


def acquire_l2(b: IRBuilder, scope: str = "agent", *, impl: str = ASM) -> None:
    """Acquire at ``scope`` after a ready flag was observed: later plain loads
    see the producer's released stores. ``impl`` as in :func:`release_l2`
    (``"ir"``: the poll before the fence is non-atomic, so the same caveat
    applies: a later plain load may legally be hoisted above the fence)."""
    if _check_impl(impl) == IR:
        b.fence(scope=scope, ordering="acquire")
        return
    bits = _scope_bits(scope)
    b.inline_asm(f"s_waitcnt vmcnt(0)\nbuffer_inv {bits}", "~{memory}", [], None)


ATOMIC_SCOPES = ("workgroup", "agent", "system")


def atomic_add_scoped(
    b: IRBuilder,
    base: Value,
    voff: Value,
    value: Value,
    *,
    scope: str = "agent",
    returns: bool = False,
) -> Value | None:
    """``global_atomic_add`` at a memory scope (monotonic), asm, both
    engines.

    ``base`` uniform i64 address (SGPR pair), ``voff`` i32 byte offset (VGPR),
    ``value`` i32. Cache bits per the gfx942/gfx950 memory model
    (``AMDGPUUsage`` "atomicrmw monotonic"): ``workgroup`` / ``agent`` none,
    ``system`` ``sc1``. ``IRBuilder.global_atomic_add`` emits an unscoped
    (system) ``atomicrmw``, i.e. always ``sc1``.

    ``returns=True`` returns the pre-add value (``sc0``, the gfx94x return
    bit) and waits ``vmcnt(0)`` inside the asm, since the compiler cannot see
    the load; ``False`` returns ``None`` and does not wait. LDS atomics need
    no scope: ``ds_add_u32`` has no cache bits (``IRBuilder.lds_atomic_add``).
    """
    if scope not in ATOMIC_SCOPES:
        raise ValueError(f"scope must be one of {ATOMIC_SCOPES}, got {scope!r}")
    if base.type != I64 or voff.type != I32 or value.type != I32:
        raise TypeError("atomic_add_scoped expects (i64 base, i32 voff, i32 value)")
    bits = ["sc1"] if scope == "system" else []
    if not returns:
        cp = (" " + " ".join(bits)) if bits else ""
        b.inline_asm(
            f"global_atomic_add $0, $1, $2{cp}",
            "v,v,s,~{memory}",
            [voff, value, base],
            None,
        )
        return None
    cp = " " + " ".join(["sc0", *bits])
    return b.inline_asm(
        f"global_atomic_add $0, $1, $2, $3{cp}\ns_waitcnt vmcnt(0)",
        "=&v,v,v,s,~{memory}",
        [voff, value, base],
        I32,
        result_name_hint="aold",
    )


def fresh(b: IRBuilder, v: Value, reg: str = "v") -> Value:
    """A copy of i32 ``v`` that LLVM cannot merge with another copy or
    rematerialize: an empty side-effecting asm (``; fresh``) tied to ``v`` in
    a ``reg`` register (``"v"`` VGPR, ``"s"`` SGPR; ``"s"`` needs a uniform
    ``v``). ATOM's ``stages.common_wide._opaque``: give each task round its
    own copies of ``tid`` / ``lane`` / ``wave`` so LLVM does not compute the
    rounds' common index math once and hold (spill) it across rounds."""
    if reg not in ("v", "s"):
        raise ValueError(f"reg must be 'v' or 's', got {reg!r}")
    if v.type != I32:
        raise TypeError(f"fresh expects i32, got {v.type}")
    return b.inline_asm(
        "; fresh $0", f"={reg},0", [v], I32, result_name_hint=f"fresh_{reg}"
    )


def fresh_v(b: IRBuilder, v: Value) -> Value:
    """:func:`fresh` in a VGPR."""
    return fresh(b, v, "v")


def fresh_s(b: IRBuilder, v: Value) -> Value:
    """:func:`fresh` in an SGPR (``v`` must be wave-uniform)."""
    return fresh(b, v, "s")


def compiler_fence(b: IRBuilder) -> None:
    """An empty asm with a memory clobber: LLVM moves no memory access across
    it. Emits no instruction."""
    b.inline_asm("", "~{memory}", [], None)


_UMAX_DPP_ASM = "\n".join(
    "s_nop 1\nv_max_u32_dpp $0, $0, $0 " + control
    for control in (
        "row_shr:1 bound_ctrl:0",
        "row_shr:2 bound_ctrl:0",
        "row_shr:4 bound_ctrl:0",
        "row_shr:8 bound_ctrl:0",
        "row_bcast:15 row_mask:0xa",
        "row_bcast:31 row_mask:0xc",
    )
)


def wave_umax_dpp(b: IRBuilder, v: Value) -> Value:
    """Wave-wide unsigned max of an i32 (gfx9 DPP row-shift + row-broadcast
    scan; lane 63 holds the result, returned as a uniform i32)."""
    if v.type != I32:
        raise TypeError(f"wave_umax_dpp expects i32, got {v.type}")
    scanned = b.inline_asm(_UMAX_DPP_ASM, "=v,0", [v], I32, result_name_hint="umax")
    return b.readlane(scanned, b.const_i32(63))


def vec2_i32(b: IRBuilder, lo: Value, hi: Value) -> Value:
    """``<2 x i32> {lo, hi}`` (one 8-byte mailbox pair)."""
    return b.vec_pack([lo, hi], I32)
