# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device ops as IR ops or as inline asm (``FrameworkKnobs.irops``).

The framework, the toy chain and the MX decode GEMV emit a few device ops
through ``rocke.helpers.device_ops``'s inline asm: runtime pointers
(``s_mov_b64``), the XCD id (``s_getreg_b32``), sleeps, clocks, scoped
atomics and the toy chain's L2-touch prefetch. Each helper here takes
``irops``: ``False`` emits that asm (the shipped build), ``True`` the IR op
the Python engine lowers to the same asm (``IRBuilder.s_getreg``,
``s_sleep``, ``memrealtime``, ``global_atomic_add_scoped``,
``l2_touch_load`` / ``l2_touch_drain``) or, for pointers, ``int_to_ptr``.
A compiler that reads rocKE IR (the direct code-object compiler) lowers the
IR ops itself; it refuses the asm templates.

Python engine only when ``irops`` is on (every op it adds is in
``rocke.core.ir.PYTHON_ONLY_OPCODES``).
"""

from __future__ import annotations

from rocke.core.ir import I32, IRBuilder, PtrType, Value
from rocke.helpers import device_ops as dops

__all__ = ["atomic_add_scoped", "l2_touch_drain", "l2_touch_load", "memrealtime", "ptr", "sleep", "xcc_id"]


def ptr(b: IRBuilder, addr: Value, irops: bool, *, pointee=I32) -> Value:
    """A global pointer from a uniform i64 address."""
    if irops:
        return b.int_to_ptr(addr, PtrType(pointee, "global"))
    return dops.ptr_from_i64(b, addr, pointee=pointee)


def xcc_id(b: IRBuilder, irops: bool) -> Value:
    """``device_ops.xcc_id``: the XCD this wave runs on."""
    if irops:
        return b.land(b.s_getreg("XCC_ID"), b.const_i32(0xF))
    return dops.xcc_id(b)


def sleep(b: IRBuilder, units: int, irops: bool) -> None:
    """``device_ops.s_sleep``."""
    if irops:
        if units > 0:
            b.s_sleep(units)
        return
    dops.s_sleep(b, units)


def memrealtime(b: IRBuilder, irops: bool) -> Value:
    """``device_ops.memrealtime``."""
    return b.memrealtime() if irops else dops.memrealtime(b)


def atomic_add_scoped(b: IRBuilder, base: Value, voff: Value, value: Value, irops: bool, *,
                      scope: str = "agent", returns: bool = False) -> Value | None:  # fmt: skip
    """``device_ops.atomic_add_scoped`` (i32 ``value``)."""
    if irops:
        return b.global_atomic_add_scoped(base, voff, value, scope=scope, returns=returns)
    return dops.atomic_add_scoped(b, base, voff, value, scope=scope, returns=returns)


def l2_touch_load(b: IRBuilder, base: Value, voff: Value, irops: bool, v4) -> Value:
    """The toy chain's L2-touch load (``global_load_dwordx4``; ``v4`` the
    ``<4 x i32>`` type)."""
    if irops:
        return b.l2_touch_load(base, voff)
    return b.inline_asm("global_load_dwordx4 $0, $1, $2", "=&v,v,s", [voff, base], v4)


def l2_touch_drain(b: IRBuilder, outs: list[Value], irops: bool) -> None:
    """``s_waitcnt vmcnt(0)`` keeping the L2-touch destinations ``outs``
    allocated until it retires."""
    if irops:
        b.l2_touch_drain(outs)
        return
    b.inline_asm("s_waitcnt vmcnt(0)", ",".join(["v"] * len(outs)), list(outs))
