# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ``ir`` mailbox implementation (Python engine only).

``rocke.helpers.mailbox`` ships the inline-asm implementation only (both
engines). This module adds the compiler-visible one, built on opcodes in
``rocke.core.ir.PYTHON_ONLY_OPCODES``:

* stores: ``IRBuilder.buffer_store_cp`` (``raw.ptr.buffer.store`` with the
  knob's ``sc1`` / ``sc0 sc1`` / ``nt`` cache bits);
* polls: the ``tile.tagged_poll`` leaf op (decision D1b): one loop of
  ``sc1`` buffer loads per batch that the compiler schedules and whose waits
  it counts, instead of the opaque asm loop's ``vmcnt(0)`` per retry;
* runtime pointers: ``IRBuilder.int_to_ptr`` (:func:`ptr_from_i64_ir`).

Importing this module registers ``"ir"`` with
:func:`rocke.helpers.mailbox.register_impl`, so
``MailboxKnobs(impl="ir")`` (kernel-name suffix ``mbir``) selects it.

Compliance (the family README has the full exemption note): a build that
uses it compiles with ``compile_kernel(..., backend="python")``
(:func:`backend_for`), carries the ``mbir`` suffix and is excluded from the
C++ byte-identity gate. It stays out of ``rocke.helpers`` because
``platform/AGENTS.md`` forbids helpers without a C++ port path.
"""

from __future__ import annotations

from rocke.core.ir import (
    CPOL_NT,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    uses_python_only_ops,
)
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as M

__all__ = ["IR_IMPL", "IrMailboxImpl", "backend_for", "ptr_from_i64_ir"]


def ptr_from_i64_ir(b: IRBuilder, addr: Value, *, pointee=I32) -> Value:
    """``inttoptr`` of a runtime i64 address to ``ptr addrspace(1)``
    (decision D4 option a). LLVM can fold it and keep the pointer in either
    register file; the asm ``device_ops.ptr_from_i64`` cannot."""
    if addr.type != I64:
        raise TypeError(f"ptr_from_i64_ir expects an i64 address, got {addr.type}")
    return b.int_to_ptr(addr, PtrType(pointee, "global"))


class IrMailboxImpl(M.MailboxImpl):
    """``buffer_store_cp`` puts and ``tagged_poll`` polls over a buffer
    resource kept in ``Mailbox.rsrc``."""

    name = dops.IR

    def rsrc_from_ptr(self, b, ptr, num_bytes):
        return b.buffer_rsrc(ptr, num_bytes)

    def rsrc_from_i64(self, b, addr, num_bytes):
        return b.buffer_rsrc(ptr_from_i64_ir(b, addr), num_bytes)

    def store(self, b, mb, voff, data, knobs) -> None:
        if mb.rsrc is None:
            raise ValueError("impl='ir' needs a mailbox built with impl='ir'")
        aux = knobs.cpol | (CPOL_NT if knobs.put_nt else 0)
        b.buffer_store_cp(data, mb.rsrc, voff, b.const_i32(0), aux=aux)

    def poll_batch(self, b, specs, tag, knobs):
        tspecs = []
        for mb, i, n in specs:
            if mb.rsrc is None:
                raise ValueError("impl='ir' needs a mailbox built with impl='ir'")
            off = M.pair_offset(b, i)
            if mb.copy_off is not None:
                off = b.add(off, mb.copy_off)
            tspecs.append((mb.rsrc, off, 2 * n))
        vecs = b.tagged_poll(
            tag, tspecs, aux=knobs.cpol, pause=dops.PAUSES[knobs.pause]
        )
        return [
            [b.vec_extract(v, 2 * q) for q in range(n)]
            for v, (_, _, n) in zip(vecs, specs)
        ]


IR_IMPL = M.register_impl(IrMailboxImpl())


def backend_for(kernel: KernelDef) -> str | None:
    """``"python"`` when ``kernel`` uses a Python-engine-only opcode (an
    ``ir`` mailbox build), else ``None`` (the caller's default engine)."""
    return "python" if uses_python_only_ops(kernel) else None
