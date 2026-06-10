# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.instruction_scheduler — C++-backed instruction scheduler.

Re-exports the ``_tensile_writer.subtile.instruction_scheduler`` nanobind
submodule (the data-only slot-placement algorithm ported from
``Tensile.Components.Subtile.InstructionScheduler``) and provides the
``instructionSchedule`` entry point that drives the C++ algorithm with *live*
rocisa instruction objects.

This is the only instruction-scheduling implementation: the pure-Python
slot-placement twin has been removed and the C++ algorithm runs unconditionally
(mirroring SubtileGeometry / TileInfo). There is no opt-in flag and no Python
fallback.

Design
------
The C++ core never touches a rocisa object. The shim:

1. Classifies each live instruction into a data-only ``Instruction`` (kind +
   the waitcnt fields the vmcnt post-pass needs). Instruction kinds the
   slot-placement rules key on (MFMA, ds_read, buffer_load, waitcnt, m0-update)
   map to their dedicated ``InstKind``; every other instruction is ``Other``
   and is placed generically, exactly as the original Python algorithm treated
   instructions that matched none of its isinstance() predicates.
2. Calls the C++ ``schedule`` to get the final emission *order* (a list of
   ``(moduleIndex, instIdx)`` references) plus the per-waitcnt vmcnt deltas.
3. Rebuilds a rocisa ``Module`` by adding the live instructions in that order
   and applying the vmcnt deltas in place — exactly mirroring the original
   ``inst.vlcnt += bufLoadCount`` post-pass.
"""

from tensile_writer import _tensile_writer as _ext

_is = _ext.subtile.instruction_scheduler

InstKind = _is.InstKind
Instruction = _is.Instruction
ModuleRef = _is.ModuleRef
ScheduleResult = _is.ScheduleResult
schedule = _is.schedule

__all__ = [
    "InstKind",
    "Instruction",
    "ModuleRef",
    "ScheduleResult",
    "schedule",
    "classifyInstruction",
    "buildModuleRefs",
    "instructionSchedule",
]


def classifyInstruction(inst):
    """Classify a live rocisa instruction into a data-only ``Instruction``.

    Mirrors the isinstance() predicates of the slot-placement algorithm. Any
    instruction that matches none of them (a plain CommonInstruction, a label,
    a branch, a comment, …) is classified as ``Other`` and placed generically —
    exactly how the original Python algorithm handled such instructions.
    """
    # Imported lazily so importing this module never hard-requires rocisa.
    from rocisa.instruction import (
        SWaitCnt,
        MFMAInstruction,
        MXMFMAInstruction,
        LocalReadInstruction,
        GlobalReadInstruction,
        CommonInstruction,
    )

    if isinstance(inst, (MFMAInstruction, MXMFMAInstruction)):
        return Instruction(InstKind.Mfma)
    if isinstance(inst, LocalReadInstruction):
        return Instruction(InstKind.LocalRead)
    if isinstance(inst, GlobalReadInstruction):
        return Instruction(InstKind.GlobalRead)
    if isinstance(inst, SWaitCnt):
        vlcnt = getattr(inst, "vlcnt", -1)
        adjust = bool(getattr(inst, "adjustVmcnt", True))
        return Instruction(InstKind.WaitCnt, vlcnt, adjust)
    if isinstance(inst, CommonInstruction):
        dst = getattr(inst, "dst", None)
        if dst is not None and getattr(dst, "regType", None) == "m":
            return Instruction(InstKind.M0Update)
    return Instruction(InstKind.Other)


def buildModuleRefs(emittedModules):
    """Build the C++ ``ModuleRef`` model from a list of live EmittedModules."""
    modules = []
    for em in emittedModules:
        insts = [classifyInstruction(i) for i in em.instructions]
        modules.append(
            ModuleRef(em.moduleId, em.opType, em.before, insts)
        )
    return modules


def instructionSchedule(emittedModules):
    """C++-backed instruction scheduler (the only implementation).

    Interleaves non-MFMA instructions between MFMAs using the C++
    slot-placement algorithm and returns a rocisa ``Module`` in the resulting
    emission order, with the waitcnt vmcnt post-pass applied in place.

    Raises ``ValueError`` (from the C++ algorithm) when the chain violates a
    structural precondition such as not having exactly one MFMA module.
    """
    from rocisa.code import Module

    if not emittedModules:
        return Module()

    modules = buildModuleRefs(emittedModules)
    result = schedule(modules)

    # Resolve each (moduleIndex, instIdx) reference back to its live instruction.
    live = [em.instructions for em in emittedModules]
    ordered = [live[mid][idx] for (mid, idx) in result.order]

    # Apply the vmcnt post-pass to the live waitcnt objects, exactly as the
    # original post-pass does (`inst.vlcnt += bufLoadCount`).
    for orderIdx, delta in result.vmcntAdjustments:
        if delta:
            ordered[orderIdx].vlcnt += delta

    out = Module()
    for inst in ordered:
        out.add(inst)
    return out
