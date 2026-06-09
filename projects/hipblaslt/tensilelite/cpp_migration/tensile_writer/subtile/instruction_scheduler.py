# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.instruction_scheduler — C++-backed instruction scheduler.

Re-exports the ``_tensile_writer.subtile.instruction_scheduler`` nanobind
submodule (the data-only slot-placement algorithm ported from
``Tensile.Components.Subtile.InstructionScheduler``) and provides an
``instructionSchedule``-compatible shim that drives the C++ algorithm with
*live* rocisa instruction objects.

Design
------
The C++ core never touches a rocisa object. The shim:

1. Classifies each live instruction into a data-only ``Instruction`` (kind +
   the waitcnt fields the vmcnt post-pass needs).
2. Calls the C++ ``schedule`` to get the final emission *order* (a list of
   ``(moduleIndex, instIdx)`` references) plus the per-waitcnt vmcnt deltas.
3. Rebuilds a rocisa ``Module`` by adding the live instructions in that order
   and applying the vmcnt deltas in place — exactly mirroring the Python
   ``inst.vlcnt += bufLoadCount`` post-pass.

If any instruction cannot be classified (an unsupported / unexpected live
rocisa object), the shim raises :class:`UnsupportedSchedule` so the caller can
fall back to the pure-Python ``instructionSchedule``.
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
    "UnsupportedSchedule",
    "classifyInstruction",
    "buildModuleRefs",
    "instructionSchedule",
]


class UnsupportedSchedule(Exception):
    """Raised when the live emitted-module chain cannot be scheduled in C++.

    Signals the caller to fall back to the pure-Python ``instructionSchedule``
    (e.g. an instruction kind we cannot classify, or a structural precondition
    the C++ algorithm rejects).
    """


def classifyInstruction(inst):
    """Classify a live rocisa instruction into a data-only ``Instruction``.

    Mirrors the isinstance() predicates of the Python scheduler. Raises
    :class:`UnsupportedSchedule` if ``inst`` is not a recognized rocisa
    instruction object (e.g. a nested container or comment item).
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
    raise UnsupportedSchedule(
        f"unschedulable instruction type: {type(inst).__name__}"
    )


def buildModuleRefs(emittedModules):
    """Build the C++ ``ModuleRef`` model from a list of live EmittedModules.

    Raises :class:`UnsupportedSchedule` if any module instruction cannot be
    classified.
    """
    modules = []
    for em in emittedModules:
        insts = [classifyInstruction(i) for i in em.instructions]
        modules.append(
            ModuleRef(em.moduleId, em.opType, em.before, insts)
        )
    return modules


def instructionSchedule(emittedModules):
    """C++-backed, ``instructionSchedule``-compatible scheduler.

    Returns a rocisa ``Module`` whose instruction order matches the pure-Python
    ``Tensile.Components.Subtile.InstructionScheduler.instructionSchedule``.

    Raises :class:`UnsupportedSchedule` (so the caller can fall back) when the
    chain contains an unclassifiable instruction or violates a structural
    precondition the C++ algorithm rejects.
    """
    from rocisa.code import Module

    if not emittedModules:
        return Module()

    modules = buildModuleRefs(emittedModules)
    try:
        result = schedule(modules)
    except (ValueError, RuntimeError) as exc:
        # Structural precondition rejected by the C++ algorithm (e.g. not
        # exactly one MFMA module). Fall back to Python.
        raise UnsupportedSchedule(str(exc)) from exc

    # Resolve each (moduleIndex, instIdx) reference back to its live instruction.
    live = [em.instructions for em in emittedModules]
    ordered = [live[mid][idx] for (mid, idx) in result.order]

    # Apply the vmcnt post-pass to the live waitcnt objects, exactly as the
    # Python post-pass does (`inst.vlcnt += bufLoadCount`).
    for orderIdx, delta in result.vmcntAdjustments:
        if delta:
            ordered[orderIdx].vlcnt += delta

    out = Module()
    for inst in ordered:
        out.add(inst)
    return out
