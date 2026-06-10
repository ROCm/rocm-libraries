# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.loop_orchestrator — C++ loop-emission facade.

Thin re-export of the ``_tensile_writer.subtile.loop_orchestrator`` nanobind
submodule.  Exposes the C++ ports of the structural loop-emission helpers from
``LogicalScheduler.py`` and the VGPR tile zero-init utility from ``Kernel.py``:

  emit_loop
    Port of ``LogicalScheduler._emitLoop``.  Iterates the
    [partition][subIterK][EmittedModule] Python list and builds a rocisa Module
    by calling the Python ``emit_fn`` (``InstructionEmitter.emit_module``) per
    EmittedModule and optionally routing through ``schedule_fn``
    (``instructionScheduleFromLists``) for interleaved scheduling.

  emit_main_and_exit_loops
    Port of ``LogicalScheduler.emitMainAndExitLoops``.  Builds the complete
    main-loop control-flow structure: optional K<DepthU skip branch, PRELOOP,
    MAINLOOP (with optional unrolling), NGLL (PGR≥2), and NLL exit paths.
    The tail loop is emitted separately by the Python caller.

  emit_tail_loop
    Port of the structural part of ``LogicalScheduler.emitTailLoop``.  Adds
    the TAILLOOP comment, mask_k_init instructions, the TAILLOOP body (via
    ``emit_loop``), and mask_k_done instructions.  The NoTailLoop guard,
    ``_realloc_tail_tiles_flat``, and mask_k_init/done generation remain in
    the Python wrapper.

  init_vgpr_tiles_to_zero
    Port of ``Kernel.initVgprTilesToZero`` + ``_zeroRegRange``.  Takes a list
    of ``(firstReg, totalRegs, isAgpr, tmpVgpr)`` tuples (pool-type grouping
    resolved in Python) and emits MFMA I8 blocks of 16 + scalar
    VMovB32/VAccvgprWrite for the remainder.

Boundary contract: the C++ orchestrator owns no writer state.  All VGPR/SGPR
indices, pool identity, and kernel flags are resolved in Python before the
call.  The ``ModuleBuilder`` argument provides the rocisa object factories.
"""

from tensile_writer import _tensile_writer as _ext

_orch = _ext.subtile.loop_orchestrator

emit_loop                = _orch.emit_loop
emit_main_and_exit_loops = _orch.emit_main_and_exit_loops
emit_tail_loop           = _orch.emit_tail_loop
init_vgpr_tiles_to_zero  = _orch.init_vgpr_tiles_to_zero

__all__ = [
    "emit_loop",
    "emit_main_and_exit_loops",
    "emit_tail_loop",
    "init_vgpr_tiles_to_zero",
]
