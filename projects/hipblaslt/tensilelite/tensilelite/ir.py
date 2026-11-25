################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Instruction IR facade: the single chokepoint for the rocisa -> StinkyTofu
logical-IR migration.

Every codegen file imports instruction classes from here
(``from .ir import VMovB32, ...``) instead of directly from
``rocisa.instruction``. Today this is a pure passthrough to rocisa, so the
emitted assembly is byte-identical to the pre-facade path: every imported name
is the real rocisa class, and ``isinstance`` / ``issubclass`` keep the native
fast path.

When an architecture's codegen moves onto logical IR, only this module changes:

  * ``_backend()`` picks the construction backend for the ISA in the current
    context -- the single "if gfx1250 -> logical" decision point, and
  * ``_MIGRATED`` / ``_MIGRATE_ALL`` choose which instruction names are routed
    through that decision at call time.

A migrated name becomes a stable call-time dispatcher: ``X(...)`` builds via the
active backend, while ``isinstance`` / ``issubclass`` delegate to the active
backend's class so type checks keep working across the migration. Names that are
not migrated stay the real rocisa class and pay no dispatch cost.

Migration happens per *region*, not per opcode. ``logical_region()`` marks a
block of codegen (e.g. ``globalOffset``) whose dispatched instructions should
build logical IR; outside the region a dispatched name still builds rocisa, so
the same opcode used elsewhere in the kernel is unaffected. While a region's
logical lowering is being validated, the logical backend is gated behind the
``TENSILELITE_LOGICALIR_TRIAL`` env var so the two run-yaml passes (trial off vs
on) A/B compare; the final migration stage drops that gate and flips gfx1250 to
logical unconditionally.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

import rocisa as _rocisa
import rocisa.instruction as _rocisa_instruction
import rocisa.code as _rocisa_code
import rocisa.container as _rocisa_container

# ISA ([major, minor, stepping]) being generated in the current context, set by
# KernelWriter around kernel generation. A ContextVar (not a mutable module
# global) so the choice is correct under Parallel.py's thread backends. None
# means "no override".
_active_version: ContextVar = ContextVar("tensilelite_ir_active_version", default=None)


def set_active_version(version):
  """Record the ISA being generated so dispatched instructions can pick a
  backend. Harmless while every architecture still builds rocisa. Returns a
  token usable with reset_active_version()."""
  return _active_version.set(tuple(version) if version is not None else None)


def reset_active_version(token):
  """Restore the ISA recorded before the matching set_active_version() call."""
  _active_version.reset(token)


def active_version():
  """The ISA recorded for the current context, or None."""
  return _active_version.get()


# Concrete arch/stepping name (StinkyTofuArchName, e.g. "gfx1250-strict") being
# generated in the current context, set by KernelWriter alongside the ISA. Some
# steppings share an ISA triple (gfx1250 v0/v1 are both 12,5,0) but resolve to
# distinct GfxArchIDs via resolveArchId(arch, archName); an empty name means the
# base triple. A logical leaf built by makeLogicalLeaf MUST resolve its archId
# with the SAME name the lazy converter (toStinkyTofuModule / ToStinkyAsmPass)
# uses, otherwise an arch-sensitive opcode (e.g. MFMA/WMMA on a colliding
# stepping) lowers differently than the rocisa it replaces. A ContextVar, like
# _active_version, so it is correct under Parallel.py's thread backends.
_active_arch_name: ContextVar = ContextVar("tensilelite_ir_active_arch_name", default="")


def set_active_arch_name(name):
  """Record the concrete arch/stepping name (StinkyTofuArchName) for the current
  context so logical leaves resolve the same archId as the lazy converter.
  Returns a token usable with reset_active_arch_name()."""
  return _active_arch_name.set(str(name) if name else "")


def reset_active_arch_name(token):
  """Restore the arch name recorded before the matching set_active_arch_name()."""
  _active_arch_name.reset(token)


def active_arch_name():
  """The concrete arch/stepping name recorded for the current context, or ""."""
  return _active_arch_name.get()


# Whether the current context is inside a region whose dispatched instructions
# should build logical IR (for a logical-capable ISA). A ContextVar, like
# _active_version, so nesting and Parallel.py's thread backends stay correct.
# False everywhere by default -> dispatched names still build rocisa.
_region_active: ContextVar = ContextVar("tensilelite_ir_logical_region", default=False)


@contextmanager
def logical_region(name=None):
  """Mark the enclosed codegen as a logical-IR migration region.

  Inside this block, instruction names in ``_MIGRATED`` build via the logical
  backend when the active ISA is logical-capable (and, while a region is still
  under validation, when TENSILELITE_LOGICALIR_TRIAL is set). Outside it the
  same names build rocisa, so a migrated opcode used elsewhere in the kernel is
  byte-identical to the pre-facade path. ``name`` is advisory, for debugging.

  Harmless on architectures that still build rocisa: the region flag is set but
  ``_backend()`` ignores it, so wrapping a block costs nothing there.
  """
  token = _region_active.set(True)
  try:
    yield
  finally:
    _region_active.reset(token)


def region_active():
  """Whether the current context is inside a logical_region()."""
  return _region_active.get()


# Some kernels still run *in-path* passes that do rocisa-specific surgery on the
# already-emitted instructions and cannot yet operate on a LogicalLeaf (that is
# Phase 4 lowering work):
#   * KernelWriter.remapMXScaleInterleave -- rewrites register operands
#     (getParams()/mxsa/mxsb/regName) of every instruction for MX scale
#     interleave; runs when self.states.mxsInterleave is set.
#   * Components/SIA.splitDSInstructionIntoSmaller -- rebuilds a ds_store_b128
#     into smaller rocisa stores via getParams()/.ds (dormant for large-MI, but
#     guarded fail-loud).
# A kernel that needs such a pass must stay pure rocisa until the pass is
# lowered -- exactly like the subtile path. KernelWriter sets this per kernel;
# when True, no region produces leaves, so the kernel is byte-identical to the
# pre-facade path. A ContextVar, like the others, so Parallel.py threads stay
# correct.
_kernel_logical_disabled: ContextVar = ContextVar(
    "tensilelite_ir_kernel_logical_disabled", default=False)


def set_kernel_logical_disabled(disabled):
  """Force the current kernel to stay pure rocisa (an un-lowered in-path pass
  needs real rocisa instructions). Set by KernelWriter per kernel; returns a
  token usable with reset_kernel_logical_disabled()."""
  return _kernel_logical_disabled.set(bool(disabled))


def reset_kernel_logical_disabled(token):
  _kernel_logical_disabled.reset(token)


def kernel_logical_disabled():
  """Whether the current kernel is forced to pure rocisa."""
  return _kernel_logical_disabled.get()


# ============================================================================
# Phase 1 scheduling introspection seam
# ============================================================================
# The in-path scheduler (KernelWriter.makeSchedule / _makeSubIterSchedule and
# Components/SIA) classifies instructions by concrete rocisa type via
# ``isinstance``. That hard-wires the scheduler to rocisa and makes a logical
# leaf (which is not any rocisa instruction type) invisible -> miscounted ->
# divergence. These ``sched_is_*`` predicates are the single place that asks
# "what kind of instruction is this?".
#
# Phase 1: each predicate wraps the *exact* rocisa ``isinstance`` it replaces, so
# the emitted assembly is byte-identical. Only the call site changes.
# Phase 2: these grow a branch that answers from a logical leaf's payload
# (``logical::Opcode`` / ``InstFlag``), so the scheduler can introspect a mixed
# rocisa/logical tree without knowing which node is which.
#
# Vocabulary note: the category names mirror the logical-IR taxonomy on purpose
# (so the Phase 2 leaf branch maps 1:1 to a logical opcode/flag).

# Phase 2 leaf answers: a LogicalLeaf reports its wrapped logical instruction's
# scheduling category (coarse) and opcode name (fine) via the C++ seam. The
# predicates below consult these so a mixed rocisa/logical tree schedules the
# same way an all-rocisa tree does.
_LogicalLeaf = _rocisa_code.LogicalLeaf
_SchedCategory = _rocisa_code.SchedCategory


def _leaf_category(item):
  """SchedCategory of a logical leaf, or None when item is not a leaf."""
  if isinstance(item, _LogicalLeaf):
    return item.schedCategory()
  return None


def _leaf_opcode(item):
  """Logical opcode name of a leaf ("DSLoadB128", ...), or None when not a leaf."""
  if isinstance(item, _LogicalLeaf):
    return item.schedOpcodeName()
  return None


def sched_is_mfma(item):
  """True for a plain MFMA instruction (not SMFMA / MXMFMA)."""
  return (isinstance(item, _rocisa_instruction.MFMAInstruction)
          or _leaf_opcode(item) == "MFMA")


def sched_is_smfma(item):
  """True for a scaled-MFMA (SMFMA) instruction."""
  return (isinstance(item, _rocisa_instruction.SMFMAInstruction)
          or _leaf_opcode(item) == "SMFMA")


def sched_is_mxmfma(item):
  """True for a mixed-precision MFMA (MXMFMA) instruction."""
  return (isinstance(item, _rocisa_instruction.MXMFMAInstruction)
          or _leaf_opcode(item) == "MXMFMA")


def sched_is_tensor_load_to_lds(item):
  """True for a tensor_load_to_lds (TDM direct-to-LDS) instruction."""
  return (isinstance(item, _rocisa_instruction.TensorLoadToLds)
          or _leaf_opcode(item) == "TensorLoadToLds")


def sched_is_ds_load(item):
  """True for any LDS (DS) read instruction."""
  return (isinstance(item, _rocisa_instruction.DSLoadInstruction)
          or _leaf_category(item) == _SchedCategory.LocalRead)


def sched_is_ds_load_b128(item):
  """True specifically for a 128-bit LDS read (ds_load_b128)."""
  return (isinstance(item, _rocisa_instruction.DSLoadB128)
          or _leaf_opcode(item) == "DSLoadB128")


def sched_is_ds_store(item):
  """True for any LDS (DS) write instruction."""
  return (isinstance(item, _rocisa_instruction.DSStoreInstruction)
          or _leaf_category(item) == _SchedCategory.LocalWrite)


def sched_is_ds_store_b128(item):
  """True specifically for a 128-bit LDS write (ds_store_b128)."""
  return (isinstance(item, _rocisa_instruction.DSStoreB128)
          or _leaf_opcode(item) == "DSStoreB128")


def sched_is_waitcnt(item):
  """True for an s_waitcnt instruction."""
  return (isinstance(item, _rocisa_instruction.SWaitCnt)
          or _leaf_opcode(item) == "SWaitCnt")


def sched_is_nop(item):
  """True for an s_nop instruction."""
  return (isinstance(item, _rocisa_instruction.SNop)
          or _leaf_opcode(item) == "SNop")


def sched_is_instruction(item):
  """True for a schedulable leaf instruction (as opposed to a Module subtree).

  This is the structural "is this an instruction, not a container" test the
  scheduler uses when walking a module's items. A logical leaf answers True
  here even though it is not a rocisa ``Instruction``."""
  return isinstance(item, (_rocisa_instruction.Instruction, _LogicalLeaf))


def sched_is_module(item):
  """True for a Module subtree (a container to recurse into), as opposed to a
  schedulable leaf. A logical leaf answers False, like any non-Module item."""
  return isinstance(item, _rocisa_code.Module)


def sched_is_register_container(item):
  """True for a register operand (RegisterContainer) the scheduler inspects for
  read/write dependency overlap."""
  return isinstance(item, _rocisa_container.RegisterContainer)


class _LogicalBackend:
  """Construction backend that builds StinkyTofu logical IR instead of rocisa.

  ``getattr(_logical_backend, "VMovB32")(dst, src, comment)`` builds the rocisa
  instruction, lowers it through ``rocisa.makeLogicalLeaf`` (the exact
  ``createLogicalFromRocisa`` path the lazy converter runs at
  ``toStinkyTofuModule`` time) and returns a ``rocisa.LogicalLeaf`` carrier. The
  leaf is a real ``rocisa.Item`` so it drops straight into the module tree, and
  its payload passes through the converter unchanged -- byte-identical to
  converting the same rocisa instruction lazily.
  """

  def __getattr__(self, name):
    real = getattr(_rocisa_instruction, name)

    def _build(*args, **kwargs):
      inst = real(*args, **kwargs)
      return _rocisa.makeLogicalLeaf(inst, _active_version.get(), _active_arch_name.get())

    return _build


_logical_backend = _LogicalBackend()


# ISAs whose codegen regions may build logical IR. Only gfx1250 for now.
_LOGICAL_ISAS = frozenset({(12, 5, 0)})

# Dev A/B gate. While a region's logical lowering is being validated, the
# logical backend is engaged only when this env var is set, so run-yaml.sh can
# compare the trial-off pass (pure rocisa) against the trial-on pass (region
# builds logical) for the same config. Step 9 removes this gate and flips the
# logical ISAs to logical unconditionally.
def _trial_enabled():
  return bool(os.environ.get("TENSILELITE_LOGICALIR_TRIAL"))


def _backend():
  """Return the namespace that builds instructions for the active context.

  This is the single arch/region dispatch point. A dispatched name builds via
  the logical backend only when all of the following hold:

    * the active ISA is logical-capable (gfx1250), and
    * we are inside a ``logical_region()``, and
    * the trial A/B gate is enabled (dropped in the final stage).

  Otherwise it builds rocisa, so codegen is byte-identical to the pre-facade
  path everywhere else. The eventual migration stage drops the trial gate:
  ``if _active_version.get() in _LOGICAL_ISAS and _region_active.get(): ...``.
  """
  if (_active_version.get() in _LOGICAL_ISAS
      and _region_active.get()
      and not _kernel_logical_disabled.get()
      and _trial_enabled()):
    return _logical_backend
  return _rocisa_instruction


def maybe_logical_leaf(inst):
  """Lower an already-built rocisa instruction to a LogicalLeaf when in-region.

  Some codegen sites do not construct through a dispatched name -- they receive a
  concrete rocisa instruction class and build it directly (e.g. the local-read
  emitter in ``Component._emitLdsRead``, whose ds_load class comes from
  ``MemoryInstruction.getInst()``, and which must set the LDS mem token on the
  instruction *after* construction). For those, construct + mutate the rocisa
  instruction as usual and pass it through here as the final step: inside a
  ``logical_region()`` on a logical ISA (trial-gated) it is lowered to a
  ``rocisa.LogicalLeaf`` via the same ``createLogicalFromRocisa`` path the lazy
  converter uses (so it is byte-identical to converting it later); everywhere
  else -- and on the subtile path, which never opens a region -- it is returned
  unchanged, so codegen stays pure rocisa.

  Set any post-construction state (mem token, etc.) on ``inst`` *before* calling
  this, so the eager lowering captures it.
  """
  if (_active_version.get() in _LOGICAL_ISAS
      and _region_active.get()
      and not _kernel_logical_disabled.get()
      and _trial_enabled()):
    return _rocisa.makeLogicalLeaf(inst, _active_version.get(), _active_arch_name.get())
  return inst


# Instruction names installed as call-time dispatchers. A dispatched name still
# builds rocisa unless _backend() routes it to logical (ISA + region + trial), so
# adding a name here is safe even when the same opcode is used outside any
# logical_region(): only calls inside a region on a logical ISA change.
#
# Step 1: the gfx1250 FFM global-read path is TDM (tensor direct-to-LDS), not the
# classic graTileAssignment/globalOffset helpers (those are the non-TDM `else`
# branch and are never on this arch's hot path). tdmGlobalOffset() ->
# TensorDataMover.calculateStartAddr() and initTDMDescriptor() build a
# self-contained, concrete-operand address/descriptor block from exactly these
# opcodes (no deferred Holder / memToken / branch -- control flow and the LDS
# load live in other, un-migrated methods of the component). They dispatch to the
# logical backend only inside those helpers' logical_region(); everywhere else --
# including the subtile path, which emits pure rocisa via str(module) and never
# runs the converter -- they still build rocisa.
#
# The set covers the clean scalar-ALU opcodes the TDM descriptor/offset block
# emits directly and via the TensorDataMover setters it calls inside the region
# (setTensorDim0/1, setPadding, setTensorStride0, ... -> s_and/s_or/s_lshl on top
# of the s_mov/s_add/s_mul/s_sub/s_cselect/readfirstlane already here). Every one
# is a concrete-operand write with no deferred Holder / memToken / branch, so it
# lowers to a self-contained logical leaf. Branch/load/waitcnt opcodes that can
# appear in the same region (e.g. the general-batched path in _resolveTDMGlobalAddr)
# are deliberately NOT migrated and stay rocisa; the logical passes treat them as
# lane-reset boundaries. Opcodes used only by not-yet-migrated siblings (e.g.
# s_mul_hi in tdmApplyTileKOffset) are added when those methods join a region.
# The mfma region (mfmaIter, wrapped in logical_region("mfma")) emits the matrix
# ops directly as MFMA/SMFMA/MXMFMA; each lowers to a self-contained logical leaf
# (concrete acc/a/b operands, no deferred Holder/memToken). The three rocisa
# classes override only schedCategory (-> MFMA); the leaf matches via the MFMA
# entries in schedCategoryForOpcode. The non-matrix scalar ALUs the same region
# may emit are already covered below. Subtile stays pure rocisa (the region
# decorator is gated on UseSubtileImpl).
_MIGRATED = {
    "SMovB32", "SMovB64", "SLShiftRightB32", "SLShiftLeftB32",
    "SMulI32", "SAddU32", "SAddCU32", "SSubI32", "SSubU32",
    "SAndB32", "SOrB32", "SCSelectB32", "VReadfirstlaneB32",
    "MFMAInstruction", "SMFMAInstruction", "MXMFMAInstruction",
}
_MIGRATE_ALL = False

_dispatch_cache = {}


def _is_dispatched(name):
  return _MIGRATE_ALL or name in _MIGRATED


def _make_dispatcher(name, real):
  """Build a stable call-time dispatcher class for instruction *name*.

  - ``X(...)``              -> construct via the backend chosen at call time.
  - ``isinstance(obj, X)``  -> delegate to the active backend's class.
  - ``issubclass(sub, X)``  -> delegate to the active backend's class.

  The dispatcher subclasses the real rocisa class so ``issubclass(X, Base)``
  (with the dispatcher itself as the first argument) also stays correct. If the
  real class forbids Python subclassing (nb::is_final), fall back to an
  independent class; only that rare first-argument case loses precision then.
  """
  def _call(cls, *args, **kwargs):
    return getattr(_backend(), name)(*args, **kwargs)

  def _instancecheck(cls, obj):
    target = getattr(_backend(), name)
    # The logical backend exposes a factory callable, not a class; type checks
    # fall back to the real rocisa class so isinstance stays non-crashing.
    if not isinstance(target, type):
      target = real
    return isinstance(obj, target)

  def _subclasscheck(cls, sub):
    target = getattr(_backend(), name)
    if not isinstance(target, type):
      target = real
    return issubclass(sub, target)

  def _getattr(cls, attr):
    # Class-level attribute / static-method access on the dispatcher (e.g.
    # X.issueLatency). The subclassing path already inherits these from real via
    # the MRO; this is the safety net for the nb::is_final fallback below, where
    # the dispatcher does NOT subclass real and would otherwise lose them.
    return getattr(real, attr)

  ns = {"__call__": _call, "__instancecheck__": _instancecheck,
        "__subclasscheck__": _subclasscheck, "__getattr__": _getattr}
  try:
    meta = type("_IRDispatchMeta_" + name, (type(real),), ns)
    return meta(name, (real,), {})
  except TypeError:
    meta = type("_IRDispatchMeta_" + name, (type,), ns)
    return meta(name, (), {})


def __getattr__(name):
  # PEP 562 module-level __getattr__: resolve instruction names lazily. Let real
  # dunder misses raise normally rather than proxying them to rocisa.
  if name.startswith("__") and name.endswith("__"):
    raise AttributeError(name)
  real = getattr(_rocisa_instruction, name)
  if not isinstance(real, type) or not _is_dispatched(name):
    # Non-class (function/constant) or not migrated: hand back the real object so
    # construction and type checks use rocisa's native fast path.
    return real
  disp = _dispatch_cache.get(name)
  if disp is None:
    disp = _make_dispatcher(name, real)
    _dispatch_cache[name] = disp
  return disp
