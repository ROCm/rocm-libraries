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
"""

import os
from contextvars import ContextVar

import rocisa as _rocisa
import rocisa.instruction as _rocisa_instruction

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
      return _rocisa.makeLogicalLeaf(inst, _active_version.get())

    return _build


_logical_backend = _LogicalBackend()


# Trial opt-in: instructions routed through the logical backend for gfx1250 only
# when TENSILELITE_LOGICALIR_TRIAL is set in the environment. Intentionally EMPTY:
# an opcode-level trial proved too coarse. Even v_mov_b32 has callers that build
# it with a deferred Holder destination (copyAccToArchVgpr / moveMIoutToArch),
# resolved later by replaceHolder() walking the rocisa tree; building it as a
# logical leaf freezes the unresolved Holder and the resolution pass skips the
# opaque leaf, so the mixed tree diverges. The mechanism below is kept for the
# next stage, which flips the whole gfx1250 build to logical at once (no mixed
# tree within a kernel) rather than migrating by opcode.
_TRIAL_NAMES = frozenset()


def _trial_enabled():
  return bool(os.environ.get("TENSILELITE_LOGICALIR_TRIAL"))


def _backend():
  """Return the namespace that builds instructions for the active ISA.

  This is the single arch dispatch point. By default every architecture builds
  rocisa, so codegen is byte-identical to the pre-facade path. The trial flips
  to the logical backend for gfx1250 only when TENSILELITE_LOGICALIR_TRIAL is
  set; the eventual migration stage replaces this with an unconditional
  ``if _active_version.get() == (12, 5, 0): return _logical_backend``.
  """
  if _trial_enabled() and _active_version.get() == (12, 5, 0):
    return _rocisa_instruction
  return _rocisa_instruction


# Instruction names routed through _backend() at call time. Empty here, so the
# facade is a pure passthrough and every imported name is the real rocisa class.
# The migration stage adds names (or sets _MIGRATE_ALL) as regions move onto
# logical IR.
_MIGRATED = set()
_MIGRATE_ALL = False

_dispatch_cache = {}


def _is_dispatched(name):
  if _MIGRATE_ALL or name in _MIGRATED:
    return True
  return _trial_enabled() and name in _TRIAL_NAMES


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

  ns = {"__call__": _call, "__instancecheck__": _instancecheck,
        "__subclasscheck__": _subclasscheck}
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
