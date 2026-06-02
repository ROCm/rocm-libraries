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
"""Shim for ``rocisa.base``.

What this file is:
    Mirrors ``rocisa/rocisa/src/base.cpp`` + ``rocisa/include/base.hpp``
    — IR-tree root types and the process-wide ``rocIsa`` singleton state.

What it does (real):
    - ``KernelInfo`` — current kernel ISA + wavefrontSize; picklable for
      ``ParallelMap2`` workers.
    - ``IsaInfo`` — asm/arch/reg/bug capability dicts holder; picklable.
      (Defined here to mirror ``base.hpp:64-70``; re-exported from
      ``rocisa_stinkytofu_adaptor.__init__`` so ``from rocisa import
      IsaInfo`` keeps working.)
    - ``OutputOptions`` — output toggles (just ``outputNoComment`` today);
      picklable.
    - Process-wide state mirror of ``rocisa::rocIsa`` (base.hpp:72-218):
      ``_current_isa`` / ``_is_init`` / ``_assembler_path`` /
      ``_kernel_info`` / ``_data`` (ISA -> IsaInfo) / ``_vgpr_idx`` /
      ``_vgpr_msb`` / ``_current_output_options``. ALL ownership lives
      here; ``rocIsa`` in ``__init__.py`` is a thin forwarding shell.
    - Accessor functions:
      ``init`` / ``isInit`` / ``getIsaInfo`` / ``getAsmCaps`` /
      ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs`` / ``setKernel`` /
      ``setKernelInfo`` / ``getKernel`` / ``getData`` / ``setData`` /
      ``getVgprIdx`` / ``setVgprIdx`` / ``getVgprMsb`` / ``setVgprMsb`` /
      ``getOutputOptions`` / ``setOutputOptions`` / ``outputNoComment``.

Not yet done (dummy):
    - ``IsaVersion`` — used as a marker only today.
    - ``Item``, ``DummyItem`` — IR-tree base nodes; arrive with Commit Y
      of the Phase 4 IR-tree work.

Design note — singleton decomposition:
    Earlier iterations housed ALL ``rocIsa`` state on the god-singleton
    class in ``__init__.py``; code that needed to read a flag had to
    ``rocIsa.getInstance().getXxx()`` even when the call site already
    lived in ``base.py`` (which inverted the dependency direction and
    forced lazy-import + try/except gymnastics to dodge cycles).
    Now state lives here, in the module that already owns the data
    classes (``KernelInfo`` / ``IsaInfo`` / ``OutputOptions``), and
    ``rocIsa`` is a forwarding shell whose only job is to preserve the
    public API surface that KernelWriter / Tensile callers use.

    For C++ parity reference, this module mirrors rocisa C++ as:
        base.py state            <-> rocisa::rocIsa::m_*  (base.hpp:212-217)
        base.py accessor fns     <-> rocisa::rocIsa::*    (base.hpp:88-207)
        __init__.py rocIsa shim  <-> nanobind binding     (base.cpp:104-150)

Thread / process semantics:
    All state below is process-wide (single instance per Python
    interpreter). ``ParallelMap2`` workers get a fresh module-level
    default and pick up Tensile's data via
    ``rocIsa.getInstance().setData(pickled_data)`` /
    ``setOutputOptions(...)`` in the worker init path. NOT thread-local.
    The C++ original IS per-thread (``std::map<std::thread::id, ...>``);
    Tensile / KernelWriter only ever read it back from the same thread
    that wrote it (and across processes goes through pickle), so a
    single per-process value is sufficient here. If true thread-locality
    is ever needed, swap the globals below for ``threading.local()`` —
    the accessor functions are the only places that need updating.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import caps as _caps
from ._dummy import make_dummy_class

_P = "rocisa.base"


IsaVersion = make_dummy_class(f"{_P}.IsaVersion")
Item = make_dummy_class(f"{_P}.Item")
DummyItem = make_dummy_class(f"{_P}.DummyItem")


# Re-export the public ISA key alias from caps for nicer typing downstream.
IsaKey = Tuple[int, int, int]


class OutputOptions:
    """Mirror of ``rocisa::OutputOptions`` (base.hpp:59-62) — mutable, picklable.

    The C++ struct only carries one bool today. We keep the Python shape
    identical so ``rocIsa.getInstance().getOutputOptions().outputNoComment =
    True`` and the subsequent ``setOutputOptions(opts)`` round-trip across
    multiprocessing pickles unchanged.
    """

    __slots__ = ("outputNoComment",)

    def __init__(self, outputNoComment: bool = False) -> None:
        self.outputNoComment = bool(outputNoComment)

    def __repr__(self) -> str:
        return f"OutputOptions(outputNoComment={self.outputNoComment})"

    # Pickle support — needed because Tensile passes this object across the
    # ParallelMap2 fork/spawn boundary.
    def __getstate__(self) -> tuple:
        return (self.outputNoComment,)

    def __setstate__(self, state: tuple) -> None:
        (self.outputNoComment,) = state


class IsaInfo:
    """Mirror of ``rocisa::IsaInfo`` (base.hpp:64-70) — asm/arch/reg/bug dicts.

    Moved here from ``__init__.py`` to honour the rocisa file layout
    (the C++ struct is declared in ``base.hpp`` alongside ``KernelInfo``
    / ``OutputOptions``). ``from rocisa import IsaInfo`` continues to
    work because ``__init__.py`` re-exports the symbol.

    Pickle support so workers spawned by ``ParallelMap2`` can rehydrate
    the data dict via ``setData(getData())``.
    """

    __slots__ = ("asmCaps", "archCaps", "regCaps", "asmBugs")

    def __init__(self, asmCaps, archCaps, regCaps, asmBugs):
        self.asmCaps = asmCaps
        self.archCaps = archCaps
        self.regCaps = regCaps
        self.asmBugs = asmBugs

    def __repr__(self) -> str:
        return (
            f"IsaInfo(asmCaps={self.asmCaps}, archCaps={self.archCaps}, "
            f"regCaps={self.regCaps}, asmBugs={self.asmBugs})"
        )

    def __getstate__(self) -> tuple:
        return (self.asmCaps, self.archCaps, self.regCaps, self.asmBugs)

    def __setstate__(self, state: tuple) -> None:
        self.asmCaps, self.archCaps, self.regCaps, self.asmBugs = state


class KernelInfo:
    """Mirror of ``rocisa::KernelInfo`` (base.hpp:47-57) — per-thread current
    kernel state.

    Only the attributes Tensile actually reads back are typed: ``isa``
    (a 3-tuple) and ``wavefrontSize``.
    """

    __slots__ = ("isa", "wavefrontSize")

    def __init__(self, isa=None, wavefrontSize: int = 0) -> None:
        self.isa = isa
        self.wavefrontSize = int(wavefrontSize)

    def __repr__(self) -> str:
        return f"KernelInfo(isa={self.isa}, wavefrontSize={self.wavefrontSize})"

    def __getstate__(self) -> tuple:
        return (self.isa, self.wavefrontSize)

    def __setstate__(self, state: tuple) -> None:
        self.isa, self.wavefrontSize = state


# ---------------------------------------------------------------------------
# Process-wide state mirror of ``rocisa::rocIsa`` (base.hpp:72-218).
# ---------------------------------------------------------------------------
#
# Module globals ARE the per-process singleton in Python — no Singleton
# class needed. The ``rocIsa`` class in ``__init__.py`` is a forwarding
# shell that preserves the rocisa API surface (KernelWriter / Tensile
# callers can keep doing ``rocIsa.getInstance().getXxx()``); the actual
# data lives down here.
#
# Each global below has a corresponding pair of accessor functions
# whose names track the C++ ``rocisa::rocIsa::*`` methods 1:1. Mutating
# the globals directly from outside this module is supported but
# discouraged — prefer the accessor functions so future refactors
# (e.g. switching to ``threading.local()`` if KernelWriter ever calls
# from worker threads) need only touch this file.
#
# Layout mirrors the order of declarations in rocisa C++ ``base.hpp``
# (lines 210-217) for easy side-by-side comparison.

_current_output_options: "OutputOptions" = OutputOptions()
"""Live ``OutputOptions`` for the codegen comment-suppression flag.
Mirror of ``rocisa::rocIsa::m_outputOptions`` (base.hpp:215)."""

_current_isa: Optional[IsaKey] = None
"""Currently-selected ISA, or ``None`` before ``init()`` / ``setKernel()``.
No direct C++ analogue (C++ derives the current ISA from the per-thread
``m_threads[id].isaVersion``); we keep a separate flag because Tensile
calls ``init()`` before any ``setKernel()`` to populate caps."""

_is_init: bool = False
"""``True`` once ``init()`` has populated at least one ISA's caps.
Mirror of ``rocisa::rocIsa::isInit()`` (base.hpp:103-106), which the
C++ derives from ``m_isainfo.size() > 0``."""

_assembler_path: str = ""
"""Path to the external assembler (``hipcc`` / ``llvm-mc``). Recorded by
``init()`` for parity with C++; the logical adaptor never shells out."""

_kernel_info: "KernelInfo" = KernelInfo()
"""Live ``KernelInfo`` for the currently-selected kernel. Mirror of
``rocisa::rocIsa::m_threads[id]`` (base.hpp:213) collapsed to a single
process-wide slot — see thread-semantics note in the module docstring."""

_data: Dict[IsaKey, "IsaInfo"] = {}
"""ISA-keyed snapshot of capability dicts. Mirror of
``rocisa::rocIsa::m_isainfo`` (base.hpp:214). ``ParallelMap2`` workers
pickle / unpickle this via ``getData()`` / ``setData()``."""

_vgpr_idx: Dict[str, int] = {}
"""Symbol -> base-VGPR-index map shared by ``RegName`` instances.
Mirror of ``rocisa::rocIsa::m_vgpridx[id]`` (base.hpp:216)."""

_vgpr_msb: int = 0
"""Current value of ``s_set_vgpr_msb`` (gfx1250 register-bank prefix).
Mirror of ``rocisa::rocIsa::m_vgprmsb[id]`` (base.hpp:217). Default 0
matches the C++ ``int()`` default-construct of a missing map entry.
``setVgprMsb`` is wired in for Commit Y (``Label.toString`` side
effect, code.hpp:122-125) — no consumer reads it today."""


# ---------------------------------------------------------------------------
# OutputOptions accessors (unchanged from the prior commit).
# ---------------------------------------------------------------------------

def getOutputOptions() -> OutputOptions:
    """Return the process-wide ``OutputOptions`` instance.

    The returned object IS the source of truth: mutating
    ``getOutputOptions().outputNoComment = True`` immediately affects
    every subsequent ``outputNoComment()`` call and every TextBlock
    rendered after this point. That mirrors rocisa C++ behaviour where
    ``rocIsa::getOutputOptions()`` returns a mutable reference.
    """
    return _current_output_options


def setOutputOptions(options: OutputOptions) -> None:
    """Replace the process-wide ``OutputOptions`` instance.

    Used by ``rocIsa.getInstance().setOutputOptions(...)`` -- which is
    what Tensile / ParallelMap2 workers call to ship a pickled
    ``OutputOptions`` from the parent into the worker process.
    """
    global _current_output_options
    _current_output_options = options


def outputNoComment() -> bool:
    """Return whether the codegen should suppress all comments / TextBlocks.

    Used by ``code.TextBlock.toString`` (gates the entire text payload,
    mirroring rocisa ``code.hpp:154-159``) and ``instruction._fmt_str``
    (drops the per-instruction ``// comment`` tail). One-line direct
    read of the module-level singleton: no lazy import, no try/except,
    no defensive ``getattr`` -- the state object is guaranteed to exist
    because we eagerly construct it at module load.
    """
    return _current_output_options.outputNoComment


# ---------------------------------------------------------------------------
# ISA init & active-ISA accessors.
# ---------------------------------------------------------------------------
#
# These four collectively mirror ``rocisa::rocIsa::init`` /
# ``isInit`` / ``getIsaInfo`` and the four ``get{Asm,Arch,Reg}Caps`` /
# ``getAsmBugs`` proxies (base.hpp:88-155).

def init(arch: Any, assemblerPath: str = "", debug: bool = False) -> None:
    """Mirror of ``rocisa::rocIsa::init`` (base.hpp:88-101).

    Selects an ISA, lazily populates its capability snapshot from
    ``caps.getCaps``, records the assembler path, and marks the
    singleton as initialised. Idempotent for the same ISA — the C++
    version short-circuits on ``m_isainfo.find(isaVersion) !=
    m_isainfo.end()``, we mirror that with ``if key not in _data``.

    ``debug`` is accepted for signature parity but ignored (the logical
    adaptor never invokes ``initAsmCaps`` / ``llvm-mc``).
    """
    global _current_isa, _assembler_path, _is_init
    key = _caps.normalize_isa_key(arch)
    _current_isa = key
    _assembler_path = assemblerPath
    if key not in _data:
        asm, arch_c, reg, bugs = _caps.getCaps(key)
        _data[key] = IsaInfo(asm, arch_c, reg, bugs)
    _is_init = True
    # ``debug`` is accepted for API parity only.
    del debug


def isInit() -> bool:
    """Mirror of ``rocisa::rocIsa::isInit`` (base.hpp:103-106).

    C++ derives this from ``m_isainfo.size() > 0``; we use the explicit
    ``_is_init`` flag so ``init()`` and ``setData()`` can both flip it
    deterministically.
    """
    return _is_init


def getIsaInfo(arch: Any) -> "IsaInfo":
    """Mirror of ``rocisa::rocIsa::getIsaInfo`` (base.hpp:125-135).

    Lazy-populates the cache so callers can ask for any supported ISA
    without a prior ``init()`` call (Tensile's
    ``Tensile.Common.Capabilities.makeIsaInfoMap`` relies on this).
    """
    key = _caps.normalize_isa_key(arch)
    info = _data.get(key)
    if info is None:
        asm, arch_c, reg, bugs = _caps.getCaps(key)
        info = IsaInfo(asm, arch_c, reg, bugs)
        _data[key] = info
    return info


def _activeCaps() -> Tuple[Dict, Dict, Dict, Dict]:
    """Internal: return ``(asmCaps, archCaps, regCaps, asmBugs)`` for the
    currently-selected ISA. Used by the four public ``get*Caps`` helpers.

    Raises ``RuntimeError`` if no ISA has been selected yet so a missing
    ``init()`` / ``setKernel()`` produces a loud error rather than a
    confusing ``KeyError`` deep inside instruction emission.
    """
    if _current_isa is None:
        raise RuntimeError(
            "rocisa.base: init(arch, ...) or setKernel(arch, ...) must "
            "be called before getAsmCaps()/getArchCaps()/getRegCaps()/"
            "getAsmBugs()."
        )
    info = _data.get(_current_isa)
    if info is None:
        asm, arch_c, reg, bugs = _caps.getCaps(_current_isa)
        info = IsaInfo(asm, arch_c, reg, bugs)
        _data[_current_isa] = info
    return (info.asmCaps, info.archCaps, info.regCaps, info.asmBugs)


def getAsmCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getAsmCaps`` (base.hpp:137-140)."""
    return _activeCaps()[0]


def getArchCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getArchCaps`` (base.hpp:147-150)."""
    return _activeCaps()[1]


def getRegCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getRegCaps`` (base.hpp:142-145)."""
    return _activeCaps()[2]


def getAsmBugs() -> Dict[str, bool]:
    """Mirror of ``rocisa::rocIsa::getAsmBugs`` (base.hpp:152-155)."""
    return _activeCaps()[3]


# ---------------------------------------------------------------------------
# KernelInfo accessors.
# ---------------------------------------------------------------------------

def setKernel(arch: Any, wavefrontSize: int) -> None:
    """Mirror of ``rocisa::rocIsa::setKernel`` (base.hpp:108-118).

    Pins the active ISA AND records the wavefront size in the live
    ``KernelInfo``. NOTE: the C++ version ALSO resets ``m_vgpridx[id]``
    and ``m_vgprmsb[id]`` on every call (base.hpp:115-116). The Python
    adaptor historically did NOT mirror that reset; preserving that
    omission here to keep this commit a pure state-sink (no behaviour
    change). Re-evaluate when wiring Label.toString in Commit Y.
    """
    global _current_isa, _kernel_info
    key = _caps.normalize_isa_key(arch)
    _current_isa = key
    _kernel_info = KernelInfo(isa=key, wavefrontSize=wavefrontSize)


def setKernelInfo(info: KernelInfo) -> None:
    """Replace the live ``KernelInfo`` directly. No C++ equivalent.

    Used by test harnesses that need to restore a previously captured
    ``KernelInfo`` -- in particular the initial ``KernelInfo()`` state
    where ``info.isa is None``, which ``setKernel`` cannot represent
    (it requires a concrete ISA tuple).
    """
    global _kernel_info
    _kernel_info = info


def getKernel() -> KernelInfo:
    """Mirror of ``rocisa::rocIsa::getKernel`` (base.hpp:120-123)."""
    return _kernel_info


# ---------------------------------------------------------------------------
# Data-dict accessors (pickled across ParallelMap2 worker boundary).
# ---------------------------------------------------------------------------

def getData() -> Dict[IsaKey, "IsaInfo"]:
    """Mirror of ``rocisa::rocIsa::getData`` (base.hpp:157-160).

    Returns the live dict (NOT a copy). Tensile pickles the result and
    ships it to ``ParallelMap2`` workers, which call ``setData`` to
    rehydrate. Mutating the returned dict mutates the process-wide
    state -- usually what you want.
    """
    return _data


def setData(data: Dict[IsaKey, "IsaInfo"]) -> None:
    """Mirror of ``rocisa::rocIsa::setData`` (base.hpp:172-175).

    Replaces the cap snapshot table wholesale. Also flips ``_is_init``
    True/False based on emptiness so ``isInit()`` matches the C++
    behaviour (``m_isainfo.size() > 0``).
    """
    global _data, _is_init
    _data = dict(data)
    _is_init = bool(_data)


# ---------------------------------------------------------------------------
# VGPR-index map accessors.
# ---------------------------------------------------------------------------

def getVgprIdx() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getVgprIdx`` (base.hpp:162-165).

    Returns the live dict so callers can ``getVgprIdx()[name]`` for
    a single lookup AND ``getVgprIdx().clear()`` for a process-wide
    reset (used by test harnesses). The C++ version returns by value
    so ``.clear()`` would not propagate there; Python dict semantics
    make the mutation visible, which matches what callers actually
    want.
    """
    return _vgpr_idx


def setVgprIdx(name: str, idx: int) -> None:
    """Mirror of ``rocisa::rocIsa::setVgprIdx`` (base.hpp:191-198)."""
    _vgpr_idx[name] = idx


# ---------------------------------------------------------------------------
# VGPR-MSB accessors.
# ---------------------------------------------------------------------------

def getVgprMsb() -> int:
    """Mirror of ``rocisa::rocIsa::getVgprMsb`` (base.hpp:167-170).

    Default ``0`` matches the C++ ``int()`` default-construct of an
    absent ``m_vgprmsb[id]`` entry.
    """
    return _vgpr_msb


def setVgprMsb(msb: int) -> None:
    """Mirror of ``rocisa::rocIsa::setVgprMsb`` (base.hpp:200-207).

    Used by ``Label::toString`` (code.hpp:122-125) as a side effect
    when the active ISA has ``HasVgprMSB``. No Python consumer reads
    it today; the accessor is wired in here so Commit Y (Label real
    implementation) can pick it up without further base.py changes.
    """
    global _vgpr_msb
    _vgpr_msb = int(msb)
