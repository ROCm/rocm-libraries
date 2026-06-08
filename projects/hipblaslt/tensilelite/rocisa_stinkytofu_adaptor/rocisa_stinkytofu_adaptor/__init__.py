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
"""``rocisa_stinkytofu_adaptor`` — Tensilelite ``rocisa`` shim on top of stinkytofu.

What this file is:
    Top-level package that mimics ``projects/hipblaslt/tensilelite/
    rocisa/rocisa`` (the nanobind C++ bindings) so KernelWriter callers
    can keep using ``from rocisa import ...`` unchanged. Activated only
    when ``ROCISA_BACKEND=stinkytofu`` (see
    ``projects/hipblaslt/tensilelite/rocisa/rocisa/__init__.py``);
    reserved for ``gfx1250`` today.

    Lives at ``projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/``
    as a sibling of ``projects/hipblaslt/tensilelite/rocisa/`` —
    intentionally a tensilelite-internal alternative backend rather than
    a piece of ``shared/stinkytofu/``, because this is a *consumer* of
    stinkytofu's Python binding (the dependency direction is
    tensilelite → adapter → stinkytofu, never the other way). Transient:
    planned to be deleted / drastically shrunk once gfx1250 KernelWriter
    is rewritten against a real stinkytofu service API.

What it does (real):
    - Re-exports ``IsaInfo`` from ``base.py`` (the class definition
      lives there to mirror ``rocisa/include/base.hpp:64-70``).
    - ``rocIsa`` — singleton forwarding shell mirroring the C++ class.
      Every method ``init`` / ``isInit`` / ``getIsaInfo`` /
      ``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs``
      / ``setKernel`` / ``getKernel`` / ``getOutputOptions`` /
      ``setOutputOptions`` / ``getData`` / ``setData`` /
      ``getVgprIdx`` / ``setVgprIdx`` / ``getVgprMsb`` / ``setVgprMsb``
      forwards to the module-level state in ``base.py``. The class is
      kept (rather than collapsed into module functions) purely to
      preserve the ``rocIsa.getInstance().method(...)`` call shape that
      KernelWriter / Tensile / the C++ binding all expose.
    - Submodules with real implementations: ``register`` (pool),
      ``enum`` (real ``IntEnum``s), ``base`` (state + accessors,
      ``KernelInfo`` / ``IsaInfo`` / ``OutputOptions``), ``caps``
      (gfx1250 snapshot).
    - Submodule registration as ``rocisa.<submodule>`` for ``base``,
      ``enum``, ``container``, ``code``, ``label``, ``instruction``,
      ``functions``, ``asmpass``, ``macro``, ``register``.

Not yet done (dummy):
    - ``getGlcBitName`` / ``getSlcBitName`` (real; gfx1250 asm caps).
    - Counters: ``count*``, ``find*``, ``getMFMAs``.
    - Interop hooks: ``isSupportedByStinkyTofu``, ``StinkyAsmModule``,
      ``toStinkyTofuModule`` — should delegate into compiled stinkytofu
      bindings once the dummy phase ends.
    - Submodules still all-dummy: ``code``, ``label``,
      ``instruction``, ``functions``, ``asmpass``, ``macro``.
    - ``container``: register-reference layer + ``Container`` ABC,
      hardware tokens, ``MemTokenData``, and ``*Modifiers``.

Design note — singleton state ownership:
    All process-wide state that the C++ ``rocisa::rocIsa`` singleton
    holds (KernelInfo / IsaInfo dict / current ISA / vgpr_idx /
    vgpr_msb / OutputOptions / is_init / assembler_path) lives as
    module-level globals in ``base.py``. The ``rocIsa`` class below
    holds NO instance state -- every method is a one-line forwarder.
    This breaks the prior cycle where ``base.py`` had to reach back
    into the package facade via lazy imports to read its own flags,
    and keeps the dependency direction one-way (``__init__`` depends
    on ``base``, never the reverse).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from . import base as _base
from . import caps as _caps
from ._dummy import make_dummy_class, make_dummy_func
from .base import IsaInfo, KernelInfo, OutputOptions

# Make submodules importable as attributes (``rocisa.code`` etc.). The
# rocisa dispatcher in ``tensilelite/rocisa/rocisa/__init__.py`` is what
# ultimately installs them under the ``rocisa.*`` name in ``sys.modules``.
from . import asmpass as asmpass
from . import base as base
from . import code as code
from . import container as container
from . import enum as enum
from . import functions as functions
from . import instruction as instruction
from . import label as label
from . import macro as macro
from . import register as register

_P = "rocisa"


# ---------------------------------------------------------------------------
# ``rocisa.rocIsa`` forwarding shell.
#
# Every method below is a one-line forwarder to the matching accessor in
# ``base.py`` (where the actual state lives -- see ``base.py`` design
# note on state sinking). The class is kept (rather than collapsed into
# module functions) purely to preserve the ``rocIsa.getInstance().X()``
# call shape that KernelWriter / Tensile / the C++ binding share.
#
# Backwards-compatible read/write of the historical private fields
# (``_vgpr_idx`` / ``_kernel_info``) is provided via ``@property``
# descriptors. New code should call the public accessors directly.
# ---------------------------------------------------------------------------


class rocIsa:
    """Singleton forwarding shell mirroring ``rocisa::rocIsa``.

    All state lives in ``base.py`` module-level globals. This class
    only exists to keep the public API surface
    (``rocIsa.getInstance().method(...)``) intact for Tensile /
    KernelWriter callers; ``__init__`` deliberately stores nothing on
    the instance.

    Implemented members (real, all forward to ``base.*``):
        - ``getInstance``
        - ``init`` / ``isInit``
        - ``getIsaInfo`` (returns ``IsaInfo``)
        - ``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs``
        - ``setKernel`` / ``getKernel``
        - ``getOutputOptions`` / ``setOutputOptions``
        - ``getData`` / ``setData`` (used by ``ParallelMap2`` workers via
          ``KernelWriter.setRocIsa(data, outOptions)``)
        - ``getVgprIdx`` / ``setVgprIdx``
        - ``getVgprMsb`` / ``setVgprMsb`` (wired for Commit Y --
          Label.toString side effect; no consumer today)

    The C++ original keeps per-thread state (``m_threads`` /
    ``m_outputOptions``); Tensile only ever reads it back via
    parameter-less getters from the same thread that wrote it, and across
    process boundaries goes through pickle, so a single per-process
    value (held in ``base.py``) is sufficient here.
    """

    _instance: "rocIsa | None" = None

    def __init__(self) -> None:
        # All state lives in ``base.py``; nothing to initialise here.
        pass

    @staticmethod
    def getInstance() -> "rocIsa":
        if rocIsa._instance is None:
            rocIsa._instance = rocIsa()
        return rocIsa._instance

    # --- ISA init / active-ISA accessors ----------------------------------

    def init(self, arch: Any, assemblerPath: str = "", debug: bool = False) -> None:
        _base.init(arch, assemblerPath, debug)

    def isInit(self) -> bool:
        return _base.isInit()

    def getIsaInfo(self, arch: Any) -> IsaInfo:
        return _base.getIsaInfo(arch)

    def getAsmCaps(self):
        return _base.getAsmCaps()

    def getArchCaps(self):
        return _base.getArchCaps()

    def getRegCaps(self):
        return _base.getRegCaps()

    def getAsmBugs(self):
        return _base.getAsmBugs()

    # --- Per-thread kernel state (used by KernelWriter / Generators). ------

    def setKernel(self, arch: Any, wavefrontSize: int) -> None:
        _base.setKernel(arch, wavefrontSize)

    def getKernel(self) -> KernelInfo:
        return _base.getKernel()

    # --- Output options (mutated in main, shipped to workers via pickle). --

    def getOutputOptions(self) -> OutputOptions:
        return _base.getOutputOptions()

    def setOutputOptions(self, options: OutputOptions) -> None:
        _base.setOutputOptions(options)

    # --- Pickle-friendly snapshot of all initialised ISAs. ----------------

    def getData(self) -> Dict[Tuple[int, int, int], IsaInfo]:
        return _base.getData()

    def setData(self, data: Dict[Tuple[int, int, int], IsaInfo]) -> None:
        _base.setData(data)

    # --- Symbol -> base-index map (consumed by ``RegName.getTotalIdx``). ---

    def getVgprIdx(self) -> Dict[str, int]:
        return _base.getVgprIdx()

    def setVgprIdx(self, name: str, idx: int) -> None:
        _base.setVgprIdx(name, idx)

    # --- VGPR-MSB (wired for Commit Y / Label.toString side effect). ------

    def getVgprMsb(self) -> int:
        return _base.getVgprMsb()

    def setVgprMsb(self, msb: int) -> None:
        _base.setVgprMsb(msb)

    # --- Backwards-compatible private-field shims --------------------------
    #
    # Test harnesses (and possibly external code) historically reached
    # into ``rocIsa.getInstance()._vgpr_idx`` / ``._kernel_info`` to
    # reset or restore state. After the state-sink refactor these are
    # exposed as ``@property`` descriptors that delegate to ``base.*``,
    # so ``._vgpr_idx.clear()`` and ``._kernel_info = info`` keep
    # working without touching callers. New code should prefer the
    # public accessors (``base.getVgprIdx()`` / ``base.setKernelInfo``).

    @property
    def _vgpr_idx(self) -> Dict[str, int]:
        return _base.getVgprIdx()

    @property
    def _kernel_info(self) -> KernelInfo:
        return _base.getKernel()

    @_kernel_info.setter
    def _kernel_info(self, info: KernelInfo) -> None:
        _base.setKernelInfo(info)

isaToGfx = _base.isaToGfx


def getGlcBitName() -> str:
    """Mirror ``rocisa::getGlcBitName()`` using the active ISA asm caps."""
    return _caps.glc_bit_name_from_caps(rocIsa.getInstance().getAsmCaps())


def getSlcBitName() -> str:
    """Mirror ``rocisa::getSlcBitName()`` using the active ISA asm caps."""
    return _caps.slc_bit_name_from_caps(rocIsa.getInstance().getAsmCaps())

countType = make_dummy_func(f"{_P}.countType")
countInstruction = make_dummy_func(f"{_P}.countInstruction")
countGlobalRead = make_dummy_func(f"{_P}.countGlobalRead")
countSMemLoad = make_dummy_func(f"{_P}.countSMemLoad")
countLocalRead = make_dummy_func(f"{_P}.countLocalRead")
countLocalWrite = make_dummy_func(f"{_P}.countLocalWrite")
countWeightedLocalRead = make_dummy_func(f"{_P}.countWeightedLocalRead")
countWeightedLocalWrite = make_dummy_func(f"{_P}.countWeightedLocalWrite")
countDSStoreB128 = make_dummy_func(f"{_P}.countDSStoreB128")
countDSStoreB192 = make_dummy_func(f"{_P}.countDSStoreB192")
countDSStoreB256 = make_dummy_func(f"{_P}.countDSStoreB256")
countVMovB32 = make_dummy_func(f"{_P}.countVMovB32")
getMFMAs = make_dummy_func(f"{_P}.getMFMAs")
findInstCount = make_dummy_func(f"{_P}.findInstCount")

# rocisa <-> stinkytofu interop (see init_stinkytofu in
# shared/stinkytofu/src/conversion/rocisa/ToStinkyTofuUtils.cpp). These
# are real logicalIR bridges; once the dummy phase is over they should
# delegate into the compiled stinkytofu bindings rather than print.
isSupportedByStinkyTofu = make_dummy_func(f"{_P}.isSupportedByStinkyTofu")
StinkyAsmModule = make_dummy_class(f"{_P}.StinkyAsmModule")
toStinkyTofuModule = make_dummy_func(f"{_P}.toStinkyTofuModule")
