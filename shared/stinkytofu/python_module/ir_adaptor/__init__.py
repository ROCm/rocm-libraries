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
"""logicalIR adaptor package — the Tensilelite ``rocisa`` shim.

What this file is:
    Top-level package that mimics ``projects/hipblaslt/tensilelite/
    rocisa/rocisa`` (the nanobind C++ bindings) so KernelWriter callers
    can keep using ``from rocisa import ...`` unchanged. Activated only
    when ``ROCISA_BACKEND=logical`` (see
    ``projects/hipblaslt/tensilelite/rocisa/rocisa/__init__.py``);
    reserved for ``gfx1250`` today.

What it does (real):
    - ``IsaInfo`` — asm/arch/reg/bug caps holder; picklable.
    - ``rocIsa`` — singleton mirror of the C++ class. Supports
      ``init`` / ``isInit``, ``getIsaInfo``,
      ``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs``,
      ``setKernel`` / ``getKernel``,
      ``getOutputOptions`` / ``setOutputOptions``,
      ``getData`` / ``setData``. Backed by ``caps.py`` snapshots.
    - Submodules with real implementations: ``register`` (pool),
      ``enum`` (real ``IntEnum``s), ``base`` (``KernelInfo`` /
      ``OutputOptions``), ``caps`` (gfx1250 snapshot).
    - Submodule registration as ``rocisa.<submodule>`` for ``base``,
      ``enum``, ``container``, ``code``, ``label``, ``instruction``,
      ``functions``, ``asmpass``, ``macro``, ``register``.

Not yet done (dummy):
    - Free functions: ``isaToGfx``, ``getGlcBitName``, ``getSlcBitName``.
    - Counters: ``count*``, ``find*``, ``getMFMAs``.
    - Interop hooks: ``isSupportedByStinkyTofu``, ``StinkyAsmModule``,
      ``toStinkyTofuModule`` — should delegate into compiled stinkytofu
      bindings once the dummy phase ends.
    - Submodules still all-dummy: ``container``, ``code``, ``label``,
      ``instruction``, ``functions``, ``asmpass``, ``macro``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from . import caps as _caps
from ._dummy import make_dummy_class, make_dummy_func
from .base import KernelInfo, OutputOptions

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
# Real (non-dummy) shims for ``rocisa.IsaInfo`` and ``rocisa.rocIsa``.
#
# These replace the structural dummies because the Tensile capability
# discovery path (``Tensile.Common.Capabilities.makeIsaInfoMap``) needs to
# read concrete dicts off ``rocIsa.getInstance().getIsaInfo(v)`` before any
# kernel can be generated. ``initAsmCaps`` belongs in the rocIsa C++ layer
# (closest to llvm-mc); here we simply hand back a snapshot captured from a
# real probe — see ``caps.py`` for the data and how to refresh it.
# ---------------------------------------------------------------------------


class IsaInfo:
    """Plain holder mirroring ``rocisa::IsaInfo`` (asm/arch/reg/bug dicts)."""

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

    # Pickle support so workers spawned by ``ParallelMap2`` can rehydrate
    # the rocIsa data dict via ``setData(getData())``.
    def __getstate__(self) -> tuple:
        return (self.asmCaps, self.archCaps, self.regCaps, self.asmBugs)

    def __setstate__(self, state: tuple) -> None:
        self.asmCaps, self.archCaps, self.regCaps, self.asmBugs = state


class rocIsa:
    """Singleton mirroring ``rocisa::rocIsa`` for the logicalIR backend.

    Implemented members (real, return data from ``caps.py``):
        - ``getInstance``
        - ``init`` / ``isInit``
        - ``getIsaInfo`` (returns ``IsaInfo``)
        - ``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs``
        - ``setKernel`` / ``getKernel``
        - ``getOutputOptions`` / ``setOutputOptions``
        - ``getData`` / ``setData`` (used by ``ParallelMap2`` workers via
          ``KernelWriter.setRocIsa(data, outOptions)``)

    The C++ original keeps per-thread state (``m_threads`` /
    ``m_outputOptions``); Tensile only ever reads it back via
    parameter-less getters from the same thread that wrote it, and across
    process boundaries goes through pickle, so a single per-instance value
    is sufficient here.
    """

    _instance: "rocIsa | None" = None

    def __init__(self) -> None:
        self._current_isa: Tuple[int, int, int] | None = None
        self._is_init: bool = False
        self._assembler_path: str = ""
        self._output_options: OutputOptions = OutputOptions()
        self._kernel_info: KernelInfo = KernelInfo()
        # ISA-keyed snapshot mirroring rocIsa::m_isainfo. Populated by
        # ``init()`` and shipped to workers by ``setData(getData())``.
        self._data: Dict[Tuple[int, int, int], IsaInfo] = {}

    @staticmethod
    def getInstance() -> "rocIsa":
        if rocIsa._instance is None:
            rocIsa._instance = rocIsa()
        return rocIsa._instance

    def init(self, arch: Any, assemblerPath: str = "", debug: bool = False) -> None:
        # No real probing in the logical adaptor. We just remember which ISA
        # was selected and stash its (statically-captured) caps so subsequent
        # parameterless getters / getData() can answer.
        key = _caps.normalize_isa_key(arch)
        self._current_isa = key
        self._assembler_path = assemblerPath
        if key not in self._data:
            asm, archc, reg, bugs = _caps.getCaps(key)
            self._data[key] = IsaInfo(asm, archc, reg, bugs)
        self._is_init = True

    def isInit(self) -> bool:
        return self._is_init

    def getIsaInfo(self, arch: Any) -> IsaInfo:
        key = _caps.normalize_isa_key(arch)
        info = self._data.get(key)
        if info is None:
            asm, archc, reg, bugs = _caps.getCaps(key)
            info = IsaInfo(asm, archc, reg, bugs)
            self._data[key] = info
        return info

    def _activeCaps(self):
        if self._current_isa is None:
            raise RuntimeError(
                "rocisa.rocIsa: init(arch, ...) or setKernel(arch, ...) must "
                "be called before getAsmCaps()/getArchCaps()/getRegCaps()/"
                "getAsmBugs()."
            )
        info = self._data.get(self._current_isa)
        if info is None:
            asm, archc, reg, bugs = _caps.getCaps(self._current_isa)
            info = IsaInfo(asm, archc, reg, bugs)
            self._data[self._current_isa] = info
        return (info.asmCaps, info.archCaps, info.regCaps, info.asmBugs)

    def getAsmCaps(self):
        return self._activeCaps()[0]

    def getArchCaps(self):
        return self._activeCaps()[1]

    def getRegCaps(self):
        return self._activeCaps()[2]

    def getAsmBugs(self):
        return self._activeCaps()[3]

    # --- Per-thread kernel state (used by KernelWriter / Generators). ------

    def setKernel(self, arch: Any, wavefrontSize: int) -> None:
        key = _caps.normalize_isa_key(arch)
        self._current_isa = key
        self._kernel_info = KernelInfo(isa=key, wavefrontSize=wavefrontSize)

    def getKernel(self) -> KernelInfo:
        return self._kernel_info

    # --- Output options (mutated in main, shipped to workers via pickle). --

    def getOutputOptions(self) -> OutputOptions:
        return self._output_options

    def setOutputOptions(self, options: OutputOptions) -> None:
        self._output_options = options

    # --- Pickle-friendly snapshot of all initialised ISAs. ----------------

    def getData(self) -> Dict[Tuple[int, int, int], IsaInfo]:
        return self._data

    def setData(self, data: Dict[Tuple[int, int, int], IsaInfo]) -> None:
        self._data = dict(data)
        self._is_init = bool(self._data)

isaToGfx = make_dummy_func(f"{_P}.isaToGfx")
getGlcBitName = make_dummy_func(f"{_P}.getGlcBitName")
getSlcBitName = make_dummy_func(f"{_P}.getSlcBitName")

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
