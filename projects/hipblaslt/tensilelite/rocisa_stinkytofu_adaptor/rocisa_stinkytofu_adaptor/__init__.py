# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocisa-compatible facade backed by stinkytofu (gfx1250-only).

Activated via ``ROCISA_BACKEND=stinkytofu``; re-exports all rocisa submodules
so KernelWriter can use ``from rocisa import ...`` unchanged.
Not yet done: module-level counters, ``insertDelayAlu``, ``getCycles``,
full ``toStinkyTofuModule`` options forwarding.
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
from .stinky_interop import toStinkyTofuModule

_P = "rocisa"

try:
    import stinkytofu as _stinkytofu  # type: ignore[import-not-found]

    StinkyAsmModule = _stinkytofu.StinkyAsmModule
    CloneSpec = _stinkytofu.CloneSpec
except ImportError:
    StinkyAsmModule = make_dummy_class(f"{_P}.StinkyAsmModule")
    CloneSpec = make_dummy_class(f"{_P}.CloneSpec")


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

# ==========================================================================
# Module-level counting / analysis functions
# ==========================================================================
# Mirrors ``rocisa/src/count.cpp``. The C++ uses dynamic_cast to traverse
# a Module tree and count instructions by type. Here we use isinstance()
# against tuples of concrete adaptor classes.

def _count_recursive(item, type_tuple, weights=None):
    """Recursively count instructions matching *type_tuple* in *item* tree."""
    from .code import Module as _Mod
    if isinstance(item, _Mod):
        total = 0
        for child in item.itemList:
            total += _count_recursive(child, type_tuple, weights)
        return total
    if isinstance(item, type_tuple):
        if weights:
            w = weights.get(type(item))
            if w is not None:
                return w
        return 1
    return 0


def _count_exact_type(item, exact_type):
    """Count items whose type is exactly *exact_type* (no subclass match)."""
    from .code import Module as _Mod
    if isinstance(item, _Mod):
        total = 0
        for child in item.itemList:
            total += _count_exact_type(child, exact_type)
        return total
    return 1 if type(item) is exact_type else 0


def _get_matching(item, type_tuple):
    """Collect all items matching *type_tuple* in tree order."""
    from .code import Module as _Mod
    result = []
    if isinstance(item, _Mod):
        for child in item.itemList:
            result.extend(_get_matching(child, type_tuple))
    elif isinstance(item, type_tuple):
        result.append(item)
    return result


def countType(item, cls):
    """Generic isinstance-based count (mirrors ``rocisa::countType``)."""
    from .code import Module as _Mod
    if isinstance(item, _Mod):
        return sum(countType(child, cls) for child in item.itemList)
    return 1 if isinstance(item, cls) else 0


def countInstruction(item):
    """Count all Instruction nodes in tree (mirrors ``countX<Instruction>``)."""
    from .instruction import Instruction as _Inst
    from .code import Module as _Mod
    if isinstance(item, _Mod):
        return sum(countInstruction(child) for child in item.itemList)
    return 1 if isinstance(item, _Inst) else 0


def countGlobalRead(item):
    """Count BufferLoad*/FlatLoad*/GlobalLoadTR* (mirrors ``countX<GlobalReadInstruction>``)."""
    return _count_recursive(item, _global_read_types())


def countSMemLoad(item):
    """Count SLoadB* instructions (mirrors ``countX<SMemLoadInstruction>``)."""
    return _count_recursive(item, _smem_load_types())


def countLocalRead(item):
    """Count DSLoad* instructions (mirrors ``countX<LocalReadInstruction>``)."""
    return _count_recursive(item, _local_read_types())


def countLocalWrite(item):
    """Count DSStore* instructions (mirrors ``countX<LocalWriteInstruction>``)."""
    return _count_recursive(item, _local_write_types())


def countWeightedLocalRead(item):
    """Count local reads with weights: DSLoadB192 counts as 2."""
    from . import instruction as _inst
    weights = {_inst.DSLoadB192: 2}
    return _count_recursive(item, _local_read_types(), weights)


def countWeightedLocalWrite(item):
    """Count local writes with weights: DSStoreB192/B256 count as 2."""
    from . import instruction as _inst
    weights = {_inst.DSStoreB192: 2, _inst.DSStoreB256: 2}
    return _count_recursive(item, _local_write_types(), weights)


def countDSStoreB128(item):
    """Count exact DSStoreB128 instances."""
    from . import instruction as _inst
    return _count_exact_type(item, _inst.DSStoreB128)


def countDSStoreB192(item):
    """Count exact DSStoreB192 instances."""
    from . import instruction as _inst
    return _count_exact_type(item, _inst.DSStoreB192)


def countDSStoreB256(item):
    """Count exact DSStoreB256 instances."""
    from . import instruction as _inst
    return _count_exact_type(item, _inst.DSStoreB256)


def countVMovB32(item):
    """Count exact VMovB32 instances."""
    from . import instruction as _inst
    return _count_exact_type(item, _inst.VMovB32)


def countMFMA(item):
    """Count all MFMA-family instructions (MFMA, SMFMA, MXMFMA)."""
    return _count_recursive(item, _mfma_types())


def getMFMAs(item):
    """Get all MFMA-family instruction objects from tree."""
    return _get_matching(item, _mfma_types())


def findInstCount(module, targetItem, count=0):
    """Find index of *targetItem* in tree, skipping TextBlocks.

    Returns (count, found) pair matching C++ semantics.
    """
    from .code import Module as _Mod, TextBlock as _TB
    if isinstance(module, _Mod):
        for child in module.itemList:
            if isinstance(child, _Mod):
                count, found = findInstCount(child, targetItem, count)
                if found:
                    return (count, True)
            elif child is targetItem:
                return (count, True)
            elif not isinstance(child, _TB):
                count += 1
    return (count, False)


# ---------------------------------------------------------------------------
# Lazy type-tuple builders (avoid circular imports at module level)
# ---------------------------------------------------------------------------

def _global_read_types():
    from . import instruction as _inst
    return (
        _inst.BufferLoadU8, _inst.BufferLoadI8,
        _inst.BufferLoadD16HIU8, _inst.BufferLoadD16U8,
        _inst.BufferLoadD16I8, _inst.BufferLoadD16HII8,
        _inst.BufferLoadD16HIB16, _inst.BufferLoadD16B16,
        _inst.BufferLoadB16, _inst.BufferLoadI16, _inst.BufferLoadU16,
        _inst.BufferLoadB32, _inst.BufferLoadB64,
        _inst.BufferLoadB96, _inst.BufferLoadB128, _inst.BufferLoadB192,
        _inst.FlatLoadU8, _inst.FlatLoadI8,
        _inst.FlatLoadD16HIU8, _inst.FlatLoadD16U8,
        _inst.FlatLoadD16I8, _inst.FlatLoadD16HII8,
        _inst.FlatLoadD16HIB16, _inst.FlatLoadD16B16,
        _inst.FlatLoadU16, _inst.FlatLoadI16,
        _inst.FlatLoadB32, _inst.FlatLoadB64,
        _inst.FlatLoadB96, _inst.FlatLoadB128, _inst.FlatLoadB192,
        _inst.GlobalLoadTR8B64, _inst.GlobalLoadTR16B128,
    )


def _smem_load_types():
    from . import instruction as _inst
    return (
        _inst.SLoadB32, _inst.SLoadB64, _inst.SLoadB128,
        _inst.SLoadB256, _inst.SLoadB512,
    )


def _local_read_types():
    from . import instruction as _inst
    return (
        _inst.DSLoadU8, _inst.DSLoadI8, _inst.DSLoadD16HIU8,
        _inst.DSLoadU16, _inst.DSLoadI16, _inst.DSLoadD16HIU16,
        _inst.DSLoadB16, _inst.DSLoadB32, _inst.DSLoadB64,
        _inst.DSLoadB96, _inst.DSLoadB96TrB6,
        _inst.DSLoadB64TrB4, _inst.DSLoadB64TrB16,
        _inst.DSLoadB128TrB16, _inst.DSLoadB64TrB8,
        _inst.DSLoadB128, _inst.DSLoadB192,
        _inst.DSLoad2B32, _inst.DSLoad2B64,
    )


def _local_write_types():
    from . import instruction as _inst
    return (
        _inst.DSStoreB8, _inst.DSStoreB16,
        _inst.DSStoreB32, _inst.DSStoreB64,
        _inst.DSStoreB96, _inst.DSStoreB128,
        _inst.DSStoreB192, _inst.DSStoreB256,
        _inst.DSStore2B32, _inst.DSStore2B64,
    )


def _mfma_types():
    from . import instruction as _inst
    return (_inst.MFMAInstruction, _inst.SMFMAInstruction, _inst.MXMFMAInstruction)

# rocisa <-> stinkytofu interop
# ---------------------------------------------------------------------------
# Architecture probes live on the standalone ``stinkytofu`` binding
# (``shared/stinkytofu/python_module/src/python_bindings.cpp``), not on
# ``_rocisa.so``. Mirror native ``hasStinkyTofuBackend`` semantics by
# checking ``hasattr(stinkytofu, "isSupportedByStinkyTofu")`` instead of
# ``hasattr(_rocisa, ...)``.


def hasStinkyTofuBackend() -> bool:
    """Return True if the standalone stinkytofu binding exposes arch probes."""
    try:
        import stinkytofu
    except ImportError:
        return False
    return hasattr(stinkytofu, "isSupportedByStinkyTofu")


def isSupportedByStinkyTofu(isa) -> bool:
    """Return True if *isa* has a registered StinkyTofu backend pipeline."""
    import stinkytofu

    return stinkytofu.isSupportedByStinkyTofu(list(_caps.normalize_isa_key(isa)))


def getRegisteredArchKeys():
    """Return arch name strings for all registered StinkyTofu backends."""
    import stinkytofu

    return stinkytofu.getRegisteredArchKeys()


# Path-1 interop: ``toStinkyTofuModule`` / ``StinkyAsmModule`` (see
# ``stinky_interop.py`` and ``code.Module.to_stinky_asm``).
