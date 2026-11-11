# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""StinkyTofu asm-IR wiring: toStinkyTofuModule / emitAssembly entry points.

Lowers Python code.Module via Module.to_stinky_asm -> stinkytofu.lower_logical_module.
The options dict is accepted for API parity but not yet forwarded to the C++ path.
"""

from __future__ import annotations

from typing import Any, List

from . import caps as _caps
from . import code as _code


def _arch_to_list(arch: Any) -> List[int]:
    return list(_caps.normalize_isa_key(arch))


def _copy_instance_fields(src: Any, dst: Any) -> None:
    """Copy slots and ``__dict__`` without pickle.

    ``SignatureBase`` refuses ``__deepcopy__`` and ``__reduce__``. Lists are
    copied so a recorder on the caller (``setGprsCalls``, ``declared``) does
    not observe writes on the emit copy. Nested objects stay shared until the
    caller replaces ``kernelDescriptor`` and ``codeMeta``.
    """
    if hasattr(src, "__dict__"):
        dst.__dict__.update(src.__dict__)
        for key, value in list(dst.__dict__.items()):
            if isinstance(value, list):
                dst.__dict__[key] = list(value)
    for cls in type(src).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if name in ("__dict__", "__weakref__"):
                continue
            try:
                value = getattr(src, name)
            except AttributeError:
                continue
            setattr(dst, name, list(value) if isinstance(value, list) else value)


def _clone_plain(obj: Any) -> Any:
    cloned = object.__new__(type(obj))
    _copy_instance_fields(obj, cloned)
    return cloned


def _clone_emit_signature(signature: Any) -> Any:
    """Signature emit can refresh without writing the one KernelWriter holds.

    Native ``toStinkySignature`` builds a new descriptor from the rocisa one
    and ``refreshVgprCount`` updates that copy. ``cmpasm/orig`` is
    ``str(moduleKernelBody)`` of the original, taken after ``emitAssembly``.
    """
    cloned = object.__new__(type(signature))
    _copy_instance_fields(signature, cloned)
    kd = getattr(signature, "kernelDescriptor", None)
    if kd is not None:
        cloned.kernelDescriptor = _clone_plain(kd)
    meta = getattr(signature, "codeMeta", None)
    if meta is not None:
        cloned.codeMeta = _clone_plain(meta)
    return cloned


# InsertDelayAluPass reads this off Function metadata. Same string as
# stinkytofu::kSigTotalVgprsMetaKey. Absent or 0 means "unknown" and the pass
# inserts s_delay_alu; native toStinkyTofuModule always stamps it.
_SIG_TOTAL_VGPRS_META_KEY = "SignatureKernelDescriptor.totalVgprs"


class StinkyAsmModuleWithAdapterSignature:
    """Behavioural subset of stinkytofu's ``StinkyAsmModuleWithSignature`` (C++).

    Prepends the Python ``SignatureBase.toString()`` banner before the
    lowered asm module's ``emitAssembly()``, matching what KernelWriter expects
    after ``toStinkyTofuModule`` on the native backend.

    The stored signature is a copy made after the pre-pipeline VGPR stamp.
    Count refresh writes that copy, so the caller's descriptor stays on the
    pool ``KernelWriter`` prints as ``cmpasm/orig``. ``getDeclaredVgprCount``
    and the emitted ``.amdhsa_next_free_vgpr`` still report the required count.
    """

    __slots__ = ("_inner", "_signature")

    def __init__(self, inner: Any, signature: Any) -> None:
        self._inner = inner
        self._signature = signature
        self._stamp_total_vgprs()
        if signature is not None:
            self._signature = _clone_emit_signature(signature)

    def _stamp_total_vgprs(self) -> None:
        """Expose per-wave VGPR allocation before the optimization pipeline.

        Port of the setMetaData(kSigTotalVgprsMetaKey) call in
        toStinkyTofuModule. The value is kernelDescriptor.totalVgprs, the
        count emitted as ``.amdhsa_next_free_vgpr``.
        """
        if self._signature is None:
            return
        kd = getattr(self._signature, "kernelDescriptor", None)
        if kd is None or not hasattr(kd, "totalVgprs"):
            return
        leaf = self._inner
        while hasattr(leaf, "_inner"):
            leaf = leaf._inner
        setter = getattr(leaf, "setMetaDataU64", None)
        if setter is None:
            return
        setter(_SIG_TOTAL_VGPRS_META_KEY, int(kd.totalVgprs))

    def runOptimizationPipeline(self) -> None:
        self._inner.runOptimizationPipeline()

    def emitAssembly(self) -> str:
        out = ""
        if self._signature is not None:
            self._refreshTotalInstructionBytes()
            self._refreshSgprCount()
            self._refreshVgprCount()
            out += self._signature.toString()
        # .set directives go between signature and instruction body.
        set_dirs = getattr(self._inner, "getSetDirectives", None)
        if set_dirs is not None:
            out += set_dirs()
        out += self._inner.emitAssembly()
        return out

    def _refreshTotalInstructionBytes(self) -> None:
        """Copy the pipeline's instruction-byte total onto the signature.

        Port of ``StinkyAsmModuleWithSignature::emitAssembly``:
        ``AccumulateInstructionSizePass`` stores the total on the asm module,
        and the signature prints ``STINKY_TOTAL_INST_BYTES`` plus
        ``.amdhsa_inst_pref_size`` from that value.
        """
        leaf = self._inner
        while hasattr(leaf, "_inner"):
            leaf = leaf._inner
        getter = getattr(leaf, "getTotalInstructionBytes", None)
        setter = getattr(self._signature, "setTotalInstructionBytes", None)
        if getter is None or setter is None:
            return
        total = int(getter())
        if total >= 0:
            setter(total)

    def _refreshSgprCount(self) -> None:
        """Take the declared SGPR count from the lowered code, as C++ does.

        Port of ``StinkyAsmModuleWithSignature::refreshSgprCount``. Tensile
        declares ``sgprPool.size()``, the pool high-water mark, which still
        counts registers that were checked back in and appear in no operand,
        so the count arrives too high. Only that number moves, and only
        downwards: everything else in the descriptor states what the hardware
        does before entry, and a flow whose registers did not move keeps the
        producer's number.
        """
        leaf = self._inner
        while hasattr(leaf, "_inner"):
            leaf = leaf._inner
        counter = getattr(leaf, "getRequiredSgprCount", None)
        if counter is None:
            return
        kd = self._signature.kernelDescriptor
        required = int(counter(kd.numSgprPreload, list(kd.sgprWorkGroup)))
        if required == 0 or required >= kd.totalSgprs:
            return
        self._signature.setGprs(kd.totalVgprs, kd.totalAgprs, required)

    def _refreshVgprCount(self) -> None:
        """Take the declared VGPR count from the lowered code, as C++ does.

        Port of ``StinkyAsmModuleWithSignature::refreshVgprCount``. The
        producer's pool still counts registers that were checked back in, so
        ``.amdhsa_next_free_vgpr`` arrives too high. A colouring can also need
        more than was reserved, so this moves in both directions. Goes through
        ``setDeclaredVgprs``, which leaves the ``Num VGPR`` comment on the
        producer's figure and does nothing when AGPRs share the register file.
        """
        leaf = self._inner
        while hasattr(leaf, "_inner"):
            leaf = leaf._inner
        counter = getattr(leaf, "getRequiredVgprCount", None)
        if counter is None:
            return
        kd = self._signature.kernelDescriptor
        required = int(counter(kd.vgprWorkItem))
        if required == 0:
            return
        setter = getattr(self._signature, "setDeclaredVgprs", None)
        if setter is None:
            return
        setter(required)

    def getDeclaredSgprCount(self) -> int:
        """SGPRs the descriptor declares, as ``.amdhsa_next_free_sgpr``.

        Ask after ``emitAssembly()``: the refresh has then replaced the
        producer's estimate with the count the emitted code uses. Matches
        ``StinkyAsmModuleWithSignature::getDeclaredSgprCount``.
        """
        if self._signature is None:
            return 0
        return int(self._signature.kernelDescriptor.totalSgprs)

    def getDeclaredVgprCount(self) -> int:
        """VGPRs the descriptor declares, as ``.amdhsa_next_free_vgpr``.

        Ask after ``emitAssembly()``. Where AGPRs share the register file the
        refresh leaves the producer's number, so this returns that number.
        Matches ``StinkyAsmModuleWithSignature::getDeclaredVgprCount``.
        """
        if self._signature is None:
            return 0
        return int(self._signature.kernelDescriptor.totalVgprs)

    def getName(self) -> str:
        return self._inner.getName()

    def setOutputName(self, name: str) -> None:
        self._inner.setOutputName(name)

    def getOutputName(self) -> str:
        return self._inner.getOutputName()

    def setOutputDir(self, directory: str) -> None:
        self._inner.setOutputDir(directory)

    def getOutputDir(self) -> str:
        return self._inner.getOutputDir()

    def getModule(self) -> Any:
        return self._inner

    def __getattr__(self, name: str) -> Any:
        """Delegate anything not overridden to the wrapped module.

        The explicit methods above exist to ADD behaviour; everything else on
        ``StinkyAsmModule`` (setPluginDataStr, pass controls, ...) should reach it
        unchanged rather than be enumerated one at a time.
        """
        return getattr(self._inner, name)


def _apply_optimization_config(signature: Any, options: Any) -> None:
    """Fill the signature the way C++ ``toStinkyTofuModule`` does.

    Native conversion does not copy ``descriptionTopic`` / ``descriptionList``.
    It calls ``setOptimizationConfig`` from ModuleOptions; the descriptor then
    prints the Optimizations block as raw text. Drop the TextBlock copies so
    they cannot print a second copy when comments are enabled.
    """
    setter = getattr(signature, "setOptimizationConfig", None)
    if setter is None or not options:
        return
    gro = options.get("UseSgprForGRO", 0)
    setter(
        (int(options.get("TileA0", 0)), int(options.get("TileB0", 0))),
        (int(options.get("SubGroup0", 0)), int(options.get("SubGroup1", 0))),
        (int(options.get("WaveGroup0", 0)), int(options.get("WaveGroup1", 0))),
        int(options.get("VectorWidthA", 0)),
        int(options.get("VectorWidthB", 0)),
        int(options.get("GlobalReadVectorWidthA", 0)),
        int(options.get("GlobalReadVectorWidthB", 0)),
        bool(options.get("DirectToLdsA", False)),
        bool(options.get("DirectToLdsB", False)),
        int(gro),
    )
    if hasattr(signature, "descriptionTopic"):
        signature.descriptionTopic = _code.TextBlock("")
    clearer = getattr(signature, "clearDescription", None)
    if clearer is not None:
        clearer()


def _convert_options(options: Any) -> dict:
    """Convert adaptor options dict to stinkytofu-binding-compatible dict.

    Replaces Python-shim CloneSpec objects with stinkytofu binding CloneSpec.
    """
    import stinkytofu as _st  # noqa: WPS433

    out = {}
    for k, v in options.items():
        if k == "CloneList" and isinstance(v, list):
            out[k] = [_st.CloneSpec(name=cs.name, startLabel=cs.startLabel) for cs in v]
        else:
            out[k] = v
    return out


def toStinkyTofuModule(
    module: Any,
    arch: Any,
    moduleName: str = "",
    *,
    signature: Any = None,
    options: Any = None,
) -> Any:
    """Lower ``code.Module`` to stinkytofu asm IR (logical path).

    Args:
        module: adapter ``rocisa.code.Module`` instance.
        arch: ISA key (tuple / list / gfx string); normalised via ``caps``.
        moduleName: forwarded as ``logical_name`` to ``LogicalModule`` /
            asm module naming (when non-empty).
        signature: optional ``SignatureBase``; when set, return value wraps
            the binding module so ``emitAssembly()`` prepends ``signature.toString()``.
        options: dict of ``ModuleOptions`` fields (same keys as native
            ``toStinkyTofuModule``: CloneList, OptLevel, wavefrontSize, etc.).
            Forwarded to ``lower_logical_module`` in the stinkytofu binding.

    Returns:
        ``stinkytofu.StinkyAsmModule`` when ``signature`` is ``None``, else
        ``StinkyAsmModuleWithAdapterSignature`` delegating pipeline/emit to
        the inner module.
    """
    if not isinstance(module, _code.Module):
        raise TypeError(
            "toStinkyTofuModule expects a rocisa.code.Module from this adapter, "
            f"got {type(module).__name__!r}",
        )
    arch_list = _arch_to_list(arch)
    logical_name = moduleName if moduleName else None
    st_options = _convert_options(options) if options else None
    inner = module.to_stinky_asm(arch_list, logical_name=logical_name, options=st_options)
    if signature is None:
        return inner
    _apply_optimization_config(signature, options)
    return StinkyAsmModuleWithAdapterSignature(inner, signature)
