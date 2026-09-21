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


class StinkyAsmModuleWithAdapterSignature:
    """Behavioural subset of stinkytofu's ``StinkyAsmModuleWithSignature`` (C++).

    Prepends the Python ``SignatureBase.toString()`` banner before the
    lowered asm module's ``emitAssembly()``, matching what KernelWriter expects
    after ``toStinkyTofuModule`` on the native backend.
    """

    __slots__ = ("_inner", "_signature")

    def __init__(self, inner: Any, signature: Any) -> None:
        self._inner = inner
        self._signature = signature

    def runOptimizationPipeline(self) -> None:
        self._inner.runOptimizationPipeline()

    def emitAssembly(self) -> str:
        out = ""
        if self._signature is not None:
            out += self._signature.toString()
        # .set directives go between signature and instruction body.
        set_dirs = getattr(self._inner, "getSetDirectives", None)
        if set_dirs is not None:
            out += set_dirs()
        out += self._inner.emitAssembly()
        return out

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
    return StinkyAsmModuleWithAdapterSignature(inner, signature)
