# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generators that expand a logic-file CustomKernel block into a custom kernel.

A solution that carries
    CustomKernel: {generator: <name>, generatorParams: {...}}
is expanded when the logic file is parsed into the same config a custom.config
provides, and its assembly is emitted when the kernel is written.
"""

from typing import Any, Dict

from . import SkinnyGEMM
from .Errors import GeneratorKernelRejected

_GENERATORS = {"SkinnyGEMM": SkinnyGEMM}


def isGeneratorKernel(customKernel: Any) -> bool:
    return isinstance(customKernel, dict) and "generator" in customKernel


def _generatorOf(customKernel: Dict[str, Any]):
    name = customKernel["generator"]
    generator = _GENERATORS.get(name) if isinstance(name, str) else None
    if generator is None:
        raise ValueError(f"unknown kernel generator {name!r}, known generators: {', '.join(_GENERATORS)}")
    return generator


def validateGeneratorKernel(customKernel: Dict[str, Any]) -> Dict[str, Any]:
    """Checks a generator block without the code generation module; returns the validated parameters."""
    return _generatorOf(customKernel).validateParams(customKernel.get("generatorParams"))


def getGeneratorKernelConfig(customKernel: Dict[str, Any], internalSupportParams: Dict[str, Any], arch: str) -> Dict[str, Any]:
    """Expands a generator block; other keys of the block are replaced by the generator's output.

    Raises GeneratorKernelRejected if the emitter rejects the parameters for arch.
    """
    generator = _generatorOf(customKernel)
    params = generator.validateParams(customKernel.get("generatorParams"))
    # Some rejections (such as the register budget) only show when the kernel is emitted; decide them here, once.
    generator.emit(params, arch)
    config = generator.kernelConfig(params)
    config["CustomKernel"]["generator"] = customKernel["generator"]
    config["CustomKernel"]["generatorParams"] = dict(params)
    isp = config["InternalSupportParams"]
    isp.setdefault("PersistentLoopArgsVersion", 0)
    for key, value in internalSupportParams.items():
        isp.setdefault(key, value)
    return config


def getGeneratorKernelSource(customKernel: Dict[str, Any], arch: str) -> str:
    generator = _generatorOf(customKernel)
    return generator.emit(generator.validateParams(customKernel.get("generatorParams")), arch)


def generatedCodeObjectName(libraryCodeObject: str, arch: str) -> str:
    """The code object, without extension, that holds the generated kernels of a library code object."""
    suffix = "_" + arch
    if libraryCodeObject.endswith(suffix):
        return libraryCodeObject[: -len(suffix)] + "_Generated" + suffix
    return libraryCodeObject + "_Generated" + suffix
