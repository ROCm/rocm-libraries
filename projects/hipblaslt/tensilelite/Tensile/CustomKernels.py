################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from .resources import custom_kernel_names, custom_kernel_text
from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters, newMIValidParameters

import yaml

import os


def isCustomKernelConfig(config):
    return "CustomKernelName" in config and config["CustomKernelName"]

def supportsUserSgprKernargPreload(rocmVersion):
    """Return whether a ROCm version passes TensileLite's preload gate.

    AMD's ROCm 6.0 compiler branch added descriptor and codegen support in
    September 2023 for feature-enabled targets. HIP recorded 6.0.32650 on
    September 29, and hipBLASLt adopted it as its 6.x floor on October 6.
    That floor is historical compatibility policy, not a complete capability
    test: target ISA, assembler, and firmware also matter.

    HIP's patch field is a build number, not globally monotonic. Official
    ROCm 7 releases can report a patch below 32650, so later major releases
    remain eligible. A locally built ROCm 6.x toolchain reporting a low build
    (for example, 6.4.0) remains ambiguous and is treated as unsupported.
    """
    return rocmVersion.major > 6 or (
        rocmVersion.major == 6 and rocmVersion.patch >= 32650
    )
def getAllCustomKernelNames(directory=None):
    if directory is None:
        return custom_kernel_names()
    # Sorted in alphabetical order so that custom-kernel enumeration (notably the CustomKernels: ["*"]
    # wildcard) does not depend on os.listdir order, which varies with the
    # filesystem and with how the package was installed.
    return sorted(fname[:-2] for fname in os.listdir(directory) if fname.endswith(".s"))

def getCustomKernelContents(name, directory=None):
    if directory is None:
        try:
            return custom_kernel_text(name)
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(f"Failed to find custom kernel: {name}") from error
    try:
        with open(os.path.join(directory, f"{name}.s")) as f:
            return f.read()
    except Exception as error:
        raise RuntimeError(
            f"Failed to find custom kernel: {os.path.join(directory, name)}"
        ) from error

def getCustomKernelSource(name, rocmVersion, directory=None):
    contents = getCustomKernelContents(name, directory)
    if supportsUserSgprKernargPreload(rocmVersion):
        return contents
    return "".join(
        line
        for line in contents.splitlines(keepends=True)
        if "amdhsa_user_sgpr_kernarg_preload" not in line
    )
def getCustomKernelConfigAndAssembly(name, directory=None):
    contents  = getCustomKernelContents(name, directory)
    config = "\n"    #Yaml configuration properties
    assembly = ""
    inConfig = False
    for line in contents.splitlines():
        if   line == "---": inConfig = True                          #Beginning of yaml section
        elif line == "...": inConfig = False                         #End of yaml section
        elif      inConfig: config   += line + "\n"
        else              : assembly += line + "\n"; config += "\n"  #Second statement to keep line numbers consistent for yaml errors

    return (config, assembly)

def readCustomKernelConfig(name, directory=None):
    rawConfig, _ = getCustomKernelConfigAndAssembly(name, directory)
    try:
        return yaml.safe_load(rawConfig)["custom.config"]
    except yaml.scanner.ScannerError as e:
        raise RuntimeError("Failed to read configuration for custom kernel: {0}\nDetails:\n{1}".format(name, e))

def getCustomKernelConfig(
    kernelName: str, internalSupportParams: dict, directory: str = None
) -> dict:
    """
    Retrieves and validates the configuration for a custom kernel.

    Args:
        kernelName: The name of the custom kernel.
        internalSupportParams: A dictionary of internal support parameters to be merged with the kernel configuration.
        directory: Optional directory where custom kernel files are located.
            Defaults to bundled package resources.

    Returns:
        dict: The validated configuration dictionary for the custom kernel.

    Raises:
        RuntimeError: If the custom kernel configuration is missing required fields or if there is an error reading the configuration.
    """
    kernelConfig = readCustomKernelConfig(kernelName, directory)
    if "InternalSupportParams" not in kernelConfig:
        raise RuntimeError(f"Custom kernel {kernelName} config must have 'InternalSupportParams'")

    if "KernArgsVersion" not in kernelConfig["InternalSupportParams"]:
        raise RuntimeError(f"Custom kernel {kernelName} config must have 'KernArgsVersion'")

    kernelIsp = kernelConfig["InternalSupportParams"]
    for key in internalSupportParams:
        if key not in kernelIsp:
            kernelIsp[key] = internalSupportParams[key]

    validParameters.update(newMIValidParameters)

    skipKeys = {"ProblemType", "InternalSupportParams", "KernelLanguage", "CustomKernelName"}
    for k, v in kernelConfig.items():
        if k not in skipKeys:
            checkParametersAreValid((k, [v]), validParameters)

    kernelConfig["KernelLanguage"] = "Assembly"
    kernelConfig["CustomKernelName"] = kernelName

    return kernelConfig
