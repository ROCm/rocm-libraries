################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import globalParameters
from Tensile.Utilities.ConditionalImports import yamlLoader

import yaml

import os

def isCustomKernelConfig(config):
    return "CustomKernelName" in config and config["CustomKernelName"]

def getCustomKernelFilepath(name, directory=globalParameters["CustomKernelDirectory"]):
    return os.path.join(directory, (name + ".s"))

def getAllCustomKernelNames(directory=globalParameters["CustomKernelDirectory"]):
    return [fname[:-2] for fname in os.listdir(directory) if fname.endswith(".s")]

def getCustomKernelContents(name, directory=globalParameters["CustomKernelDirectory"]):
    try:
        with open(getCustomKernelFilepath(name, directory)) as f:
            return f.read()
    except:
        raise RuntimeError("Failed to find custom kernel: {}".format(os.path.join(directory, name)))

def getCustomKernelConfigAndAssembly(name, directory=globalParameters["CustomKernelDirectory"]):
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

def getCustomKernelConfig(name, directory=globalParameters["CustomKernelDirectory"]):
    rawConfig, _ = getCustomKernelConfigAndAssembly(name, directory)
    try:
        return yaml.load(rawConfig, yamlLoader)["custom.config"]
    except yaml.scanner.ScannerError as e:
        raise RuntimeError("Failed to read configuration for custom kernel: {0}\nDetails:\n{1}".format(name, e))
