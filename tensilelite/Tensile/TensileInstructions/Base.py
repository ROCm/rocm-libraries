################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import pickle
import threading

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

from ..Common import initAsmCaps, initArchCaps, initRegisterCaps, initAsmBugs
from .Formatting import __TI_DEBUG_LEVEL__, printExit


def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

class TensileInstructions:

    _instance  = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._isaInfo = {}  # type: ignore
                cls._instance._kernelInfo = {}
        return cls._instance

    def __reduce__(self):
        return (TensileInstructions, ())

    @dataclass
    class IsaInfo:
        assemblerPath: str
        asmCaps: dict
        archCaps: dict
        regCaps: dict
        asmBugs: dict

    @dataclass
    class kernelInfo:
        isa: Tuple[int, int, int]
        wavefrontSize: int = 64

    def init(self, isaVersion: Tuple[int, int, int], assemblerPath: str, debug: bool=False) -> None:
        with self._lock:
            if len(self._kernelInfo) > 1000:
                self._kernelInfo = _removeIdent(self._kernelInfo)
            self._kernelInfo[threading.get_ident()] = TensileInstructions.kernelInfo(isa=isaVersion)
            if isaVersion not in self._isaInfo: # type: ignore
                asmCaps  = initAsmCaps(isaVersion, assemblerPath, debug)
                archCaps = initArchCaps(isaVersion)
                regCaps  = initRegisterCaps(isaVersion, archCaps)
                asmBugs  = initAsmBugs(asmCaps)
                self._isaInfo[isaVersion] = TensileInstructions.IsaInfo(assemblerPath, # type: ignore
                    asmCaps, archCaps, regCaps, asmBugs)

    def setDebugLevel(self, level: int) -> None:
        __TI_DEBUG_LEVEL__ = level

    def setKernelInfo(self, isaVersion: Tuple[int, int, int], wavefrontSize: int) -> None:
        if isaVersion not in self._isaInfo: # type: ignore
            import traceback
            printExit(f"Current isa {str(isaVersion)} not initialized. Initialized isas are {str(self._isaInfo.keys())}, traceback: {traceback.format_stack()}")
        with self._lock:
            if len(self._kernelInfo) > 1000:
                self._kernelInfo = _removeIdent(self._kernelInfo)
            tid = threading.get_ident()
            if tid not in self._kernelInfo:
                self._kernelInfo[threading.get_ident()] = \
                    TensileInstructions.kernelInfo(isa=isaVersion, wavefrontSize=wavefrontSize)
            else:
                self._kernelInfo[threading.get_ident()].isa           = isaVersion
                self._kernelInfo[threading.get_ident()].wavefrontSize = wavefrontSize

    def getCurrentIsa(self) -> Tuple[int]:
        return self._kernelInfo[threading.get_ident()].isa

    def getAsmCaps(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].asmCaps # type: ignore

    def getArchCaps(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].archCaps # type: ignore

    def getRegCaps(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].regCaps

    def getAsmBugs(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].asmBugs # type: ignore

    def getKernel(self) -> kernelInfo:
        return self._kernelInfo[threading.get_ident()]

    def isInit(self):
        return len(self._isaInfo) > 0

def printItemList(listOfItems, tag="__unnamed__") -> None:
    header = "="*40
    print("%s\nbegin list %s\n%s"%(header, tag, header))
    for i, item in enumerate(listOfItems):
        item = list(item) if isinstance(item, tuple) else [item]
        print("list[%s] %s"%(i, "-"*30))
        for j, t in enumerate(item):
            ostream = t.prettyPrint()
            ostream = ostream[:-1] if len(ostream)>0 and ostream[-1:] == '\n' else ostream
            print(ostream)
    print("%s\nend list %s\n%s"%(header, tag, header))

# Global
_global_ti = TensileInstructions()

class Item:
    """
    Base class for Modules, Instructions, etc
    Item is a atomic collection of or more instructions and commentsA
    """

    def __init__(self, name: str="") -> None:
        self.parent = ""
        self.name = name

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def asmCaps(self) -> dict:
        return _global_ti.getAsmCaps()

    @property
    def archCaps(self) -> dict:
        return _global_ti.getArchCaps()
    
    @property
    def regCaps(self) -> dict:
        return _global_ti.getRegCaps()

    @property
    def asmBugs(self) -> dict:
        return _global_ti.getAsmBugs()

    @property
    def kernel(self) -> TensileInstructions.kernelInfo:
        return _global_ti.getKernel()

    def countType(self, ttype) -> int:
        return int(isinstance(self, ttype))

    def prettyPrint(self, indent="") -> str:
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        ostream += str(self)
        return ostream

def getGlcBitName(hasGLCModifier):
  if hasGLCModifier:
    return "glc"
  return "sc0"

def getSlcBitName(hasGLCModifier):
  if hasGLCModifier:
    return "slc"
  return "sc1"

def _removeIdent(isaDict) -> list:
    ids = [th.ident for th in threading.enumerate()]
    isaDict = [id for id in isaDict if id in ids]
    return isaDict
