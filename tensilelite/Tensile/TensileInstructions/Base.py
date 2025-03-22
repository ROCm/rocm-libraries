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

from rocisa import rocIsa, base
from rocisa.base import KernelInfo
from rocisa.base import Item

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
_global_ti = rocIsa.getInstance()

# This is a temporary wrapper.Will remove when TensileInstructions are all moved to rocisa
# class Item(base.Item):
#     def __init__(self, name=""):
#         super().__init__(name)

#     def __getstate__(self):
#         base_state = super().__getstate__()
#         py_state = {key: value for key, value in self.__dict__.items() if not key.startswith("__")}
#         return (base_state, py_state)
    
#     def __setstate__(self, state):
#         base_state, py_state = state
#         super().__setstate__(base_state)
#         self.__dict__.update(py_state)

#     def __deepcopy__(self, memo):
#         assert 0, "Not implemented"

#     def countType(self, ttype) -> int:
#         return int(isinstance(self, ttype))

def _removeIdent(isaDict) -> list:
    ids = [th.ident for th in threading.enumerate()]
    isaDict = [id for id in isaDict if id in ids]
    return isaDict
