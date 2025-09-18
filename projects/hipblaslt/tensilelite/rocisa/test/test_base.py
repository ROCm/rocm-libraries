################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import rocisa
from copy import deepcopy
import pickle
import os

isa = (9,0,10)

def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

def getInstance():
    return rocisa.rocIsa.getInstance()

def getGfxName(isa):
    return rocisa.isaToGfx(isa)

def test_rocisa():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    global_isa = rocisa.rocIsa.getInstance()
    global_isa.init(isa, rocm_path + "/bin/amdclang++", False)
    global_isa.setKernel(isa, 64)

    ki = global_isa.getKernel()

    assert global_isa.getAsmCaps()["v_mac_f16"] == 1
    assert global_isa.getKernel().isa == isa

    assert ki.isa == isa
    assert isa == ki.isa
    assert (0,0,0,0) != ki.isa
    assert ki.isa == ki.isa

class Item2(rocisa.base.Item):
    def __init__(self, name):
        super().__init__(name)
        self.itemList = []

    def __getstate__(self):
        base_state = super().__getstate__()
        py_state = {key: value for key, value in self.__dict__.items() if not key.startswith("__")}
        return (base_state, py_state)

    def __setstate__(self, state):
        base_state, py_state = state
        super().__setstate__(base_state)
        self.__dict__.update(py_state)

    def __deepcopy__(self, memo):
        assert 0, "Not implemented"

    def print(self):
        for i in self.itemList:
            print(i)

def test_item():
    item = rocisa.base.Item("Tested Item")
    print("Test property 1:", item.asmCaps["v_mac_f16"])
    print("Test property 2:", item.kernel.isa)
    print("PrettyPrint:", item.prettyPrint(""))


    item2 = Item2("Tested Item 2")
    item2.itemList.append(rocisa.base.Item("Tested Item 3"))
    item2.print()

def test_copy():
    getInstance2 = fastdeepcopy(getInstance)
    global_isa   = rocisa.rocIsa.getInstance()
    global_isa2  = getInstance2()
    assert global_isa is global_isa2

    ki = global_isa.getKernel()
    ki2 = fastdeepcopy(ki)

    item = rocisa.base.Item("Tested Item")
    item2 = deepcopy(item)
    item3 = fastdeepcopy(item)
    print("Copied item using deepcopy:", item2.kernel.isa)
    print("Copied item using pickle:", item3.kernel.isa)
    item2 = Item2("Tested Item 2")
    item2.itemList.append(Item2("Tested Item 3"))
    item2.itemList[0].parent = item2
    print(item2.itemList[0].parent.name)
    item3 = fastdeepcopy(item2)
    print("Copied item using pickle:", item3.name)

    deepcopiedFunction = fastdeepcopy(getGfxName)
    print("This is a deepcopied function:", deepcopiedFunction(isa))

def test_functions():
    print("GLC:", rocisa.getGlcBitName(True))
    print("SLC:", rocisa.getSlcBitName(False))

test_rocisa()
test_item()
test_copy()
test_functions()
