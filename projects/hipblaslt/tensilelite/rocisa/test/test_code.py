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

from rocisa import code
from rocisa.container import sgpr
from rocisa.instruction import SMovB32
from copy import deepcopy
import pickle
import timeit

def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

def test_code():
    label = code.Label("label", comment="comment")
    print(label)
    tesxtblock = code.TextBlock("textblock")
    print(tesxtblock)

    #Test Module
    module = code.Module("module")
    module.add(SMovB32(dst=sgpr(1), src=sgpr(2)))
    module2 = module.add(code.Module("module2"))
    module2.add(SMovB32(dst=sgpr(3), src=sgpr(4)))
    print(module)
    print(module.prettyPrint(""))

    for item in module.items():
        item.name = "newname"
    for item in module.items():
        assert item.name == "newname"

    flat = module.flatitems()
    testpop = []
    while flat:
        item = flat.pop(0)
        testpop.append(item)
    print("Test pop", testpop)


    srdUpperValue = code.SrdUpperValue((9,4,2))
    print(srdUpperValue.desc())
    assert srdUpperValue.getValue() == 131072

    signature = code.SignatureBase(kernelName="123",
                                    kernArgsVersion=1,
                                    codeObjectVersion="4",
                                    groupSegmentSize=256,
                                    sgprWorkGroup=(1, 1, 100),
                                    vgprWorkItem=1,
                                    flatWorkGroupSize=(256),
                                    numSgprPreload=16)
    print(signature)

    vs = code.ValueSet("BufferLimit", 0xFFFFFFFF, format=1)
    print(vs)

def _metadata_version(codeObjectVersion):
    signature = code.SignatureBase(kernelName="k",
                                   kernArgsVersion=1,
                                   codeObjectVersion=codeObjectVersion,
                                   groupSegmentSize=0,
                                   sgprWorkGroup=(1, 1, 0),
                                   vgprWorkItem=0,
                                   flatWorkGroupSize=256)
    text = str(signature)
    start = text.index("amdhsa.version:\n")
    return text[start:].split("\n")[1:3]

def test_signature_metadata_version():
    # The runtime reads the code object version from amdhsa.version: 1.1 is v4, 1.2 is v5 and v6.
    assert _metadata_version("4") == ["  - 1", "  - 1"]
    assert _metadata_version("default") == ["  - 1", "  - 1"]
    assert _metadata_version("5") == ["  - 1", "  - 2"]
    assert _metadata_version("6") == ["  - 1", "  - 2"]
    try:
        _metadata_version("7")
    except RuntimeError as e:
        assert "Unknown code object version: 7" in str(e)
    else:
        assert False, "an unknown code object version must not emit metadata"

def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def copyfunc(obj, noPickle=False):
    obj2 = deepcopy(obj)
    print("Copied item using deepcopy:", obj2)
    if noPickle:
        return
    obj3 = fastdeepcopy(obj)
    print("Copied item using pickle:", obj3)

def test_copy():
    label = code.Label("label", comment="comment")
    copyfunc(label)
    tesxtblock = code.TextBlock("textblock")
    copyfunc(tesxtblock)

    module = code.Module("module")
    module.add(SMovB32(dst=sgpr(1), src=sgpr(2)))
    module2 = module.add(code.Module("module2"))
    module2.add(SMovB32(dst=sgpr(3), src=sgpr(4)))
    copyfunc(module, True)

    valueset = code.ValueSet("Test", 1)
    copyfunc(valueset)

test_code()
test_copy()
