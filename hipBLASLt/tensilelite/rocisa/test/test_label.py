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

name = "Test"

def test_label():
    lm = rocisa.label.LabelManager()
    # Basic
    lm.addName(name)
    print("Get name:", lm.getName(name))
    print("Get name inc:", lm.getNameInc(name))
    print("Get name with index:", lm.getNameIndex(name, 0))
    print("Get unique name:", lm.getUniqueName())
    print("Get unique name with prefix:", lm.getUniqueNamePrefix("Yeee"))

def test_label_copy():
    lm = rocisa.label.LabelManager()
    lm2 = deepcopy(lm)
    print("Copied lm using deepcopy:", lm2.getName(name))
    lm3 = pickle.loads(pickle.dumps(lm))
    print("Copied lm using pickle:", lm3.getName(name))
