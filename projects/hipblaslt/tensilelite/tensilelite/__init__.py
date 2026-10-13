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

# Even though we don't support python 2, this is still packaged sometimes with python 2.
from __future__ import print_function
from os import path

# Hardcoded tensilelite version, also in tensilelite/Source/TensileConfigVersion.cmake
__version__ = "5.0.0"

# Compatibility-only filesystem-style root. Production code that reads bundled
# resources should use tensilelite.resources.
ROOT_PATH: str = path.dirname(__file__)

def _extend_extension_path():
    """Find tensilelite_write next to this package on PYTHONPATH.

    The extension is built into the build or install tree, not the source
    package. create_device_library puts that tree's parent on PYTHONPATH.
    """
    import os
    import sys

    here = os.path.abspath(os.path.dirname(__file__))
    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.abspath(os.path.join(entry, "tensilelite"))
        if candidate == here or not os.path.isdir(candidate):
            continue
        try:
            names = os.listdir(candidate)
        except OSError:
            continue
        if any(name.startswith("tensilelite_write") for name in names):
            if candidate not in __path__:
                __path__.append(candidate)
            return

_extend_extension_path()

def PrintTensileRoot():
    print(ROOT_PATH, end='')

__all__ = ["__version__", "ROOT_PATH"]
