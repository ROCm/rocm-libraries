# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from enum import Enum
from rocm_arch import architectures


class lds_config(Enum):
    SIZE_64KiB = 64 * 1024
    SIZE_160KiB = 160 * 1024


# Architecture identity comes from the common catalog. Tuned configuration
# availability remains in the per-kernel configuration records.
def _enum_name(name):
    return "GFX_" + name[3:].upper().replace("-", "_")


_members = {"GFX_GENERIC": "gfx_generic"}
for _name in architectures():
    _member = _enum_name(_name)
    if _member in _members:
        raise ValueError(f"Architecture enum name collision: {_name}")
    _members[_member] = _name
supported_arch = Enum("supported_arch", _members, module=__name__)
