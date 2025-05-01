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

from Tensile.Common import tryAssembler

def test_Simple(useGlobalParameters):
    with useGlobalParameters():
        assert tryAssembler((9,0,0), "")
# disable for now due to a clang-12 assembler bug
#       assert not tryAssembler((20,0,0), "")
# disable for now due to targetid changes

#def test_Options(useGlobalParameters):
#    with useGlobalParameters():
#        assert tryAssembler((9,0,6), "", False, "-mllvm --amdhsa-code-object-version=2")

def test_Macro(useGlobalParameters):
    """
    Test a multi-line kernel that defines a macro.
    """
    with useGlobalParameters():

        thekernel = r"""
            .text
            .macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
            v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
            .endm

            .set vgprLocalReadAddrB, 93

            _v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x400, v[vgprLocalReadAddrB+0]
            a_label:

            v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x400, v[vgprLocalReadAddrB+0]
            _v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x400, v[vgprLocalReadAddrB+0]

            """

# disable for now due to targetid changes
#        assert tryAssembler((10,1,0), thekernel.format(arch="gfx1010"), True, '-mllvm --amdhsa-code-object-version=4')
        assert tryAssembler((10,1,1), thekernel.format(arch="gfx1011"))
        assert not tryAssembler((8,0,3), thekernel.format(arch="gfx803"))
