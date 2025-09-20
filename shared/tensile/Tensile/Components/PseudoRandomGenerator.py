################################################################################
#
# Copyright (C) 2022-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import PseudoRandomGenerator
from ..AsmUtils import inst, vgpr, sgpr

class CustomRandomGenerator(PseudoRandomGenerator):
    """
    Custom Pseudo Random Generator
    """
    def __call__(self, writer):
        kStr = ""

        kStr += writer.comment3("PRND_GENERATOR: vRand=RND(vAcc, sSeed, vTid)")
        kStr += ".macro PRND_GENERATOR vRand vAcc vTemp0 vTemp1 %s" % writer.endLine
        ## Previous implementation
        #kStr += inst("v_lshrrev_b32", "v[\\vRand]",  hex(5), "\\vAcc", "vRand = vAcc >> 5")
        #kStr += inst("v_lshl_or_b32", "v[\\vRand]", "\\vAcc", hex(27), "v[\\vRand]", "vRand = vRand | vAcc << 27")
        #kStr += inst("v_mov_b32", "v[\\vTemp0]", "0x42fe83a3", "" )
        #kStr += inst("v_mul_u32_u24", "v[\\vRand]", "v[\\vRand]", "v[\\vTemp0]", "VRand = vRand * vTemp0")   # mult lower 24 bits should be enough
        #kStr += inst("v_mov_b32", "v[\\vTemp0]", "0x6acc2047", "" )
        #kStr += inst("v_mul_u32_u24", "v[\\vTemp1]", vgpr("Serial"), "v[\\vTemp0]", "VRand = vTid * vTemp0")
        #kStr += inst("v_mov_b32", "v[\\vTemp0]", "0xdfc231fd", "" )
        #kStr += inst("v_xor_b32", "v[\\vRand]", "v[\\vRand]", "v[\\vTemp0]", "VRand = vRand ^ vTemp0")
        #kStr += inst("v_xor_b32", "v[\\vRand]", "v[\\vRand]", "v[\\vTemp1]", "VRand = vRand ^ vTemp0")
        #kStr += inst("v_xor_b32", "v[\\vRand]", "v[\\vRand]", sgpr("RNDSeed"), "VRand = vRand ^ sSeed")

        ## New implementation (instruction scheduling to minimize the dependencies?)
        kStr += inst("v_and_b32", "v[\\vTemp0]", "0xFFFF", "\\vAcc" ,"vTemp0 = vAcc & 0xFFFF")
        kStr += inst("v_lshrrev_b32", "v[\\vTemp1]",  hex(16), "\\vAcc", "vTemp1 = vAcc >> 16")
        kStr += inst("v_xor_b32", "v[\\vTemp0]", "v[\\vTemp0]", "v[\\vTemp1]", "VTemp0 = vTemp0 ^ vTemp1")
        kStr += inst("v_and_b32", "v[\\vTemp1]", "v[\\vTemp0]", "31" ,"vTemp1 = vTemp0 & 31")
        kStr += inst("v_lshlrev_b32", "v[\\vTemp1]",  hex(11), "v[\\vTemp1]", "vTemp1 = vTemp1 << 11")
        kStr += inst("v_lshl_or_b32", "v[\\vTemp0]", "v[\\vTemp0]", hex(5), "v[\\vTemp1]", "vTemp0 = vTemp0 << 5 | vTemp1")
        kStr += inst("v_mul_u32_u24", "v[\\vTemp0]","0x700149" , "v[\\vTemp0]", "VTemp0 = vTemp0 * 0x700149")   # mult lower 24 bits should be enough??
        kStr += inst("v_mul_u32_u24", "v[\\vTemp1]", 229791 , vgpr("Serial"), "VTemp1 = vTid * 229791")  # TODO: use index of C/D instead of local Tid
        kStr += inst("v_xor_b32", "v[\\vRand]", "0x1337137", "v[\\vTemp0]" , "VRand = vTemp0 ^ 0x1337137")
        kStr += inst("v_xor_b32", "v[\\vRand]", "v[\\vRand]", "v[\\vTemp1]", "VRand = vRand ^ vTemp1")
        kStr += inst("v_xor_b32", "v[\\vRand]", "v[\\vRand]", sgpr("RNDSeed"), "VRand = vRand ^ sSeed")

        ## NOTE: Some ideas on validation:
        #     1. to test with existing validator: if we use integer initialization pattern and the output is <=16, it will work since no rounding for int up to 16.0 for fp8.
        #     2. We can use same RND (e.g., 0) in both reference and gpu kernel by commenting out following line.
        #     3. If we use 0xFFFFFFFF, cvt_sr will always round the value up. So, tests with existing validator may fail if we don't ensure this in reference kernel of Tensile host
        #     4. A better way to validate:
        #        Fix the value of RNDSeed from the caller, Save the output of this macro-function and compare it with quantization kernel's (TF-SIM's) output.
        #kStr += inst("v_mov_b32", "v[\\vRand]", "0x0", "vRand = 0x0" )
        #kStr += inst("v_mov_b32", "v[\\vRand]", "0xFFFFFFFF", "VRand = 0xffffffff" )
        ###kStr += inst("v_mov_b32", "v[\\vRand]", sgpr("RNDSeed"), "vRand = RNDSeed" )
        kStr += ".endm%s" % writer.endLine

        return kStr
