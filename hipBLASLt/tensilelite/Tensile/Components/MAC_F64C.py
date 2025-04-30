################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import DataType, Module
from ..Component import Component, MAC

class FMA_F64C_Plain(MAC):
    asmCaps = {"v_fma_f64": True}
    kernel = {"ProblemType": {"DataType": DataType(DataType.complexDouble)}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("FMA_F64C_Plain")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        vars = {}
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]

        for b in range(0, kernel["ThreadTile1"]):
            vars["b"] = b
            for a in range(0, kernel["ThreadTile0"]):
                vars["a"] = a
                for iui in range(0, innerUnroll):
                    vars["iui"] = iui
                    # c.real += a.real * b.real
                    cStr = "v[vgprValuC+({a}+{b}*{ThreadTile0})*4+0:(vgprValuC+{a}+{b}*{ThreadTile0})*4+1]".format_map(vars)
                    aStr = "v[vgprValuA_X{m}_I{iui}+{a}*4+0:vgprValuA_X{m}_I{iui}+{a}*4+1]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui}+{b}*4+0:vgprValuB_X{m}_I{iui}+{b}*4+1]".format_map(vars)
                    module.addInst("v_fma_f64", cStr, aStr, bStr, cStr, "")
                    # c.real -= a.imag * b.imag
                    cStr = "v[vgprValuC+({a}+{b}*{ThreadTile0})*4+0:(vgprValuC+{a}+{b}*{ThreadTile0})*4+1]".format_map(vars)
                    aStr = "v[vgprValuA_X{m}_I{iui}+{a}*4+2:vgprValuA_X{m}_I{iui}+{a}*4+3]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui}+{b}*4+2:vgprValuB_X{m}_I{iui}+{b}*4+3]".format_map(vars)
                    sign = "-" if (not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]) or \
                            (kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]) else ""
                    module.addInst("v_fma_f64", cStr, aStr, sign + bStr, cStr, "")
                    # c.imag += a.real * b.imag
                    cStr = "v[vgprValuC+({a}+{b}*{ThreadTile0})*4+2:(vgprValuC+{a}+{b}*{ThreadTile0})*4+3]".format_map(vars)
                    aStr = "v[vgprValuA_X{m}_I{iui}+{a}*4+0:vgprValuA_X{m}_I{iui}+{a}*4+1]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui}+{b}*4+2:vgprValuB_X{m}_I{iui}+{b}*4+3]".format_map(vars)
                    sign = "-" if kernel["ProblemType"]["ComplexConjugateB"] else ""
                    module.addInst("v_fma_f64", cStr, aStr, sign + bStr, cStr, "")
                    # c.imag += a.imag * b.real
                    cStr = "v[vgprValuC+({a}+{b}*{ThreadTile0})*4+2:(vgprValuC+{a}+{b}*{ThreadTile0})*4+3]".format_map(vars)
                    aStr = "v[vgprValuA_X{m}_I{iui}+{a}*4+2:vgprValuA_X{m}_I{iui}+{a}*4+3]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui}+{b}*4+0:vgprValuB_X{m}_I{iui}+{b}*4+1]".format_map(vars)
                    sign = "-" if kernel["ProblemType"]["ComplexConjugateA"] else ""
                    module.addInst("v_fma_f64", cStr, sign + aStr, bStr, cStr, "")

                    module.add(priority(writer, 1, "Raise priority while processing macs"))

        module.add(priority(writer, 0, "Reset priority after macs"))
        return module
