/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "instruction/extension.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void ext_inst(nb::module_ m)
{
    m.def("SLongBranch",
          &rocisa::SLongBranch,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("positiveLabelStr"),
          nb::arg("comment") = "");
    m.def("SGetPositivePCOffset",
          &rocisa::SGetPositivePCOffset,
          nb::arg("sgprIdx"),
          nb::arg("label"),
          nb::arg("ContinuousRegister"));
    m.def("SLongBranchPositive",
          &rocisa::SLongBranchPositive,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("comment") = "");
    m.def("SLongBranchNegative",
          &rocisa::SLongBranchNegative,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("comment") = "");
    m.def("SCLongBranchScc0",
          &rocisa::SCLongBranchScc0,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("noBranchLabelStr"),
          nb::arg("positiveLabelStr"),
          nb::arg("posNeg")  = 0,
          nb::arg("comment") = "");
    m.def("SCLongBranchScc1",
          &rocisa::SCLongBranchScc1,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("noBranchLabelStr"),
          nb::arg("positiveLabelStr"),
          nb::arg("posNeg")  = 0,
          nb::arg("comment") = "");
    m.def("SCLongBranchVccnz",
          &rocisa::SCLongBranchVccnz,
          nb::arg("label"),
          nb::arg("tmpSgprRes"),
          nb::arg("noBranchLabelStr"),
          nb::arg("positiveLabelStr"),
          nb::arg("posNeg")  = 0,
          nb::arg("comment") = "");
    m.def("SMulInt64to32",
          &rocisa::SMulInt64to32,
          nb::arg("dst0"),
          nb::arg("dst1"),
          nb::arg("src0"),
          nb::arg("src1"),
          nb::arg("tmpVgprRes"),
          nb::arg("hasSMulHi") = false,
          nb::arg("sign")      = false,
          nb::arg("comment")   = "");
    m.def("VCvtBF16toFP32",
          &rocisa::VCvtBF16toFP32,
          nb::arg("dst"),
          nb::arg("src"),
          nb::arg("vgprMask").none(),
          nb::arg("vi"),
          nb::arg("comment") = "");
}
