// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "compute_scheme.h"

#include <map>
#include <set>
#include <stdexcept>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)
#define STRENUM(x) TO_STR(x), x

static const std::map<ComputeScheme, const char*>& ComputeSchemetoStringMap()
{
    static const std::map<ComputeScheme, const char*> ComputeSchemetoString
        = {{ENUMSTR(CS_NONE)},
           {ENUMSTR(CS_KERNEL_STOCKHAM)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_RC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CR)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_USING_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_HERM)},
           {ENUMSTR(CS_KERNEL_COPY_HERM_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_R)},

           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_EVEN)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_CMPLX_TO_R)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_CMPLX_TO_R)},
           {ENUMSTR(CS_REAL_2D_EVEN)},
           {ENUMSTR(CS_REAL_3D_EVEN)},

           {ENUMSTR(CS_BLUESTEIN)},
           {ENUMSTR(CS_KERNEL_CHIRP)},
           {ENUMSTR(CS_KERNEL_PAD_MUL)},
           {ENUMSTR(CS_KERNEL_FFT_MUL)},
           {ENUMSTR(CS_KERNEL_RES_MUL)},
           {ENUMSTR(CS_KERNEL_BLUESTEIN_SINGLE)},

           {ENUMSTR(CS_L1D_TRTRT)},
           {ENUMSTR(CS_L1D_CC)},
           {ENUMSTR(CS_L1D_CRT)},

           {ENUMSTR(CS_2D_STRAIGHT)},
           {ENUMSTR(CS_2D_RTRT)},
           {ENUMSTR(CS_2D_RC)},
           {ENUMSTR(CS_KERNEL_2D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_2D_SINGLE)},

           {ENUMSTR(CS_3D_STRAIGHT)},
           {ENUMSTR(CS_3D_TRTRTR)},
           {ENUMSTR(CS_3D_RTRT)},
           {ENUMSTR(CS_3D_BLOCK_RC)},
           {ENUMSTR(CS_3D_BLOCK_CR)},
           {ENUMSTR(CS_3D_RC)},
           {ENUMSTR(CS_KERNEL_3D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_3D_SINGLE)}};
    return ComputeSchemetoString;
}

std::string PrintScheme(ComputeScheme cs)
{
    return ComputeSchemetoStringMap().at(cs);
}

static std::map<std::string, ComputeScheme> StrToComputeSchemeMap()
{
    std::map<std::string, ComputeScheme> String2ComputeScheme;
    for(auto i : ComputeSchemetoStringMap())
        String2ComputeScheme.emplace(i.second, i.first);
    return String2ComputeScheme;
}

ComputeScheme StrToComputeScheme(const std::string& str)
{
    static auto csmap = StrToComputeSchemeMap();
    return csmap.at(str);
}

// schemes that could be a root problem, not a kernel
// TODO- It would be better to refactor the ComputeScheme, might be good to define
// things like PROB_DESC (such as 3D_C2C, LARGE_1D_C2C) , ALGORITHM (STOCKHAM...),
// DECOMPOSITION (2D_RTRT, 1D_CC, 1D_TRTRT)...
static const std::set<ComputeScheme>& ProblemScheme()
{
    static const std::set<ComputeScheme> ProblemSchemeSet = {(CS_KERNEL_STOCKHAM),
                                                             (CS_REAL_TRANSFORM_USING_CMPLX),
                                                             (CS_REAL_TRANSFORM_EVEN),
                                                             (CS_REAL_2D_EVEN),
                                                             (CS_REAL_3D_EVEN),
                                                             (CS_BLUESTEIN),
                                                             (CS_L1D_TRTRT),
                                                             (CS_L1D_CC),
                                                             (CS_L1D_CRT),
                                                             (CS_2D_RTRT),
                                                             (CS_2D_RC),
                                                             (CS_KERNEL_2D_SINGLE),
                                                             (CS_3D_TRTRTR),
                                                             (CS_3D_RTRT),
                                                             (CS_3D_BLOCK_RC),
                                                             (CS_3D_BLOCK_CR),
                                                             (CS_3D_RC)};

    return ProblemSchemeSet;
}

bool ComputeSchemeIsAProblem(ComputeScheme cs)
{
    return ProblemScheme().count(cs) != 0;
}

std::string PrintKernelSchemeAbbr(ComputeScheme cs)
{
    switch(cs)
    {
    case CS_KERNEL_STOCKHAM:
        return "sbrr";
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
        return "sbcc";
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
        return "sbcr";
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
        return "sbrc";
    case CS_KERNEL_2D_SINGLE:
        return "2d_single";
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        return "sbrc_xy_z";
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        return "sbrc_z_xy";
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
        return "sbrc_erc_z_xy";
    default:
        throw std::runtime_error("unsupported scheme in PrintKernelSchemeAbbr");
    }

    return "";
}
