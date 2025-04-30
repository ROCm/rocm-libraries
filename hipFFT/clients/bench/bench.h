// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPFFT_BENCH_H
#define HIPFFT_BENCH_H

#include "../hipfft_params.h"
#include "hipfft/hipfft.h"

#include <vector>

// This is used to either wrap a HIP function call, or to explicitly check a variable
// for an error condition.  If an error occurs, we throw.
// Note: std::runtime_error does not take unicode strings as input, so only strings
// supported
inline void
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

inline void lib_V_Throw(hipfftResult       res,
                        const std::string& msg,
                        size_t             lineno,
                        const std::string& fileName)
{
    if(res != HIPFFT_SUCCESS)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

#endif // HIPFFT_BENCH_H
