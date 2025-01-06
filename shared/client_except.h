// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_CLIENT_EXCEPT_H
#define ROCFFT_CLIENT_EXCEPT_H

#include <string>

// exception type to throw when we want to skip a problem
struct ROCFFT_SKIP
{
    const std::string msg;
    ROCFFT_SKIP(std::string&& s)
        : msg(std::move(s))
    {
    }
    ROCFFT_SKIP(const std::string& s)
        : msg(s)
    {
    }
};

// exception type to throw when we want to consider a problem failed
struct ROCFFT_FAIL
{
    const std::string msg;
    ROCFFT_FAIL(std::string&& s)
        : msg(std::move(s))
    {
    }
    ROCFFT_FAIL(const std::string& s)
        : msg(s)
    {
    }
};

#endif
