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

#ifndef ROCFFT_RTC_GENERATOR_H
#define ROCFFT_RTC_GENERATOR_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

// runtime_compile dispatches to subclasses, those subclasses
// return callables to do the code generation, so that the
// compilation and caching can live in the cached_compile method.
//
// function to generate the name of a kernel, given no arguments
using kernel_name_gen_t = std::function<std::string()>;
// functor to generate the source code of a kernel, given the
// kernel name.  but remember the source in case we're asked to
// generate it again.
struct kernel_src_gen_t
{
    using generator_func = std::function<std::string(const std::string&)>;

    // allow default ctor + assign, plus ctor + assign from function
    kernel_src_gen_t() = default;
    kernel_src_gen_t(generator_func f)
        : f(f)
    {
    }
    kernel_src_gen_t& operator=(const kernel_src_gen_t&) = default;
    kernel_src_gen_t& operator                           =(generator_func f)
    {
        this->f = f;
        kernel_src.clear();
        return *this;
    }
    // forward a call to the function, if we don't already have the source
    std::string operator()(const std::string& kernel_name)
    {
        if(kernel_src.empty())
            kernel_src = f(kernel_name);
        return kernel_src;
    }
    operator bool() const
    {
        return static_cast<bool>(f);
    }

private:
    generator_func f;
    std::string    kernel_src;
};

#endif
