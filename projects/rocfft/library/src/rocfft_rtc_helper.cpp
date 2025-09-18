// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_compile.h"
#include <iostream>
#include <iterator>

#ifdef WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, const char* const* argv)
{
#ifdef WIN32
    // stdout on Windows defaults to text mode and will mangle our code objects
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    try
    {
        if(argc != 2)
        {
            // GPU architecture is passed as a command line argument
            std::cerr << "usage: rocfft_rtc_helper gfxNNN\n";
            throw std::runtime_error("rocfft_rtc_helper: invalid command line");
        }

        std::string gpu_arch = argv[1];

        // collect stdin as kernel source
        std::string kernel_src;
        std::noskipws(std::cin);
        std::copy(std::istream_iterator<char>(std::cin),
                  std::istream_iterator<char>(),
                  std::back_inserter(kernel_src));

        // compile and write code object to stdout
        auto code = compile_inprocess(kernel_src, gpu_arch);
        std::cout.write(code.data(), code.size());
        std::cout.flush();
        if(!std::cout.good())
            return 1;
        return 0;
    }
    catch(std::exception& e)
    {
        // write exception content to stdout - exit status will tell
        // the caller that stdout is an error message instead of
        // compiled code
        std::cout << e.what();
        return 1;
    }
}
