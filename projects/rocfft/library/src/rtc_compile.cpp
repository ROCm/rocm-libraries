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

#include <hip/hiprtc.h>
#include <stdexcept>

std::vector<char> compile_inprocess(const std::string& kernel_src, const std::string& gpu_arch)
{
    struct RaiiState
    {
        hiprtcProgram prog = nullptr;
        ~RaiiState()
        {
            if(prog)
            {
                hiprtcDestroyProgram(&prog);
            }
        }
    };
    RaiiState state;

    // give it a .hip extension so it'll be compiled as HIP code
    if(hiprtcCreateProgram(&state.prog, kernel_src.c_str(), "rocfft_rtc.hip", 0, nullptr, nullptr)
       != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }

    std::string gpu_arch_arg = "--gpu-architecture=" + gpu_arch;

    std::vector<const char*> options;
    options.push_back("-O3");
    options.push_back("-std=c++14");
    options.push_back(gpu_arch_arg.c_str());
    options.push_back("-mcumode");
#ifdef ADDRESS_SANITIZER
    options.push_back("-fsanitize=address");
#endif

    auto compileResult = hiprtcCompileProgram(state.prog, options.size(), options.data());
    if(compileResult != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(state.prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(state.prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(std::string(log.begin(), log.end()));
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetCodeSize(state.prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetCode(state.prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    return code;
}
