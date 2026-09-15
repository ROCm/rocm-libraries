//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "hiprtcc_common.h"
#include <stdio.h>

// Arch configs are strings and bools determining architecture and ptx/sass compilation
using ArchConfig          = std::tuple<std::string, bool>;
constexpr auto archString = [](const ArchConfig& a) -> const auto& {
  return std::get<0>(a);
};
constexpr auto isArchReal = [](const ArchConfig& a) -> const auto& {
  return std::get<1>(a);
};

using ArgList = std::vector<std::string>;

using GpuProg = std::vector<char>;

// Takes arguments for building a file and returns the path to the output file
GpuProg nvrtc_build_prog(const std::string& testCu, const ArchConfig& config, const ArgList& argList)
{
  // Assemble arguments
  std::vector<const char*> optList;
  // NOTE(HIPRTC): -fgpu-rdc is required to generate LLVM IR
  optList.emplace_back("-fgpu-rdc");
  // Be careful with lifetimes here
  std::for_each(argList.begin(), argList.end(), [&](const auto& it) {
    optList.emplace_back(it.c_str());
  });
  
  // Use the translated architecture
  std::string gpu_arch("--offload-arch=" + archString(config));
  optList.emplace_back(gpu_arch.c_str());

  fprintf(stderr, "NVRTC opt list:\r\n");
  for (const auto& it : optList)
  {
    fprintf(stderr, "  %s\r\n", it);
  }
  
  fprintf(stderr, "Compiling program...\r\n");
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, testCu.c_str(), "test.cu", 0, nullptr, nullptr));

  nvrtcResult compile_result = nvrtcCompileProgram(prog, optList.size(), optList.data());

  fprintf(stderr, "Collecting logs...\r\n");
  size_t log_size;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));

  {
    std::unique_ptr<char[]> log{new char[log_size]};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    printf("%s\r\n", log.get());
  }

  if (compile_result != NVRTC_SUCCESS)
  {
    exit(1);
  }

  size_t codeSize;
  GpuProg code;
  
  if (isArchReal(config))
  {
    NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
    code.resize(codeSize);
    NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code.data()));
  }
  else
  {
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &codeSize));
    code.resize(codeSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, code.data()));
  }

  std::vector<hiprtcJIT_option> option_keys;
  std::vector<void*> option_vals;
  int arch = std::atoi(archString(config).substr(3).c_str()); 
  option_keys.push_back(HIPRTC_JIT_TARGET);
  printf("++++ arch %d\n", arch);
  option_vals.push_back((void*)(intptr_t)&arch);
  hiprtcLinkState rtc_link_state;
  NVRTC_SAFE_CALL(hiprtcLinkCreate((unsigned)option_keys.size(), // number of options
                  option_keys.data(),            // Array of options
                  option_vals.data(),            // Array of option values cast to void*
                  &rtc_link_state ));             // hiprtc link state created upon success

  NVRTC_SAFE_CALL(hiprtcLinkAddData(rtc_link_state,        // hiprtc link state
                  HIPRTC_JIT_INPUT_LLVM_BITCODE,            // type of the input data or bitcode
                  code.data(),          // input data which is null terminated
                  codeSize,         // size of the input data
                  "test.cu",                   // optional name for this input
                  0,                     // size of the options
                  0,                     // Array of options applied to this input
                  0));
  
  void* codePtr = nullptr;
  size_t codeSize2 = 0;

  NVRTC_SAFE_CALL(hiprtcLinkComplete(rtc_link_state,       // hiprtc link state
                  &codePtr,              // upon success, points to the output binary
                  &codeSize2));         // size of the binary is stored (optional)
  GpuProg code2(reinterpret_cast<char*>(codePtr), reinterpret_cast<char*>(codePtr) + codeSize2);
  code = code2;

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return code;
}
