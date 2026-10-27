/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

// The COMGR wrapper of the rocKE backend's JIT (HIPTENSOR_ENABLE_ROCKE_JIT): compiles LLVM IR text
// to a code object with the COMGR library hipTensor links, and names that compiler.
//
// The compiler is the linked library, never one found through the environment: the LLVM IR flavor
// the rocKE lowering emits is derived from the linked library's LLVM version, and the compile
// options are fixed here. Calls are serialized behind one mutex.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_COMGR_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_COMGR_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace hiptensor
{
    namespace rockebackend
    {
        // The linked COMGR library.
        struct CompilerInfo
        {
            bool        available = false;
            std::string reason; // why it is not available
            unsigned    llvmMajor = 0, llvmMinor = 0, llvmPatch = 0;
            size_t      comgrMajor = 0, comgrMinor = 0;
            // "llvm-<major>.<minor>.<patch>;comgr-<major>.<minor>": part of every kernel cache
            // entry's identity
            std::string id;
            // The rocKE LLVM IR flavor of this compiler: "llvm20", "llvm22" or "llvm23"
            std::string flavor;
        };

        // Probed once per process.
        CompilerInfo const& compilerInfo();

        // The compile options of every JIT kernel.
        std::vector<std::string> const& jitCompileOptions();

        // The COMGR ISA name of a target, e.g. "amdgcn-amd-amdhsa--gfx950".
        std::string isaName(std::string const& baseArch);

        // Compiles one LLVM IR module (text) to a code object: SOURCE (LLVM IR) -> BC ->
        // relocatable -> executable. False with `log` set on failure.
        bool compileLlvmIr(std::string const&              llvmText,
                           std::string const&              isa,
                           std::vector<std::string> const& options,
                           std::vector<uint8_t>&           codeObject,
                           std::string&                    log);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_COMGR_HPP
