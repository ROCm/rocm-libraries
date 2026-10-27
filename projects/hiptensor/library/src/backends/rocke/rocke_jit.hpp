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

// The JIT of the rocKE backend (HIPTENSOR_ENABLE_ROCKE_JIT) and its kernel cache.
//
// Plans created with HIPTENSOR_JIT_MODE_DEFAULT replace the launches of the pack's generic
// elementwise kernel (permutations, elementwise binary / trinary operations, the operand packing
// and epilogues of contractions) by kernels specialized to the problem: every size, stride and
// operator code a constant, and vector accesses along the output's fastest mode where the layout
// allows them. A specialized kernel comes from a recipe (rocke.recipes, recorded from the rocKE
// builder at build time) that the rocKE recipe VM replays for the problem's values; it is lowered
// to LLVM IR for the linked compiler's flavor and compiled with COMGR (rocke_comgr.hpp). No Python
// runs. Plans created with HIPTENSOR_JIT_MODE_NONE never reach this code.
//
// Kernel cache: process-wide, content addressed. An entry's identity is the SHA-256 of (ISA, the
// SHA-256 of the LLVM IR, compile options, fresh-process flag, compiler id); the recipe key and
// its values (the spec key) are aliases that find it without replaying the recipe. The cache
// files (hiptensorWriteKernelCacheToFile / hiptensorReadKernelCacheFromFile) hold the entries of
// one device class (architecture, target features, compute unit count, partition modes); reading
// merges, writing overwrites. Entries compiled by another compiler are kept but not served.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP

#include <cstdint>
#include <string>

#include <hiptensor/hiptensor_types.h>

#include "../../backend/problem.hpp"
#include "rocke_runtime.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // What the JIT did for a plan.
        struct JitReport
        {
            int         launches     = 0; // elementwise launches of the program
            int         specialized  = 0; // replaced by specialized kernels
            int         compiled     = 0; // of which compiled now
            int         memoryHits   = 0; // found in the kernel cache (compiled or read before)
            int         fileHits     = 0; // of the memory hits: entries read from a cache file
            int         declined     = 0; // kept on the pack kernel (no gain, refused, budget)
            double      milliseconds = 0.0;
            std::string note;
        };

        // Specializes the elementwise launches of a program whose functions are loaded
        // (loadFunctions). Never fails the plan: a launch that does not specialize keeps its pack
        // kernel. Every specialized launch keeps its pack launch as the fallback execution takes
        // when a buffer is not aligned for the vector accesses, or on another device.
        void specializeProgram(Program&                  program,
                               Pack const&               pack,
                               backend::DeviceKey const& device,
                               JitReport&                report);

        // The kernel cache files (cuTENSOR semantics; see hiptensor.h).
        hiptensorStatus_t writeKernelCacheFile(char const*               fileName,
                                               backend::DeviceKey const& device);
        hiptensorStatus_t readKernelCacheFile(char const*               fileName,
                                              backend::DeviceKey const& device);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP
