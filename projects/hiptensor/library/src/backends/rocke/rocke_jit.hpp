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
// allows them. A specialized kernel is built for the problem's values by rocKE's C++ kernel
// builder (rocke/instance_hiptensor_aot.h, the twin of rocKE's jit_ew.py, linked into the
// library), lowered to LLVM IR for the linked compiler's flavor and compiled with COMGR
// (rocke_comgr.hpp). No Python runs, and the JIT needs no data file of its own. Plans created with
// HIPTENSOR_JIT_MODE_NONE never reach this code. The contraction JIT (rocke_gett_jit.hpp) builds the
// launches of rocKE's tensor GETT planner through jitKernel() and keeps them in the same cache.
//
// Kernel cache: process-wide, content addressed. An entry's identity is the SHA-256 of (ISA, the
// SHA-256 of the LLVM IR, compile options, fresh-process flag, compiler id); the kernel class, the
// rocKE engine build and the values (the spec key) are aliases that find it without building the
// kernel. The cache files (hiptensorWriteKernelCacheToFile / hiptensorReadKernelCacheFromFile)
// hold the entries of one device class (architecture, target features, compute unit count,
// partition modes); reading merges, writing overwrites. Entries compiled by another compiler are
// kept but not served.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP

#include <cstdint>
#include <functional>
#include <string>

#include <hiptensor/hiptensor_types.h>

#include "rocke/ir.h"

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
        // `spentMs`: JIT time the plan spent before (its contraction route's kernels), counted
        // against HIPTENSOR_JIT_MAX_MS.
        void specializeProgram(Program&                  program,
                               backend::DeviceKey const& device,
                               JitReport&                report,
                               double                    spentMs = 0.0);

        // The scheme of the spec keys of the GETT JIT's kernels (rocke_gett_jit.hpp):
        // "<prefix>|<arch>|<kernel name>|<rocKE build id>|<SHA-256 of the kernel spec>". Bump the
        // version when a key's kernel can change for the same name, build and spec (the
        // lowering or compile options of the JIT changing without changing the LLVM IR).
        inline constexpr char kGettJitSpecKeyPrefix[] = "hiptensor-rocke-gett-jit/v1";

        // A kernel the JIT builds with one of rocKE's C++ kernel builders, compiled with rocKE's
        // COMGR options for that kernel (rocke_comgr_options_for_kernel) and kept in the kernel
        // cache under `specKey`.
        struct JitKernelSource
        {
            std::string specKey;
            // the PackKernel family the kernel gets (logs, plan descriptions)
            std::string family;
            // what the logs and cache file entries call the kernel
            std::string label;
            // builds the kernel into an uninitialized builder; nullptr with the builder's error
            // set when it refuses
            std::function<rocke_kernel_def_t*(rocke_ir_builder_t*)> build;
        };

        // A kernel of the cache, its function on the current device, and how it was obtained.
        struct JitKernel
        {
            PackKernel const* kernel    = nullptr;
            hipFunction_t     function  = nullptr;
            bool              compiled  = false; // compiled now (else a cache hit)
            bool              fileHit   = false; // the hit's entry was read from a cache file
            double            frontMs   = 0.0; // build and lowering (when compiled)
            double            compileMs = 0.0; // COMGR (when compiled)
        };

        // Whether the JIT can compile (the linked COMGR library); false with the reason.
        bool jitCompilerAvailable(std::string& reason);

        // The kernel of `source` from the kernel cache (and HIPTENSOR_KERNEL_CACHE_DIR), else
        // built and compiled now when `allowCompile` (single flight per spec key: concurrent
        // callers of one key wait for one compile). False with `reason` when it cannot be had;
        // a refusal of the builder or the compiler is remembered for the process.
        bool jitKernel(JitKernelSource const&    source,
                       backend::DeviceKey const& device,
                       bool                      allowCompile,
                       JitKernel&                out,
                       std::string&              reason);

        // The kernel cache files (cuTENSOR semantics; see hiptensor.h).
        hiptensorStatus_t writeKernelCacheFile(char const*               fileName,
                                               backend::DeviceKey const& device);
        hiptensorStatus_t readKernelCacheFile(char const*               fileName,
                                              backend::DeviceKey const& device);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_JIT_HPP
