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

// The contraction JIT of the rocKE backend (HIPTENSOR_ENABLE_ROCKE_JIT): for a contraction that
// rocKE's GETT route tables (the pack's tuned and AOT routes) do not hold, plans created with
// HIPTENSOR_JIT_MODE_DEFAULT run rocKE's tensor GETT product planner (rocke/tensor_gett_planner.h,
// the C++ twin of rocKE's plan_tensor_gett: the MN-layout GETT on the GEMM dispatcher's pick,
// single launch or with tensor_pack copies, or the K-contiguous route with both operands packed)
// and build the launches it chooses with rocKE's C++ kernel builders, every size and stride a
// constant. The kernels are compiled with COMGR and kept in the JIT's kernel cache
// (rocke_jit.hpp; spec keys kGettJitSpecKeyPrefix).
//
// The planner ranks GEMM bodies with rocKE's selection models (LightGBM text models), which the
// pack build ships next to each pack that has a planner (<pack dir>/gemm_models, with an index of
// SHA-256 digests the loader checks). Without them, or on a target the planner does not serve,
// no contraction takes this route.
//
// Plans created with HIPTENSOR_JIT_MODE_NONE never reach this code.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_GETT_JIT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_GETT_JIT_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "../../backend/problem.hpp"
#include "rocke_runtime.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // The planned route of one contraction: the launches (pack steps, then the GETT) and
        // the kernel specs they need. Immutable; shared by the plans of one problem.
        struct GettJitPlan;

        // What planGettJit decided.
        struct GettJitPlanInfo
        {
            uint64_t    workspaceBytes = 0; // the launches' workspace (256-byte pack offsets)
            int         launches       = 0; // pack steps + 1
            int64_t     M = 0, N = 0, K = 0; // the GETT's GEMM view (K padded as the kernel runs)
            int         block = 0; // the GETT's threads per workgroup
            int64_t     ctas  = 0; // the GETT's workgroups
            int         tileM = 0, tileN = 0, tileK = 0;
            std::string route; // "<kind>/<candidate>" of the planner, e.g. "single/tensor_gett_mn"
            std::string kernel; // the GETT kernel's name
            // the route as a selection feature (CandidateFacts::gettKind) and the planner's
            // predicted time of the route in nanoseconds (rounded; 0 when the planner's time model
            // does not predict it)
            int     kind        = 0;
            int64_t predictedNs = 0;
        };

        // Plans `problem` (a contraction: A, B, optional C, D) with rocKE's tensor GETT planner
        // for the device the problem names. False with `reason` when the problem is outside the
        // planner's scope (f16 / bf16 data of one type with the 32F compute type, no operators,
        // C laid out as D, the target, the layouts its kernels read) or the planner finds no
        // route, or the route needs more than `workspaceLimit` bytes of workspace. Planning
        // decisions are kept per process (the same problem plans once).
        bool planGettJit(backend::Problem const&             problem,
                         Pack const&                         pack,
                         uint64_t                            workspaceLimit,
                         std::shared_ptr<GettJitPlan const>& out,
                         GettJitPlanInfo&                    info,
                         std::string&                        reason);

        // What building a planned route did.
        struct GettJitReport
        {
            int    kernels      = 0; // distinct kernels of the route
            int    compiled     = 0; // compiled now
            int    memoryHits   = 0; // found in the kernel cache
            int    fileHits     = 0; // of the memory hits: read from a cache file
            double milliseconds = 0.0;
        };

        // The launches of a planned route (the kernels from the kernel cache, else compiled now
        // while `budgetMs` (0: none) is not spent), appended to `program` (whose workspace and
        // buffers the steps use as planGettJit laid them out). Never runs a kernel. False with
        // `reason` when a kernel cannot be had (the builder or the compiler refuses it, the time
        // budget is spent, HIPTENSOR_ROCKE_JIT_NO_COMPILE).
        bool buildGettJitProgram(GettJitPlan const&        plan,
                                 backend::DeviceKey const& device,
                                 double                    budgetMs,
                                 Program&                  program,
                                 GettJitReport&            report,
                                 std::string&              reason);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_GETT_JIT_HPP
