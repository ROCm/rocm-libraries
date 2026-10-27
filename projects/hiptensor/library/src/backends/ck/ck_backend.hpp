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

// The Composable Kernel (CK) backend: the default backend. It wraps the CK solution code of
// the library unchanged. Its kernel selection and its per-plan state stay where they were
// (the plan preference): contraction kernels are selected at plan creation, elementwise and
// reduction kernels at the first execution of the plan.

#ifndef HIPTENSOR_BACKENDS_CK_CK_BACKEND_HPP
#define HIPTENSOR_BACKENDS_CK_CK_BACKEND_HPP

#include <memory>

#include <ck/stream_config.hpp>

#include "../../backend/backend.hpp"

namespace hiptensor
{
    namespace ckbackend
    {
        // Kernel argument version of the CK backend's kernels, recorded in plan-cache files.
        inline constexpr uint32_t kKernelAbi = 1u;

        std::unique_ptr<backend::IBackend> makeCkBackend();

        // The one conversion of the backend launch configuration to CK's.
        inline StreamConfig toStreamConfig(backend::LaunchConfig const& cfg)
        {
            if(!cfg.timeKernel)
            {
                return StreamConfig{cfg.stream, false};
            }
            return StreamConfig{cfg.stream,
                                true,
                                0,
                                cfg.coldIters,
                                cfg.hotIters,
                                cfg.flushCache,
                                cfg.rotatingBuffers};
        }

        // The launch configuration of a performance-traced execution: timed with the cold and
        // hot run counts of the library options.
        backend::LaunchConfig timedLaunch(backend::LaunchConfig const& cfg);

        // The CK executions behind the public API entry points, with the same arguments, checks,
        // logs and statuses.
        hiptensorStatus_t contract(hiptensorHandle_t            handle,
                                   hiptensorPlan_t              plan,
                                   void const*                  alpha,
                                   void const*                  A,
                                   void const*                  B,
                                   void const*                  beta,
                                   void const*                  C,
                                   void*                        D,
                                   void*                        workspace,
                                   uint64_t                     workspaceSize,
                                   backend::LaunchConfig const& cfg);

        hiptensorStatus_t contractTrinary(hiptensorHandle_t            handle,
                                          hiptensorPlan_t              plan,
                                          void const*                  alpha,
                                          void const*                  A,
                                          void const*                  B,
                                          void const*                  C,
                                          void const*                  beta,
                                          void const*                  D,
                                          void*                        E,
                                          void*                        workspace,
                                          uint64_t                     workspaceSize,
                                          backend::LaunchConfig const& cfg);

        hiptensorStatus_t permute(hiptensorHandle_t            handle,
                                  hiptensorPlan_t              plan,
                                  void const*                  alpha,
                                  void const*                  A,
                                  void*                        B,
                                  backend::LaunchConfig const& cfg);

        hiptensorStatus_t elementwiseBinary(hiptensorHandle_t            handle,
                                            hiptensorPlan_t              plan,
                                            void const*                  alpha,
                                            void const*                  A,
                                            void const*                  gamma,
                                            void const*                  C,
                                            void*                        D,
                                            backend::LaunchConfig const& cfg);

        hiptensorStatus_t elementwiseTrinary(hiptensorHandle_t            handle,
                                             hiptensorPlan_t              plan,
                                             void const*                  alpha,
                                             void const*                  A,
                                             void const*                  beta,
                                             void const*                  B,
                                             void const*                  gamma,
                                             void const*                  C,
                                             void*                        D,
                                             backend::LaunchConfig const& cfg);

        hiptensorStatus_t reduce(hiptensorHandle_t            handle,
                                 hiptensorPlan_t              plan,
                                 void const*                  alpha,
                                 void const*                  A,
                                 void const*                  beta,
                                 void const*                  C,
                                 void*                        D,
                                 void*                        workspace,
                                 uint64_t                     workspaceSize,
                                 backend::LaunchConfig const& cfg);
    } // namespace ckbackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_CK_CK_BACKEND_HPP
