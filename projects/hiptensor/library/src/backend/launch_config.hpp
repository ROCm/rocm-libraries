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

#ifndef HIPTENSOR_BACKEND_LAUNCH_CONFIG_HPP
#define HIPTENSOR_BACKEND_LAUNCH_CONFIG_HPP

#include <hip/hip_runtime_api.h>

namespace hiptensor
{
    namespace backend
    {
        // How a backend launches the kernels of one execution. It replaces the kernel library's
        // own stream configuration at the backend interface, so the solution layer carries no
        // kernel library type. Every kernel of an execution runs on `stream`.
        struct LaunchConfig
        {
            //! The caller's stream; nullptr is the legacy default stream
            hipStream_t stream = nullptr;
            //! Time the kernels (performance traces and measured selection only)
            bool timeKernel = false;
            //! Untimed warm-up launches and timed launches when timeKernel is set
            int coldIters = 0;
            int hotIters  = 1;
            //! Flush the caches before each timed launch
            bool flushCache = false;
            //! Number of rotating input buffers for cold-cache timing
            int rotatingBuffers = 1;
        };
    } // namespace backend
} // namespace hiptensor

#endif // HIPTENSOR_BACKEND_LAUNCH_CONFIG_HPP
