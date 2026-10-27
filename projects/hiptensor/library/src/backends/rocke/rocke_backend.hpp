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

// The rocKE backend, built with HIPTENSOR_ENABLE_ROCKE_BACKEND=ON. Plans select it with
// HIPTENSOR_BACKEND=rocke.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_BACKEND_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_BACKEND_HPP

#include <memory>

#include "../../backend/backend.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // Kernel argument version of the rocKE backend's kernels, recorded in plan-cache files.
        inline constexpr uint32_t kKernelAbi = 1u;

        std::unique_ptr<backend::IBackend> makeRockeBackend();
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_BACKEND_HPP
