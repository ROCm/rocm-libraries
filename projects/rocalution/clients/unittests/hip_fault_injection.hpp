/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

#pragma once

// Fault injection for the HIP, rocBLAS and rocSPARSE calls of the HIP backend.
// The executable defines these functions, so the dynamic linker binds the
// references of librocalution_hip to them instead of to the ROCm libraries.
// A fault applies only to calls from librocalution_hip; all other calls,
// including those of the tests and of the ROCm libraries themselves, are
// forwarded unchanged.

namespace rocalution_unit
{
    enum class HipFault
    {
        none,
        device_count_fails, // hipGetDeviceCount() returns an error
        no_devices, // hipGetDeviceCount() reports zero devices
        set_device_fails, // hipSetDevice() returns an unspecific error
        context_in_use, // hipSetDevice() returns hipErrorContextAlreadyInUse
        blas_handle_fails, // rocblas_create_handle() returns an error
        sparse_handle_fails, // rocsparse_create_handle() returns an error
        low_compute_capability, // hipGetDeviceProperties() reports major 2
        no_registers_per_block, // hipGetDeviceProperties() reports regsPerBlock 0
        blas_destroy_fails, // rocblas_destroy_handle() destroys, then returns an error
        sparse_destroy_fails // rocsparse_destroy_handle() destroys, then returns an error
    };

    // Arms a fault for its lifetime
    class ScopedHipFault
    {
    public:
        explicit ScopedHipFault(HipFault fault);
        ~ScopedHipFault();

        ScopedHipFault(const ScopedHipFault&)            = delete;
        ScopedHipFault& operator=(const ScopedHipFault&) = delete;

        // Number of calls the fault has been applied to
        int injected() const;
    };
}
