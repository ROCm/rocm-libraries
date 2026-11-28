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

// Reads the kernel descriptions of an AMDGPU code object: the NT_AMDGPU_METADATA note (msgpack)
// of the ELF file. The JIT takes every kernel's argument layout from it, never from a second
// description, as the pack builder does for the kernel pack.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_CODEOBJECT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_CODEOBJECT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "rocke_runtime.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // The kernel `name` of a code object: its arguments (hidden arguments dropped), kernarg
        // segment size, LDS, scratch, workgroup and wavefront sizes. False with `reason` set
        // when the code object is not a valid AMDGPU ELF file or has no such kernel.
        bool readKernelMetadata(uint8_t const*     code,
                                size_t             size,
                                std::string const& name,
                                PackKernel&        out,
                                std::string&       reason);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_CODEOBJECT_HPP
