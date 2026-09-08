/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <gtest/gtest.h>
#include "platform.hpp"

#define THROW_PLATFORM_EXCEPTION(what) \
    throw std::runtime_error(what + std::string(" at ") + __FILE__ + ":" + std::to_string(__LINE__))

// ==================== Device ====================

Device::Device(miopenHandle_t) {}

Device::~Device() {}

DevMem Device::Malloc(size_t size) const { return {*this, size}; }

bool Device::Synchronize() const
{
    auto status = hipDeviceSynchronize();
    if(status != hipSuccess)
        return false;
    return true;
}

// ==================== DevMem ====================

DevMem::DevMem(const Device&, size_t size)
{
    auto status = hipMalloc(&ptr, size);
    if(status != hipSuccess)
    {
        THROW_PLATFORM_EXCEPTION("hipMalloc error");
    }
}

DevMem::~DevMem()
{
    // ASSERT_* cannot be used in destructors (generates illegal return-void).
    auto err = hipFree(ptr);
    if(err != hipSuccess)
    {
        fprintf(
            stderr, "hipFree failed: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);
        abort();
    }
}

void* DevMem::Data() const { return ptr; }

bool DevMem::CopyToDevice(const void* src, size_t size) const
{
    auto status = hipMemcpy(ptr, src, size, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        return false;
    return true;
}

bool DevMem::CopyFromDevice(void* dst, size_t size) const
{
    auto status = hipMemcpy(dst, ptr, size, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        return false;
    return true;
}
