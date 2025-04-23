/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_HANDLE_HPP
#define HIPTENSOR_HANDLE_HPP

#include <new>

#include <hip/hip_runtime_api.h>

#include "hip_device.hpp"

namespace hiptensor
{
    // hiptensorHandle_t wrapper object
    struct Handle
    {
    public:
        Handle()  = default;
        ~Handle() = default;

        static Handle  createHandle(int64_t* buff); // Calls constructor for all member variables
        static void    destroyHandle(int64_t* buff); // Calls destructor for all member variables
        static Handle* toHandle(int64_t* buff); // Reinterprets input buffer as Handle class

        HipDevice getDevice();

    private:
        HipDevice mDevice;
    };
} // namespace hiptensor

#endif // HIPTENSOR_HANDLE_HPP
