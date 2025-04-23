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

#ifndef HIPTENSOR_VERSION_HPP_
#define HIPTENSOR_VERSION_HPP_

// clang-format off
#define HIPTENSOR_MAJOR_VERSION       @hiptensor_VERSION_MAJOR@
#define HIPTENSOR_MINOR_VERSION       @hiptensor_VERSION_MINOR@
#define HIPTENSOR_PATCH_VERSION       @hiptensor_VERSION_PATCH@
// clang-format on

//! @brief Returns the version number of hipTensor
//! @details Return the version with three least significant digits for patch version,
//! the next three digits for minor version, and the most significant digits for major version.
//! @returns The version number calculated as major * 10000 + minor * 100 + patch.
inline size_t hiptensorGetVersion()
{
    return HIPTENSOR_MAJOR_VERSION * 1e6 + HIPTENSOR_MINOR_VERSION * 1e3 + HIPTENSOR_PATCH_VERSION;
}

#endif // HIPTENSOR_VERSION_HPP_
