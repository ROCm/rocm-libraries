//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda/std/__cccl/os.h>

#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#  if _CCCL_OS(WINDOWS)
#    include <windows.h>
#  endif

#  if _CCCL_OS(LINUX)
#    include <unistd.h>
#  endif

#  if _CCCL_OS(ANDROID)
#    include <android/api-level.h>
#  endif

#  if _CCCL_OS(QNX)
#    include <qnx.h>
#  endif
#endif

int main(int, char**)
{
  static_assert(_CCCL_OS(WINDOWS) + _CCCL_OS(LINUX) == 1, "");
#if _CCCL_OS(ANDROID) || _CCCL_OS(QNX)
  static_assert(_CCCL_OS(LINUX) == 1, "");
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) == 1, "");
#endif
#if _CCCL_OS(LINUX)
  static_assert(_CCCL_OS(WINDOWS) == 0, "");
#endif
#if _CCCL_OS(WINDOWS)
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) + _CCCL_OS(LINUX) == 0, "");
#endif
  return 0;
}
