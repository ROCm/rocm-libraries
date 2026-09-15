//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/std/cassert>
#include <cuda/stream_ref>

#include <atomic>
#include <chrono>
#include <thread>

void CUDART_CB callback(cudaStream_t, cudaError_t, void* flag)
{
  std::chrono::milliseconds sleep_duration{1000};
  std::this_thread::sleep_for(sleep_duration);
  assert(!reinterpret_cast<std::atomic_flag*>(flag)->test_and_set());
}

void test_wait(cuda::stream_ref& ref)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  try
  {
    ref.wait();
  }
  catch (...)
  {
    assert(false && "Should not have thrown");
  }
#else
  ref.wait();
#endif // !TEST_HAS_NO_EXCEPTIONS
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    ( // passing case
      cudaStream_t stream; cudaStreamCreate(&stream); std::atomic_flag flag = ATOMIC_FLAG_INIT;
      cudaStreamAddCallback(stream, callback, &flag, 0);
      cuda::stream_ref ref{stream};
      test_wait(ref);
      assert(flag.test_and_set());
      cudaStreamDestroy(stream);))

  return 0;
}
