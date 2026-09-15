//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

static_assert(cuda::std::is_default_constructible<cuda::stream_ref>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, int>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, cuda::std::nullptr_t>::value, "");

template <class T, class = void>
constexpr bool has_value_type = false;

template <class T>
constexpr bool has_value_type<T, cuda::std::void_t<typename T::value_type>> = true;
static_assert(has_value_type<cuda::stream_ref>, "");

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      { // default construction
        cuda::stream_ref ref;
        static_assert(noexcept(cuda::stream_ref{}), "");
        assert(ref.get() == reinterpret_cast<cudaStream_t>(0));
      }

      { // from stream
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(42);
        cuda::stream_ref ref{stream};
        static_assert(noexcept(cuda::stream_ref{stream}), "");
        assert(ref.get() == reinterpret_cast<cudaStream_t>(42));
      }))

  return 0;
}
