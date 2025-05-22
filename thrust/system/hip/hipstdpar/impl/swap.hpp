// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/hipstdpar/impl/swap.hpp
 *  \brief <tt>Swap operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#  include <thrust/execution_policy.h>
#  include <thrust/swap.h>

#  include <algorithm>
#  include <execution>
#  include <utility>

#  include "hipstd.hpp"

namespace std
{
// BEGIN SWAP_RANGES
template <typename I0, typename I1, enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
inline I1 swap_ranges(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
{
  return ::thrust::swap_ranges(::thrust::device, f0, l0, f1);
}

template <typename I0, typename I1, enable_if_t<!::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
inline I1 swap_ranges(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I0>::iterator_category,
                                          typename iterator_traits<I0>::iterator_category>();

  return ::std::swap_ranges(::std::execution::par, f0, l0, f1);
}
// END SWAP_RANGES
} // namespace std
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
