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

/*! \file thrust/system/hip/hipstdpar/hipstdpar_lib.hpp
 *  \brief Forwarding header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

// Interposed allocations
#  if defined(__HIPSTDPAR_INTERPOSE_ALLOC__)
#    include "impl/interpose_allocations.hpp"
#  endif
// Parallel STL algorithms
#  include "impl/batch.hpp"
#  include "impl/copy.hpp"
#  include "impl/generation.hpp"
#  include "impl/heap.hpp"
#  include "impl/lexicographical_comparison.hpp"
#  include "impl/merge.hpp"
#  include "impl/min_max.hpp"
#  include "impl/numeric.hpp"
#  include "impl/order_changing.hpp"
#  include "impl/partitioning.hpp"
#  include "impl/removing.hpp"
#  include "impl/search.hpp"
#  include "impl/set.hpp"
#  include "impl/sorting.hpp"
#  include "impl/swap.hpp"
#  include "impl/transformation.hpp"
#  include "impl/uninitialized.hpp"

#endif
