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

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN SWAP_RANGES
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    I1 swap_ranges(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        return ::thrust::swap_ranges(::thrust::device, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    I1 swap_ranges(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I0>::iterator_category>();

        return ::std::swap_ranges(::std::execution::par, f0, l0, f1);
    }
    // END SWAP_RANGES
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
