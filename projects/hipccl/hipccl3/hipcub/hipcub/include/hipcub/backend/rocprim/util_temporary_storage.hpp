/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2026, Advanced Micro Devices, Inc.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_UTIL_TEMPORARY_STORAGE_HPP_
#define HIPCUB_ROCPRIM_UTIL_TEMPORARY_STORAGE_HPP_

#include "../../config.hpp"
#include "../../util_deprecated.hpp"

#include <rocprim/detail/temp_storage.hpp> // IWYU pragma: export

#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
// Base case: When N == 0
template<size_t N, typename Generator, typename... Ts>
HIPCUB_HOST_DEVICE
typename std::enable_if<N == 0, hipError_t>::type generate_partition(void*   d_temp_storage,
                                                                     size_t& temp_storage_bytes,
                                                                     Generator /*gen*/,
                                                                     Ts... args)
{
    return rocprim::detail::temp_storage::partition(
        d_temp_storage,
        temp_storage_bytes,
        rocprim::detail::temp_storage::linear_partition<Ts...>(args...));
}

// Recursive case: When N > 0
template<size_t N, typename Generator, typename... Ts>
HIPCUB_HOST_DEVICE
typename std::enable_if<(N > 0), hipError_t>::type
    generate_partition(void* d_temp_storage, size_t& temp_storage_bytes, Generator gen, Ts... args)
{
    return generate_partition<N - 1>(d_temp_storage, temp_storage_bytes, gen, gen(N - 1), args...);
}

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_TEMPORARY_STORAGE_HPP_
