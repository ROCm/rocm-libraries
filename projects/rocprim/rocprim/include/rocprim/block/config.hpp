// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_BLOCK_CONFIG_HELPER_HPP_
#define ROCPRIM_BLOCK_CONFIG_HELPER_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Padding hints for algorithms. Padding can be used to reduce bank
/// conflicts at the cost of increasing LDS usage and potentially reducing
/// occupancy.
enum class block_padding_hint {
    /// Use padding to avoid bank conflicts, if applicable. This allows an
    /// algorithm to use more shared memory to reduce bank conflicts.
    avoid_conflicts = 0,

    /// Never use padding. This is useful when occupancy needs to be
    /// maximized, and bank conflicts are known to be not an issue.
    never_pad = 1,

    /// Similar to \p block_padding_hint::avoid_conflicts , but only allows
    /// padding when it does not affect theorethical occupancy limited by
    /// shared memory. It's advised to use this when LDS usage is restricting
    /// occupancy.
    lds_occupancy_bound = 2,
};

namespace detail
{
/// \brief Utility wrapper to expose a static constexpr member occupancy of
/// type T as a static constexpr value.
template<typename T>
struct map_occupancy_to_value
{
    /// \brief The original type.
    using type = T;

    /// \brief The value to order this by.
    static constexpr auto value = T::occupancy;
};

/// \brief Selects the config depending on the padding hint.
template<block_padding_hint PaddingHint, typename PaddedConfig, typename UnpaddedConfig>
using select_block_padding_config
    = std::conditional_t<PaddingHint == block_padding_hint::avoid_conflicts,
                         PaddedConfig,
                         std::conditional_t<PaddingHint == block_padding_hint::never_pad,
                                            UnpaddedConfig,
                                            typename detail::select_max_by_value_t<
                                                map_occupancy_to_value<PaddedConfig>,
                                                map_occupancy_to_value<UnpaddedConfig>>::type>>;
} // namespace detail

END_ROCPRIM_NAMESPACE
#endif
