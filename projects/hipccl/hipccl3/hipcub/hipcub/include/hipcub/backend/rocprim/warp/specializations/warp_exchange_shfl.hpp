// MIT License
//
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

#ifndef HIPCUB_ROCPRIM_WARP_SPECIALIZATIONS_WARP_EXCHANGE_SHFL_HPP_
#define HIPCUB_ROCPRIM_WARP_SPECIALIZATIONS_WARP_EXCHANGE_SHFL_HPP_

#include "../../../../config.hpp"
#include "../../util_type.hpp"

#include <rocprim/warp/warp_exchange.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<typename InputT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS>
class WarpExchangeShfl
{
    using warp_exchange =
        typename rocprim::warp_exchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

public:
    using TempStorage = NullType;

public:
    explicit HIPCUB_DEVICE __forceinline__ WarpExchangeShfl(TempStorage& /*temp_storage*/) {}

    template<typename OutputT>
    HIPCUB_DEVICE __forceinline__ void
        BlockedToStriped(const InputT (&input_items)[ITEMS_PER_THREAD],
                         OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        warp_exchange{}.blocked_to_striped_shuffle(input_items, output_items);
    }

    template<typename OutputT>
    HIPCUB_DEVICE __forceinline__ void
        StripedToBlocked(const InputT (&input_items)[ITEMS_PER_THREAD],
                         OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        warp_exchange{}.striped_to_blocked_shuffle(input_items, output_items);
    }

    template<typename OffsetT>
    HIPCUB_DEVICE __forceinline__ void ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD],
                                                        OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        (void)items;
        (void)ranks;
        // Always false
        static_assert(sizeof(OffsetT) == 0,
                      "Shuffle specialization of warp exchange does not support\n"
                      "ScatterToStriped(const InputT (&input_items)[ITEMS_PER_THREAD],\n"
                      "                 OutputT (&output_items)[ITEMS_PER_THREAD],\n"
                      "                 OffsetT (&ranks)[ITEMS_PER_THREAD])");
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE __forceinline__ void
        ScatterToStriped(const InputT (&input_items)[ITEMS_PER_THREAD],
                         OutputT (&output_items)[ITEMS_PER_THREAD],
                         OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        (void)input_items;
        (void)output_items;
        (void)ranks;
        // Always false
        static_assert(sizeof(OffsetT) == 0,
                      "Shuffle specialization of warp exchange does not support\n"
                      "ScatterToStriped(const InputT (&input_items)[ITEMS_PER_THREAD],\n"
                      "                 OutputT (&output_items)[ITEMS_PER_THREAD],\n"
                      "                 OffsetT (&ranks)[ITEMS_PER_THREAD])");
    }
};

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_SPECIALIZATIONS_WARP_EXCHANGE_SHFL_HPP_
