/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_WMMA_RESOLVE_HPP
#define ROCWMMA_WMMA_RESOLVE_HPP

#include "adapters.hpp"
#include "descriptor.hpp"
#include "target_traits.hpp"

// Maps a requested WMMA operation to its implementation for one target: the
// unique native operation with that key, or the unique native operation that
// lists the requested K as paddable. List order never decides between
// candidates; two candidates are a configuration error.
namespace rocwmma
{
    namespace detail
    {
        // Whether Op serves Query, natively or by padding K
        template <typename Op, typename Query>
        constexpr bool wmma_claims_v
            = is_same_v<typename Op::Key::template with_block_k<Query::BlockK>, Query>
              && (Op::Key::BlockK == Query::BlockK || Op::PaddableK::contains(Query::BlockK));

        template <typename Query, typename OpList>
        struct wmma_claim_count;

        template <typename Query, typename... Ops>
        struct wmma_claim_count<Query, wmma_op_list<Ops...>>
        {
            constexpr static uint32_t value = (0u + ... + uint32_t(wmma_claims_v<Ops, Query>));
        };

        // Every key that Op claims has no other claimant in OpList
        template <typename Op, typename OpList, typename PaddableK = typename Op::PaddableK>
        struct wmma_op_is_unique;

        template <typename Op, typename OpList, uint32_t... Ks>
        struct wmma_op_is_unique<Op, OpList, wmma_k_list<Ks...>>
        {
            constexpr static bool value
                = wmma_claim_count<typename Op::Key, OpList>::value == 1u
                  && ((wmma_claim_count<typename Op::Key::template with_block_k<Ks>, OpList>::value
                       == 1u)
                      && ...);
        };

        template <typename OpList>
        struct wmma_ops_are_unique;

        template <typename... Ops>
        struct wmma_ops_are_unique<wmma_op_list<Ops...>>
        {
            constexpr static bool value
                = (wmma_op_is_unique<Ops, wmma_op_list<Ops...>>::value && ... && true);
        };

        // The first operation in the list that claims Query, or void
        template <typename Query, typename OpList>
        struct wmma_find_op
        {
            using type = void;
        };

        template <typename Query, typename Op, typename... Ops>
        struct wmma_find_op<Query, wmma_op_list<Op, Ops...>>
        {
            using type = conditional_t<wmma_claims_v<Op, Query>,
                                       Op,
                                       typename wmma_find_op<Query, wmma_op_list<Ops...>>::type>;
        };

        template <typename Op, uint32_t BlockK>
        struct wmma_implement
        {
            using type = conditional_t<Op::Key::BlockK == BlockK, Op, wmma_pad_k<Op, BlockK>>;
        };

        template <uint32_t BlockK>
        struct wmma_implement<void, BlockK>
        {
            using type = void;
        };

        // Implementation of the requested operation on TargetId, or void if the
        // target has none
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t TargetId>
        struct wmma_resolve
        {
        private:
            using Ops   = wmma_target_ops_t<TargetId>;
            using Query = wmma_key<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>;

            static_assert(wmma_ops_are_unique<Ops>::value,
                          "rocWMMA: the WMMA capability list of this target has two operations "
                          "that serve the same types, block sizes and K");

        public:
            using type =
                typename wmma_implement<typename wmma_find_op<Query, Ops>::type, BlockK>::type;
        };

        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t TargetId>
        using wmma_resolve_t =
            typename wmma_resolve<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK, TargetId>::
                type;

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_RESOLVE_HPP
