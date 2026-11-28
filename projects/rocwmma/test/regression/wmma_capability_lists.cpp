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
// Capability lists compose: a target can have another target's operations plus
// its own, as superset targets do. Checked on a made-up target ID.
#include <gtest/gtest.h>
#include <rocwmma/rocwmma.hpp>

#include "wmma_target_backend.hpp"

namespace rocwmma
{
    namespace detail
    {
        constexpr uint32_t SupersetTargetId = 0xFFFEu;

        template <>
        struct wmma_target_ops<SupersetTargetId>
        {
            using type
                = wmma_op_list_concat_t<WmmaGfx12Ops, wmma_op_list<wmma_f32_16x16x4_f32_gfx125x>>;
        };

        struct wmma_test_op_a;
        struct wmma_test_op_b;
        struct wmma_test_op_c;

        static_assert(is_same_v<wmma_op_list_concat_t<>, wmma_op_list<>>);
        static_assert(is_same_v<wmma_op_list_concat_t<wmma_op_list<wmma_test_op_a>,
                                                      wmma_op_list<>,
                                                      wmma_op_list<wmma_test_op_b, wmma_test_op_c>>,
                                wmma_op_list<wmma_test_op_a, wmma_test_op_b, wmma_test_op_c>>);
    }

    namespace
    {
        template <typename InputT, typename ComputeT, uint32_t BlockKTest>
        using SelectedForSuperset = SelectedWmmaFor<detail::SupersetTargetId,
                                                    InputT,
                                                    InputT,
                                                    ComputeT,
                                                    16u,
                                                    16u,
                                                    BlockKTest>;

        // From the gfx12 list, including padding
        static_assert(MmaTraits<SelectedForSuperset<float16_t, float32_t, 64u>>::BlockK == 16u);
        static_assert(MmaTraits<SelectedForSuperset<uint8_t, int32_t, 8u>>::BlockK == 8u);
        // From the added operation, including padding
        static_assert(MmaTraits<SelectedForSuperset<float32_t, float32_t, 64u>>::BlockK == 4u);
        static_assert(MmaTraits<SelectedForSuperset<float32_t, float32_t, 2u>>::BlockK == 2u);
        static_assert(MmaTraits<SelectedForSuperset<float32_t, float32_t, 2u>>::is_supported);
    }
}

TEST(WmmaCapabilityLists, Composition)
{
    // All checks are static_asserts, evaluated in every compilation pass.
    SUCCEED();
}
