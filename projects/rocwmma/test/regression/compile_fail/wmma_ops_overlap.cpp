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
// Two operations that serve the same key, one natively and one by padding K, are a
// configuration error: list order never decides between them.
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    namespace detail
    {
        constexpr uint32_t TestTargetId = 0xFFFFu;

        template <>
        struct wmma_target_ops<TestTargetId>
        {
            using type = wmma_op_list<wmma_f32_16x16x16_f16_gfx12, wmma_f32_16x16x32_f16_gfx1250>;
        };

        static_assert(
            MmaTraits<amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 32u, TestTargetId>>::
                is_supported);
    }
}
