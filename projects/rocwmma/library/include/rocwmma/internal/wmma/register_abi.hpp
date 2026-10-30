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
#ifndef ROCWMMA_WMMA_REGISTER_ABI_HPP
#define ROCWMMA_WMMA_REGISTER_ABI_HPP

#include "../config.hpp"
#include "../vector_types.hpp"

// Whether the compiler provides __builtin_amdgcn_<name> in the pass being compiled.
// Only meaningful in device passes: the host pass reports no amdgcn builtins.
// Tests may simulate a compiler that lacks builtins: defining
// ROCWMMA_TEST_NO_AMDGCN_BUILTIN_<name>=1 hides one, and defining
// ROCWMMA_TEST_NO_AMDGCN_BUILTINS hides all of them.
#if defined(ROCWMMA_TEST_NO_AMDGCN_BUILTINS)
#define ROCWMMA_HAS_BUILTIN_IMPL(builtin) 0
#elif defined(__has_builtin)
#define ROCWMMA_HAS_BUILTIN_IMPL(builtin) __has_builtin(builtin)
#else
#define ROCWMMA_HAS_BUILTIN_IMPL(builtin) 0
#endif
#define ROCWMMA_HAS_AMDGCN_BUILTIN(name)              \
    ((ROCWMMA_TEST_NO_AMDGCN_BUILTIN_##name + 0) == 0 \
     && ROCWMMA_HAS_BUILTIN_IMPL(__builtin_amdgcn_##name))

namespace rocwmma
{
    namespace detail
    {
        // Native vector type that a builtin takes or returns
        template <typename DataT, uint32_t Rank>
        using wmma_native_vec_t = DataT __attribute__((ext_vector_type(Rank)));

        // Views rocWMMA registers as the native vector type of a builtin operand.
        // The bits are unchanged; the sizes must match.
        struct wmma_abi
        {
            template <typename NativeT, typename RegsT>
            ROCWMMA_DEVICE static inline NativeT const& as(RegsT const& regs)
            {
                static_assert(sizeof(NativeT) == sizeof(RegsT),
                              "rocWMMA: register and builtin operand sizes differ");
                return reinterpret_cast<NativeT const&>(to_native_vector(regs));
            }

            template <typename NativeT, typename RegsT>
            ROCWMMA_DEVICE static inline NativeT& as(RegsT& regs)
            {
                static_assert(sizeof(NativeT) == sizeof(RegsT),
                              "rocWMMA: register and builtin operand sizes differ");
                return reinterpret_cast<NativeT&>(to_native_vector(regs));
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_REGISTER_ABI_HPP
