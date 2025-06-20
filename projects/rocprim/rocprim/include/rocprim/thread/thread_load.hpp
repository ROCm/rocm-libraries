/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define ROCPRIM_THREAD_THREAD_LOAD_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../type_traits.hpp" // not be used
#include "thread_copy.hpp"

#include <iterator>
#include <stdint.h>
#include <type_traits>
#include <utility>

BEGIN_ROCPRIM_NAMESPACE

/// \defgroup thread_load Thread Load Functions
/// \ingroup threadmodule

/// \addtogroup thread_load
/// @{

/// \brief These enum values are used to specify caching behaviour on load
enum cache_load_modifier : int
{
    load_default     = 0, ///< Default (no modifier)
    load_ca          = 1, ///< Cache at all levels
    load_cg          = 2, ///< Cache at global level
    load_nontemporal = 3, ///< Cache streaming (likely not to be accessed again after loading)
    load_cv          = 4, ///< Cache as volatile (including cached system lines)
    load_ldg         = 5, ///< Cache as texture
    load_volatile    = 6, ///< Volatile (any memory space)
    load_count       = 7
};

/// @}
// end of group thread_load

namespace detail
{

template<cache_load_modifier CacheLoadModifier = load_default, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T asm_thread_load(void* ptr)
{
    T retval{};
    __builtin_memcpy(static_cast<void*>(&retval), ptr, sizeof(T));
    return retval;
}

#if ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS == 1

    // Important for syncing. Check section 9.2.2 or 7.3 in the following document
    // http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
    #define ROCPRIM_ASM_THREAD_LOAD(cache_modifier,                                             \
                                    llvm_cache_modifier,                                        \
                                    type,                                                       \
                                    interim_type,                                               \
                                    asm_operator,                                               \
                                    output_modifier,                                            \
                                    wait_inst,                                                  \
                                    wait_cmd)                                                   \
        template<>                                                                              \
        ROCPRIM_DEVICE ROCPRIM_INLINE type asm_thread_load<cache_modifier, type>(void* ptr)     \
        {                                                                                       \
            interim_type retval;                                                                \
            asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier "\n\t" wait_inst wait_cmd \
                                       "(%2)"                                                   \
                         : "=" #output_modifier(retval)                                         \
                         : "v"(ptr), "I"(0x00));                                                \
            return *bit_cast<type*>(&retval);                                                   \
        }

    // TODO Add specialization for custom larger data types
    // clang-format off
#define ROCPRIM_ASM_THREAD_LOAD_GROUP(cache_modifier, llvm_cache_modifier, wait_inst, wait_cmd)                                  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int8_t, int32_t, flat_load_sbyte, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int16_t, int32_t, flat_load_sshort, v, wait_inst, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint8_t, uint32_t, flat_load_ubyte, v, wait_inst, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint16_t, uint32_t, flat_load_ushort, v, wait_inst, wait_cmd);  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_load_dword, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_load_dword, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_load_dwordx2, v, wait_inst, wait_cmd); \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_load_dwordx2, v, wait_inst, wait_cmd);
    // clang-format on

    #if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "sc0 nt", "s_waitcnt", "");
    #elif defined(__GFX12__)
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "th:TH_DEFAULT scope:SCOPE_DEV", "s_wait_loadcnt_dscnt", "");
    #else
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "glc slc", "s_waitcnt", "");
    #endif

#endif

} // namespace detail

/// \addtogroup thread_load
/// @{

/// \brief Load data using the load instruction specified by CacheLoadModifier.
/// \tparam CacheLoadModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam InputIteratorT Type of Output Iterator
/// \param itr [in] Iterator to location where data is to be stored
/// \return Data that is loaded from memory
template<cache_load_modifier CacheLoadModifier = load_default, typename InputIteratorT>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::iterator_traits<InputIteratorT>::value_type thread_load(InputIteratorT itr)
{
    using T = typename std::iterator_traits<InputIteratorT>::value_type;
    return thread_load<CacheLoadModifier, T>(&(*itr));
}

/// \brief Load data using the default load instruction.
/// \tparam CacheLoadModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of Data to be loaded
/// \tparam Alignment Explicit alignment of the source data.
/// \param ptr [in] Pointer to data to be loaded
/// \return Data that is loaded from memory
template<cache_load_modifier CacheLoadModifier = load_default,
         typename T,
         size_t Alignment = alignof(T)>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheLoadModifier == load_ca || CacheLoadModifier == load_default
                     || CacheLoadModifier == load_ldg,
                 T> thread_load(T* ptr)
{
    using decay_type = typename std::remove_const_t<T>;
    alignas(Alignment) decay_type result;
    detail::thread_fused_copy<decay_type, T, Alignment>(&result,
                                                        ptr,
                                                        [](auto& dst, const auto& src)
                                                        { dst = src; });
    return result;
}

/// \brief Global thread load.
///
/// \tparam CacheLoadModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of Data to be loaded
/// \tparam Alignment (Unused)
/// \param ptr [in] Input pointer for data that will be loaded
/// \return returns loaded value
template<cache_load_modifier CacheLoadModifier, typename T, size_t Alignment = alignof(T)>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheLoadModifier == load_cg, T> thread_load(T* ptr)
{
    return detail::asm_thread_load<CacheLoadModifier, typename std::remove_const<T>::type>(ptr);
}

/// \brief Volatile thread load.
///
/// \tparam CacheLoadModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of Data to be copied
/// \tparam Alignment Explicit alignment of the source data.
/// \param ptr [in] Input pointer for data that will be copied
/// \return returns loaded value
template<cache_load_modifier CacheLoadModifier, typename T, size_t Alignment = alignof(T)>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheLoadModifier == load_volatile || CacheLoadModifier == load_cv, T>
    thread_load(T* ptr)
{
    using decay_type = typename std::remove_const_t<T>;
    alignas(Alignment) decay_type result;
    detail::thread_fused_copy<decay_type, T, Alignment>(
        &result,
        ptr,
        [](auto& dst, const auto& src)
        {
            using U = std::remove_reference_t<decltype(src)>;
            dst     = *static_cast<const volatile U*>(&src);
        });
    return result;
}

/// \brief Load with non-temporal hint.
///
/// Non-temporal loads help the compiler and hardware to optimize loading
/// data which is not expected to be re-used, for example by bypassing
/// the data cache.
///
/// \tparam CacheLoadModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of data to be loaded.
/// \tparam Alignment Explicit alignment of the source data.
/// \param ptr [in] Pointer to data to be loaded.
/// \return Returns loaded value.
template<cache_load_modifier CacheLoadModifier, typename T, size_t Alignment = alignof(T)>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheLoadModifier == load_nontemporal, T> thread_load(T* ptr)
{
#if __has_builtin(__builtin_nontemporal_load)
    using decay_type = typename std::remove_const_t<T>;
    alignas(Alignment) decay_type result;
    detail::thread_fused_copy<decay_type, T, Alignment>(
        &result,
        ptr,
        [](auto& dst, const auto& src) { dst = __builtin_nontemporal_load(&src); });
    return result;
#else
    return thread_load(ptr);
#endif
}

namespace detail
{

template<cache_load_modifier CacheLoadModifier, typename T, int... Is>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_thread_load_impl(T* src, T* dst, std::integer_sequence<int, Is...>)
{
    // Unroll multiple thread loads by unpacking an integer sequence
    // into a dummy array. We assign the destination values inside the
    // constructor of this dummy array.
    int dummy[] = {(dst[Is] = thread_load<CacheLoadModifier>(src + Is), 0)...};
    (void)dummy;
}

} // namespace detail

/// \brief Load Count number of items from src to dst.
/// \tparam Count number of items to load
/// \tparam CacheLoadModifier the modifier used for the thread_load
/// \tparam T Type of Data to be copied to
/// \param src [in] Input iterator for data that will be loaded in
/// \param dst [out] The pointer the data will be loaded to.
template<int Count, cache_load_modifier CacheLoadModifier, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_thread_load(T* src, T* dst)
{
    detail::unrolled_thread_load_impl<CacheLoadModifier>(src,
                                                         dst,
                                                         std::make_integer_sequence<int, Count>{});
}

/// @}
// end of group thread_load

END_ROCPRIM_NAMESPACE

#endif
