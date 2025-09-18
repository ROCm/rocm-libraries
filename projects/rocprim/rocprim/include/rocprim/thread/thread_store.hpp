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

#ifndef ROCPRIM_THREAD_THREAD_STORE_HPP_
#define ROCPRIM_THREAD_THREAD_STORE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"
#include "thread_copy.hpp"

#include <stdint.h>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

/// \defgroup thread_store Thread Store Functions
/// \ingroup threadmodule

/// \addtogroup thread_store
/// @{

/// \brief These enum values are used to specify caching behaviour on store
enum cache_store_modifier
{
    store_default     = 0, ///< Default (no modifier)
    store_wb          = 1, ///< Cache write-back all coherent levels
    store_cg          = 2, ///< Cache at global level
    store_nontemporal = 3, ///< Cache streaming (likely not to be accessed again after storing)
    store_wt          = 4, ///< Cache write-through (to system memory)
    store_volatile    = 5, ///< Volatile (any memory space)
    store_count       = 6
};

/// @}
// end group thread_store

namespace detail
{

template<cache_store_modifier CacheStoreModifier = store_default, typename T>
ROCPRIM_DEVICE __forceinline__
void asm_thread_store(void* ptr, T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

#if ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS == 1

    // NOTE: the reason there is an interim_type is because of a bug for 8bit types.
    // TODO fix flat_store_ubyte and flat_store_sbyte issues

    // Important for syncing. Check section 9.2.2 or 7.3 in the following document
    // http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
    #define ROCPRIM_ASM_THREAD_STORE(cache_modifier,                                            \
                                     llvm_cache_modifier,                                       \
                                     type,                                                      \
                                     interim_type,                                              \
                                     asm_operator,                                              \
                                     output_modifier,                                           \
                                     wait_inst,                                                 \
                                     wait_cmd)                                                  \
        template<>                                                                              \
        ROCPRIM_DEVICE __forceinline__ void asm_thread_store<cache_modifier, type>(void* ptr,   \
                                                                                   type  val)   \
        {                                                                                       \
            interim_type temp_val = *bit_cast<interim_type*>(&val);                             \
            asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier "\n\t" wait_inst wait_cmd \
                                       "(%2)"                                                   \
                         :                                                                      \
                         : "v"(ptr), #output_modifier(temp_val), "I"(0x00));                    \
        }

    // TODO fix flat_store_ubyte and flat_store_sbyte issues
    // TODO Add specialization for custom larger data types
    // clang-format off
#define ROCPRIM_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_inst, wait_cmd)                                   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_inst, wait_cmd);       \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_inst, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_inst, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_inst, wait_cmd); \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_inst, wait_cmd);
    // clang-format on

    #if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "sc0 nt", "s_waitcnt", "");
    #elif defined(__GFX12__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg,
                               "th:TH_DEFAULT scope:SCOPE_DEV",
                               "s_wait_storecnt_dscnt",
                               "");
    #else
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "glc slc", "s_waitcnt", "");
    #endif

#endif

} // namespace detail

/// \addtogroup thread_store
/// @{

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam CacheStoreModifier        Value in enum for determine which type of cache store modifier to be used
/// \tparam OutputIteratorT Type of Output Iterator
/// \tparam T               Type of Data to be stored
/// \param itr [in]         Iterator to location where data is to be stored
/// \param val [in]         Data to be stored
template<cache_store_modifier CacheStoreModifier = store_default,
         typename OutputIteratorT,
         typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
void thread_store(OutputIteratorT itr, T val)
{
    thread_store<CacheStoreModifier>(&(*itr), val);
}

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam CacheStoreModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T        Type of Data to be stored
/// \param ptr [in]  Pointer to location where data is to be stored
/// \param val [in]  Data to be stored
template<cache_store_modifier CacheStoreModifier = store_default, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheStoreModifier == store_default || CacheStoreModifier == store_wb, void>
    thread_store(T* ptr, T val)
{
    detail::thread_fused_copy(ptr, &val, [](auto& dst, const auto& src) { dst = src; });
}

/// \brief Global cached store.
///
/// \tparam CacheStoreModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of data to be stored.
/// \param ptr [in] Pointer to place where to store data.
/// \param val [in] Data to be stored.
template<cache_store_modifier CacheStoreModifier, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheStoreModifier == store_cg, void> thread_store(T* ptr, T val)
{
    detail::asm_thread_store(ptr, val);
}

/// \brief Volatile thread store.
///
/// \tparam CacheStoreModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T Type of data to be stored.
/// \param ptr [in] Pointer to place where to store data.
/// \param val [in] Data to be stored.
template<cache_store_modifier CacheStoreModifier, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheStoreModifier == store_volatile || CacheStoreModifier == store_wt, void>
    thread_store(T* ptr, T val)
{
    detail::thread_fused_copy(ptr,
                              &val,
                              [](auto& dst, const auto& src)
                              {
                                  using U = std::remove_reference_t<decltype(dst)>;
                                  *static_cast<volatile U*>(&dst) = src;
                              });
}

/// \brief Store with non-temporal hint.
///
/// Non-temporal stores help the compiler and hardware to optimize storing
/// data which not expected to be re-used, for example by bypassing the
/// data cache.
///
/// \tparam CacheStoreModifier Value in enum for determine which type of cache store modifier to be used
/// \tparam T - Type of data to be stored.
/// \param ptr [in] Pointer to place where to store data.
/// \param val [in] Data to be stored.
template<cache_store_modifier CacheStoreModifier, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
std::enable_if_t<CacheStoreModifier == store_nontemporal, void> thread_store(T* ptr, T val)
{
#if __has_builtin(__builtin_nontemporal_store)
    detail::thread_fused_copy(ptr,
                              &val,
                              [](auto& dst, const auto& src)
                              { __builtin_nontemporal_store(src, &dst); });
#else
    thread_store(ptr, val);
#endif
}

/// @}
// end group thread_store

END_ROCPRIM_NAMESPACE

#endif
