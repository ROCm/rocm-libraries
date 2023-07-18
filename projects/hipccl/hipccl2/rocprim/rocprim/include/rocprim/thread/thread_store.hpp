/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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

BEGIN_ROCPRIM_NAMESPACE

enum cache_store_modifier
{
    store_default,   ///< Default (no modifier)
    store_wb,        ///< Cache write-back all coherent levels
    store_cg,        ///< Cache at global level
    store_cs,        ///< Cache streaming (likely to be accessed once)
    store_wt,        ///< Cache write-through (to system memory)
    store_volatile,  ///< Volatile shared (any memory space)
};

namespace detail
{

template<cache_store_modifier MODIFIER = store_default, typename T>
ROCPRIM_DEVICE __forceinline__ void AsmThreadStore(void * ptr, T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

#if ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS == 1

// NOTE: the reason there is an interim_type is because of a bug for 8bit types.
// TODO fix flat_store_ubyte and flat_store_sbyte issues

// Important for syncing. Check section 9.2.2 or 7.3 in the following document
// http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
#define ROCPRIM_ASM_THREAD_STORE(cache_modifier,                                                              \
                                llvm_cache_modifier,                                                         \
                                type,                                                                        \
                                interim_type,                                                                \
                                asm_operator,                                                                \
                                output_modifier,                                                             \
                                wait_cmd)                                                                    \
    template<>                                                                                               \
    ROCPRIM_DEVICE __forceinline__ void AsmThreadStore<cache_modifier, type>(void * ptr, type val)            \
    {                                                                                                        \
        interim_type temp_val = val;                                                          \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier : : "v"(ptr), #output_modifier(temp_val)); \
        asm volatile("s_waitcnt " wait_cmd "(%0)" : : "I"(0x00));                                            \
    }

// TODO fix flat_store_ubyte and flat_store_sbyte issues
// TODO Add specialization for custom larger data types
#define ROCPRIM_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_cmd);       \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_cmd);      \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_cmd); \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_cmd);

// [HIP-CPU] MSVC: erronous inline assembly specification (Triggers error C2059: syntax error: 'volatile')

#ifndef __HIP_CPU_RT__
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "sc0 sc1", ""); // TODO: gfx942 validation
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "sc0 sc1", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "sc0 sc1", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "sc0 sc1", "vmcnt");
#else
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "glc", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "glc slc", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "glc", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "glc", "vmcnt");
#endif
// TODO find correct modifiers to match these
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cs, "", "");
#endif // __HIP_CPU_RT__

#endif

}

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER        - Value in enum for determine which type of cache store modifier to be used
/// \tparam OutputIteratorT - Type of Output Iterator
/// \tparam T               - Type of Data to be stored
/// \param itr [in]         - Iterator to location where data is to be stored
/// \param val [in]         - Data to be stored
template <
    cache_store_modifier MODIFIER = store_default,
    typename OutputIteratorT,
    typename T
>
ROCPRIM_DEVICE ROCPRIM_INLINE void thread_store(
    OutputIteratorT itr,
    T               val)
{
    thread_store<MODIFIER>(&(*itr), val);
}

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER        - Value in enum for determine which type of cache store modifier to be used
/// \tparam T               - Type of Data to be stored
/// \param ptr [in] - Pointer to location where data is to be stored
/// \param val [in] - Data to be stored
template <
    cache_store_modifier MODIFIER = store_default,
    typename T
>
ROCPRIM_DEVICE ROCPRIM_INLINE void thread_store(
    T *ptr,
    T val)
{
#ifndef __HIP_CPU_RT__
    detail::AsmThreadStore<MODIFIER, T>(ptr, val);
#else
    std::memcpy(ptr, &val, sizeof(T));
#endif
}

END_ROCPRIM_NAMESPACE

#endif
