/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2024, Advanced Micro Devices, Inc.  All rights reserved.
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

BEGIN_ROCPRIM_NAMESPACE

/// \defgroup thread_store Thread Store Functions
/// \ingroup threadmodule

/// \addtogroup thread_store
/// @{

/// \brief These enum values are used to specify caching behaviour on store
enum cache_store_modifier
{
    store_default,   ///< Default (no modifier)
    store_wb,        ///< Cache write-back all coherent levels
    store_cg,        ///< Cache at global level
    store_cs,        ///< Cache streaming (likely to be accessed once)
    store_wt,        ///< Cache write-through (to system memory)
    store_volatile,  ///< Volatile shared (any memory space)
};

/// @}
// end group thread_store

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
    #define ROCPRIM_ASM_THREAD_STORE(cache_modifier,                                            \
                                     llvm_cache_modifier,                                       \
                                     type,                                                      \
                                     interim_type,                                              \
                                     asm_operator,                                              \
                                     output_modifier,                                           \
                                     wait_inst,                                                 \
                                     wait_cmd)                                                  \
        template<>                                                                              \
        ROCPRIM_DEVICE __forceinline__ void AsmThreadStore<cache_modifier, type>(void* ptr,     \
                                                                                 type  val)     \
        {                                                                                       \
            interim_type temp_val = *bit_cast<interim_type*>(&val);                             \
            asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier "\n\t" wait_inst wait_cmd \
                                       "(%2)"                                                   \
                         :                                                                      \
                         : "v"(ptr), #output_modifier(temp_val), "I"(0x00));                    \
        }

// TODO fix flat_store_ubyte and flat_store_sbyte issues
// TODO Add specialization for custom larger data types
#define ROCPRIM_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_inst, wait_cmd)                                   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_inst, wait_cmd);       \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_inst, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_inst, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_inst, wait_cmd); \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_inst, wait_cmd);

#if defined(__gfx940__) || defined(__gfx941__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "sc0 sc1", "s_waitcnt", ""); // TODO: gfx942 validation
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "sc0 sc1", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "sc0 sc1", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "sc0 sc1", "s_waitcnt", "vmcnt");
#elif defined(__gfx942__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "sc0", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "sc0 nt", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "sc0", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "sc0", "s_waitcnt", "vmcnt");
#elif defined(__gfx1200__) ||  defined(__gfx1201__)
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "scope:SCOPE_DEV", "s_wait_storecnt_dscnt", ""); // TODO: gfx942 validation
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "th:TH_DEFAULT scope:SCOPE_DEV", "s_wait_storecnt_dscnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "scope:SCOPE_DEV", "s_wait_storecnt_dscnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "scope:SCOPE_DEV", "s_wait_storecnt_dscnt", "");
#else
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "glc", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "glc slc", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "glc", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "glc", "s_waitcnt", "vmcnt");
#endif
// TODO find correct modifiers to match these
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cs, "", "s_waitcnt", "");

#endif

}

/// \addtogroup thread_store
/// @{

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER        Value in enum for determine which type of cache store modifier to be used
/// \tparam OutputIteratorT Type of Output Iterator
/// \tparam T               Type of Data to be stored
/// \param itr [in]         Iterator to location where data is to be stored
/// \param val [in]         Data to be stored
template<cache_store_modifier MODIFIER = store_default, typename OutputIteratorT, typename T>
[[deprecated("Use a dereference instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE void
    thread_store(OutputIteratorT itr, T val)
{
    thread_store<MODIFIER>(&(*itr), val);
}

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER Value in enum for determine which type of cache store modifier to be used
/// \tparam T        Type of Data to be stored
/// \param ptr [in]  Pointer to location where data is to be stored
/// \param val [in]  Data to be stored
template<cache_store_modifier MODIFIER = store_default, typename T>
[[deprecated("Use a dereference instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE void thread_store(T* ptr,
                                                                                             T  val)
{
    detail::AsmThreadStore<MODIFIER, T>(ptr, val);
}

/// @}
// end group thread_store

END_ROCPRIM_NAMESPACE

#endif
