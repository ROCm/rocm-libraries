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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
BEGIN_HIPCUB_NAMESPACE

enum CacheStoreModifier
{
    STORE_DEFAULT,   ///< Default (no modifier)
    STORE_WB,        ///< Cache write-back all coherent levels
    STORE_CG,        ///< Cache at global level
    STORE_CS,        ///< Cache streaming (likely to be accessed once)
    STORE_WT,        ///< Cache write-through (to system memory)
    STORE_VOLATILE,  ///< Volatile shared (any memory space)
};

// TODO add to detail namespace
// TODO cleanup
template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ void AsmThreadStore(void * ptr, T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

#if HIPCUB_THREAD_STORE_USE_CACHE_MODIFIERS == 1

// NOTE: the reason there is an interim_type is because of a bug for 8bit types.
// TODO fix flat_store_ubyte and flat_store_sbyte issues

// Important for syncing. Check section 9.2.2 or 7.3 in the following document
// https://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
#define HIPCUB_ASM_THREAD_STORE(cache_modifier,                                                              \
                                llvm_cache_modifier,                                                         \
                                type,                                                                        \
                                interim_type,                                                                \
                                asm_operator,                                                                \
                                output_modifier,                                                             \
                                wait_cmd)                                                                    \
    template<>                                                                                               \
    HIPCUB_DEVICE __forceinline__ void AsmThreadStore<cache_modifier, type>(void * ptr, type val)            \
    {                                                                                                        \
        interim_type temp_val = val;                                                          \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier : : "v"(ptr), #output_modifier(temp_val)); \
        asm volatile("s_waitcnt " wait_cmd "(%0)" : : "I"(0x00));                                            \
    }

// TODO fix flat_store_ubyte and flat_store_sbyte issues
// TODO Add specialization for custom larger data types
#define HIPCUB_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_cmd);       \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_cmd);     \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_cmd);     \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_cmd); \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_cmd);

#if defined(__gfx940__) || defined(__gfx941__)
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WB, "sc0 sc1", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CG, "sc0 sc1", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WT, "sc0 sc1", "vmcnt");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_VOLATILE, "sc0 sc1", "vmcnt");
#elif defined(__gfx942__)
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WB, "sc0", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CG, "sc0 nt", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WT, "sc0", "vmcnt");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_VOLATILE, "sc0", "vmcnt");
#else
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WB, "glc", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CG, "glc slc", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WT, "glc", "vmcnt");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_VOLATILE, "glc", "vmcnt");
#endif

// TODO find correct modifiers to match these
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CS, "", "");

#endif

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename OutputIteratorT, typename T>
__device__ __forceinline__ void ThreadStore(OutputIteratorT itr, T val)
{
    ThreadStore<MODIFIER>(&(*itr), val);
}

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename T>
__device__ __forceinline__ void ThreadStore(T * ptr, T val)
{
    AsmThreadStore<MODIFIER, T>(ptr, val);
}

END_HIPCUB_NAMESPACE
#endif
