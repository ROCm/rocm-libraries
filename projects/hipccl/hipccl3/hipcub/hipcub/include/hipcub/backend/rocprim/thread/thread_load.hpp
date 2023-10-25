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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_
BEGIN_HIPCUB_NAMESPACE

enum CacheLoadModifier : int32_t
{
    LOAD_DEFAULT,   ///< Default (no modifier)
    LOAD_CA,        ///< Cache at all levels
    LOAD_CG,        ///< Cache at global level
    LOAD_CS,        ///< Cache streaming (likely to be accessed once)
    LOAD_CV,        ///< Cache as volatile (including cached system lines)
    LOAD_LDG,       ///< Cache as texture
    LOAD_VOLATILE,  ///< Volatile (any memory space)
};

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval = 0;
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

#if HIPCUB_THREAD_LOAD_USE_CACHE_MODIFIERS == 1

// Important for syncing. Check section 9.2.2 or 7.3 in the following document
// https://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
#define HIPCUB_ASM_THREAD_LOAD(cache_modifier,                                                                \
                               llvm_cache_modifier,                                                           \
                               type,                                                                          \
                               interim_type,                                                                  \
                               asm_operator,                                                                  \
                               output_modifier,                                                               \
                               wait_cmd)                                                                      \
    template<>                                                                                                \
    HIPCUB_DEVICE __forceinline__ type AsmThreadLoad<cache_modifier, type>(void * ptr)                        \
    {                                                                                                         \
        interim_type retval;                                                                                  \
        asm volatile(                                                                                         \
            #asm_operator " %0, %1 " llvm_cache_modifier "\n"                                                 \
            "\ts_waitcnt " wait_cmd "(0)" : "=" #output_modifier(retval) : "v"(ptr)                            \
        );                                                                                                    \
        return retval;                                                                                        \
    }

// TODO Add specialization for custom larger data types
#define HIPCUB_ASM_THREAD_LOAD_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                  \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_load_sbyte, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_load_sshort, v, wait_cmd);    \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_load_ubyte, v, wait_cmd);    \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_load_ushort, v, wait_cmd);  \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_load_dword, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_load_dword, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_load_dwordx2, v, wait_cmd); \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_load_dwordx2, v, wait_cmd);

#if defined(__gfx940__) || defined(__gfx941__)
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CA, "sc0", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CG, "sc1", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CV, "sc0 sc1", "vmcnt");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_VOLATILE, "sc0 sc1", "vmcnt");
#elif defined(__gfx942__)
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CA, "sc0", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CG, "sc0 nt", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CV, "sc0", "vmcnt");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_VOLATILE, "sc0", "vmcnt");
#else
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CA, "glc", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CG, "glc slc", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CV, "glc", "vmcnt");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_VOLATILE, "glc", "vmcnt");
#endif

// TODO find correct modifiers to match these
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_LDG, "", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CS, "", "");

#endif

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename InputIteratorT>
HIPCUB_DEVICE __forceinline__
typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    using T  = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = ThreadLoad<MODIFIER>(&(*itr));
    return retval;
}

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T
ThreadLoad(T * ptr)
{
    return AsmThreadLoad<MODIFIER, T>(ptr);
}

END_HIPCUB_NAMESPACE
#endif
