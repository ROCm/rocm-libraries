// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_AMD_SEMAPHORE_HPP
#define CK_AMD_SEMAPHORE_HPP

#include "amd_address_space.hpp"

// #define CK_USE_AMD_SEMAPHORE_ASM 1

namespace ck {

enum SemaphoreAddressSpaceMask : index_t
{
    SemaphoreAddressSpaceGlobal     = 1 << index_t(AddressSpaceEnum::Global),
    SemaphoreAddressSpaceShared     = 1 << index_t(AddressSpaceEnum::Lds),
    SemaphoreAddressSpaceLaneShared = 1 << 8,
    SemaphoreAddressSpaceAsync      = 1 << 9,
};

enum WaveUsageId : index_t
{
    WaveIdLoad    = 0,
    WaveIdRun     = 1,
    WaveIdPostRun = 2,
    WaveIdOutput  = 3
};

template <unsigned WaveIdInWavegroup
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
          ,
          unsigned SemId
#endif
          >
class WavegroupSemaphore
{
    public:
    __device__ void init(unsigned count = 0, unsigned limit = 1, bool done = 0)
    {
#if defined(__gfx13__)
        if(__builtin_amdgcn_wave_id_in_wavegroup() == WaveIdInWavegroup)
        {
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
            static_assert(SemId >= 1 && SemId <= 7, "GFX13 only support Sem 1-7");
            constexpr uint32_t SemaHwReg   = 0x00000024UL + SemId - 1;
            constexpr uint32_t HwRegOffset = 0;
            constexpr uint32_t HwRegBits   = 31;
            __builtin_amdgcn_s_setreg(SemaHwReg | (HwRegOffset << 6) | (HwRegBits << 11),
                                      count | (limit << 16) | (done << 28));
#else
            __builtin_amdgcn_s_sema_set_state(&_sema, count | (limit << 16) | (done << 28));
#endif
        }
#else
        ignore = count;
        ignore = limit;
        ignore = done;
#endif
    }

    template <index_t MemorySpaces>
    __device__ void wait()
    {
#if defined(__gfx13__)
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
        asm("s_sema_wait %0" : : "n"(1 << (SemId - 1)) : "memory");
#else
        __builtin_amdgcn_s_sema_wait(&_sema);
#endif
        if constexpr(MemorySpaces & SemaphoreAddressSpaceGlobal)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceShared)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "shared");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceLaneShared)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "laneshared");
#endif
    }

    template <index_t MemorySpaces>
    __device__ void signal()
    {
#if defined(__gfx13__)
        if constexpr(MemorySpaces & SemaphoreAddressSpaceGlobal)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceShared)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "shared");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceLaneShared)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "laneshared");

#if defined(CK_USE_AMD_SEMAPHORE_ASM)
        constexpr unsigned imm = SemId | (WaveIdInWavegroup << 4);
        asm("s_sema_signal %0" : : "n"(imm) : "memory");
#else
#if defined(__clang_major__) && (__clang_major__ >= 23)
        // The second argument is ignored for GFX13.
        __builtin_amdgcn_s_sema_signal(&_sema, 0);
#else
        __builtin_amdgcn_s_sema_signal(&_sema);
#endif
#endif
#endif
    }

#if defined(__gfx13__)
#if !defined(CK_USE_AMD_SEMAPHORE_ASM)
    template <unsigned WaveIdInWavegroup_>
    struct SemaphoreType;

    template <>
    struct SemaphoreType<0>
    {
        using Type = __amdgpu_semaphore0_t;
    };

    template <>
    struct SemaphoreType<1>
    {
        using Type = __amdgpu_semaphore1_t;
    };

    template <>
    struct SemaphoreType<2>
    {
        using Type = __amdgpu_semaphore2_t;
    };

    template <>
    struct SemaphoreType<3>
    {
        using Type = __amdgpu_semaphore3_t;
    };

    template <>
    struct SemaphoreType<4>
    {
        using Type = __amdgpu_semaphore4_t;
    };

    template <>
    struct SemaphoreType<5>
    {
        using Type = __amdgpu_semaphore5_t;
    };

    template <>
    struct SemaphoreType<6>
    {
        using Type = __amdgpu_semaphore6_t;
    };

    template <>
    struct SemaphoreType<7>
    {
        using Type = __amdgpu_semaphore7_t;
    };
    using Type = typename SemaphoreType<WaveIdInWavegroup>::Type;

    __device__ WavegroupSemaphore() {}

    private:
    Type _sema;
#endif
#endif
};
} // namespace ck

#endif // CK_AMD_SEMAPHORE_HPP
