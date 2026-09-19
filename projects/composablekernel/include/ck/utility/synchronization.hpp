// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

// block_sync_lds() -- workgroup barrier with LDS visibility
//
// This function implements a barrier that ensures all LDS writes issued before
// the call are visible to all threads in the workgroup after the call.
// Specifically, it waits for lgkmcnt=0 (all LDS/SMEM ops complete) but does
// NOT wait for vmcnt (global memory ops), which is intentional: this barrier
// is only needed for LDS coherence.
//
// CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM (always 1, see ck.hpp)
// selects architecture-specific implementations that avoid an unnecessary
// vmcnt=0 wait. Each target requires a different approach -- see below.
//
// WARNING: The implementation is subtle and architecture-specific. Do NOT
// replace the gfx9 path with __builtin_amdgcn_fence -- it does not lower to
// s_waitcnt lgkmcnt(0) on gfx9 and causes LDS read-before-write races that
// produce incorrect numerical results. See LLVM issue #120131.
__device__ void block_sync_lds()
{
#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#if defined(__gfx12__)
    // gfx12 uses a two-phase barrier protocol: signal then wait.
    // The release fence before signal ensures LDS writes are visible before
    // the barrier, and the acquire fence after wait ensures LDS writes from
    // other waves are visible after the barrier. __builtin_amdgcn_fence
    // correctly lowers to the appropriate s_waitcnt on gfx12.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#elif defined(__gfx11__)
    // gfx11 uses a single s_barrier, wrapped in release/acquire fences.
    // __builtin_amdgcn_fence correctly lowers to s_waitcnt lgkmcnt(0) on
    // gfx11 and is the preferred approach over explicit magic constants.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    // gfx9 and earlier: use explicit magic constant 0xc07f which encodes
    // lgkmcnt=0 (wait for all LDS/SMEM ops) with vmcnt left unconstrained.
    //
    // WARNING: Do NOT replace this with __builtin_amdgcn_fence. On gfx9,
    // the compiler does not lower __builtin_amdgcn_fence("workgroup","local")
    // to s_waitcnt lgkmcnt(0). Instead it may emit s_waitcnt 0 (waiting on
    // both lgkmcnt AND vmcnt) or nothing at all, both of which are wrong.
    // This is a known LLVM bug (issue #120131). Using the fence here causes
    // LDS read-before-write races and incorrect numerical results, verified
    // on gfx90a (MI210).
    //
    // Instead, use inline asm with a "memory" clobber. The clobber gives the
    // compiler a real ordering constraint (preventing LDS store sinking)
    // while the explicit lgkmcnt(0) encoding (0xc07f) ensures we wait only
    // for LDS/SMEM ops and not vmcnt. This is the gfx9-safe equivalent of
    // the release/acquire fence pattern used on gfx11/gfx12.
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
    asm volatile("s_barrier"            : : : "memory");
#endif
#else
    __syncthreads();
#endif
}

__device__ void block_sync_lds_direct_load()
{
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
#elif defined(__gfx12__)
    asm volatile("\
    s_wait_loadcnt 0x0 \n \
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);
#else
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif
}

__device__ void block_sync_lds_async_load()
{
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
    __syncthreads();
#else
    // fall back
    block_sync_lds();
#endif
}

__device__ void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace ck
