/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include <cstdio>
#include <cstdint>

#include <hip/hip_runtime.h>

#define __syncwarp()

// Define min for ssize_t
inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  v = __atomic_load_n(ptr, __ATOMIC_RELAXED);
  return v;
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  //if (BytePerPack == 0) __trap();

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    // minSrcs[0] cannot be nullptr so we always process it
    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    #pragma unroll Unroll
    for (int s=1; s < MinSrcs; s++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      BytePack<BytePerPack> tmp[Unroll];
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          // coverity[dead_error_line]
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll Unroll
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_condition]
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
      uintptr_t dst = dstPtr + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps-1)*BytePerHunk;
    }
    #pragma unroll
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    for (int d=0; d < MinDsts; d++) {
      minDsts[d] += (nWarps-1)*BytePerHunk;
    }
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes nElts
  ) {
  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts, "Multimem pointers cannot exceed respective Min values.");
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;
  // If a multimem src is present then our biggest pack size is limited to what
  // is supported for this redfn/type.
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts==0) return;
  if (MinDsts==0 && nDsts==0) return;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  #if __cpp_if_constexpr
  if constexpr (BigPackSize > sizeof(T)) {
  #else
  if (BigPackSize > sizeof(T)) {
  #endif
    // Check that all pointers are BigPackSize aligned.
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane)) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane)) % (BigPackSize + !BigPackSize);
    aligned = !(__any(!aligned));
    if (aligned) {
#if defined(__gfx90a__)
      reduceCopyPacks<RedFn, T, ((MinSrcs > 1) ? 2 : Unroll), BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
#else
      reduceCopyPacks<RedFn, T, Unroll*((MinSrcs == 1 && MinDsts == 1) ? 2 : 1), BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
#endif
      if (nBytesAhead == 0) return;

      reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }

/*
 * For gfx90a,
* Before we had `Unroll/2*(16/sizeof(T))/2`, which does not work with unroll=1
* as unroll=1; `Unroll/2` = 0, which results in the above expression to 0, and is not supported
* This was reformulated to `(Unroll*4 + sizeof(T) - 1)/sizeof(T)`
*
* Before: `Unroll/2*(16/sizeof(T))/2`
*         sizeof(T)
* unroll  1   2   4   8
*   4     16  8   4   2
*   2     8   4   2   1
*   1     0   0   0   0
*
* After: `(Unroll*4 + sizeof(T) - 1)/sizeof(T)`
*         sizeof(T)
* unroll  1   2   4   8
*   4     16  8   4   2
*   2     8   4   2   1
*   1     4   2   1   1
*/
#if defined(__gfx90a__)
  if (MinSrcs > 1) {
    reduceCopyPacks<RedFn, T, (Unroll*4 + sizeof(T) - 1)/sizeof(T), sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
  } else {
    reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  }
#else
  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
#endif
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs,
    IntBytes nElts
  ) {
  reduceCopy<Unroll, RedFn, T,
             MultimemSrcs, MinSrcs, MaxSrcs,
             MultimemDsts, MinDsts, MaxDsts, PreOpSrcs, IntBytes>
    (thread, nThreads, redArg, preOpArgs, postOp,
     nSrcs, [=]__device__(int i) { return srcPtrs[i]; },
     nDsts, [=]__device__(int i) { return dstPtrs[i]; }, nElts);
}

#endif // COMMON_KERNEL_H_
