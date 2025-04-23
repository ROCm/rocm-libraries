/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
    const int bid = ncclShmem.channelId - work->channelLo;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    size_t size;
    size_t chunkCount;
    size_t channelCount;
    size_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &size, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

#if defined(ENABLE_NPKIT)
    int npKitCtxIdx = bid;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_BROADCAST_RING_ENTRY, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, work->connIndex, work->connIndex, work);

#if defined(ENABLE_NPKIT)
    if (tid == 0) {
      prims.npKitCtxIdx = npKitCtxIdx;
    }
#endif

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);

      if (rank == root) {
        if (inputBuf == outputBuf) {
          prims.directSend(offset, offset, nelem);
        } else {
          prims.directCopySend(offset, offset, nelem);
        }
      } else if (nextRank == root) {
        prims.directRecv(offset, offset, nelem);
      } else {
        prims.directRecvCopyDirectSend(offset, nelem);
      }
    }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_BROADCAST_RING_EXIT, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};