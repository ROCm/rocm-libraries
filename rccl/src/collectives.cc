/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"

#include "msccl/msccl_lifecycle.h"

using namespace rccl;

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsAllGather {
    size_t bytes;
    ncclDataType_t datatype;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsAllGather, datatype)}
  };
  NvtxParamsAllGather payload{sendcount * ncclTypeSize(datatype), datatype};
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, payload)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllGather, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      sendcount, datatype, 0, 0, ncclSum, mscclFuncAllGather, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);


ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
    ncclDataType_t datatype;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0, offsetof(NvtxParamsAllReduce, op)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsAllReduce, datatype)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op, datatype};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllReduce, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, 0, op, mscclFuncAllReduce, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

RCCL_PARAM(AllToAllPivotEnable, "ALL_TO_ALL_PIVOT_ENABLE", 0);

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclAllToAll_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllToAll, sendbuff, recvbuff, count, datatype, comm, stream));
  }

  struct NvtxParamsAllToAll {
    size_t bytes;
    ncclDataType_t datatype;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllToAllSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsAllToAll, datatype)}
  };
  NvtxParamsAllToAll payload{count * ncclTypeSize(datatype), datatype};
  NVTX3_FUNC_WITH_PARAMS(AllToAll, AllToAllSchema, payload)

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, 0, ncclSum, mscclFuncAllToAll, comm, stream);
  }

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);
  // Determine Pivot A2A support now that we know number of channels
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAllToAllPivotEnable()) {
    struct ncclInfo info = { ncclFuncAllToAllPivot, "AllToAllPivot",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    if (count == 0) return ncclSuccess;
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }
}

NCCL_API(ncclResult_t, ncclAllToAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclAllToAllv_impl(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream) {

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllToAllv, sendbuff, recvbuff, 0, datatype, comm, stream, -1, sendcounts, sdispls, recvcounts, rdispls));
  }

  struct NvtxParamsAllToAllv {
    size_t sendbytes;
    size_t recvbytes;
    ncclDataType_t datatype;
  };
  // Just pass the size of one send/recv messages and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllToAllvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes] (Send)"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes] (Recv)", nullptr, 0, 
      offsetof(NvtxParamsAllToAllv, recvbytes)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsAllToAllv, datatype)}
  };
  NvtxParamsAllToAllv payload{sendcounts[comm->rank] * ncclTypeSize(datatype), recvcounts[comm->rank] * ncclTypeSize(datatype), datatype};
  NVTX3_FUNC_WITH_PARAMS(AllToAllv, AllToAllvSchema, payload)

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
      0, datatype, 0, 0, ncclSum, mscclFuncAllToAllv, comm, stream);
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(
        ((char*)sendbuff) + sdispls[r]*ncclTypeSize(datatype),
        sendcounts[r],
        datatype,
        r,
        comm,
        stream));
    NCCLCHECK(ncclRecv(
        ((char*)recvbuff) + rdispls[r]*ncclTypeSize(datatype),
        recvcounts[r],
        datatype,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
    ncclDataType_t datatype;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsBroadcast, datatype)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root, datatype};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrBroadcast, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, ncclSum, mscclFuncBroadcast, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(Recorder::instance().record(rrBcast, buff, buff, count, datatype, comm, stream, root));
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream) {
    struct NvtxParamsGather {
      size_t bytes;
      int root;
      ncclDataType_t datatype;
    };
    constexpr nvtxPayloadSchemaEntry_t GatherSchema[] = {
      {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
      {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsGather, root)},
      {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
        offsetof(NvtxParamsGather, datatype)}
    };
    NvtxParamsGather payload{sendcount * ncclTypeSize(datatype), root, datatype};
    NVTX3_FUNC_WITH_PARAMS(Gather, GatherSchema, payload)

    if (!mscclIsCaller()) // when msccl falls back to
    {
      NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, sendcount, datatype, comm, stream, root));
    }

    if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
      return mscclEnqueueCheck(
        sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
        sendcount, datatype, root, 0, ncclSum, mscclFuncGather, comm, stream);
    }

    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = sendcount * ncclTypeSize(datatype);
    if (sendcount == 0) return ncclSuccess;
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++)
        NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, sendcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclSend(sendbuff, sendcount, datatype, root, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsReduce {
    size_t bytes;
    int root;
    ncclRedOp_t op;
    ncclDataType_t datatype;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduce, op)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsReduce, datatype)}
  };
  NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op, datatype};
  NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrReduce, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, op, mscclFuncReduce, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);


ncclResult_t ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
    ncclDataType_t datatype;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsReduceScatter, datatype)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op, datatype};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      recvcount, datatype, 0, 0, op, mscclFuncReduceScatter, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
    struct NvtxParamsScatter {
      size_t bytes;
      int root;
      ncclDataType_t datatype;
    };
    constexpr nvtxPayloadSchemaEntry_t ScatterSchema[] = {
      {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
      {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsScatter, root)},
      {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
        offsetof(NvtxParamsScatter, datatype)}
    };
    NvtxParamsScatter payload{recvcount * ncclTypeSize(datatype), root, datatype};
    NVTX3_FUNC_WITH_PARAMS(Scatter, ScatterSchema, payload)

    if (!mscclIsCaller()) // when msccl falls back to
    {
      NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, recvcount, datatype, comm, stream, root));
    }

    if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
      return mscclEnqueueCheck(
        sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
        recvcount, datatype, root, 0, ncclSum, mscclFuncScatter, comm, stream);
    }

    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = recvcount * ncclTypeSize(datatype);
    if (recvcount == 0) return ncclSuccess;
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++)
        NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, recvcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclRecv(recvbuff, recvcount, datatype, root, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
    ncclDataType_t datatype;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "Data type", nullptr, 0, 
      offsetof(NvtxParamsSendRecv, datatype)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);


ncclResult_t ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer, datatype};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrSend, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, nullptr, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncSend, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer, datatype};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrRecv, info));
  }

  if (mscclAvailable(comm->rank) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      nullptr, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncRecv, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}
