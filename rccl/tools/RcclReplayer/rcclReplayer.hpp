#pragma once
#include <map>
#include <cstring>

#include <rccl/rccl.h>
#include <hip/hip_bfloat16.h>
#include "hip/hip_fp16.h"
#include "rccl_float8.h"

// NOTE: Parsing is based on this line logging collective information in enqueue.cc
// INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d \
                   root %d comm %p [nranks=%d] stream %p task %d globalrank %d",
//                info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
//                info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream,
//                info->comm->tasks.nTasksP2p + info->comm->tasks.nTasksColl,
//                info->comm->localRankToRank[info->comm->localRank]);

#define HIP_CALL(cmd)                                                   \
  do {                                                                  \
      hipError_t error = (cmd);                                         \
      if (error != hipSuccess) {                                        \
        printf("Encountered HIP error (%s) at line %d in file %s\n",    \
               hipGetErrorString(error), __LINE__, __FILE__);           \
        exit(-1);                                                       \
      }                                                                 \
  } while (0)

#define NCCL_CALL(cmd)                                          \
  do {                                                          \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
      printf("NCCL failure %s:%d '%s'\n",                       \
             __FILE__,__LINE__,ncclGetErrorString(res));        \
    }                                                           \
  } while(0)

struct LineItem
{
  char   hostname[MPI_MAX_PROCESSOR_NAME];
  int    pid;
  int    tid;
  int    cudaDev;
  char   opName[32];
  int    opCount;
  char   sendbuff[32];
  char   recvbuff[32];
  size_t count;
  int    datatype;
  int    op;
  int    root;
  char   comm[32];
  int    nRanks;
  void*  stream;
  int    task;
  int    globalRank;
};

// Enumeration of all collective functions currently supported
typedef enum
{
  ncclCollBroadcast = 0,
  ncclCollReduce,
  ncclCollAllGather,
  ncclCollReduceScatter,
  ncclCollAllReduce,
  ncclCollGather,
  ncclCollScatter,
  ncclCollAllToAll,
  ncclCollAllToAllv,
  ncclCollSend,
  ncclCollRecv,
  ncclNumFuncs
} ncclFunc_t;

char const ncclFuncNames[ncclNumFuncs][32] =
{
  "Broadcast",
  "Reduce",
  "AllGather",
  "ReduceScatter",
  "AllReduce",
  "Gather",
  "Scatter",
  "AllToAll",
  "AllToAllv",
  "Send",
  "Recv"
};

char const mscclFuncNames[ncclNumFuncs][32] =
{
  "mscclFuncBroadcast",
  "mscclFuncReduce",
  "mscclFuncAllGather",
  "mscclFuncReduceScatter",
  "mscclFuncAllReduce",
  "mscclFuncGather",
  "mscclFuncScatter",
  "mscclFuncAllToAll",
  "mscclFuncAllToAllv",
  "mscclFuncSend",
  "mscclFuncRecv"
};

union PtrUnion
{
  void*          ptr;
  int8_t*        I1; // ncclInt8
  uint8_t*       U1; // ncclUint8
  int32_t*       I4; // ncclInt32
  uint32_t*      U4; // ncclUint32
  int64_t*       I8; // ncclInt64
  uint64_t*      U8; // ncclUint64
  __half*        F2; // ncclFloat16
  rccl_float8*   F1; // ncclFloat8e4m3
  float*         F4; // ncclFloat32
  double*        F8; // ncclFloat64
  rccl_bfloat8*  B1; // ncclFloat8e5m2
  hip_bfloat16*  B2; // ncclBfloat16

  constexpr PtrUnion() : ptr(nullptr) {}
};

struct TaskInfo
{
  ncclFunc_t     funcType;
  bool           inPlace;
  size_t         count;
  ncclDataType_t datatype;
  ncclRedOp_t    op;
  int            root;
  PtrUnion       inputGpu;
  PtrUnion       outputCpu;
  PtrUnion       outputGpu;
  PtrUnion       expected;
};

struct RankData
{
  int                   lineNum;
  int                   commIdx;
  std::vector<TaskInfo> tasks;
};

struct GroupCall
{
  bool isValid;
  int opCount;
  std::map<int, RankData> rankData;
};

struct CollectiveCalls
{
  int numGlobalRanks;
  int numGpusPerMpiRank;
  std::vector<std::vector<std::string>> globalRankComms;  // Set of comms used by each global rank
  std::vector<GroupCall>                groupCalls;       // List of group calls for each global rank

  int localGpuOffset;                                     // First local GPU device idx for this MPI process
  int firstGlobalRank;                                    // First global rank for this MPI process
  int numCommsPerRank;                                    // Number of communicators per rank
  std::vector<std::vector<ncclComm_t>>  localRankComms;   // comms per local rank
  std::vector<std::vector<hipStream_t>> localRankStreams; // streams per local rank
};

std::string DataTypeToName(ncclDataType_t const dataType)
{
  switch (dataType) {
  case ncclInt8:     return "Int8";
  case ncclUint8:    return "Uint8";
  case ncclInt32:    return "Int32";
  case ncclUint32:   return "Uint32";
  case ncclInt64:    return "Int64";
  case ncclUint64:   return "Uint64";
  case ncclFloat16:  return "Float16";
  case ncclFloat32:  return "Float32";
  case ncclFloat64:  return "Float64";
  case ncclBfloat16: return "Bfloat16";
  case ncclFloat8e4m3:  return "Fp8E4M3";
  case ncclFloat8e5m2:  return "Fp8E5M2";
  default:
    printf("Unsupported datatype (%d)\n", dataType);
    exit(0);
  }
}

size_t DataTypeToBytes(ncclDataType_t const dataType)
{
  switch (dataType) {
  case ncclInt8:     return 1;
  case ncclUint8:    return 1;
  case ncclInt32:    return 4;
  case ncclUint32:   return 4;
  case ncclInt64:    return 8;
  case ncclUint64:   return 8;
  case ncclFloat16:  return 2;
  case ncclFloat32:  return 4;
  case ncclFloat64:  return 8;
  case ncclBfloat16: return 2;
  case ncclFloat8e4m3:  return 1;
  case ncclFloat8e5m2:  return 1;
  default:
    printf("Unsupported datatype (%s)\n", DataTypeToName(dataType).c_str());
    exit(0);
  }
}

std::string RedOpToName(ncclRedOp_t const op)
{
  switch (op) {
  case ncclSum:       return "Sum";
  case ncclProd:      return "Product";
  case ncclMax:       return "Max";
  case ncclMin:       return "Min";
  case ncclAvg:       return "Average";
  case ncclNumOps:    return "Number of built-in reduction ops";
  case ncclMaxRedOp:  return "Largest value for ncclRedOp_t";
  default:
    printf("Unsupported redOp (%d)\n", op);
    exit(0);
  }
}

ncclFunc_t GetFuncType(char* func)
{
  for (int i = 0; i < ncclNumFuncs; i++)
    if (!strcmp(func, ncclFuncNames[i]) || !strcmp(func, mscclFuncNames[i])) return (ncclFunc_t)i;
  printf("[ERROR] Unrecognized func %s\n", func);
  exit(1);
}

// Set data for ptrUnion (Used during fillPattern)
void SetPtr(PtrUnion& ptrUnion, ncclDataType_t const dataType, int const idx, int valueI, double valueF) {
  switch (dataType)
  {
    case ncclInt8:     ptrUnion.I1[idx] = valueI; break;
    case ncclUint8:    ptrUnion.U1[idx] = valueI; break;
    case ncclInt32:    ptrUnion.I4[idx] = valueI; break;
    case ncclUint32:   ptrUnion.U4[idx] = valueI; break;
    case ncclInt64:    ptrUnion.I8[idx] = valueI; break;
    case ncclUint64:   ptrUnion.U8[idx] = valueI; break;
    case ncclFloat8e4m3:  ptrUnion.F1[idx] = rccl_float8(valueF); break;
    case ncclFloat16:  ptrUnion.F2[idx] = __float2half(static_cast<float>(valueF)); break;
    case ncclFloat32:  ptrUnion.F4[idx] = valueF; break;
    case ncclFloat64:  ptrUnion.F8[idx] = valueF; break;
    case ncclFloat8e5m2:  ptrUnion.B1[idx] = rccl_bfloat8(valueF); break;
    case ncclBfloat16: ptrUnion.B2[idx] = hip_bfloat16(static_cast<float>(valueF)); break;
    default:
      printf("Unsupported datatype (%s)\n", DataTypeToName(dataType).c_str());
      exit(0);
  }
}

// Check if each element in actual equals to expected
bool IsEqual(PtrUnion const& actual, PtrUnion const& expected, ncclDataType_t const dataType, size_t const numElements, int const globalRank) {
  bool isMatch = true;
  size_t idx = 0;
  for (idx = 0; idx < numElements; ++idx)
  {
    switch (dataType)
    {
    case ncclInt8:     isMatch = (actual.I1[idx] == expected.I1[idx]); break;
    case ncclUint8:    isMatch = (actual.U1[idx] == expected.U1[idx]); break;
    case ncclInt32:    isMatch = (actual.I4[idx] == expected.I4[idx]); break;
    case ncclUint32:   isMatch = (actual.U4[idx] == expected.U4[idx]); break;
    case ncclInt64:    isMatch = (actual.I8[idx] == expected.I8[idx]); break;
    case ncclUint64:   isMatch = (actual.U8[idx] == expected.U8[idx]); break;
    case ncclFloat8e4m3:  isMatch = (fabs(float(actual.F1[idx]) - float(expected.F1[idx])) < 9e-2); break;
    case ncclFloat16:  isMatch = (fabs(__half2float(actual.F2[idx]) - __half2float(expected.F2[idx])) < 9e-2); break;
    case ncclFloat32:  isMatch = (fabs(actual.F4[idx] - expected.F4[idx]) < 1e-5); break;
    case ncclFloat64:  isMatch = (fabs(actual.F8[idx] - expected.F8[idx]) < 1e-12); break;
    case ncclFloat8e5m2:  isMatch = (fabs(float(actual.B1[idx]) - float(expected.B1[idx])) < 9e-2); break;
    case ncclBfloat16: isMatch = (fabs((float)actual.B2[idx] - (float)expected.B2[idx]) < 9e-2); break;
    default:
      printf("Unsupported datatype (%s)\n", DataTypeToName(dataType).c_str());
      isMatch = false;
    }
    if (!isMatch) {
      switch (dataType)
      {
      case ncclInt8:
        printf("[Error Rank = %d] Expected output: %d.  Actual output: %d at index %lu\n", globalRank, expected.I1[idx], actual.I1[idx], idx); break;
      case ncclUint8:
        printf("[Error Rank = %d] Expected output: %u.  Actual output: %u at index %lu\n", globalRank, expected.U1[idx], actual.U1[idx], idx); break;
      case ncclInt32:
        printf("[Error Rank = %d] Expected output: %d.  Actual output: %d at index %lu\n", globalRank, expected.I4[idx], actual.I4[idx], idx); break;
      case ncclUint32:
        printf("[Error Rank = %d] Expected output: %u.  Actual output: %u at index %lu\n", globalRank, expected.U4[idx], actual.U4[idx], idx); break;
      case ncclInt64:
        printf("[Error Rank = %d] Expected output: %ld.  Actual output: %ld at index %lu\n", globalRank, expected.I8[idx], actual.I8[idx], idx); break;
      case ncclUint64:
        printf("[Error Rank = %d] Expected output: %lu.  Actual output: %lu at index %lu\n", globalRank, expected.U8[idx], actual.U8[idx], idx); break;
      case ncclFloat8e4m3:
        printf("[Error Rank = %d] Expected output: %f.  Actual output: %f at index %lu\n", globalRank, (float)expected.F1[idx], (float)actual.F1[idx], idx); break;
      case ncclFloat16:
        printf("[Error Rank = %d] Expected output: %f.  Actual output: %f at index %lu\n", globalRank, __half2float(expected.F2[idx]), __half2float(actual.F2[idx]), idx); break;
      case ncclFloat32:
        printf("[Error Rank = %d] Expected output: %f.  Actual output: %f at index %lu\n", globalRank, expected.F4[idx], actual.F4[idx], idx); break;
      case ncclFloat64:
        printf("[Error Rank = %d] Expected output: %lf.  Actual output: %lf at index %lu\n", globalRank, expected.F8[idx], actual.F8[idx], idx); break;
      case ncclFloat8e5m2:
        printf("[Error Rank = %d] Expected output: %f.  Actual output: %f at index %lu\n", globalRank, (float)expected.B1[idx], (float)actual.B1[idx], idx); break;
      case ncclBfloat16:
        printf("[Error Rank = %d] Expected output: %f.  Actual output: %f at index %lu\n", globalRank, (float)expected.B2[idx], (float)actual.B2[idx], idx); break;
      default:
        break;
      }
      return isMatch;
    }
  }

  return isMatch;
}

// Performs the various basic reduction operations
template <typename T>
T ReduceOp(ncclRedOp_t const op, T const A, T const B)
{
  switch (op)
  {
  case ncclSum:  return A + B;
  case ncclProd: return A * B;
  case ncclMax:  return std::max(A, B);
  case ncclMin:  return std::min(A, B);
  default:
    printf("Unsupported reduction operator (%s)\n", RedOpToName(op).c_str());
    exit(0);
  }
}

// Perform various reduction ops to ptrUnion
void Reduce(PtrUnion& ptrUnion, PtrUnion const& otherPtrUnion, size_t const numElements, ncclDataType_t const dataType, ncclRedOp_t const op) {
  for (size_t idx = 0; idx < numElements; ++idx)
  {
    switch (dataType)
    {
    case ncclInt8:     ptrUnion.I1[idx] = ReduceOp(op, ptrUnion.I1[idx], otherPtrUnion.I1[idx]); break;
    case ncclUint8:    ptrUnion.U1[idx] = ReduceOp(op, ptrUnion.U1[idx], otherPtrUnion.U1[idx]); break;
    case ncclInt32:    ptrUnion.I4[idx] = ReduceOp(op, ptrUnion.I4[idx], otherPtrUnion.I4[idx]); break;
    case ncclUint32:   ptrUnion.U4[idx] = ReduceOp(op, ptrUnion.U4[idx], otherPtrUnion.U4[idx]); break;
    case ncclInt64:    ptrUnion.I8[idx] = ReduceOp(op, ptrUnion.I8[idx], otherPtrUnion.I8[idx]); break;
    case ncclUint64:   ptrUnion.U8[idx] = ReduceOp(op, ptrUnion.U8[idx], otherPtrUnion.U8[idx]); break;
    case ncclFloat8e4m3:  ptrUnion.F1[idx] = rccl_float8(ReduceOp(op, float(ptrUnion.F1[idx]), float(otherPtrUnion.F1[idx]))); break;
    case ncclFloat16:  ptrUnion.F2[idx] = __float2half(ReduceOp(op, __half2float(ptrUnion.F2[idx]), __half2float(otherPtrUnion.F2[idx]))); break;
    case ncclFloat32:  ptrUnion.F4[idx] = ReduceOp(op, ptrUnion.F4[idx], otherPtrUnion.F4[idx]); break;
    case ncclFloat64:  ptrUnion.F8[idx] = ReduceOp(op, ptrUnion.F8[idx], otherPtrUnion.F8[idx]); break;
    case ncclFloat8e5m2:  ptrUnion.B1[idx] = rccl_bfloat8(ReduceOp(op, float(ptrUnion.B1[idx]), float(otherPtrUnion.B1[idx]))); break;
    case ncclBfloat16: ptrUnion.B2[idx] = ReduceOp(op, ptrUnion.B2[idx], otherPtrUnion.B2[idx]); break;
    default:
      printf("Unsupported datatype (%s)\n", DataTypeToName(dataType).c_str());
      exit(0);
    }
  }
}

// Divide each element in ptrUnion by divisor
void DivideByInt(PtrUnion& ptrUnion, ncclDataType_t const dataType, size_t const numElements, int const divisor) {
  for (size_t idx = 0; idx < numElements; ++idx)
  {
    switch (dataType)
    {
    case ncclInt8:     ptrUnion.I1[idx] /= divisor; break;
    case ncclUint8:    ptrUnion.U1[idx] /= divisor; break;
    case ncclInt32:    ptrUnion.I4[idx] /= divisor; break;
    case ncclUint32:   ptrUnion.U4[idx] /= divisor; break;
    case ncclInt64:    ptrUnion.I8[idx] /= divisor; break;
    case ncclUint64:   ptrUnion.U8[idx] /= divisor; break;
    case ncclFloat8e4m3:  ptrUnion.F1[idx] = (rccl_float8((float)(ptrUnion.F1[idx]) / divisor)); break;
    case ncclFloat16:  ptrUnion.F2[idx] = __float2half(__half2float(ptrUnion.F2[idx])/divisor); break;
    case ncclFloat32:  ptrUnion.F4[idx] /= divisor; break;
    case ncclFloat64:  ptrUnion.F8[idx] /= divisor; break;
    case ncclFloat8e5m2:  ptrUnion.B1[idx] = (rccl_bfloat8((float)(ptrUnion.B1[idx]) / divisor)); break;
    case ncclBfloat16: ptrUnion.B2[idx] = (hip_bfloat16((float)(ptrUnion.B2[idx]) / divisor)); break;
    default:
      printf("Unsupported datatype (%s)\n", DataTypeToName(dataType).c_str());
      exit(0);
    }
  }
}

// Check if a collective uses a root
bool IsRootUsed(ncclFunc_t funcType) {
  return (funcType == ncclCollBroadcast || funcType == ncclCollReduce ||
          funcType == ncclCollGather    || funcType == ncclCollScatter);
}

// parse the logs and assign them into lineItem
bool ParseLineItem(char const* line, LineItem& li);

// this covers grouping the logs based on opCount and task number,
// validatation of the groupCalls for both non-send/recv collectives and send/recv
void ParseCollectives(char const* logFilename, bool isFirstRank, CollectiveCalls& collectiveCalls);

// allocates send/recv buff, sets the device based on which rank the task belongs to,
// syncronize devices after executing all the tasks and free device memory.
double ReplayRccl(CollectiveCalls& collCall, int groupIdx, int& numInvalid);

// Print information about a group call
void PrintGroupCall(GroupCall const& gc);

// Records performance data of each group call in a csv file named replayer_data.csv
void dataToCsv(GroupCall const& gc, std::ofstream &datafile, double runTime);

// size differ for each collective call and getSize gives a specific size in bytes depending on type of task,
// global rank, element count and data type
std::pair<size_t, size_t> GetSize(TaskInfo taskInfo, int numGlobalRanks);

// executes the collective call (task)
void ExecuteCollective(TaskInfo& task, ncclComm_t const& comm, hipStream_t stream);

// Allocate CPU/GPU memory for ptrUnion
void AllocateMem(PtrUnion& ptrUnion, size_t const numBytes, bool isGpu = false);

// Free CPU/GPU memory for ptrUnion
void FreeMem(PtrUnion& ptrUnion, bool isGpu = false);

// Fill buffers based on pattern using globalRank
void FillPattern(PtrUnion& ptrUnion, ncclDataType_t const dataType, size_t const numElements, int globalRank, bool isGpu = false);

// PrepareData functions are responsible for setting up input / expected for the given taskInfo
void PrepareDataFunc(TaskInfo& taskInfo, int globalRank, int totalRanks);
void PrepData_Broadcast(TaskInfo& taskInfo, int globalRank);
void PrepData_Reduce(TaskInfo& taskInfo, int globalRank, int totalRanks, bool isAllReduce);
void PrepData_ReduceScatter(TaskInfo& taskInfo, int globalRank, int totalRanks);
void PrepData_Gather(TaskInfo& taskInfo, int globalRank, int totalRanks, bool isAllGather);
void PrepData_Scatter(TaskInfo& taskInfo, int globalRank, int totalRanks);
void PrepData_AlltoAll(TaskInfo& taskInfo, int globalRank, int totalRanks);
void PrepData_Send(TaskInfo& taskInfo, int globalRank);
void PrepData_Recv(TaskInfo& taskInfo, int globalRank);