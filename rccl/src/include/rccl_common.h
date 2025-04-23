/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef RCCL_COMMON_H_
#define RCCL_COMMON_H_
#include "nccl_common.h"
#include "nccl.h"
typedef enum RcclTunableColls {
  RCCL_UNSUPPORTED_TUNABLE = -1,
  RCCL_RS_TUNABLE = 0,    // reduce_scatter index
  RCCL_AG_TUNABLE = 1,    // all_gather index
  RCCL_AR_TUNABLE = 2,    // all_reduce index
  RCCL_RE_TUNABLE = 3,    // reduce index
  RCCL_TUNABLE_COLLS = 4  // LL/LL64/LL128 tunable collectives count
} rcclTunableIndex_t;

#define RCCL_LL_LIMITS_UNDEFINED 0
#define RCCL_PROTOCOL_ENTRY_SIZE 3
#define RCCL_PROTOCOL_MIN_IDX 0
#define RCCL_PROTOCOL_MAX_IDX 1
#define RCCL_PROTOCOL_FACTOR_IDX 2

#ifdef RCCL_EXPOSE_STATIC
#define RCCL_STATIC_EXPOSE_CHECK()
#else
#define RCCL_STATIC_EXPOSE_CHECK() \
  do { \
    WARN("Attempting to use internal logic while required static functions are not exposed. Rebuild with RCCL_EXPOSE_STATIC enabled"); \
    return ncclInvalidUsage; \
  } while (0)
#endif

inline rcclTunableIndex_t rcclGetTunableIndex(ncclFunc_t const& func) {
  switch (func) {
    case ncclFuncReduceScatter:
      return RCCL_RS_TUNABLE;
    case ncclFuncAllGather:
      return RCCL_AG_TUNABLE;
    case ncclFuncAllReduce:
      return RCCL_AR_TUNABLE;
    case ncclFuncReduce:
      return RCCL_RE_TUNABLE;
    default:
      return RCCL_UNSUPPORTED_TUNABLE; // Invalid or unsupported function
  }
}

inline size_t rcclGetSizePerRank(ncclFunc_t const& func, size_t const& nBytes, int const& nRanks) {
  // Normalize the comparison to sizePerRank as this is essentially what matters in determining protocol choice for the impacted collectives
  // For AG, this is the send size per rank
  // For RS, this is the recv size per rank
  // For AR, this is the send/recv size per rank
  return (func == ncclFuncReduceScatter || func == ncclFuncAllGather) ? nBytes / nRanks : nBytes;
}
void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info);


ncclResult_t rcclGetAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             int collNetSupport, int nvlsSupport, int numPipeOps,
                             int* algo, int* protocol, int* maxChannels);

ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount);

#endif