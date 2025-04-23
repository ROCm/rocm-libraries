/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include "device.h"
#include "collectives.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif
#include "tuner.h"
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "graph/topo.h"
#include "graph/xml.h"
#include "archinfo.h"
#include "param.h"

// [RCCL]
#include "git_version.h"
#include "rccl_vars.h"
#include "hip_rocm_version_info.h"
//#include <hsa/hsa_ext_amd.h>
#ifdef ENABLE_MSCCLPP
#include "mscclpp/mscclpp_nccl.h"
#endif
#include "rocm_smi_wrap.h"
// [/RCCL]

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_status.h"

#ifndef STR2
  #define STR2(v) #v
#endif

#ifndef STR
  #define STR(v) STR2(v)
#endif

#if CUDART_VERSION >= 9020 || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

using namespace rccl;

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2] = { "AllGather", "AllReduce", "AllToAllPivot", "Broadcast", "Reduce", "ReduceScatter", "SendRecv"};
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };
const char* ncclDevRedOpStr[ncclNumDevRedOps] = { "Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv" };
const char *ncclTypeStr[ncclNumTypes] = {"_i8", "_u8", "_i32", "_u32", "_i64", "_u64", "_f16", "_f32", "_f64", "_b16"};

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
static ncclResult_t commReclaim(ncclComm_t comm);

static uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) {
  hipDeviceProp_t devProp;
  CUDACHECK(hipGetDeviceProperties(&devProp, comm->cudaDev));
  if(IsArchMatch(devProp.gcnArchName, "gfx950"))
    comm->unroll = NCCL_UNROLL_1;
  else if(IsArchMatch(devProp.gcnArchName, "gfx908") || ((IsArchMatch(devProp.gcnArchName, "gfx942") && devProp.multiProcessorCount > 80)))
    comm->unroll = NCCL_UNROLL_2;
  else
    comm->unroll = NCCL_UNROLL_4;
  return ncclSuccess;
}

#ifdef ENABLE_MSCCLPP
size_t std::hash<ncclUniqueId>::operator ()(const ncclUniqueId& uniqueId) const noexcept {
  return (size_t)hashUniqueId(uniqueId);
}

bool operator ==(const ncclUniqueId& a, const ncclUniqueId& b) {
  return memcmp(a.internal, b.internal, NCCL_UNIQUE_ID_BYTES) == 0;
}

RCCL_PARAM(MscclppThreshold, "MSCCLPP_THRESHOLD", (size_t)(16*1024*1024));
#endif

static constexpr int64_t defaultEnableMscclpp = 0;
RCCL_PARAM(MscclppEnabled, "MSCCLPP_ENABLE", defaultEnableMscclpp);
RCCL_PARAM(MscclppForceEnabled, "MSCCLPP_FORCE_ENABLE", 0);

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}

static ncclResult_t initResult = ncclSuccess;
static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;

static void initOnceFunc() {
  initEnv();
  initGdrCopy();
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);

#ifndef NVTX_NO_IMPL
  initNvtxRegisteredEnums();
#endif
exit:;
}

static ncclResult_t ncclInit() {
    char strValue[2048];
    NCCLCHECK(ncclTopoGetStrFromSys("/proc/sys/kernel", "numa_balancing", strValue));
    if (strcmp(strValue, "1") == 0)
      WARN("NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by \"sudo sysctl kernel.numa_balancing=0\"");
    NCCLCHECK(ncclTopoGetStrFromSys("/proc", "version", strValue));
    char *verStr, *state;
    verStr = strtok_r(strValue, " ", &state);
    for (int i = 0; i < 2; i ++) {
      verStr = strtok_r(NULL, " ", &state);
      if (verStr == NULL) break;
    }
    INFO(NCCL_INIT, "Kernel version: %s", verStr);
    if (strstr(verStr, "cray") == NULL) {
      NCCLCHECK(ncclTopoGetStrFromSys("/sys/devices/virtual/dmi/id", "bios_version", strValue));
      if (strncmp("Hyper-V UEFI Release", strValue, 20) != 0) {
        FILE* file;
        if ((file = fopen("/proc/cmdline", "r")) != NULL) {
          if (feof(file) == 0 && ferror(file) == 0) {
            int len = fread(strValue, 1, 2047, file);
            strValue[len] = '\0';
          }
          fclose(file);
        }
        if (strstr(strValue, "iommu=pt") == NULL)
          WARN("Missing \"iommu=pt\" from kernel command line which can lead to system instablity or hang!");
      }
#ifndef HIP_UNCACHED_MEMORY
      char *env = getenv("HSA_FORCE_FINE_GRAIN_PCIE");
      if (env == NULL || strcmp(env, "1") != 0)
        WARN("Missing \"HSA_FORCE_FINE_GRAIN_PCIE=1\" from environment which can lead to low RCCL performance, system instablity or hang!");
#endif
    }
  pthread_once(&initOnceControl, initOnceFunc);
  return initResult;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion_impl(int* version) {
  Recorder::instance().record("GetVersion");
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId_impl(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  struct ncclBootstrapHandle handle;
  NCCLCHECK(bootstrapGetUniqueId(&handle));
  // ncclUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset(out, 0, sizeof(*out));
  // copy to avoid alignment mismatch
  memcpy(out, &handle, sizeof(handle));
  Recorder::instance().record(rrGetUniqueId, -1, -1, out);
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return ncclSuccess;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
  comm->startMagic = comm->endMagic = 0;
}

RCCL_PARAM(KernelCollTraceEnable, "KERNEL_COLL_TRACE_ENABLE", 0);
RCCL_PARAM(KernelCollTraceThreadEnable, "KERNEL_COLL_TRACE_THREAD_ENABLE", 0);

#ifdef ENABLE_COLLTRACE
// Should be in sync with 'ALL_COLLS' in Generator.cmake
void *ncclCommThreadMain(void *arg) {
  ncclComm_t comm = (ncclComm_t)arg;
  int head[MAXCHANNELS];
  double vega_gpu_rtc_freq;

  vega_gpu_rtc_freq = GetDeviceWallClockRateInKhz(comm->cudaDev) * 1.0E3;
  for (int channel = 0; channel < MAXCHANNELS; channel++) {
    int tail = comm->collTraceTail[channel].tail;
    if (tail < COLLTRACE_NUM_ITEMS)
      head[channel] = 0;
    else
      head[channel] = tail - COLLTRACE_NUM_ITEMS;
  }
  do {
    int numActiveChans = MAXCHANNELS;
    for (int channel = 0; channel < MAXCHANNELS; channel++) {
      int tail = comm->collTraceTail[channel].tail;
      int count;
      count = tail - head[channel];
      if (count == 0) {
        numActiveChans--;
        continue;
      }
      for (int i = 0; i < count; i++) {
        volatile struct ncclCollTrace *td = comm->collTrace+COLLTRACE_NUM_ITEMS*channel+head[channel]%COLLTRACE_NUM_ITEMS;
        head[channel] ++;
        uint8_t type = td->type;
        if (type == ncclCollTraceNotReady)
          continue;
        char line[1024];
        int offset = 0;
        uint16_t fIdx = td->funcIndex;
        if (type == ncclCollTraceDataType) {
          sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] L:%04d DT %08x %016lx %016lx",
            (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid,             fIdx, td->data_0, td->opCount, td->data_1);
        } else {
          if (type & ncclCollTraceP2pElemType)
            sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] %06x-%06x", (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid, td->p2pOpCount[0], td->p2pOpCount[1]);
          else
            sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] %06lx", (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid, td->opCount);
          offset = strlen(line);
          if (type == ncclCollTraceCollElemType) {
            sprintf(line+offset, " CE %s nw %d bi %d nc %d root %d busId %lx nRanks %d", funcNames[fIdx], td->coll.nWarps, td->coll.bid, td->coll.nChannels, td->coll.root, comm->busId, comm->nRanks);
          } else if (type == ncclCollTraceP2pElemType) {
            sprintf(line+offset, " Recv %d -> %d/%d/%d/%d ConnIdx/LL/Reg/nc %d/%d/%d/%d -> Send %d cb %d busId %lx nRanks %d",
              td->p2p.recvRank, td->p2p.recvConnIndex, td->p2p.recvProtoLL, td->p2p.recvRegistered, td->p2p.nRecvChannels, td->p2p.sendConnIndex, td->p2p.sendProtoLL, td->p2p.sendRegistered, td->p2p.nSendChannels, td->p2p.sendRank, td->p2p.channelBase,
              comm->busId, comm->nRanks);
          } else {
            switch (type&0xf) {
              case ncclCollTraceKernelLaunchType:
              case ncclCollTraceCollLaunchType:
                if ((type&0xf) == ncclCollTraceKernelLaunchType)
                  sprintf(line+offset, " KL HWID %8x %s", td->data_0, funcNames[fIdx]);
                else if ((type&0xf) == ncclCollTraceCollLaunchType)
                  sprintf(line+offset, " CL %d %s", td->batchIx, funcNames[fIdx]);
                offset = strlen(line);
                if ((type&0xf0) == ncclCollTraceCollElemType)
                  sprintf(line+offset, " nw %d bi %d nc %d root %d busId %lx nRanks %d", td->coll.nWarps, td->coll.bid, td->coll.nChannels, td->coll.root, comm->busId, comm->nRanks);
                else if ((type&0xf0) == ncclCollTraceP2pElemType)
                  sprintf(line+offset, " Recv %d -> %d/%d/%d/%d ConnIdx/LL/Reg/nc %d/%d/%d/%d -> Send %d cb %d busId %lx nRanks %d",
                    td->p2p.recvRank, td->p2p.recvConnIndex, td->p2p.recvProtoLL, td->p2p.recvRegistered, td->p2p.nRecvChannels, td->p2p.sendConnIndex, td->p2p.sendProtoLL, td->p2p.sendRegistered, td->p2p.nSendChannels, td->p2p.sendRank, td->p2p.channelBase,
                    comm->busId, comm->nRanks);
                break;
              case ncclCollTraceKernelEndType:
                sprintf(line+offset, " KE busId %lx nRanks %d", comm->busId, comm->nRanks);
                break;
              case ncclCollTraceAbortType:
                sprintf(line+offset, " Abort");
                break;
              default:
                sprintf(line+offset, " unknown collective trace data type");
                break;
            }
          }
        }
        INFO(NCCL_COLL, "%s", line);
        td->type = ncclCollTraceNotReady;
      }
    }
    if (comm->collTraceExit && numActiveChans == 0)
      break;
    usleep(1000); //sleep 1ms
  } while(true);
  if (comm->collTraceThread)
    pthread_exit(NULL);
  else
    return 0;
}
#endif

#undef NCCL_NO_OPTIMIZE


static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;
  /* commFree() should not involve any sync among ranks. */
  if (comm == NULL)
    return ncclSuccess;

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
    PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
    if (comm->proxyState->threadUDS) {
      // UDS support
      PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
    }
  }

  CUDACHECK(cudaMemPoolDestroy(comm->memPool));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

#ifdef ENABLE_PROFILING
  struct ncclProf *prof, *prof_seq;
  prof = (struct ncclProf*)malloc(sizeof(struct ncclProf)*MAXCHANNELS*PROFILE_NUM_LAUNCHES);
  CUDACHECK(hipMemcpy(prof, comm->devComm->devProf, sizeof(struct ncclProf)*MAXCHANNELS*PROFILE_NUM_LAUNCHES, hipMemcpyDeviceToHost));
  #define VEGA_GPU_RTC_FREQUENCY 2.5E7
  for (int i=0; i<comm->nChannels; i++) {
    for (int s=0; s<prof[MAXCHANNELS*i].seq; s++) {
      if (prof[MAXCHANNELS*s+i].count == 0) continue;
      for (int j=0; j<prof[MAXCHANNELS*s+i].count; j++) {
        INFO(NCCL_INIT, "# [%02d:%02d] %02d-%02d L:%04u %6.2fus", comm->rank, i, s, j, prof[MAXCHANNELS*s+i].elem[j].line, (prof[MAXCHANNELS*s+i].elem[j].timeStamp-prof[MAXCHANNELS*s+i].elem[0].timeStamp)/VEGA_GPU_RTC_FREQUENCY*1.0E6);
      }
    }
  }
  free(prof);
  CUDACHECK(hipFree(comm->devComm->devProf));
#endif

#ifdef ENABLE_COLLTRACE
  comm->collTraceExit = 1;
  if (comm->collTraceEnabled) {
    if (comm->collTraceThread)
      pthread_join(comm->collTraceThread, NULL);
    else
      ncclCommThreadMain((void *)comm);
  }
  NCCLCHECK(ncclCudaFree((void *)comm->collTrace));
  NCCLCHECK(ncclCudaFree((void *)comm->collTraceTail));
#endif

  free(comm->peerInfo);
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->collNetHeads);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));

  if (comm->doneEvent != NULL)
    CUDACHECK(hipEventDestroy(comm->doneEvent));

  if (comm->sharedRes) {
    if (ncclAtomicRefCountDecrement(&comm->sharedRes->refCount) == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      NCCLCHECK(ncclProxyDestroy(comm));
      free(comm->sharedRes);
    }
  }

#if CUDART_VERSION >= 12010
  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));
#endif

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }
  CUDACHECK(hipStreamDestroy(comm->sideStream));

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  abort = *comm->abortFlag;
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    free(comm->abortFlag);
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  free(comm->gproxyConn);

  NCCLCHECK(ncclRegCleanup(comm));

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");

  commPoison(comm); // poison comm before free to avoid comm reuse.
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  NCCLCHECK(ncclNetFinalize(comm));
  NCCLCHECK(ncclNetPluginUnload(comm));
  free(comm);

  return ncclSuccess;
}

RCCL_PARAM(P2pNetDisable, "P2P_NET_DISABLE", 0);
RCCL_PARAM(PivotAlltoallEnable, "PIVOT_ALLTOALL_ENABLE", 1);
RCCL_PARAM(LL128ForceEnable, "LL128_FORCE_ENABLE", 0);
NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);
NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<20)
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
enum ncclLaunchMode ncclParamLaunchMode;


// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  if (comm->ncclNet->regMrDmaBuf == NULL || rocmLibraryInit() != ncclSuccess) return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // Query device to see if DMA-BUF support is available
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#else
  return pfn_hsa_amd_portable_export_dmabuf != NULL ? ncclSuccess : ncclInternalError;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret != ncclSuccess) {
      /* if ret is not ncclInProgress, we just keep it. */
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      if (ret == ncclInProgress) ret = ncclInvalidArgument;
      goto exit;
    }
    /* if there is linked group job, we should complete it. */
    if (comm->groupJob) {
      NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
      comm->groupJob = NULL;
    }
  }

exit:
  return ret;
}

RCCL_PARAM(InjectFaults, "INJECT_FAULTS", 0);

static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  NCCLCHECK(ncclNetPluginLoad(comm));
  NCCLCHECK(ncclNetInit(comm));
  NCCLCHECK(ncclProfilerPluginInit(comm));
  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  if (parent && parent->config.splitShare) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }
  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  hipEvent_t doneEvent;
  CUDACHECK(hipEventCreateWithFlags(&doneEvent, hipEventDisableTiming));

  comm->doneEvent = doneEvent;
  comm->lastStream = nullptr;
  CUDACHECK(cudaGetDevice(&comm->cudaDev));

  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  char busId[]="0000:00:00.0";
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  NCCLCHECK(rocm_smi_init());
  NCCLCHECK(rocm_smi_getDeviceIndexByPciBusId(busId, (unsigned int*)&comm->nvmlDev));

  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  // RCCL: create persistent stream for calloc
  CUDACHECK(hipStreamCreateWithFlags(&comm->sideStream, hipStreamNonBlocking));
  // RCCL: determine and set unroll factor for comm
  NCCLCHECK(commSetUnrollFactor(comm));
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

#ifdef ENABLE_COLLTRACE
  NCCLCHECK(ncclCudaCalloc(&comm->collTraceTail, MAXCHANNELS));
  NCCLCHECK(ncclCudaCalloc(&comm->collTrace, COLLTRACE_NUM_ITEMS*MAXCHANNELS));
  comm->collTraceExit = 0;
  comm->collTraceEnabled = false; // we can enable colltrace without starting a thread
  if ((ncclDebugLevel >= NCCL_LOG_INFO) && rcclParamKernelCollTraceEnable()) {
    comm->collTraceEnabled = true;
    if (rcclParamKernelCollTraceThreadEnable())
      pthread_create(&comm->collTraceThread, NULL, ncclCommThreadMain, (void *)comm);
    else
      comm->collTraceThread = 0;
  }
#endif

  if (rcclParamInjectFaults() != 0) {
#ifdef ENABLE_FAULT_INJECTION
    comm->faults = rcclParamInjectFaults();
    if (comm->rank == 0) INFO(NCCL_INIT, "Enabled RCCL faults injection with value 0x%lx", comm->faults);
#else
    WARN("Ignore faults injection of value 0x%lx as RCCL is not compiled to support it", rcclParamInjectFaults());
#endif
  }

  comm->collNetSupport = 0;
  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);

  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks*NCCL_MAX_CONNS));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks*NCCL_MAX_CONNS));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  if (parent == NULL || !parent->config.splitShare) {
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));
    /* most of attributes are assigned later in initTransportsRank(). */
    sharedRes->owner = comm;
    sharedRes->tpNRanks = comm->nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));
    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;
  } else {
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
  }

  CUDACHECK(hipDeviceGetAttribute(&comm->WarpSize, hipDeviceAttributeWarpSize, comm->cudaDev));
  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

  do {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  //  setup intraComm0 and intraRanks 0 to default values to ensure proper cleanup of the communicator
  comm->intraComm0 = comm;
  comm->intraRank = 0;
  comm->intraRanks = 1;

  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels tmpCommAndChans;
  struct ncclDevCommAndChannels *devCommAndChans = NULL;
  struct ncclNvmlCCStatus ccStatus;

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream), ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
  tmpCommAndChans.comm.isNvlink = ncclTopoPathAllNVLink(comm->topo);
  tmpCommAndChans.comm.p2pnChannelsPerPeer = comm->p2pnChannelsPerPeer;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  memset(&ccStatus, 0, sizeof(ccStatus));
  if (ncclNvmlGetCCStatus(&ccStatus) == ncclSuccess && ccStatus.CCEnabled) {
    comm->workFifoBytes = 0;
    if (ccStatus.multiGpuCCEnabled == false && comm->rank == 0) {
      WARN("CC On, Multi-GPU CC Off (No inter-GPU communication protection)");
    }
  } else {
    comm->workFifoBytes = ncclParamWorkFifoBytes();
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
    }
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
  }
#else
  comm->workFifoBytes = ncclParamWorkFifoBytes();
  if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
    WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
    comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
  }
  comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
#endif

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, Multi-GPU CC %s, workFifoBytes %d", ccStatus.CCEnabled ? "On" : "Off", ccStatus.multiGpuCCEnabled ? "On" : "Off", comm->workFifoBytes);
  }

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle, comm->sideStream), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);
  } else {
    // The workFifoBuf lives in cudaHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoConsumed, MAXCHANNELS), ret, fail);
  ncclCommPushCudaHostFree(comm, comm->workFifoConsumed);
  comm->workFifoProduced = 0;
  comm->workFifoConsumedLeast = 0;
  tmpCommAndChans.comm.workConsumed = comm->workFifoConsumed;

  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  }

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].binTree = comm->channels[c].binTree;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
    }
  }

#ifdef ENABLE_COLLTRACE
  tmpCommAndChans.comm.collTrace = comm->collTrace;
  tmpCommAndChans.comm.collTraceTail = comm->collTraceTail;
  tmpCommAndChans.comm.collTraceThread = comm->collTraceThread;
#endif

#if defined(ENABLE_NPKIT)
  // Init NPKit
  NCCLCHECK(NpKit::Init(comm->rank));
  tmpCommAndChans.comm.npKitEventCollectContexts = NpKit::GetGpuEventCollectContexts();
  tmpCommAndChans.comm.cpuTimestamp = NpKit::GetCpuTimestamp();
#endif

#ifdef ENABLE_PROFILING
  NCCLCHECK(ncclCudaCalloc(&tmpCommAndChans.comm.devProf, MAXCHANNELS*PROFILE_NUM_LAUNCHES, comm->sideStream));
#endif

#ifdef ENABLE_FAULT_INJECTION
  tmpCommAndChans.comm.faults = comm->faults;
#endif

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream), ret, fail);
exit:
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define VERSION_STRING "RCCL version : " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX
#define VERSION_STRING_EXTENDED "HIP version  : " HIP_BUILD_INFO "\nROCm version : " ROCM_BUILD_INFO
#else
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX
#define VERSION_STRING_EXTENDED "CUDA version " STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
#endif
static void showVersion() {
  char versionInfo[2048+2*HOST_NAME_MAX], hostInfo[HOST_NAME_MAX], libPathInfo[2048];

  // Retrieve Hostname info
  if (gethostname(hostInfo, sizeof(hostInfo)-1) != 0) {
    // Returns Unknown in hostInfo if function call unsuccessful
    strncpy(hostInfo, "Unknown", sizeof(hostInfo)-1);
  }

  // Retrieve librccl path
  Dl_info pathInfo;
  if (dladdr((void*)ncclCommInitRank, &pathInfo)) {
    strncpy(libPathInfo, pathInfo.dli_fname, sizeof(libPathInfo)-1);
  } else {
    // Sets libPath to Unknown if the above function call is not successful
    strncpy(libPathInfo, "Unknown", sizeof(libPathInfo)-1);
  }

  snprintf(versionInfo, sizeof(versionInfo),
    "%s-%s\n%s\n"
    "%-12s : %s\n%12s : %s",
    VERSION_STRING, rcclGitHash, VERSION_STRING_EXTENDED,
    "Hostname", hostInfo, "Librccl path", libPathInfo
  );

  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    VERSION("%s", versionInfo);
  } else {
    INFO(NCCL_ALL,"%s", versionInfo);
  }
}

NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->nvmlDev = comm->nvmlDev;
  NCCLCHECK(ncclGetVersion(&info->version));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;
  info->cuMemSupport = ncclCuMemEnable();

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  // detect if fine grained memory is available on this GPU
  int *ptr;
#if defined(HIP_UNCACHED_MEMORY)
  if (hipExtMallocWithFlags((void**)&ptr, sizeof(int), hipDeviceMallocUncached) == hipSuccess) {
#else
  if (hipExtMallocWithFlags((void**)&ptr, sizeof(int), hipDeviceMallocFinegrained) == hipSuccess) {
#endif
    CUDACHECK(hipFree(ptr));
    info->hasFineGrain = true;
    // GPU supports GDR if DMABUF is supported
    if (dmaBufSupported(comm) == ncclSuccess)
      info->gdrSupport = 1;
    else
      NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  }
  else {
    info->hasFineGrain = false;
    info->gdrSupport = 0;
  }
  comm->hasFineGrain = info->hasFineGrain;

  info->comm = comm;
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  // MNNVL support
  {
    // MNNVL: Request the fabric UUID and partition info
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    nvmlDevice_t nvmlDev;
    NCCLCHECK(int64ToBusId(info->busId, busId));
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);
    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId,
           ((long *)&info->fabricInfo.clusterUuid)[0], ((long *)&info->fabricInfo.clusterUuid)[1],
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
    if (ncclParamMNNVLCliqueId() != -1) info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
  }
#endif

  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  if (comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();
  else if (ncclTopoPathAllNVLink(comm->topo)) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  else comm->p2pChunkSize = ncclParamP2pPciChunkSize();

  // Make sure P2P chunksize is not larger than coll chunksize.
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE]) comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  if (comm->sharedRes->owner != comm) {
    /* make sure split comm p2pChunkSize won't exceed shared p2pChunkSize. */
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 0);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

// Determine if MNNVL support is available
static int checkMNNVL(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // MNNVL requires cuMem to be enabled
  if (!ncclCuMemEnable()) return 0;

  // MNNVL also requires FABRIC handle support
  int cudaDev;
  int flag = 0;
  CUdevice currentDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));;
  if (!flag) return 0;
  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) return 0;
  }

  // Determine our MNNVL domain/clique
  NCCLCHECKGOTO(ncclCalloc(&comm->clique.ranks, comm->nRanks), ret, fail);
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  for (int i = 0; i < comm->nRanks; i++) {
    nvmlGpuFabricInfoV_t *fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    nvmlGpuFabricInfoV_t *fabricInfo2 = &comm->peerInfo[i].fabricInfo;
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
    if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) goto fail;
    if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
        (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
      if (i == comm->rank) {
        comm->cliqueRank = comm->clique.size;
      }
      comm->clique.ranks[comm->clique.size++] = i;
    }
  }
  // Determine whether to enable MNNVL or not
  comm->MNNVL = ncclParamMNNVLEnable() == 2 ? comm->clique.size > 1 : ncclParamMNNVLEnable();
  INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d ", comm->MNNVL, comm->clique.id, comm->clique.size, comm->cliqueRank);

  if (comm->MNNVL) {
    // Force the CUMEM handle type to be FABRIC for MNNVL
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
  }

  return comm->MNNVL;

fail:
  if (comm->clique.ranks) free(comm->clique.ranks);
  return 0;
}

#else
static int checkMNNVL(struct ncclComm* comm) {
  return 0;
}
#endif

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;
  cpu_set_t affinitySave;
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN];
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT];
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  struct graphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
    int crossNic;
  };

  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
    int cpuArch;
    int cpuVendor;
    int nc;
    bool pivotA2AEnabled;
    bool ll128Enabled;
    bool mscclEnabled;
  };

  int nChannelsOrig;
  struct allGatherInfo *allGather3Data = NULL;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;

  bool needsProxy = false;
  bool mscclNeedsProxy = needsProxy;

  timers[TIMER_INIT_ALLGATHER] = clockNano();
  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  comm->cuMemSupport = 1;
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }

  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // MNNVL support
  if (nNodes > 1 && !checkMNNVL(comm) && ncclParamMNNVLEnable() == 1) {
    // Return an error if the user specifically requested MNNVL support
    WARN("MNNVL is not supported on this system");
    ret = ncclSystemError;
    goto fail;
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
    for (int i = 0; i < nranks; i++) comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);

    comm->nvlsRegSupport = 1;
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      if (comm->nvlsRegSupport) {
        for (int j = i + 1; j < nranks; j++) {
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
            comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0;
            break;
          }
        }
      }
    }

    // Buffer Registration is not supported with MNNVL
    if (comm->MNNVL) comm->nvlsRegSupport = 0;

    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    #if defined(ENABLE_NPKIT)
    if (intraProcRanks != 1) {
      WARN("NPKit currently does not support more than 1 device per process");
      ret = ncclInternalError;
      goto fail;
    }
    #endif
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  timers[TIMER_INIT_TOPO] = clockNano();
  // Topo detection / System graph creation
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);
  // save nRanks to ncclTopoSystem as indicator of multi-node
  comm->topo->nRanks = comm->nRanks;
  // init netGdrLevel
  comm->topo->netGdrLevel = -2;
  // init Pivot A2A related fields
  comm->topo->pivotA2AEnabled = false;
  comm->topo->pivotA2ANumBiRings = 0;
  // LL128
  comm->topo->ll128Enabled = false;
  // Topology hint for MSCCL internal scheduler about whether to enable MSCCL
  comm->topo->mscclEnabled = false;
  // Topology hint if tree has been defined by model or User
  comm->topo->treeDefined = false;
  // Compute paths between GPUs and NICs
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);
  // Recompute paths after trimming
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);
  // Decide on comm's CPU architecture.
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  // Determine local CollNet support
  if (collNetSupport(comm)) {
    const char *collNetEnable = ncclGetEnv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }

  // Determine local Nvls support
  NCCLCHECK(ncclNvlsInit(comm));

  // [RCCL] Compute hostIdx (based on hostHash)
  {
    comm->topo->nHosts = 0;
    for (int r = 0; r < nranks; r++) {
      int isNewHost = 1;
      // Check if this is the first time this hostname has been used
      for (int i = 0; i < r && isNewHost; i++) {
        if (comm->peerInfo[i].hostHash == comm->peerInfo[r].hostHash) {
          isNewHost = 0;
        }
      }
      if (isNewHost)
      {
        // Check if this is the same hostname associated with this rank
        if (comm->peerInfo[r].hostHash == comm->peerInfo[rank].hostHash)
          comm->topo->hostIdx = comm->topo->nHosts;
        comm->topo->nHosts++;
      }
    }
  }

  timers[TIMER_INIT_GRAPHS] = clockNano();
  // Get rings and trees
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0;
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph->minChannels = 1;
  ringGraph->maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1;
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;
  collNetChainGraph->collNet = 1;
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;
  if (comm->collNetSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }
  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];

  bool allXgmi, hasPeerAccess;
  allXgmi = true;
  hasPeerAccess = true;
  // Check that all the GPUs have peer access to one another and are XGMI connected
  for (int i = 0; i < nranks && hasPeerAccess; i++) {
    int cudaDev1 = comm->peerInfo[i].cudaDev;
    for (int j = 0; j < nranks; j++) {
      if (i == j) continue;
      int cudaDev2 = comm->peerInfo[j].cudaDev;
      int p2p;
      if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess || !p2p)
      {
        hasPeerAccess = false;
        break;
      }

      bool isXGMI;
      // Limit to single intermediate GPU for enabling clique
      NCCLCHECK(ncclTopoGetLinkType(comm->topo, i, j, &isXGMI, 1));
      allXgmi &= isXGMI;
    }
  }
  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  if ((comm->topo->type & RCCL_TOPO_4P2H_ROME) && (comm->topo->type & RCCL_TOPO_GDR_ALL)) {
    if (rcclParamP2pNetDisable() == 0) {
      if (!(comm->topo->type & RCCL_TOPO_FORCE_INTRA)) comm->p2pNet = 1;
      INFO(NCCL_INIT, "RCCL enabled same node P2P over network");
    }
    else
      INFO(NCCL_INIT, "RCCL force disabled same node P2P over network");
  }
  // Because timers[[TIMER_INIT_ALLGATHER] already contains the timing of the first allgather,
  // we temporarily store the start time of the subsequent one in an as-of-yet unused CONNECT timer.
  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, comm->busId, &idx));
  allGather3Data[rank].nc = 2;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
       IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx906") && allXgmi)
    allGather3Data[rank].nc = 4;
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx908"))
    allGather3Data[rank].nc = std::max(4/ringGraph->nChannels, 2);
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
       (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
      IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx90a"))
    allGather3Data[rank].nc = 4;
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx90a"))
    allGather3Data[rank].nc = std::max(allGather3Data[rank].nc, 4/ringGraph->nChannels);
  if (ringGraph->nChannels > MAXCHANNELS/2)
    allGather3Data[rank].nc = 1;
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx950")) {
    // Multi-node MI300A
    int managed = 0;
    CUDACHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
    if (managed && nNodes > 1) {
      // This forces the minimum channels to 24
      allGather3Data[rank].nc = 6;
    } else {
      // MI300X
      if (nranks == 2)
        // NCCL_MIN_NCHANNELS=32
        allGather3Data[rank].nc = 16;
      else if (nranks == 4)
        // NCCL_MIN_NCHANNELS=24
        allGather3Data[rank].nc = 4;
    }
  }

  allGather3Data[rank].pivotA2AEnabled = comm->topo->pivotA2AEnabled && rcclParamPivotAlltoallEnable();
  comm->topo->ll128Enabled =  comm->topo->ll128Enabled || rcclParamLL128ForceEnable();
  allGather3Data[rank].ll128Enabled = comm->topo->ll128Enabled;
  allGather3Data[rank].mscclEnabled = comm->topo->mscclEnabled;

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;

  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[r] = node;

    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    if (comm->cpuVendor != allGather3Data[r].cpuVendor &&
        comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
  }

  // Alert the user to the presence of mixed CPUs. In the past this has caused
  // locks in some collective routines. This may help debug issues in the future.
  if (rank==0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d",
       comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  int nc;
  nc = allGather3Data[0].nc;
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    nc = std::min(allGather3Data[i].nc, nc);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && allGather3Data[i].pivotA2AEnabled;
    comm->topo->ll128Enabled = comm->topo->ll128Enabled && allGather3Data[i].ll128Enabled;
    comm->topo->mscclEnabled = comm->topo->mscclEnabled && allGather3Data[i].mscclEnabled;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
  }
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->collNetSupport = 0;
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;

  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels =
    (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph->nChannels, ringGraph->nChannels) : ringGraph->nChannels;
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->collNetSupport == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->collNetSupport = 0;
    }
    // As long as there is more than 1 rank on any node, we need to disable collnet reg
    comm->collNetRegSupport = (comm->maxLocalRanks == 1);
  }

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);

  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent, nc), ret, fail);
  if (comm->topo->treeDefined) NCCLCHECK(ncclTreeBasePostset(comm, treeGraph));

  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[4096];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 2047-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %d : %d -> %d -> %d comm %p nRanks %02d busId %lx", c, comm->channels[c].ring.prev,
         comm->rank, comm->channels[c].ring.next, comm, comm->nRanks, comm->busId);
  }
  line[4095] = '\0';
  INFO(NCCL_INIT, "Trees%s comm %p nRanks %02d busId %lx", line, comm, comm->nRanks, comm->busId);

  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->intraNodeP2pSupport, &comm->directMode), ret, fail);
  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->config.splitShare) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();
  do { // Build p2p schedule
    int node = comm->node;
    int nNodes = comm->nNodes;
    int nRanks = comm->nRanks;
    int local = comm->localRank;
    int nLocals = comm->maxLocalRanks;
    struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
    bool flat = false;
    for (int node = 0; node < nNodes; node++) {
      if (nodeRanks[node].localRanks != nLocals) {
        flat = true;
        nNodes = 1; node = 0;
        nLocals = nRanks; local = rank;
        break;
      }
    }
    int nNodesPow2 = pow2Up(nNodes);
    int nLocalsPow2 = pow2Up(nLocals);
    comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, nRanks);
    comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, nRanks);
    uint32_t nodeRound = 0;
    uint32_t nodeDelta = 0;
    int round = 0;
    // When enumerating peer deltas we use the quadratic formula (x*x+x)/2 mod N.
    // Since that formula only produces valid permutations when N is a pow of 2,
    // we let N = pow2Up(n) and filter out results greater-eq to n.
    // Example sequence for 16 ranks: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
    do {
      if (nodeDelta < nNodes) { // Filter nonsensical node deltas
        int sendNode = (node + nodeDelta) % nNodes;
        int recvNode = (node - nodeDelta + nNodes) % nNodes;
        uint32_t localRound = 0;
        uint32_t localDelta = 0;
        do {
          if (localDelta < nLocals) { // Filter nonsensical node-local deltas
            int sendLocal = (local + localDelta) % nLocals;
            int recvLocal = (local - localDelta + nLocals) % nLocals;
            comm->p2pSchedule[round].sendRank = flat ? sendLocal : nodeRanks[sendNode].localRankToRank[sendLocal];
            comm->p2pSchedule[round].recvRank = flat ? recvLocal : nodeRanks[recvNode].localRankToRank[recvLocal];
            round += 1;
          }
          localRound += 1;
          localDelta = (localDelta + localRound) & (nLocalsPow2 - 1); // Quadratic update
        } while (localRound != nLocalsPow2);
      }
      nodeRound += 1;
      nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1); // Quadratic update
    } while (nodeRound != nNodesPow2);

    if (round != nRanks) {
      WARN("P2p schedule creation has bugs.");
      ret = ncclInternalError;
      goto fail;
    }
  } while (0);

  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (comm->runtimeConn) {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    // Setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // Check if we can setup CollNet
    if (comm->collNetSupport > 0) ncclCollNetSetup(comm, parent, graphs);
  } else {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // Connect NET for intranode use
    if (comm->graphs[NCCL_ALGO_RING].nIntraChannels && rcclParamP2pNetDisable() == 0) {
      comm->useIntraNet = 1;
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels+c;
        if (comm->nRanks == 1) continue;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, NCCL_CONN_IDX_P2P_NET), ret, fail);
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], NCCL_CONN_IDX_P2P_NET), ret, fail);
    }

    // Connect Trees
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // Connect PAT only for communicators with 1 GPU per node
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // Setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // And NVLS trees if needed
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // Check if we can setup CollNet
    if (comm->collNetSupport > 0) {
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY+1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // Connect to local net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // Then to remote ones when using PXN
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    if (ncclParamNvbPreconnect()) {
      // Connect p2p when using NVB path
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r=0; r<nvbNpeers; r++) {
        int peer = nvbPeers[r];
        int sendRound=0, recvRound=0;
        while (comm->p2pSchedule[sendRound].sendRank != peer) sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer) recvRound++;
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c, comm->p2pnChannelsPerPeer, comm->nNodes);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer].masks[channelId/64] |= (1UL<<(channelId%64));
          }
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c, comm->p2pnChannelsPerPeer, comm->nNodes);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer].masks[channelId/64] |= (1UL<<(channelId%64));
          }
        }
      }

      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);
  timers[TIMER_INIT_CONNECT] = clockNano() -  timers[TIMER_INIT_CONNECT];

  if (mscclEnabled() && (comm->topo->mscclEnabled || mscclForceEnabled())) {
    NCCLCHECK(mscclInit(comm));
    mscclStatus& status = mscclGetStatus(comm->rank);
    status.needsProxy |= mscclNeedsProxy;
  }

  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  if (comm->sharedRes->owner == comm && !comm->config.splitShare && ret == ncclSuccess && !ncclCuMemEnable()) ncclProxyShmUnlink(comm);
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

#ifdef USE_INDIRECT_FUNCTION_CALL
NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 1);
#else
NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
#endif
RCCL_PARAM(StackSizeOverride, "STACK_SIZE_OVERRIDE", 0);

NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

#define NCCL_COMMINIT_FUNCNAME_LEN 128
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  // name of the function calling
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

typedef struct{
  int key;
  int color;
} commSplitInfo;
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
#ifdef ENABLE_MSCCLPP
  ncclUniqueId origUniqueId = *job->commId;
#endif
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;
  int64_t stackSize;
  hipDeviceProp_t devProp;

  timers[TIMER_INIT_TOTAL] = clockNano();
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  cudaArch = 100*archMajor + 10*archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
#ifdef USE_INDIRECT_FUNCTION_CALL
  CUDACHECK(hipGetDeviceProperties(&devProp, 0));
  if (ncclParamSetStackSize() == 1 && !IsArchMatch(devProp.gcnArchName,"gfx942") && !IsArchMatch(devProp.gcnArchName,"gfx950")) {
    stackSize = rcclParamStackSizeOverride() ? rcclParamStackSizeOverride() : maxLocalSizeBytes;
    if (stackSize == 0) {
      if (IsArchMatch(devProp.gcnArchName,"gfx906"))
        stackSize = 1024;
      else
        stackSize = 512;
    }
    INFO(NCCL_INIT, "Setting cudaLimitStackSize to %zi maxLocalSizeBytes %zi", stackSize, maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
  }
#endif
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  if (job->parent) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
    // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
    if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    // obtain a unique hash for the comm, re-using part of the parent's hash, commHash is a 64bit struct (=16 hex), add the color
    ncclUniqueId tmpId;
    memset(&tmpId,0,sizeof(ncclUniqueId));// must set 0 here to avoid undefined bits
    snprintf((char*)&tmpId, NCCL_UNIQUE_ID_BYTES, "%016lx-%d", job->parent->commHash, job->color);
    comm->commHash = getHash(tmpId.internal, NCCL_UNIQUE_ID_BYTES);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p color %d key %d- Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info, no commId was used
    commIdHash = 0;
  } else {
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    // obtain a unique hash using the first commId
    comm->commHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    commIdHash = hashUniqueId(job->commId[0]);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

#ifdef ENABLE_MSCCLPP
  if (job->parent) {
    if (job->parent->mscclppCompatible) {
      INFO(NCCL_INIT, "MSCCL++: Splitting a compatible communicator; using parent mscclpp_comm");
      comm->mscclppCompatible = true;
      comm->mscclpp_threshold = job->parent->mscclpp_threshold;
      comm->mscclpp_comm = job->parent->mscclpp_comm;
      auto& mscclppUniqueId = mscclpp_uniqueIdMap[*job->commId];
      mscclpp_uniqueIdMap[*job->commId] = mscclppUniqueId;
      mscclpp_uniqueIdReverseMap[mscclppUniqueId].insert(*job->commId);
      ncclCommToUniqueIdMap[comm] = *job->commId;
    }
  }
  else
#endif
  if (rcclParamMscclppEnabled()) {
#ifdef ENABLE_MSCCLPP
    if (mscclEnabled() && (comm->topo->mscclEnabled || mscclForceEnabled()) && mscclppCommCompatible(comm)) {
      hipDeviceProp_t devProp;
      CUDACHECK(hipGetDeviceProperties(&devProp, cudaDev));
      comm->mscclppCompatible = IsArchMatch(devProp.gcnArchName, "gfx942") || IsArchMatch(devProp.gcnArchName, "gfx950");
      if (comm->mscclppCompatible) {
        bool mapContainsId = (mscclpp_uniqueIdMap.count(*job->commId) > 0);
        auto& mscclppUniqueId = mscclpp_uniqueIdMap[*job->commId];
        if (comm->localRank == 0 && !mapContainsId) {
          NCCLCHECKGOTO(mscclpp_ncclGetUniqueId(&mscclppUniqueId), res, fail);
          TRACE_CALL("mscclpp_ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(mscclppUniqueId));
        }

        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, &mscclppUniqueId, sizeof(mscclppUniqueId)), res, fail);
        unsigned long long mscclppUniqueIdHash; (void)mscclppUniqueIdHash;
        TRACE_CALL("bootstrapIntraNodeBroadcast(rank=%d, nranks=%d, root=%d, bcastData=hash:0x%llx)", comm->localRank, comm->localRanks, 0, (mscclppUniqueIdHash = (unsigned long long)hashUniqueId(mscclppUniqueId)));
        mscclpp_uniqueIdReverseMap[mscclppUniqueId].insert(*job->commId);

        comm->mscclpp_threshold = rcclParamMscclppThreshold();
        INFO(NCCL_INIT, "MSCCL++: Enabled! Msg size threshold=%zu", comm->mscclpp_threshold);

        NCCLCHECKGOTO(mscclpp_ncclCommInitRank(&(comm->mscclpp_comm), job->nranks, mscclppUniqueId, job->myrank), res, fail);
        TRACE_CALL("mscclpp_ncclCommInitRank (*comm=%p, nranks=%d, commId=hash:0x%llx, myrank=%d)", comm->mscclpp_comm, job->nranks, mscclppUniqueIdHash, job->myrank);
        mscclpp_commToUniqueIdMap[comm->mscclpp_comm] = mscclppUniqueId;
        ncclCommToUniqueIdMap[comm] = *job->commId;
        if (rcclParamMscclppForceEnabled()) {
          comm->mscclppForceEnable = true;
        } else {
          comm->mscclppForceEnable = false;
        }
      } else {
        WARN("MSCCL++: Cannot enable MSCCL++ on %s architecture", devProp.gcnArchName);
      }
    } else {
      comm->mscclppCompatible = false;
      WARN("MSCCL++: Cannot enable MSCCL++; environment is not MSCCL compatible");
    }
#else
    WARN("MSCCL++: Feature not enabled. ENABLE_MSCCLPP must be defined at compile-time to enable this feature.");
#endif
  }

  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), res, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(comm->nRanks, comm->nNodes, ncclDebugLog, &comm->tunerContext));
  }

  // update communicator state
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->color, job->key);
  } else {
    // the name for the replay tool is ncclCommInitRank for all the variations
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);
  return res;
fail:
  comm->initState = res;
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;

  /* override configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    WARN("NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.minCTAs = minCTAsEnv;
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.maxCTAs = maxCTAsEnv;
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.splitShare = splitShareEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    WARN("minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    WARN("maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    WARN("minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    WARN("splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parnet) {
  memcpy(&childComm->config, &parnet->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* check version. */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT &&
    internalConfigPtr->minCTAs <= 0) ||
    (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT &&
      internalConfigPtr->maxCTAs <= 0) ||
    (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs, internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;

  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  free(job->commId);
  free(_job);
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument;
  }
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob* job = NULL;
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), res, fail);

  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, showVersion);
  }
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  // for loggin only, not ready for replaying
  // !recording at sink
  NCCLCHECK(Recorder::instance().record(rrCommInitDev, nranks, myrank, commId, comm, cudaDev));
  return ncclGroupErrCheck(res);
fail:
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

struct NvtxParamsCommInitRank
{
  int rank;
  int nranks;
  int cudaDev;
};
constexpr nvtxPayloadSchemaEntry_t CommInitRankSchema[] = {
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Rank"},
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of ranks", nullptr, 0, offsetof(NvtxParamsCommInitRank, nranks)},
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "CUDA device", nullptr, 0, offsetof(NvtxParamsCommInitRank, cudaDev)},
};

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank_impl(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(Recorder::instance().record(rrCommInitRank, nranks, myrank, &commId));
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  rocmLibraryInit();

  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECK(cudaGetDevice(&cudaDev));

  NvtxParamsCommInitRank payload{myrank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommInitRank, CommInitRankSchema, payload)

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll_impl(ncclComm_t* comms, int ndev, const int* devlist) {
  Recorder::instance().record(comms, ndev, devlist);
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  constexpr nvtxPayloadSchemaEntry_t CommInitAllSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of devices"}
  };
  NVTX3_FUNC_WITH_PARAMS(CommInitAll, CommInitAllSchema, ndev)

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  rocmLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* duplicate device check. */
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStart(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms+i, ndev,1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEnd(), ret, fail);

exit:
  cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig_impl(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  Recorder::instance().record(rrCommInitRankConfig, nranks, myrank, &commId, config);
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  rocmLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  NvtxParamsCommInitRank payload{myrank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommInitRankConfig, CommInitRankSchema, payload)

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  rocmLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  NvtxParamsCommInitRank payload{myrank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommInitRankScalable, CommInitRankSchema, payload)

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  int savedDevice;
  int commDevice = comm->cudaDev;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaGetDevice(&savedDevice), ret, fail);
  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(commDevice), ret, fail);
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d\n", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d\n", comm, comm->rank, ret);
    }

    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
    // And keep polling until all graphs referencing us die.
    while (comm->persistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(savedDevice), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  int savedDevice;
  int commDevice = comm->cudaDev;
  bool mscclEnabledForTopo = comm->topo->mscclEnabled;

  CUDACHECK(cudaGetDevice(&savedDevice));
  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->destroy(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }

  if (mscclEnabled() && (mscclEnabledForTopo || mscclForceEnabled())) {
    NCCLCHECK(mscclTeardown(comm->rank));
  }

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(savedDevice));
  }

#if defined(ENABLE_NPKIT)
  // Dump NPKit events and shutdown
  const char* npkitDumpDir = getenv("NPKIT_DUMP_DIR");
  if (npkitDumpDir == nullptr) {
    WARN("NPKIT_DUMP_DIR is empty");
  } else {
    NCCLCHECK(NpKit::Dump(npkitDumpDir));
  }
  NCCLCHECK(NpKit::Shutdown());
#endif

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize_impl(ncclComm_t comm) {
  NCCLCHECK(Recorder::instance().record(rrCommFinalize, comm));
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, NULL, free, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(comm, &ret)); }
  return ret;
fail:
  free(job);
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* this is  the last call to ncclCommDestroy/Abort, we need to make sure all comms
       * in the process have been finalized before we free local resources. */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy_impl(ncclComm_t comm) {
  NCCLCHECK(Recorder::instance().record(rrCommDestroy, comm));
  if (comm == NULL) {
    NVTX3_FUNC_RANGE_IN(nccl_domain);
    return ncclSuccess;
  }

#ifdef ENABLE_MSCCLPP
  if (comm->mscclppCompatible) {
    auto& mscclppUniqueId = mscclpp_commToUniqueIdMap[comm->mscclpp_comm];
    auto& uniqueIds = mscclpp_uniqueIdReverseMap[mscclppUniqueId];
    auto& ncclUniqueId = ncclCommToUniqueIdMap[comm];
    if (uniqueIds.find(ncclUniqueId) == uniqueIds.end()) {
      WARN("MSCCL++: comm=%p not found in mscclpp_uniqueIdReverseMap for key=%p", comm, comm->mscclpp_comm);
    }
    uniqueIds.erase(ncclUniqueId);
    if (uniqueIds.size() == 0) {
      mscclpp_uniqueIdReverseMap.erase(mscclppUniqueId);
      ncclResult_t res = mscclpp_ncclCommDestroy(comm->mscclpp_comm);
      TRACE_CALL("mscclpp_ncclCommDestroy");
      if (res != ncclSuccess) {
        WARN("MSCCL++: mscclpp_ncclCommDestroy failed (%s)", ncclGetErrorString(res));
      }
    }

    comm->mscclppCompatible = false;
    comm->mscclpp_comm = nullptr;
  }
#endif

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NvtxParamsCommInitRank payload{rank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommDestroy, CommInitRankSchema, payload)

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  return res;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort_impl(ncclComm_t comm) {
  NCCLCHECK(Recorder::instance().record(rrCommAbort, comm));
  if (comm == NULL) {
    NVTX3_FUNC_RANGE_IN(nccl_domain);
    return ncclSuccess;
  }

  // Ask anything that might still be running on the device to quit
  if (comm->childAbortFlag != nullptr) {
    __atomic_store_n(comm->childAbortFlag, 1, __ATOMIC_RELEASE);
    __atomic_store_n(comm->childAbortFlagDev, 1, __ATOMIC_RELEASE);
  }
  __atomic_store_n(comm->abortFlag, 1, __ATOMIC_RELEASE);
  __atomic_store_n(comm->abortFlagDev, 1, __ATOMIC_RELEASE);
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NvtxParamsCommInitRank payload{rank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommAbort, CommInitRankSchema, payload)

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  return ncclSuccess;
fail:
  goto exit;
}

struct NvtxParamsCommSplit {
  int rank;
  int nranks;
  int cudaDev;
  int color;
  int key;
};
constexpr nvtxPayloadSchemaEntry_t CommSplitSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Rank"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of ranks", nullptr, 0, offsetof(NvtxParamsCommSplit, nranks)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "CUDA device", nullptr, 0, offsetof(NvtxParamsCommSplit, cudaDev)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "color", nullptr, 0, offsetof(NvtxParamsCommSplit, color)},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "key", nullptr, 0, offsetof(NvtxParamsCommSplit, key)},
};

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit_impl(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  NvtxParamsCommSplit payload{comm->rank, comm->nRanks, comm->cudaDev, color, key};
  NVTX3_FUNC_WITH_PARAMS(CommSplit, CommSplitSchema, payload)

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));

  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(CommCheck(comm, "CommSplit", "comm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(newcomm, "CommSplit", "newcomm"), res, fail);
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, fail);

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, fail);
  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;
    if (comm->config.splitShare) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", __func__);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, free, comm), res, fail);

exit:
  // for loggin only, not ready for replaying
  // TODO: further integrate overloaded record header
  // !recording at sink
  Recorder::instance().record(rrCommSplit, color, key, (ncclUniqueId*)comm, config, *newcomm);
  cudaSetDevice(oldDev);
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  if (childComm) {
    if (comm && !comm->config.splitShare) {
      free(childComm->abortFlag);
      if (childComm->abortFlagDev) ncclCudaHostFree(childComm->abortFlagDev);
      free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString_impl(ncclResult_t code) {
  Recorder::instance().record("GetErrorString");
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError_impl(ncclComm_t comm) {
  Recorder::instance().record("GetLastEror");
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t *asyncError) {
  Recorder::instance().record("GetAsyncError");
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount_impl(const ncclComm_t comm, int* count) {
  Recorder::instance().record("CommCount");
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice_impl(const ncclComm_t comm, int* devid) {
  Recorder::instance().record("CuDevice");
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank_impl(const ncclComm_t comm, int* rank) {
  Recorder::instance().record("CommUserRank");
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc_impl(void **ptr, size_t size) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  size_t mcGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmulticastObjectProp mcprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag;
  int dcnt;
  int mcSupport = 0;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));

  if (mcSupport) {
    int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // Query device to see if FABRIC handle support is available
    flag = 0;
    (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));;
    if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    flag = 0;
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    /* mc property */
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    mcprop.size = size;
    /* device cnt is a dummy value right now, it might affect mc granularity in the future. */
    mcprop.numDevices = dcnt;
    mcprop.handleTypes = requestedHandleTypes;
    mcprop.flags = 0;
    CUCHECK(cuMulticastGetGranularity(&mcGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

    /* only size needs to be aligned to mcGran */
    ALIGN_SIZE(size, mcGran);
    if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
      /* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
      CUresult err = CUPFN(cuMemCreate(&handle, size, &memprop, 0));
      if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
        requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
        memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
        /* Allocate the physical memory on the device */
        CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
      }
    } else {
      /* Allocate the physical memory on the device */
      CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
    }
    /* Reserve a virtual address range */
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, cudaDev, i) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
      }
    }
    goto exit;
  }

fallback:
#endif
  // Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
  // we want CUDA to return an error to the caller.
  // coverity[var_deref_model]
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

exit:
  NCCLCHECK(Recorder::instance().record(rrMemAlloc, *ptr, size));
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree_impl(void *ptr) {
  NCCLCHECK(Recorder::instance().record(rrMemFree, ptr));
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010
  CUdevice ptrDev = 0;
  int mcSupport = 0;

  if (ptr == NULL) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECKGOTO(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ptrDev), ret, fail);

  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  if (mcSupport) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
  CUDACHECK(cudaSetDevice(saveDevice));
  return ret;
fail:
  goto exit;
}
