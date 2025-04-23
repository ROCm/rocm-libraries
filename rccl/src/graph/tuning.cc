/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "device.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def, int WarpSize) {
  int nt = env;
  if (nt > 0) {
    if (nt % WarpSize != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WarpSize);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
    }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
      { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Tree, Ring
      { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Collnet Direct, Chain
      {    0,    0,    0 }, {    0,    0,    0 }};  // NVLS, NVLS Tree

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2



struct tuningModel {
  float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bwRatio [2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  uint64_t llProtoRanges[RCCL_TUNABLE_COLLS][NCCL_NUM_PROTOCOLS - 1][RCCL_PROTOCOL_ENTRY_SIZE];
};

static struct tuningModel tuning_model_0 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 1.4 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 11.8, 18.2, 20.8 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 19.8, 15.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 11.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 18.2 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.04, 0.22, 0.91 }, /* Ring (LL/LL128/Simple)*/ { 0.04, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.04, 0.22, 0.95 }, /* Ring (LL/LL128/Simple)*/ { 0.04, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.2, 0.1, 0.1, 0.9, 0.3, 0.4, 0.1, 0.2, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.3, 1.0, 0.1, 0.5, 1.0, 0.9, 1.0, 1.0, 1.0, 0.3, 0.1, 0.4, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.2, 1.0, 0.1, 0.1, 0.7, 0.2, 0.4, 0.1, 0.1, 0.3, 0.4, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.2, 0.3, 0.5, 0.3, 0.1, 0.5, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, },
    { 1.0, 0.8, 0.2, 1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.5, 1.0, 0.8, 0.8, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {RCCL_LL_LIMITS_UNDEFINED},
};

static struct tuningModel tuning_model_1 {
  .hwLat =
  { /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 4.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 4.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 33.0, 33.0, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 5.1, 5.1, 68.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 15.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 15.8 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio =
  { /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.12, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.12, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.15, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.5, 0.1, 0.6, 1.0, 1.0, 1.0, 0.6, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.3, },
  },

  .ringCorrectionFactor = {
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 0.3, 1.0, 0.3, 0.1, 0.1, 0.1, 0.3, 0.7, 1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {RCCL_LL_LIMITS_UNDEFINED},
};

static struct tuningModel tuning_model_2 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 4.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 4.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 27.9, 27.9, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 12.1, 12.1, 68.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 15.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 15.8 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.6, 0.9, 0.8, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 1.0, },
  },

  .ringCorrectionFactor = {
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {RCCL_LL_LIMITS_UNDEFINED},
};

static struct tuningModel tuning_model_3 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 0.0, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.0, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 12.5, 0.0, 22.4 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 0.0, 19.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 12.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.20, 0.00, 1.75 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.20, 0.00, 0.96 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.2, 1.0, 0.9, 1.0, 0.6, 0.4, 0.6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.2, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.8, 0.9, 0.7, 0.7, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.2, 0.1, 0.4, 0.4, 0.2, 0.2, 0.3, 0.7, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {RCCL_LL_LIMITS_UNDEFINED},
};

static struct tuningModel tuning_model_4 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNetDirect (Simple)*/ { 0.8, 1.4, 2.5 }, /* CollNetChain (Simple)*/ { 0.8, 1.4, 2.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 32.2, 34.4, 47.6 }, /* Ring (LL/LL128/Simple)*/ { 35.4, 87.8, 209.2 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 47.6 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 47.6 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.61 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.08 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.2, 0.4, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 1.0, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.4, 0.3, 0.3, 0.1, 0.1, 1.0, 1.0, 0.7, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.7, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1, 0.3, 1.0, 1.0, 0.7, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.3, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.5, 0.1, 0.7, 0.2, 0.4, 0.4, 0.6, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {RCCL_LL_LIMITS_UNDEFINED},
};

static struct tuningModel tuning_model_5 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.9, 0.9, 2.3 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.8, 2.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.9 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 10.5, 10.5, 25.0 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 9.5, 320.0 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 10.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.11 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.59 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.6, 0.6, 0.6, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.8, 1.0, 1.0, 1.0, },
  },
  // Follow order in RcclTunableColls
  .llProtoRanges = {
    /*ReduceScatter*/
    {/*LL (min/max/factor)*/ {0, 655360, 1}, /*LL64/128 (min/max/factor)*/ {131072, 3211264, 1}},
    /*AllGather*/
    {/*LL (min/max/factor)*/ {0, 98304, 1},  /*LL64/128 (min/max/factor)*/ {98304, 5046272, 1}},
    /*AllReduce*/
    {/*LL (min/max/factor)*/ {0, 1048576, 1},/*LL64/128 (min/max/factor)*/ {1048576, 9437184, 3145728}},
  },
};

static struct tuningModel rcclTuningModel[] = {
  tuning_model_0,
  tuning_model_1,
  tuning_model_2,
  tuning_model_3,
  tuning_model_4,
  tuning_model_5,
};

/* Array indexes used below */
#define VOLTA_COMPCAP_IDX 0
#define AMPERE_COMPCAP_IDX 1
#define HOPPER_COMPCAP_IDX 2

// LL128 max BW per channel
static const double llMaxBws[3][3] = {
  /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4},
  /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
  /* Hopper-N1/AMD-N2/AMD-N4) */ {141.0, 45.0 /*avg of ring & tree*/, 35.0}
};

static const double perChMaxRingLL128Bws[3][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7},
};
static const double perChMaxTreeLL128Bws[3][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0},
};
static const double perChMaxTreeBws[3][3] = {
  /* Volta (N1/N2/N4) */  {26.5, 18.5, 10.0},
  /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
  /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0},
};

NCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
static int ncclPatEnable(struct ncclComm* comm) {
  int patEnable = ncclParamPatEnable();
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;   // PAT doesn't support net device offload
  return 1;
}

// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  return 1.0;
}

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) {
  int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra*graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, NCCL_MAX_NTHREADS, simpleDefaultThreads, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), 4*comm->WarpSize, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS, comm->WarpSize);
#else
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
#endif

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;

  int compCapIndex = minCompCap >= 90 ? HOPPER_COMPCAP_IDX : minCompCap >= 80 ? AMPERE_COMPCAP_IDX : VOLTA_COMPCAP_IDX;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = nNodes == 1 ? compCapIndex : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  //if (cpuArch == NCCL_TOPO_CPU_ARCH_POWER) hwLat[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = hwLat[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  float ppn = (float)nRanks / nNodes;

  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  memcpy(comm->minMaxLLRange,
        rcclTuningModel[comm->topo->tuning].llProtoRanges,
        sizeof(rcclTuningModel[comm->topo->tuning].llProtoRanges));

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
      if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
          && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
          && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
      if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_SIMPLE && (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) && comm->topo->nodes[GPU].count == comm->topo->nRanks) continue;
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
            && a == NCCL_ALGO_PAT && (p != NCCL_PROTO_SIMPLE || ncclPatEnable(comm) == 0)) continue;
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;
        float busBw = comm->topo->baseBw != 0.0 ? comm->topo->baseBw : graphs[a]->nChannels * bw;
        //INFO(NCCL_INIT, "algo %s proto %s busBw %f baseBw %f bw %f nChannels %d bwIntra %f bwInter %f", ncclAlgoStr[a], ncclProtoStr[p], busBw, comm->topo->baseBw, bw, graphs[a]->nChannels, graphs[a]->bwIntra, graphs[a]->bwInter);

        if (a == NCCL_ALGO_NVLS) bw = std::min(graphs[a]->bwIntra, graphs[a]->bwInter);
        if (a == NCCL_ALGO_NVLS_TREE) bw = std::min(graphs[a]->bwIntra, nNodes <= 2 ? graphs[a]->bwInter : graphs[a]->bwInter/2);

        // Various model refinements
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (nNodes <= 2)
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[0][a][p];
        else
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[1][a][p];
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL && (coll == ncclFuncBroadcast || coll == ncclFuncReduce) && (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) && comm->topo->nodes[GPU].count == comm->topo->nRanks) { busBw = busBw * 1.65; }
#else
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * .5); }
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);
        if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);
        if (a == NCCL_ALGO_TREE && graphs[a]->pattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
        if (a == NCCL_ALGO_PAT) busBw *= .85;
        if (a == NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_CHAIN && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_DIRECT && p == NCCL_PROTO_SIMPLE) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            busBw = ppn * bw;
            // AllGather/ReduceScatter requires 1:1 GPU:NIC
            int nicPerNode = comm->collNetHeadsNum;
            if (coll == ncclFuncAllGather && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->iallgather || ppn > nicPerNode) busBw = 0;
            }
            if (coll == ncclFuncReduceScatter && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter || ppn > nicPerNode) busBw = 0;
            }
            // Measured corrective ratio needed at 1 ppn and 8ppn. Here we hackishly
            // interpolate the two.
            float w = (ppn-1)/(8-1);
            busBw *= w*0.85 + (1-w)*0.95;
          } else {
            // Collnet+Direct requires all GPUs to have a local NIC to work at full speed
            float factor = ppn / (1.0*graphs[a]->nChannels); // GPU/NIC ratio
            factor -= (factor-1)/2;
            busBw /= factor;
            if (minCompCap >= 90) busBw *= .85;
          }
        }
#endif

        // Convert bus BW to algorithm BW
        if (!(a != NCCL_ALGO_RING && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING) ratio *= (1.0 * nRanks) / nsteps;
          else if (a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= 5.0/6.0;
          else ratio *= .5;
          busBw *= ratio;
        }
        comm->bandwidths[coll][a][p] = busBw;
        /* Ring bandwidth backup */
        if (a == NCCL_ALGO_RING)
          comm->ringbdw[coll][p] = comm->bandwidths[coll][NCCL_ALGO_RING][p];

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = rcclTuningModel[comm->topo->tuning].hwLat[intraHw[a]][a][p];
        float interLat =  ppn == 1 ? rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][NCCL_ALGO_TREE][p] : rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        interLat += graphs[a]->latencyInter;
        // Also add the flush extra latency
        if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

        if (a == NCCL_ALGO_RING) {
          float lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a]->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE) lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = getNetOverhead(comm);
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            int nInterSteps = nNodes == 1 ? 0 : coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          if (coll == ncclFuncAllReduce) {
            comm->latencies[coll][a][p] +=
              2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
          }
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          comm->latencies[coll][a][p] +=
            2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;  // Add 0.4 us arity serialization latency
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          comm->latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          if (nNodes > 1) comm->latencies[coll][a][p] += rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          comm->latencies[coll][a][p] += 2*(nNodes-1)*rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        } else if (a == NCCL_ALGO_PAT) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            comm->latencies[coll][a][p] = 8 // Base time
              + log2i(nNodes) * (interLat/3.5) // Log latency
              + nRanks * 2.8; // Still a linear part; hopefully we'll manage to remove it at some point.
          }
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1, 2, 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1, 1, 1, 1, 1, 1 };

  const char *protoStr = ncclGetEnv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = ncclGetEnv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  if (comm->nNodes == 1) algoEnable[NCCL_ALGO_NVLS_TREE] = 0;

  // Disable CollNet if it is not supported
  if (comm->collNetSupport == 0) {
    algoEnable[NCCL_ALGO_COLLNET_DIRECT] = 0;
    algoEnable[NCCL_ALGO_COLLNET_CHAIN] = 0;
    if (nNodes > 1) algoEnable[NCCL_ALGO_NVLS] = 0;
    // If user has hard set NCCL_ALGO=COLLNET, ignore it
    if (algoEnable[NCCL_ALGO_RING] == 0 && algoEnable[NCCL_ALGO_TREE] == 0 &&
        algoEnable[NCCL_ALGO_NVLS] == 0 && algoEnable[NCCL_ALGO_NVLS_TREE] == 0) {
      algoEnable[NCCL_ALGO_RING] = algoEnable[NCCL_ALGO_TREE] = 1;
    }
  } else {
    // Disable CollNet+Direct if not on an NVSwitch system
    int nvsCount = 0;
    NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));
    if (nvsCount == 0) algoEnable[NCCL_ALGO_COLLNET_DIRECT] = 0;
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // Disable LL protocol on gfx12xx
    int pEnable = (p == NCCL_PROTO_LL && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx12")) ? 0 : protoEnable[p];
    if (p == NCCL_PROTO_LL128) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#if defined(ENABLE_LL128)
      // Enable LL128 by default only on gfx90a with available tuning table
      pEnable = (graphs[a]->typeInter <= PATH_PXB) && graphs[a]->typeIntra <= PATH_NVL &&
        ((IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx90a") ||
          IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942")) && comm->topo->ll128Enabled) ? 1 : 0;
#else
      pEnable = 0;
#endif
#else
      // Enable LL128 by default only on Volta/Ampere/Hopper+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = 1;
      pEnable &= (graphs[a]->typeInter <= PATH_PXB || (minCompCap >= 90 && graphs[a]->typeInter <= PATH_PXN));
      pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
      pEnable &= (minCompCap == maxCompCap);
      switch (minCompCap) {
      case 70: pEnable &= 1; break;
      case 80: pEnable &= 1; break;
      case 90: pEnable &= !(CUDART_VERSION == 11080 && c == ncclFuncAllReduce && a == NCCL_ALGO_RING && comm->nRanks == 2); break;
      default: pEnable &= 0; break;
      }
#endif
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    if (algoEnable[a] == 0) comm->bandwidths[c][a][p] = 0;
    //if (a == NCCL_ALGO_RING && pEnable == 0) comm->ringbdw[c][p] = 0;
  }

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    bool available = false;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
        if (comm->bandwidths[c][a][p] != 0) {
          available = true;
          goto check_avail;
        }
  check_avail:
    if (available == false) {
      /* at least set ring algo available */
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
        comm->bandwidths[c][NCCL_ALGO_RING][p] = comm->ringbdw[c][p];
    }
  }

  if (comm->rank == 0) {
    char line[1024];
    for (int block=0; block<DIVUP(NCCL_NUM_ALGORITHMS, 3); block++) {
      sprintf(line, "  Algorithm   |");
      for (int ba=0; ba<3; ba++) {
  int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        sprintf(line+strlen(line), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      sprintf(line, "  Protocol    |");
      for (int ba=0; ba<3; ba++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      sprintf(line, " Max NThreads |");
      for (int ba=0; ba<3; ba++) {
  int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), " %14d |", comm->maxThreads[a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
        sprintf(line, "%13s |", ncclFuncStr[c]);
        for (int ba=0; ba<3; ba++) {
    int a = block*3+ba;
          if (a >= NCCL_NUM_ALGORITHMS) continue;
          for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
            sprintf(line+strlen(line), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 256;
  comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 256;

  // Override defaults with user env
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2 }};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<2; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time, bool* backup) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];

  if (backup) {
    *backup = false;
    if (algorithm == NCCL_ALGO_RING && bw == 0.0f) {
      /* try back up RING algorithm */
      bw = comm->ringbdw[coll][protocol];
      *backup = true;
    }
  }

  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(nBytes>>6);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (algorithm == NCCL_ALGO_TREE) {
    if (logSize < 27) bw *= rcclTuningModel[comm->topo->tuning].treeCorrectionFactor[protocol][logSize];
    else bw *= rcclTuningModel[comm->topo->tuning].treeCorrectionFactor[protocol][26];
  }
  else if (algorithm == NCCL_ALGO_RING && comm->nNodes > 1) {
    if(logSize < 27) bw *= rcclTuningModel[comm->topo->tuning].ringCorrectionFactor[protocol][logSize];
    else bw *= rcclTuningModel[comm->topo->tuning].ringCorrectionFactor[protocol][26];
  }
#else
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1
      && coll == ncclFuncAllReduce && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
#endif
  // Tree pipelining saves latency in aggregation cases
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
}
