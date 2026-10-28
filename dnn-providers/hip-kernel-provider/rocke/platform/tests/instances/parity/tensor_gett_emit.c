/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/tensor_gett_emit.c -- C-side emitter for the tensor GETT and its
 * tuned route table. Builds the rows of tensor_gett_emit.py (same order)
 * through rocke/instance_tensor_gett.h: a tuned row parses the shipped table
 * (embedded below), checks the route key of the entry's problem and builds the
 * spec of tuned_plan; an explicit row builds its TensorGettSpec.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/instance_tensor_gett.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{
using namespace rocke::gett;

const char* const TUNED_JSON[] = {
    "{\n \"entries\": {\n  \"gfx950\": {\n   \"0f9482785a920350e45f\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_"
    "barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_"
    "interleave=True,mn_chunk=128\",\n    \"problem\": {\n     \"bilinear\": true,\n     "
    "\"dtype\": \"fp16\",\n     \"extents\": {\n      \"K0\": 6,\n      \"K1\": 6,\n      \"K2\": "
    "4,\n      \"K3\": 4,\n      \"M0\": 12,\n      \"M1\": 8,\n      \"M2\": 8,\n      \"M3\": "
    "6,\n      \"N0\": 10,\n      \"N1\": 8,\n      \"N2\": 6,\n      \"N3\": 8\n     },\n     "
    "\"modes_a\": [\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M3\",\n   "
    "   \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      \"K3\",\n      "
    "\"K2\",\n      \"K1\",\n      \"K0\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\"\n     ],\n     \"modes_d\": [\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    "
    "\"route\": \"mn\"\n   },\n   \"23f2bbb805ad34ff92ea\": {\n    \"body\": "
    "\"t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,dtl_"
    "frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_"
    "after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_interleave=True,mn_chunk="
    "128,waves_per_eu=1\",\n    \"problem\": {\n     \"bilinear\": false,\n     \"dtype\": "
    "\"bf16\",\n     \"extents\": {\n      \"K0\": 24,\n      \"K1\": 24,\n      \"M0\": 48,\n     "
    " \"M1\": 32,\n      \"N0\": 40,\n      \"N1\": 36\n     },\n     \"modes_a\": [\n      "
    "\"K1\",\n      \"K0\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K1\",\n      \"K0\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      "
    "\"N1\",\n      \"N0\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n  "
    " },\n   \"3d6bbc99c041dd301738\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_",
    "stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
    "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128\",\n    \"problem\": {\n     "
    "\"bilinear\": false,\n     \"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 6,\n      "
    "\"K1\": 6,\n      \"K2\": 4,\n      \"K3\": 4,\n      \"M0\": 12,\n      \"M1\": 8,\n      "
    "\"M2\": 8,\n      \"M3\": 6,\n      \"N0\": 10,\n      \"N1\": 8,\n      \"N2\": 6,\n      "
    "\"N3\": 8\n     },\n     \"modes_a\": [\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      "
    "\"K0\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     "
    "\"modes_b\": [\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"N3\",\n   "
    "   \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N3\",\n      "
    "\"N2\",\n      \"N1\",\n      \"N0\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      "
    "\"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   \"4907717213385e0ec490\": {\n    "
    "\"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
    "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,lds_swizzle="
    "True,mfma_interleave=True,mn_chunk=128\",\n    \"problem\": {\n     \"bilinear\": true,\n     "
    "\"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": "
    "3,\n      \"K3\": 3,\n      \"K4\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": "
    "6,\n      \"M3\": 4,\n      \"M4\": 4,\n      \"N0\": 8,\n      \"N1\": 6,\n      \"N2\": "
    "4,\n      \"N3\": 6,\n      \"N4\": 4\n     },\n     \"modes_a\": [\n      \"K4\",\n      "
    "\"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M4\",\n      \"M3\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      \"K4\",\n      "
    "\"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"N4\",\n      \"N3\",\n      "
    "\"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N4\",\n      "
    "\"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M4\",\n      \"M3\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"52afee352c8cc1485d4f\": {\n    \"body\": \"t128x128x32_w2x",
    "4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b=0,dtl_"
    "frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_merge="
    "True,interwave_setprio=True,mfma_interleave=True\",\n    \"problem\": {\n     \"bilinear\": "
    "true,\n     \"dtype\": \"fp16\",\n     \"extents\": {\n      \"K0\": 12,\n      \"K1\": 8,\n  "
    "    \"K2\": 6,\n      \"M0\": 24,\n      \"M1\": 16,\n      \"M2\": 8,\n      \"N0\": 20,\n   "
    "   \"N1\": 12,\n      \"N2\": 8\n     },\n     \"modes_a\": [\n      \"K2\",\n      \"K1\",\n "
    "     \"K0\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K2\",\n      \"K1\",\n      \"K0\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n  "
    "   \"modes_d\": [\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M2\",\n      "
    "\"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"5ef8541a5e73dbe91da4\": {\n    \"body\": "
    "\"t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_lds=True,"
    "dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
    "default,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_interleave=True\",\n    "
    "\"problem\": {\n     \"bilinear\": false,\n     \"dtype\": \"fp16\",\n     \"extents\": {\n   "
    "   \"K0\": 12,\n      \"K1\": 8,\n      \"K2\": 6,\n      \"M0\": 24,\n      \"M1\": 16,\n    "
    "  \"M2\": 8,\n      \"N0\": 20,\n      \"N1\": 12,\n      \"N2\": 8\n     },\n     "
    "\"modes_a\": [\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M2\",\n      \"M1\",\n   "
    "   \"M0\"\n     ],\n     \"modes_b\": [\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      "
    "\"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    "
    "\"route\": \"mn\"\n   },\n   \"6a5951db87850882819f\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
    "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_"
    "setprio=True,lds_swizzle=True,mfma_interleave=Tru",
    "e,mn_chunk=64\",\n    \"problem\": {\n     \"bilinear\": false,\n     \"dtype\": \"fp16\",\n  "
    "   \"extents\": {\n      \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": 3,\n      \"K3\": 3,\n   "
    "   \"K4\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 6,\n      \"M3\": 4,\n      "
    "\"M4\": 4,\n      \"N0\": 8,\n      \"N1\": 6,\n      \"N2\": 4,\n      \"N3\": 6,\n      "
    "\"N4\": 4\n     },\n     \"modes_a\": [\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      "
    "\"K1\",\n      \"K0\",\n      \"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      "
    "\"M0\"\n     ],\n     \"modes_b\": [\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      "
    "\"K1\",\n      \"K0\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\"\n     ],\n     \"modes_d\": [\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\",\n      \"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      "
    "\"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   \"6ca39588a7df7b98780c\": {\n    "
    "\"body\": "
    "\"t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
    "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_"
    "setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True\",\n    \"problem\": {\n   "
    "  \"bilinear\": false,\n     \"dtype\": \"fp16\",\n     \"extents\": {\n      \"K0\": 4,\n    "
    "  \"K1\": 4,\n      \"K2\": 3,\n      \"K3\": 2,\n      \"K4\": 2,\n      \"K5\": 2,\n      "
    "\"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 4,\n      \"M3\": 4,\n      \"M4\": 3,\n      "
    "\"M5\": 2,\n      \"N0\": 6,\n      \"N1\": 6,\n      \"N2\": 4,\n      \"N3\": 4,\n      "
    "\"N4\": 4,\n      \"N5\": 2\n     },\n     \"modes_a\": [\n      \"K5\",\n      \"K4\",\n     "
    " \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M5\",\n      \"M4\",\n      "
    "\"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K5\",\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      "
    "\"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n  "
    "   \"modes_d\": [\n      \"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\",\n      \"M5\",\n      \"M4\",\n      \"M3\",\n      \"M",
    "2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"826372e76a2ca4dc4d44\": {\n    \"body\": "
    "\"t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,dtl_"
    "frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_merge="
    "True,interwave_setprio=True,lds_swizzle=True,mfma_interleave=True\",\n    \"problem\": {\n    "
    " \"bilinear\": true,\n     \"dtype\": \"fp16\",\n     \"extents\": {\n      \"K0\": 24,\n     "
    " \"K1\": 24,\n      \"M0\": 48,\n      \"M1\": 32,\n      \"N0\": 40,\n      \"N1\": 36\n     "
    "},\n     \"modes_a\": [\n      \"K1\",\n      \"K0\",\n      \"M1\",\n      \"M0\"\n     ],\n "
    "    \"modes_b\": [\n      \"K1\",\n      \"K0\",\n      \"N1\",\n      \"N0\"\n     ],\n     "
    "\"modes_d\": [\n      \"N1\",\n      \"N0\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n   "
    " \"route\": \"mn\"\n   },\n   \"a3f922690fb55c4f4d8d\": {\n    \"body\": "
    "\"t256x256x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=4,chiplet_swizzle=True,direct_"
    "to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
    "3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,"
    "lds_swizzle_r=1,mfma_interleave=True\",\n    \"problem\": {\n     \"bilinear\": true,\n     "
    "\"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 16,\n      \"K1\": 12,\n      "
    "\"K2\": 8,\n      \"M0\": 32,\n      \"M1\": 20,\n      \"M2\": 12,\n      \"N0\": 24,\n      "
    "\"N1\": 20,\n      \"N2\": 14\n     },\n     \"modes_a\": [\n      \"K2\",\n      \"K1\",\n   "
    "   \"K0\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K2\",\n      \"K1\",\n      \"K0\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n  "
    "   \"modes_d\": [\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M2\",\n      "
    "\"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"acde940489d8c678e4cf\": {\n    \"body\": "
    "\"t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds=True,"
    "dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
    "default,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_interleav",
    "e=True,waves_per_eu=1\",\n    \"problem\": {\n     \"bilinear\": true,\n     \"dtype\": "
    "\"bf16\",\n     \"extents\": {\n      \"K0\": 12,\n      \"K1\": 8,\n      \"K2\": 6,\n      "
    "\"M0\": 24,\n      \"M1\": 16,\n      \"M2\": 8,\n      \"N0\": 20,\n      \"N1\": 12,\n      "
    "\"N2\": 8\n     },\n     \"modes_a\": [\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      \"K2\",\n      "
    "\"K1\",\n      \"K0\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     "
    "\"modes_d\": [\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M2\",\n      \"M1\",\n   "
    "   \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   \"bc2dca5ed43e3e3fabd2\": {\n    "
    "\"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
    "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,lds_swizzle="
    "True,mfma_interleave=True,mn_chunk=128\",\n    \"problem\": {\n     \"bilinear\": false,\n    "
    " \"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": "
    "3,\n      \"K3\": 3,\n      \"K4\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": "
    "6,\n      \"M3\": 4,\n      \"M4\": 4,\n      \"N0\": 8,\n      \"N1\": 6,\n      \"N2\": "
    "4,\n      \"N3\": 6,\n      \"N4\": 4\n     },\n     \"modes_a\": [\n      \"K4\",\n      "
    "\"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M4\",\n      \"M3\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      \"K4\",\n      "
    "\"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"N4\",\n      \"N3\",\n      "
    "\"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N4\",\n      "
    "\"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M4\",\n      \"M3\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"c040717c86e077567e5d\": {\n    \"body\": "
    "\"t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_"
    "barrier_after=2,ilv_merge=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleav",
    "e=True\",\n    \"problem\": {\n     \"bilinear\": true,\n     \"dtype\": \"fp16\",\n     "
    "\"extents\": {\n      \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": 3,\n      \"K3\": 2,\n      "
    "\"K4\": 2,\n      \"K5\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 4,\n      "
    "\"M3\": 4,\n      \"M4\": 3,\n      \"M5\": 2,\n      \"N0\": 6,\n      \"N1\": 6,\n      "
    "\"N2\": 4,\n      \"N3\": 4,\n      \"N4\": 4,\n      \"N5\": 2\n     },\n     \"modes_a\": "
    "[\n      \"K5\",\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n "
    "     \"M5\",\n      \"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     "
    "],\n     \"modes_b\": [\n      \"K5\",\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      "
    "\"K1\",\n      \"K0\",\n      \"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N5\",\n      \"N4\",\n      "
    "\"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      \"M5\",\n      \"M4\",\n      "
    "\"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n  "
    " },\n   \"c3afaac231cdfe126504\": {\n    \"body\": "
    "\"t128x128x32_w2x4x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_"
    "stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
    "swizzle=True,mfma_interleave=True,waves_per_eu=1\",\n    \"problem\": {\n     \"bilinear\": "
    "false,\n     \"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 12,\n      \"K1\": 8,\n "
    "     \"K2\": 6,\n      \"M0\": 24,\n      \"M1\": 16,\n      \"M2\": 8,\n      \"N0\": 20,\n  "
    "    \"N1\": 12,\n      \"N2\": 8\n     },\n     \"modes_a\": [\n      \"K2\",\n      "
    "\"K1\",\n      \"K0\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     "
    "\"modes_b\": [\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"N2\",\n      \"N1\",\n   "
    "   \"N0\"\n     ],\n     \"modes_d\": [\n      \"N2\",\n      \"N1\",\n      \"N0\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"ce0756fc3571cca809a1\": {\n    \"body\": "
    "\"t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,dtl_"
    "frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_st",
    "ages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_"
    "interleave=True\",\n    \"problem\": {\n     \"bilinear\": false,\n     \"dtype\": "
    "\"fp16\",\n     \"extents\": {\n      \"K0\": 24,\n      \"K1\": 24,\n      \"M0\": 48,\n     "
    " \"M1\": 32,\n      \"N0\": 40,\n      \"N1\": 36\n     },\n     \"modes_a\": [\n      "
    "\"K1\",\n      \"K0\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K1\",\n      \"K0\",\n      \"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      "
    "\"N1\",\n      \"N0\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n  "
    " },\n   \"d010a031b720ee7d5362\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_"
    "barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_"
    "interleave=True,mn_chunk=128\",\n    \"problem\": {\n     \"bilinear\": true,\n     "
    "\"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 6,\n      \"K1\": 6,\n      \"K2\": "
    "4,\n      \"K3\": 4,\n      \"M0\": 12,\n      \"M1\": 8,\n      \"M2\": 8,\n      \"M3\": "
    "6,\n      \"N0\": 10,\n      \"N1\": 8,\n      \"N2\": 6,\n      \"N3\": 8\n     },\n     "
    "\"modes_a\": [\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M3\",\n   "
    "   \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      \"K3\",\n      "
    "\"K2\",\n      \"K1\",\n      \"K0\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\"\n     ],\n     \"modes_d\": [\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    "
    "\"route\": \"mn\"\n   },\n   \"de86aebf39d3e2ff25d0\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_"
    "barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_"
    "interleave=True,mn_chunk=128\",\n    \"problem\": {\n     \"bilinear\": false,\n     "
    "\"dtype\": \"fp16\",\n     \"extents\": {\n      \"K0\": 6,\n      \"K1\": 6,\n  ",
    "    \"K2\": 4,\n      \"K3\": 4,\n      \"M0\": 12,\n      \"M1\": 8,\n      \"M2\": 8,\n     "
    " \"M3\": 6,\n      \"N0\": 10,\n      \"N1\": 8,\n      \"N2\": 6,\n      \"N3\": 8\n     "
    "},\n     \"modes_a\": [\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      "
    "\"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\"\n     ],\n     \"modes_d\": [\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n   "
    " },\n    \"route\": \"mn\"\n   },\n   \"e22621813ed5a21095d8\": {\n    \"body\": "
    "\"t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,dtl_"
    "frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_"
    "after=2,ilv_merge=True,interwave_setprio=True,mfma_interleave=True\",\n    \"problem\": {\n   "
    "  \"bilinear\": true,\n     \"dtype\": \"bf16\",\n     \"extents\": {\n      \"K0\": 24,\n    "
    "  \"K1\": 24,\n      \"M0\": 48,\n      \"M1\": 32,\n      \"N0\": 40,\n      \"N1\": 36\n    "
    " },\n     \"modes_a\": [\n      \"K1\",\n      \"K0\",\n      \"M1\",\n      \"M0\"\n     "
    "],\n     \"modes_b\": [\n      \"K1\",\n      \"K0\",\n      \"N1\",\n      \"N0\"\n     ],\n "
    "    \"modes_d\": [\n      \"N1\",\n      \"N0\",\n      \"M1\",\n      \"M0\"\n     ]\n    "
    "},\n    \"route\": \"mn\"\n   },\n   \"eb9949e94e6710bf165a\": {\n    \"body\": "
    "\"t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
    "direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_"
    "stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,lds_swizzle=True,mfma_interleave="
    "True,waves_per_eu=2\",\n    \"problem\": {\n     \"bilinear\": true,\n     \"dtype\": "
    "\"fp16\",\n     \"extents\": {\n      \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": 3,\n      "
    "\"K3\": 3,\n      \"K4\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 6,\n      "
    "\"M3\": 4,\n      \"M4\": 4,\n      \"N0\": 8,\n      \"N1\": 6,\n      \"N2\": 4,\n      "
    "\"N3\": 6,\n      \"N4\": 4\n     },\n     \"modes_a\": [\n      \"K4\",\n      \"K3\",\n     "
    " \"K2\",\n      \"K1\",\n      \"K0\"",
    ",\n      \"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     "
    "\"modes_b\": [\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n   "
    "   \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n     "
    "\"modes_d\": [\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\",\n   "
    "   \"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    "
    "\"route\": \"mn\"\n   },\n   \"f476c18bd8d31703491c\": {\n    \"body\": "
    "\"t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_swizzle=True,chiplet_wgm=16,direct_to_"
    "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
    "epilogue=default,ilv_merge=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True\",\n    "
    "\"problem\": {\n     \"bilinear\": false,\n     \"dtype\": \"bf16\",\n     \"extents\": {\n   "
    "   \"K0\": 4,\n      \"K1\": 4,\n      \"K2\": 3,\n      \"K3\": 2,\n      \"K4\": 2,\n      "
    "\"K5\": 2,\n      \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 4,\n      \"M3\": 4,\n      "
    "\"M4\": 3,\n      \"M5\": 2,\n      \"N0\": 6,\n      \"N1\": 6,\n      \"N2\": 4,\n      "
    "\"N3\": 4,\n      \"N4\": 4,\n      \"N5\": 2\n     },\n     \"modes_a\": [\n      \"K5\",\n  "
    "    \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M5\",\n      "
    "\"M4\",\n      \"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     "
    "\"modes_b\": [\n      \"K5\",\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n   "
    "   \"K0\",\n      \"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      "
    "\"N0\"\n     ],\n     \"modes_d\": [\n      \"N5\",\n      \"N4\",\n      \"N3\",\n      "
    "\"N2\",\n      \"N1\",\n      \"N0\",\n      \"M5\",\n      \"M4\",\n      \"M3\",\n      "
    "\"M2\",\n      \"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   },\n   "
    "\"fb8844773bff9c3e7321\": {\n    \"body\": "
    "\"t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,"
    "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_"
    "barrier_after=2,ilv_merge=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True\",\n    "
    "\"problem\": {\n     \"bilinear\": true,\n     \"dtype\": \"bf16\",\n     \"extents\": {\n    "
    "  \"K0\": 4,\n",
    "      \"K1\": 4,\n      \"K2\": 3,\n      \"K3\": 2,\n      \"K4\": 2,\n      \"K5\": 2,\n    "
    "  \"M0\": 8,\n      \"M1\": 6,\n      \"M2\": 4,\n      \"M3\": 4,\n      \"M4\": 3,\n      "
    "\"M5\": 2,\n      \"N0\": 6,\n      \"N1\": 6,\n      \"N2\": 4,\n      \"N3\": 4,\n      "
    "\"N4\": 4,\n      \"N5\": 2\n     },\n     \"modes_a\": [\n      \"K5\",\n      \"K4\",\n     "
    " \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      \"M5\",\n      \"M4\",\n      "
    "\"M3\",\n      \"M2\",\n      \"M1\",\n      \"M0\"\n     ],\n     \"modes_b\": [\n      "
    "\"K5\",\n      \"K4\",\n      \"K3\",\n      \"K2\",\n      \"K1\",\n      \"K0\",\n      "
    "\"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      \"N1\",\n      \"N0\"\n     ],\n  "
    "   \"modes_d\": [\n      \"N5\",\n      \"N4\",\n      \"N3\",\n      \"N2\",\n      "
    "\"N1\",\n      \"N0\",\n      \"M5\",\n      \"M4\",\n      \"M3\",\n      \"M2\",\n      "
    "\"M1\",\n      \"M0\"\n     ]\n    },\n    \"route\": \"mn\"\n   }\n  }\n },\n \"schema\": "
    "1\n}\n",
};

std::string tuned_json()
{
    std::string s;
    for(const char* p : TUNED_JSON)
        s += p;
    return s;
}

[[noreturn]] void reject(const std::string& why)
{
    fprintf(stderr, "%s\n", why.c_str());
    exit(1);
}

TensorGettSpec tuned(const char* key)
{
    std::vector<TunedEntry> entries;
    std::string why;
    /* the table the engine carries (CMake-embedded) is the shipped one */
    if(tuned_json() != rocke_tensor_gett_tuned_json())
        reject("the engine's tuned GETT table differs from the shipped one");
    if(!tuned_entries(tuned_json(), "gfx950", &entries, &why))
        reject(why);
    for(const auto& e : entries)
    {
        if(e.key != key)
            continue;
        if(route_key(tuned_problem(e)) != key)
            reject("route key mismatch");
        TensorGettSpec s;
        if(!tuned_plan(e, "gfx950", &s, &why))
            reject(why);
        return s;
    }
    reject(std::string("no tuned entry ") + key);
}

void mn_body(TensorGettSpec* s, const char* cfg)
{
    std::string why;
    if(!parse_config(cfg, &s->mn_tile, &s->mn_trait, &why))
        reject(why);
    s->has_mn_tile = true;
    s->has_mn_trait = true;
}

ModeMap mm(std::vector<std::pair<std::string, int64_t>> v)
{
    return ModeMap(v.begin(), v.end());
}

bool make(int idx, TensorGettSpec* s)
{
    switch(idx)
    {
    case 0:
        *s = tuned("0f9482785a920350e45f");
        return true;
    case 1:
        *s = tuned("23f2bbb805ad34ff92ea");
        return true;
    case 2:
        *s = tuned("3d6bbc99c041dd301738");
        return true;
    case 3:
        *s = tuned("4907717213385e0ec490");
        return true;
    case 4:
        *s = tuned("52afee352c8cc1485d4f");
        return true;
    case 5:
        *s = tuned("5ef8541a5e73dbe91da4");
        return true;
    case 6:
        *s = tuned("6a5951db87850882819f");
        return true;
    case 7:
        *s = tuned("6ca39588a7df7b98780c");
        return true;
    case 8:
        *s = tuned("826372e76a2ca4dc4d44");
        return true;
    case 9:
        *s = tuned("a3f922690fb55c4f4d8d");
        return true;
    case 10:
        *s = tuned("acde940489d8c678e4cf");
        return true;
    case 11:
        *s = tuned("bc2dca5ed43e3e3fabd2");
        return true;
    case 12:
        *s = tuned("c040717c86e077567e5d");
        return true;
    case 13:
        *s = tuned("c3afaac231cdfe126504");
        return true;
    case 14:
        *s = tuned("ce0756fc3571cca809a1");
        return true;
    case 15:
        *s = tuned("d010a031b720ee7d5362");
        return true;
    case 16:
        *s = tuned("de86aebf39d3e2ff25d0");
        return true;
    case 17:
        *s = tuned("e22621813ed5a21095d8");
        return true;
    case 18:
        *s = tuned("eb9949e94e6710bf165a");
        return true;
    case 19:
        *s = tuned("f476c18bd8d31703491c");
        return true;
    case 20:
        *s = tuned("fb8844773bff9c3e7321");
        return true;
    case 21:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 22:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.pad = 0;
        return true;
    case 23:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_order = "ni";
        return true;
    case 24:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_pair = true;
        return true;
    case 25:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.lane_split = false;
        return true;
    case 26:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.acc_pin = 1;
        return true;
    case 27:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.acc_pin = 2;
        return true;
    case 28:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.frag_pin = true;
        return true;
    case 29:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 30:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        return true;
    case 31:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        return true;
    case 32:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->tile_k = 64;
        return true;
    case 33:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->dtl_stages = 2;
        return true;
    case 34:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->dtl_stages = 4;
        return true;
    case 35:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->persistent_ctas = 256;
        return true;
    case 36:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->chiplet_wgm = 4;
        s->chiplet_chunk_size = 32;
        return true;
    case 37:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"M3", "M2", "M1", "M0", "N3", "N2", "N1", "N0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 38:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->has_strides_a = true;
        s->strides_a = mm({{"K0", 9216},
                           {"K1", 55296},
                           {"K2", 331776},
                           {"K3", 1327104},
                           {"M0", 1},
                           {"M1", 24},
                           {"M2", 192},
                           {"M3", 1536}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 39:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->has_strides_d = true;
        s->strides_d = mm({{"M0", 1},
                           {"M1", 36},
                           {"M2", 288},
                           {"M3", 2304},
                           {"N0", 13824},
                           {"N1", 138240},
                           {"N2", 1105920},
                           {"N3", 6635520}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 40:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 41:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 5},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        return true;
    case 42:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        return true;
    case 43:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 44:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.pad = 0;
        return true;
    case 45:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_order = "ni";
        return true;
    case 46:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_pair = true;
        return true;
    case 47:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.lane_split = false;
        return true;
    case 48:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.acc_pin = 1;
        return true;
    case 49:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.acc_pin = 2;
        return true;
    case 50:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.frag_pin = true;
        return true;
    case 51:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        return true;
    case 52:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        return true;
    case 53:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        return true;
    case 54:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->tile_k = 64;
        return true;
    case 55:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->dtl_stages = 2;
        return true;
    case 56:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->dtl_stages = 4;
        return true;
    case 57:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->persistent_ctas = 256;
        return true;
    case 58:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->chiplet_wgm = 4;
        s->chiplet_chunk_size = 32;
        return true;
    case 59:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"M1", "M0", "N1", "N0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 60:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->has_strides_a = true;
        s->strides_a = mm({{"K0", 3072}, {"K1", 73728}, {"M0", 1}, {"M1", 96}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 61:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->has_strides_d = true;
        s->strides_d = mm({{"M0", 1}, {"M1", 144}, {"N0", 4608}, {"N1", 184320}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 62:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 50}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 63:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 25}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 64:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        return true;
    case 65:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 66:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.pad = 0;
        return true;
    case 67:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_order = "ni";
        return true;
    case 68:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_pair = true;
        return true;
    case 69:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.lane_split = false;
        return true;
    case 70:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.acc_pin = 1;
        return true;
    case 71:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.acc_pin = 2;
        return true;
    case 72:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.frag_pin = true;
        return true;
    case 73:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        return true;
    case 74:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        return true;
    case 75:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        return true;
    case 76:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->tile_k = 64;
        return true;
    case 77:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->dtl_stages = 2;
        return true;
    case 78:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->dtl_stages = 4;
        return true;
    case 79:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->persistent_ctas = 256;
        return true;
    case 80:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->chiplet_wgm = 4;
        s->chiplet_chunk_size = 32;
        return true;
    case 81:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"M3", "M2", "M1", "M0", "N3", "N2", "N1", "N0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 82:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->has_strides_a = true;
        s->strides_a = mm({{"K0", 9216},
                           {"K1", 55296},
                           {"K2", 331776},
                           {"K3", 1327104},
                           {"M0", 1},
                           {"M1", 24},
                           {"M2", 192},
                           {"M3", 1536}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 83:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->has_strides_d = true;
        s->strides_d = mm({{"M0", 1},
                           {"M1", 36},
                           {"M2", 288},
                           {"M3", 2304},
                           {"N0", 13824},
                           {"N1", 138240},
                           {"N2", 1105920},
                           {"N3", 6635520}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 84:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 85:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 5},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        return true;
    case 86:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        return true;
    case 87:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 88:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 89:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 90:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.store_nt = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 91:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_order = "ni";
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 92:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.pad = 0;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 93:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 94:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 95:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 96:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.store_nt = true;
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 97:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_order = "ni";
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 98:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.pad = 0;
        s->tile_k = 64;
        mn_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,mn_chunk=128,waves_per_eu=1");
        return true;
    case 99:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 100:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 101:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 102:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 103:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_order = "ni";
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 104:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.pad = 0;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 105:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 106:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 107:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 108:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 109:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_order = "ni";
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 110:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.pad = 0;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 111:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 112:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 113:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 114:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.store_nt = true;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 115:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_order = "ni";
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 116:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.pad = 0;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_lds="
            "True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages="
            "3,epilogue=default,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 117:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 118:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 119:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 120:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.store_nt = true;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 121:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_order = "ni";
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 122:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.pad = 0;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=4,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True");
        return true;
    case 123:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(
            s,
            "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
            "chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,"
            "dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
            "interwave_setprio=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=64");
        return true;
    case 124:
        *s = TensorGettSpec::make(
            {"K5", "K4", "K3", "K2", "K1", "K0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {"K5", "K4", "K3", "K2", "K1", "K0", "N5", "N4", "N3", "N2", "N1", "N0"},
            {"N5", "N4", "N3", "N2", "N1", "N0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {{"K0", 4},
             {"K1", 4},
             {"K2", 3},
             {"K3", 2},
             {"K4", 2},
             {"K5", 2},
             {"M0", 8},
             {"M1", 6},
             {"M2", 4},
             {"M3", 4},
             {"M4", 3},
             {"M5", 2},
             {"N0", 6},
             {"N1", 6},
             {"N2", 4},
             {"N3", 4},
             {"N4", 4},
             {"N5", 2}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(
            s,
            "t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
            "chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,"
            "dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
            "interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    case 125:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 64;
        mn_body(
            s,
            "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,"
            "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,"
            "ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_interleave=True");
        return true;
    case 126:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 16},
                                   {"K1", 12},
                                   {"K2", 8},
                                   {"M0", 32},
                                   {"M1", 20},
                                   {"M2", 12},
                                   {"N0", 24},
                                   {"N1", 20},
                                   {"N2", 14}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x256x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=4,chiplet_swizzle="
                "True,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
                "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
                "interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    case 127:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 0;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|chiplet_chunk_size=32,chiplet_swizzle=True,direct_to_"
                "lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,"
                "dtl_stages=3,epilogue=default,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,mfma_interleave=True,waves_per_eu=1");
        return true;
    case 128:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,chiplet_wgm=4,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_swizzle=True,mfma_interleave=True,mn_chunk=128");
        return true;
    case 129:
        *s = TensorGettSpec::make(
            {"K5", "K4", "K3", "K2", "K1", "K0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {"K5", "K4", "K3", "K2", "K1", "K0", "N5", "N4", "N3", "N2", "N1", "N0"},
            {"N5", "N4", "N3", "N2", "N1", "N0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {{"K0", 4},
             {"K1", 4},
             {"K2", 3},
             {"K3", 2},
             {"K4", 2},
             {"K5", 2},
             {"M0", 8},
             {"M1", 6},
             {"M2", 4},
             {"M3", 4},
             {"M4", 3},
             {"M5", 2},
             {"N0", 6},
             {"N1", 6},
             {"N2", 4},
             {"N3", 4},
             {"N4", 4},
             {"N5", 2}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=16,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,lds_swizzle=True,lds_swizzle_"
                "r=1,mfma_interleave=True");
        return true;
    case 130:
        *s = TensorGettSpec::make({"K2", "K1", "K0", "M2", "M1", "M0"},
                                  {"K2", "K1", "K0", "N2", "N1", "N0"},
                                  {"N2", "N1", "N0", "M2", "M1", "M0"},
                                  {{"K0", 12},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"M0", 24},
                                   {"M1", 16},
                                   {"M2", 8},
                                   {"N0", 20},
                                   {"N1", 12},
                                   {"N2", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t128x128x32_w2x4x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
                "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
                "interwave_setprio=True,lds_swizzle=True,mfma_interleave=True,waves_per_eu=1");
        return true;
    case 131:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 64;
        mn_body(
            s,
            "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,"
            "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,"
            "ilv_merge=True,interwave_setprio=True,lds_swizzle=True,mfma_interleave=True");
        return true;
    case 132:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 133:
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=4,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_"
                "swizzle=True,lds_swizzle_r=1,mfma_interleave=True,mn_chunk=128");
        return true;
    case 134:
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->tile_k = 64;
        mn_body(
            s,
            "t128x128x64_w2x4x1_a16x16x32|chiplet_chunk_size=32,direct_to_lds=True,dtl_cache_b=0,"
            "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,"
            "ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,mfma_interleave=True");
        return true;
    case 135:
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 4},
                                   {"K1", 4},
                                   {"K2", 3},
                                   {"K3", 3},
                                   {"K4", 2},
                                   {"M0", 8},
                                   {"M1", 6},
                                   {"M2", 6},
                                   {"M3", 4},
                                   {"M4", 4},
                                   {"N0", 8},
                                   {"N1", 6},
                                   {"N2", 4},
                                   {"N3", 6},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        mn_body(s,
                "t256x128x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
                "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
                "lds_swizzle=True,mfma_interleave=True,waves_per_eu=2");
        return true;
    case 136:
        *s = TensorGettSpec::make(
            {"K5", "K4", "K3", "K2", "K1", "K0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {"K5", "K4", "K3", "K2", "K1", "K0", "N5", "N4", "N3", "N2", "N1", "N0"},
            {"N5", "N4", "N3", "N2", "N1", "N0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {{"K0", 4},
             {"K1", 4},
             {"K2", 3},
             {"K3", 2},
             {"K4", 2},
             {"K5", 2},
             {"M0", 8},
             {"M1", 6},
             {"M2", 4},
             {"M3", 4},
             {"M4", 3},
             {"M5", 2},
             {"N0", 6},
             {"N1", 6},
             {"N2", 4},
             {"N3", 4},
             {"N4", 4},
             {"N5", 2}});
        s->dtype = "bf16";
        s->knobs.bilinear = false;
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        mn_body(s,
                "t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_swizzle=True,chiplet_wgm=16,"
                "direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
                "prefetch=True,dtl_stages=3,epilogue=default,ilv_merge=True,lds_swizzle=True,lds_"
                "swizzle_r=1,mfma_interleave=True");
        return true;
    case 137:
        *s = TensorGettSpec::make(
            {"K5", "K4", "K3", "K2", "K1", "K0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {"K5", "K4", "K3", "K2", "K1", "K0", "N5", "N4", "N3", "N2", "N1", "N0"},
            {"N5", "N4", "N3", "N2", "N1", "N0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {{"K0", 4},
             {"K1", 4},
             {"K2", 3},
             {"K3", 2},
             {"K4", 2},
             {"K5", 2},
             {"M0", 8},
             {"M1", 6},
             {"M2", 4},
             {"M3", 4},
             {"M4", 3},
             {"M5", 2},
             {"N0", 6},
             {"N1", 6},
             {"N2", 4},
             {"N3", 4},
             {"N4", 4},
             {"N5", 2}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.epi_v16 = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        mn_body(s,
                "t128x192x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_wgm=16,direct_to_lds=True,dtl_"
                "cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,"
                "epilogue=default,ilv_barrier_after=2,ilv_merge=True,lds_swizzle=True,lds_swizzle_"
                "r=1,mfma_interleave=True");
        return true;
    default:
        return false;
    }
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    TensorGettSpec s;
    if(!make(idx, &s))
        return NULL;
    *known = 1;
    std::string why;
    if(!is_valid_spec(s, "gfx950", &why))
        reject("invalid spec: " + why);
    return build_tensor_gett(b, s, "gfx950");
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: tensor_gett_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch = "gfx950";
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    int known = 0;
    rocke_kernel_def_t* kernel = build(&b, idx, &known);
    if(!known)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    if(kernel == NULL)
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rc = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
