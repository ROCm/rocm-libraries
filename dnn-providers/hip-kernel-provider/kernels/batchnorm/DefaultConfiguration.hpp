// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ---------- general configs ----------

// normalization of input macros (give default value to undefined macros)
// TODO: we can consider to remove all of these default values, force the
// user to define all of them.
#ifndef HIP_PLUGIN_LAYOUT_NHWC
#define HIP_PLUGIN_LAYOUT_NHWC 0 // Default value
#endif

#ifndef HIP_PLUGIN_SAVE_MEAN_VARIANCE
#define HIP_PLUGIN_SAVE_MEAN_VARIANCE 0
#endif

#ifndef HIP_PLUGIN_RUNNING_RESULT
#define HIP_PLUGIN_RUNNING_RESULT 0
#endif

#ifndef HIP_PLUGIN_USE_FP32
#define HIP_PLUGIN_USE_FP32 0
#endif

#ifndef HIP_PLUGIN_USE_FP16
#define HIP_PLUGIN_USE_FP16 0
#endif

#ifndef HIP_PLUGIN_USE_FPMIX
#define HIP_PLUGIN_USE_FPMIX 0
#endif

#ifndef HIP_PLUGIN_USE_BFPMIX
#define HIP_PLUGIN_USE_BFPMIX 0
#endif

#ifndef HIP_PLUGIN_BN_NODPP
#define HIP_PLUGIN_BN_NODPP 0
#endif

#ifndef HIP_PLUGIN_USE_AMDGCN
#define HIP_PLUGIN_USE_AMDGCN 1
#endif

#ifndef HIP_PLUGIN_NRN_OP_ID
#define HIP_PLUGIN_NRN_OP_ID 0
#endif

// ---------- batchnorm configs ----------

#ifndef HALF_MAX
#define HALF_MAX 65504
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#ifndef HIP_PLUGIN_BN_GRP0
#define HIP_PLUGIN_BN_GRP0 1
#endif

#ifndef HIP_PLUGIN_BN_GRP1
#define HIP_PLUGIN_BN_GRP1 1
#endif

#ifndef HIP_PLUGIN_BN_GRP2
#define HIP_PLUGIN_BN_GRP2 1
#endif

#ifndef HIP_PLUGIN_BN_GFX103X
#define HIP_PLUGIN_BN_GFX103X 0
#endif

#ifndef HIP_PLUGIN_BN_GFX110X
#define HIP_PLUGIN_BN_GFX110X 0
#endif

#ifndef HIP_PLUGIN_BN_GFX120X
#define HIP_PLUGIN_BN_GFX120X 0
#endif

#ifndef HIP_PLUGIN_BN_GFX115X
#define HIP_PLUGIN_BN_GFX115X 0
#endif

#ifndef HIP_PLUGIN_BN_VARIANT
#define HIP_PLUGIN_BN_VARIANT 255
#endif

#ifndef HIP_PLUGIN_BN_NCHW
#define HIP_PLUGIN_BN_NCHW 1
#endif

#ifndef HIP_PLUGIN_BN_MAXN
#define HIP_PLUGIN_BN_MAXN 65
#endif

#ifndef HIP_PLUGIN_BN_VECTORIZE
#define HIP_PLUGIN_BN_VECTORIZE 0
#endif

#ifndef HIP_PLUGIN_BN_VEC_SIZE
#define HIP_PLUGIN_BN_VEC_SIZE 1
#endif

#ifndef HIP_PLUGIN_BN_STASH_METHOD
#define HIP_PLUGIN_BN_STASH_METHOD 0
#endif

#ifndef HIP_PLUGIN_BN_LOOP_UNROLL_MAXN
#define HIP_PLUGIN_BN_LOOP_UNROLL_MAXN 768
#endif

#ifndef HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW
#define HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW 2500
#endif

#ifndef HIP_PLUGIN_BN_LDSGCN_SIZE
#define HIP_PLUGIN_BN_LDSGCN_SIZE 16
#endif

#ifndef HIP_PLUGIN_BN_LDS_SIZE
#define HIP_PLUGIN_BN_LDS_SIZE 256
#endif

#ifndef HIP_PLUGIN_BN_C
#define HIP_PLUGIN_BN_C 1
#endif

#ifndef HIP_PLUGIN_BN_N
#define HIP_PLUGIN_BN_N 1
#endif

#ifndef HIP_PLUGIN_BN_N_ELEMENTS
#define HIP_PLUGIN_BN_N_ELEMENTS HIP_PLUGIN_BN_N
// This is determined as such in the heuristics that select the kernels
// (engines/plans/BatchnormCommon.hpp: defaultConfigSpatialMultiple)
#endif

#ifndef HIP_PLUGIN_BN_NHW
#define HIP_PLUGIN_BN_NHW 1
#endif

#ifndef HIP_PLUGIN_BN_INHW
#define HIP_PLUGIN_BN_INHW 1
#endif

#ifndef HIP_PLUGIN_BN_CHW
#define HIP_PLUGIN_BN_CHW 1
#endif

#ifndef HIP_PLUGIN_BN_HW
#define HIP_PLUGIN_BN_HW 1
#endif
