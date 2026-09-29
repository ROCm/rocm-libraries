// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Analysis-only configuration for the hip_mlops kernels.
//
// The kernels in ../ are compiled by hipRTC at runtime, with every HIP_PLUGIN_* macro
// supplied as a -D option built from the problem being solved (see
// compilation/KernelCompileOptions.hpp and the plans under ../../plans). Nothing defines
// them at build time, so the kernels do not even parse outside hipRTC, which is why
// clang-tidy could not be pointed at them directly.
//
// The values below are a representative instantiation, NOT the values any plan produces.
// They exist so that the kernels parse under clang-tidy; they are never compiled into the
// provider. Where a plan has a documented default (see
// plans/batchnorm/BatchnormKernelCompileOptions.hpp) that default is used, so the
// configuration stays recognisable to someone reading the plan code.
//
// Consequence to keep in mind: clang-tidy sees exactly one configuration of each kernel.
// Code behind a different HIP_PLUGIN_BN_VARIANT, or specialised for a half/bfloat16
// instantiation, is parsed but not necessarily instantiated, so it is checked less
// thoroughly than the configuration named here. Adding a second prelude and a second
// tidy target is the way to widen that coverage.
//
// A kernel that starts using a new HIP_PLUGIN_* macro must add it here, otherwise it no
// longer parses and the `tidy` target fails.

#pragma once

// The kernels rely on the HIP device runtime (blockIdx, __syncthreads, __launch_bounds__,
// the __half types). hipRTC injects it implicitly; under clang-tidy it has to be
// included, and this header is force-included ahead of every kernel.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// This file is an analysis fixture, not code the provider ships, and the macros below
// stand in for -D options: they cannot become the enums modernize-macro-to-enum asks for
// without ceasing to do their job. Checking the fixture would only ever report on itself,
// so it is excluded; the kernels it enables are still fully checked.
// NOLINTBEGIN

// --- provider-wide options (compilation/KernelCompileOptions.hpp) ---------------------
#define HIP_PLUGIN_LAYOUT_NHWC 0
#define HIP_PLUGIN_USE_AMDGCN 0
#define HIP_PLUGIN_USE_FP16 0
#define HIP_PLUGIN_USE_FP32 1
#define HIP_PLUGIN_USE_FPMIX 0
#define HIP_PLUGIN_USE_BFPMIX 0
#define HIP_PLUGIN_GFX103X 0
#define HIP_PLUGIN_GFX110X 0
#define HIP_PLUGIN_GFX115X 0
#define HIP_PLUGIN_GFX120X 0

// --- batchnorm (plans/batchnorm/BatchnormKernelCompileOptions.hpp) --------------------
#define HIP_PLUGIN_BN_INPUT_TYPE float
#define HIP_PLUGIN_BN_OUTPUT_TYPE float
#define HIP_PLUGIN_BN_SCALE_TYPE float
#define HIP_PLUGIN_BN_MEAN_VAR_TYPE float
#define HIP_PLUGIN_BN_NRN_OP_ID 0
// Both stats paths are enabled so the code they guard is analysed rather than
// preprocessed away.
#define HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE 1
#define HIP_PLUGIN_BN_RUNNING_RESULT 1
#define HIP_PLUGIN_BN_USESAVED 1
#define HIP_PLUGIN_BN_NODPP 0
#define HIP_PLUGIN_BN_VARIANT 1
#define HIP_PLUGIN_BN_STASH_METHOD 0
#define HIP_PLUGIN_BN_VEC_SIZE 1
#define HIP_PLUGIN_BN_GRP0 1
#define HIP_PLUGIN_BN_GRP1 1024
#define HIP_PLUGIN_BN_GRP2 1
#define HIP_PLUGIN_BN_GRP0_FINAL 1
#define HIP_PLUGIN_BN_GRP1_FINAL 1024
#define HIP_PLUGIN_BN_GRP2_FINAL 1
#define HIP_PLUGIN_BN_NGRPS 1
#define HIP_PLUGIN_BN_NGRPS2 1
#define HIP_PLUGIN_BN_LDS_SIZE 256
#define HIP_PLUGIN_BN_LDSGCN_SIZE 16
#define HIP_PLUGIN_BN_LOOP_UNROLL_MAXN 768
#define HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW 2500
#define HIP_PLUGIN_BN_N 1
#define HIP_PLUGIN_BN_C 1
#define HIP_PLUGIN_BN_HW 1
#define HIP_PLUGIN_BN_NHW 1
#define HIP_PLUGIN_BN_CHW 1
#define HIP_PLUGIN_BN_NCHW 1
#define HIP_PLUGIN_BN_N_ELEMENTS HIP_PLUGIN_BN_N

// --- RMSnorm (plans/RMSnorm) ----------------------------------------------------------
#define HIP_PLUGIN_RMSNORM_INPUT_TYPE float
#define HIP_PLUGIN_RMSNORM_OUTPUT_TYPE float
#define HIP_PLUGIN_RMSNORM_SCALE_TYPE float
// The kernels static_assert that the compute type is float.
#define HIP_PLUGIN_RMSNORM_COMPUTE_TYPE float
#define HIP_PLUGIN_RMSNORM_X_TYPE float
#define HIP_PLUGIN_RMSNORM_Y_TYPE float
#define HIP_PLUGIN_RMSNORM_DX_TYPE float
#define HIP_PLUGIN_RMSNORM_DY_TYPE float
#define HIP_PLUGIN_RMSNORM_NRN_OP_ID 0
#define HIP_PLUGIN_RMSNORM_LOCAL_SIZE 256
#define HIP_PLUGIN_RMSNORM_INNER_SIZE 256
#define HIP_PLUGIN_RMSNORM_OUTER_SIZE 1
#define HIP_PLUGIN_RMSNORM_STRIDE 1

// --- layernorm (plans/layernorm) ------------------------------------------------------
#define HIP_PLUGIN_LAYERNORM_INPUT_TYPE float
#define HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE float
#define HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE float
#define HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE float
#define HIP_PLUGIN_LAYERNORM_LOCAL_SIZE 256
#define HIP_PLUGIN_LAYERNORM_INNER_SIZE 256
#define HIP_PLUGIN_LAYERNORM_OUTER_SIZE 1
#define HIP_PLUGIN_LAYERNORM_STRIDE 1
#define HIP_PLUGIN_LAYERNORM_PARALLEL_SIZE 1

// --- resample (plans/resample) --------------------------------------------------------
// A 2D 8x8 -> 4x4 pooling with a 2x2 window and packed NCHW strides.
#define HIP_PLUGIN_RESAMPLE_INPUT_TYPE float
#define HIP_PLUGIN_RESAMPLE_OUTPUT_TYPE float
#define HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE float
#define HIP_PLUGIN_RESAMPLE_INDEX_TYPE int
#define HIP_PLUGIN_RESAMPLE_DX_TYPE float
#define HIP_PLUGIN_RESAMPLE_DY_TYPE float
#define HIP_PLUGIN_RESAMPLE_MODE 0
#define HIP_PLUGIN_RESAMPLE_PADDING_MODE 0
#define HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS 2
#define HIP_PLUGIN_RESAMPLE_HAS_INDEX 0
#define HIP_PLUGIN_RESAMPLE_GENERATE_INDEX 0
#define HIP_PLUGIN_RESAMPLE_N 1
#define HIP_PLUGIN_RESAMPLE_C 1
#define HIP_PLUGIN_RESAMPLE_X_D 1
#define HIP_PLUGIN_RESAMPLE_X_H 8
#define HIP_PLUGIN_RESAMPLE_X_W 8
#define HIP_PLUGIN_RESAMPLE_Y_D 1
#define HIP_PLUGIN_RESAMPLE_Y_H 4
#define HIP_PLUGIN_RESAMPLE_Y_W 4
#define HIP_PLUGIN_RESAMPLE_DX_D 1
#define HIP_PLUGIN_RESAMPLE_DX_H 8
#define HIP_PLUGIN_RESAMPLE_DX_W 8
#define HIP_PLUGIN_RESAMPLE_DY_D 1
#define HIP_PLUGIN_RESAMPLE_DY_H 4
#define HIP_PLUGIN_RESAMPLE_DY_W 4
#define HIP_PLUGIN_RESAMPLE_X_STRIDE_N 64
#define HIP_PLUGIN_RESAMPLE_X_STRIDE_C 64
#define HIP_PLUGIN_RESAMPLE_X_STRIDE_D 64
#define HIP_PLUGIN_RESAMPLE_X_STRIDE_H 8
#define HIP_PLUGIN_RESAMPLE_X_STRIDE_W 1
#define HIP_PLUGIN_RESAMPLE_Y_STRIDE_N 16
#define HIP_PLUGIN_RESAMPLE_Y_STRIDE_C 16
#define HIP_PLUGIN_RESAMPLE_Y_STRIDE_D 16
#define HIP_PLUGIN_RESAMPLE_Y_STRIDE_H 4
#define HIP_PLUGIN_RESAMPLE_Y_STRIDE_W 1
#define HIP_PLUGIN_RESAMPLE_DX_STRIDE_N 64
#define HIP_PLUGIN_RESAMPLE_DX_STRIDE_C 64
#define HIP_PLUGIN_RESAMPLE_DX_STRIDE_D 64
#define HIP_PLUGIN_RESAMPLE_DX_STRIDE_H 8
#define HIP_PLUGIN_RESAMPLE_DX_STRIDE_W 1
#define HIP_PLUGIN_RESAMPLE_DY_STRIDE_N 16
#define HIP_PLUGIN_RESAMPLE_DY_STRIDE_C 16
#define HIP_PLUGIN_RESAMPLE_DY_STRIDE_D 16
#define HIP_PLUGIN_RESAMPLE_DY_STRIDE_H 4
#define HIP_PLUGIN_RESAMPLE_DY_STRIDE_W 1
#define HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N 16
#define HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C 16
#define HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D 16
#define HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H 4
#define HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W 1
#define HIP_PLUGIN_RESAMPLE_WINDOW_D 1
#define HIP_PLUGIN_RESAMPLE_WINDOW_H 2
#define HIP_PLUGIN_RESAMPLE_WINDOW_W 2
#define HIP_PLUGIN_RESAMPLE_STRIDE_D 1
#define HIP_PLUGIN_RESAMPLE_STRIDE_H 2
#define HIP_PLUGIN_RESAMPLE_STRIDE_W 2
#define HIP_PLUGIN_RESAMPLE_PRE_PAD_D 0
#define HIP_PLUGIN_RESAMPLE_PRE_PAD_H 0
#define HIP_PLUGIN_RESAMPLE_PRE_PAD_W 0
#define HIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT 16
#define HIP_PLUGIN_RESAMPLE_DX_ELEMENT_COUNT 64

// NOLINTEND
