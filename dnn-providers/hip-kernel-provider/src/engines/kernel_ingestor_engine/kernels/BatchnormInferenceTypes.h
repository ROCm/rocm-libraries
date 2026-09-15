// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
//
// Type and geometry bindings for BatchnormInference.cpp.
//
// Every macro below is supplied by the compile command (`-D`), never defaulted. An
// unbound token would otherwise paste into something that happens to exist and the
// kernel would fail only in the numbers, which is diagnosed nowhere.
//
// Modelled on dnn-providers/hip-kernel-provider/config/pointwise_hiprtc_dropin/
// pointwise_dropin_sources/PointwiseDropinTypes.h.

#pragma once

#ifndef HIPDNN_BN_IO_DTYPE
#error "HIPDNN_BN_IO_DTYPE must be supplied by the compile command (x/y element type tag)"
#endif

#ifndef HIPDNN_BN_PARAM_DTYPE
#error \
    "HIPDNN_BN_PARAM_DTYPE must be supplied by the compile command (mean/inv_variance/scale/bias element type tag)"
#endif

#ifndef HIPDNN_BN_BLOCK
#error "HIPDNN_BN_BLOCK must be supplied by the compile command (threads per block)"
#endif

// Tag -> device type. hipRTC has no <hip/hip_fp16.h> (VectorTypes.hpp guards it behind
// __HIPCC_RTC__), so the 16-bit types are spelled as the compiler builtins.
#define HIPDNN_BN_T_FLOAT float
#define HIPDNN_BN_T_HALF _Float16
#define HIPDNN_BN_T_BFLOAT16 __bf16

// Two levels, deliberately. `##` suppresses expansion of its operands, so a one-level
// paste yields HIPDNN_BN_T_HIPDNN_BN_IO_DTYPE, which does not exist. HIPDNN_BN_CAT
// expands its argument first and HIPDNN_BN_PASTE does the paste.
#define HIPDNN_BN_PASTE(tag) HIPDNN_BN_T_##tag
#define HIPDNN_BN_CAT(tag) HIPDNN_BN_PASTE(tag)

using BnIoElement = HIPDNN_BN_CAT(HIPDNN_BN_IO_DTYPE);
using BnParamElement = HIPDNN_BN_CAT(HIPDNN_BN_PARAM_DTYPE);

// Compute always happens in float, independent of storage width. Precedent:
// kernel_ingestor_engine/kernels/ConvFwd.cpp:48 and
// hip_mlops_engine/kernels/batchnorm/BatchNormFwdInferSpatial.cpp:11.
using BnCompute = float;
