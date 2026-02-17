// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "Bfloat16Dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

/// If HIP_PLUGIN_USE_DOUBLE_ACCUM is defined as 1 when "float_types.h" is included,
/// then all the ACCUM macros (the represent operations and types) will use FP64
/// instead of FP32. In other words, the computations will be
/// performed using the native datatype even if ACCUM macros are used.
/// This functionality is indended mostly for debugging.
#ifdef HIP_PLUGIN_USE_DOUBLE_ACCUM
#if !(HIP_PLUGIN_USE_DOUBLE_ACCUM == 0 || HIP_PLUGIN_USE_DOUBLE_ACCUM == 1)
#error "Invalid value of HIP_PLUGIN_USE_DOUBLE_ACCUM"
#endif
#else
#define HIP_PLUGIN_USE_DOUBLE_ACCUM 0
#endif

#if HIP_PLUGIN_USE_DOUBLE_ACCUM
#define FLOAT_ACCUM double
#define MAX_VAL_ACCUM DBL_MAX
#else // HIP_PLUGIN_USE_DOUBLE_ACCUM
#define FLOAT_ACCUM float
#ifndef FLT_MAX
#define MAX_VAL_ACCUM 3.402823466e+38F
#else
#define MAX_VAL_ACCUM FLT_MAX
#endif
#endif // HIP_PLUGIN_USE_DOUBLE_ACCUM

#if(HIP_PLUGIN_USE_FP16 == 1) || (HIP_PLUGIN_USE_FPMIX == 1)
#define FLOAT _Float16
#define SIZEOF_FLOAT 2
// Max value for the main datatype
#ifndef HALF_MAX
#define MAX_VAL 65504
#else
#define MAX_VAL HALF_MAX
#endif
#endif // HIP_PLUGIN_USE_FP16 || HIP_PLUGIN_USE_FPMIX

#if HIP_PLUGIN_USE_FP32 == 1
#define FLOAT float
#define SIZEOF_FLOAT 4
// Max value for the main datatype
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F
#else
#define MAX_VAL FLT_MAX
#endif
#endif // HIP_PLUGIN_USE_FP32

#if(HIP_PLUGIN_USE_BFP16 == 1) || (HIP_PLUGIN_USE_BFPMIX == 1)
#define FLOAT ushort
#define SIZEOF_FLOAT 2
// Max value for the main datatype
#define MAX_VAL 0x7F7F
#endif // HIP_PLUGIN_USE_BFP16 || HIP_PLUGIN_USE_BFPMIX

#if(HIP_PLUGIN_USE_FP16 == 1) || (HIP_PLUGIN_USE_FPMIX == 1)
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
// These two are required to uniformly initialize
// variables with non-zero literal constants of FP32 type
// regardless of the actual type of the variable.
// This is especially complicated for BF16, because
// the compiler lacks the support of BF16 literals.
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#define CVT_FLOAT2FP32(x) (CVT_FLOAT2ACCUM(x))
#define CVT_ACCUM2FP32(x) (x)
#endif // HIP_PLUGIN_USE_FP16 || HIP_PLUGIN_USE_FPMIX

#if HIP_PLUGIN_USE_FP32 == 1
/// \todo Basically, conversions from float to accum and vice versa
/// should be removed because FLOAT_ACCUM and FLOAT are identical.
/// However this may lead to problems if these macros are used in
/// inappropriate contexts (e.g. with integral types), so this
/// refactoring should be considered as nontrivial and requires
/// a separate PR. Let's keep this historical stuff for now.
/// --atamazov 30.08.2023
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#define CVT_FLOAT2FP32(x) (CVT_FLOAT2ACCUM(x))
#define CVT_ACCUM2FP32(x) (x)
#endif // HIP_PLUGIN_USE_FP32

#if(HIP_PLUGIN_USE_BFP16 == 1) || (HIP_PLUGIN_USE_BFPMIX == 1)
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#define CVT_FLOAT2FP32(x) (CVT_FLOAT2ACCUM(x))
#define CVT_ACCUM2FP32(x) (x)
#endif

/// If HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM is defined as 1 when "float_types.h" is included,
/// then all the ACCUM macros (the represent operations and types) will use the native
/// datatype (BF16 or FP16) instead of FP32. In other words, the computations will be
/// performed using the native datatype even if ACCUM macros are used. This allows for
/// building both mixed-precision and "pure" kernels from the single source.
/// Note: This macro has higher priority than HIP_PLUGIN_USE_DOUBLE_ACCUM.
#ifdef HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM
#if !(HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM == 0 || HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM == 1)
#error "Invalid value of HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM"
#endif
#else
#define HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM 0
#endif

#if HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM

#undef FLOAT_ACCUM
#define FLOAT_ACCUM FLOAT

#undef MAX_VAL_ACCUM
#define MAX_VAL_ACCUM MAX_VAL
#undef CVT_FLOAT2ACCUM
#define CVT_FLOAT2ACCUM(x) (x)
#undef CVT_ACCUM2FLOAT
#define CVT_ACCUM2FLOAT(x) (x)
#undef CVT_FP32_2ACCUM
#define CVT_FP32_2ACCUM(x) (CVT_FP32_2FLOAT(x))
#undef CVT_ACCUM2FP32
#define CVT_ACCUM2FP32(x) (CVT_FLOAT2FP32(x))

#undef CVT_INTEGRAL2ACCUM
#if(HIP_PLUGIN_USE_BFP16 == 1) || (HIP_PLUGIN_USE_BFPMIX == 1)
// No direct conversion from integral types to BF16 is available.
// WARNING: Precision loss when integral type is wider than 16 bits.
#define CVT_INTEGRAL2ACCUM(x) (float_to_bfloat16(static_cast<float>(x)))
#else
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT>(x))
#endif

#endif // HIP_PLUGIN_USE_NATIVE_DATATYPE_ACCUM
