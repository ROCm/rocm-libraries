/*
 *  Copyright 2021 NVIDIA Corporation
 *  Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/device_system.h>
#include <thrust/version.h>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#  include <thrust/rocthrust_version.hpp>
#endif

/**
 * \file namespace.h
 * \brief Utilities that allow `thrust::` to be placed inside an
 * application-specific namespace.
 */

/**
 * \def THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` and `cub::` namespaces.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#  define THRUST_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def THRUST_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` namespace.
 * If THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_WRAPPED_NAMESPACE
#  define THRUST_NS_PREFIX             \
    namespace THRUST_WRAPPED_NAMESPACE \
    {

#  define THRUST_NS_POSTFIX }

#  define THRUST_NS_QUALIFIER ::THRUST_WRAPPED_NAMESPACE::thrust
#endif

/**
 * \def THRUST_NS_PREFIX
 * This macro is inserted prior to all `namespace thrust { ... }` blocks. It is
 * derived from THRUST_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case THRUST_NS_PREFIX,
 * THRUST_NS_POSTFIX, and THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_PREFIX
#  define THRUST_NS_PREFIX
#endif

/**
 * \def THRUST_NS_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace thrust { ... }` block. It is defined appropriately when
 * THRUST_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_POSTFIX
#  define THRUST_NS_POSTFIX
#endif

/**
 * \def THRUST_NS_QUALIFIER
 * This macro is used to qualify members of thrust:: when accessing them from
 * outside of their namespace. By default, this is just `::thrust`, and will be
 * set appropriately when THRUST_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_QUALIFIER
#  define THRUST_NS_QUALIFIER ::thrust
#endif

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#  define THRUST_DETAIL_PP_EXPAND(...) __VA_ARGS__
#  define THRUST_PP_EVAL(_M, ...)      _M(__VA_ARGS__)

#  define THRUST_PP_CAT(_X, _Y) THRUST_PP_CAT_IMPL0(_X, _Y)

#  if defined(_MSC_VER) && (defined(__EDG__) || defined(__EDG_VERSION__)) \
    && (defined(__INTELLISENSE__) || __EDG_VERSION__ >= 308)
#    define THRUST_PP_CAT_IMPL0(_X, _Y) THRUST_PP_CAT_IMPL1(~, _X##_Y)
#    define THRUST_PP_CAT_IMPL1(_X, _Y) _Y
#  else
#    define THRUST_PP_CAT_IMPL0(_X, _Y) _X##_Y
#  endif

///////////////////////////////////////////////////////////////////////////////

// Count the number of arguments. There must be at least one argument and fewer
// than 126 arguments.
// clang-format off
#  define THRUST_PP_COUNT_IMPL(                                                                     \
    _125, _124, _123, _122, _121, _120, _119, _118, _117, _116, _115, _114, _113, _112, _111, _110, \
    _109, _108, _107, _106, _105, _104, _103, _102, _101, _100, _99, _98, _97, _96, _95, _94,       \
    _93, _92, _91, _90, _89, _88, _87, _86, _85, _84, _83, _82, _81, _80, _79, _78,                 \
    _77, _76, _75, _74, _73, _72, _71, _70, _69, _68, _67, _66, _65, _64, _63, _62,                 \
    _61, _60, _59, _58, _57, _56, _55, _54, _53, _52, _51, _50, _49, _48, _47, _46,                 \
    _45, _44, _43, _42, _41, _40, _39, _38, _37, _36, _35, _34, _33, _32, _31, _30,                 \
    _29, _28, _27, _26, _25, _24, _23, _22, _21, _20, _19, _18, _17, _16, _15, _14,                 \
    _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, _1, _0, ...) _0

#  define THRUST_PP_COUNT(...)                                                        \
    THRUST_DETAIL_PP_EXPAND(THRUST_PP_COUNT_IMPL( __VA_ARGS__,                        \
      125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, \
      109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94,       \
      93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78,                 \
      77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62,                 \
      61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,                 \
      45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,                 \
      29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,                 \
      13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
// clang-format on

///////////////////////////////////////////////////////////////////////////////

#  define THRUST_PP_SPLICE_WITH_IMPL1(SEP, P1) P1
#  define THRUST_PP_SPLICE_WITH_IMPL2(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL1(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL3(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL2(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL4(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL3(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL5(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL4(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL6(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL5(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL7(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL6(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL8(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL7(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL9(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL8(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL10(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL9(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL11(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL10(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL12(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL11(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL13(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL12(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL14(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL13(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL15(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL14(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL16(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL15(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL17(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL16(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL18(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL17(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL19(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL18(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL20(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL19(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL21(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL20(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL22(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL21(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL23(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL22(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL24(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL23(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL25(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL24(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL26(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL25(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL27(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL26(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL28(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL27(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL29(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL28(SEP, __VA_ARGS__))
#  define THRUST_PP_SPLICE_WITH_IMPL30(SEP, P1, ...) \
    THRUST_PP_CAT(P1##SEP, THRUST_PP_SPLICE_WITH_IMPL29(SEP, __VA_ARGS__))

#  define THRUST_PP_SPLICE_WITH_IMPL_DISPATCH(N) THRUST_PP_SPLICE_WITH_IMPL##N

// Splices a pack of arguments into a single token, separated by SEP
// E.g., THRUST_PP_SPLICE_WITH(_, A, B, C) will evaluate to A_B_C
#  define THRUST_PP_SPLICE_WITH(SEP, ...) \
    THRUST_DETAIL_PP_EXPAND(              \
      THRUST_PP_EVAL(THRUST_PP_SPLICE_WITH_IMPL_DISPATCH, THRUST_PP_COUNT(__VA_ARGS__))(SEP, __VA_ARGS__))

#  if defined(THRUST_DISABLE_ABI_NAMESPACE) || defined(THRUST_WRAPPED_NAMESPACE)
#    if !defined(THRUST_WRAPPED_NAMESPACE)
#      if !defined(THRUST_IGNORE_ABI_NAMESPACE_ERROR)
#        error "Disabling ABI namespace is unsafe without wrapping namespace"
#      endif // !defined(THRUST_IGNORE_ABI_NAMESPACE_ERROR)
#    endif // !defined(THRUST_WRAPPED_NAMESPACE)
#    define THRUST_DETAIL_ABI_NS_BEGIN
#    define THRUST_DETAIL_ABI_NS_END
#  else // not defined(THRUST_DISABLE_ABI_NAMESPACE)
#    if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#      if defined(_NVHPC_CUDA)
#        define THRUST_DETAIL_ABI_NS_BEGIN                                                                            \
          inline namespace THRUST_PP_SPLICE_WITH(_, THRUST, THRUST_VERSION, SM, NV_TARGET_SM_INTEGER_LIST, NVHPC, NS) \
          {
#        define THRUST_DETAIL_ABI_NS_END }
#      else // not defined(_NVHPC_CUDA)
#        define THRUST_DETAIL_ABI_NS_BEGIN                                                              \
          inline namespace THRUST_PP_SPLICE_WITH(_, THRUST, THRUST_VERSION, SM, __CUDA_ARCH_LIST__, NS) \
          {
#        define THRUST_DETAIL_ABI_NS_END }
#      endif // not defined(_NVHPC_CUDA)
#    else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#      define THRUST_DETAIL_ABI_NS_BEGIN                                                         \
        inline namespace THRUST_PP_SPLICE_WITH(_, THRUST, THRUST_VERSION, ROCTHRUST_VERSION, NS) \
        {
#      define THRUST_DETAIL_ABI_NS_END }
#    endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  endif // not defined(THRUST_DISABLE_ABI_NAMESPACE)
#else // !(THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP)
#  define THRUST_DETAIL_ABI_NS_BEGIN
#  define THRUST_DETAIL_ABI_NS_END
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP

/**
 * \def THRUST_NAMESPACE_BEGIN
 * This macro is used to open a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_BEGIN \
  THRUST_NS_PREFIX             \
  namespace thrust             \
  {                            \
  THRUST_DETAIL_ABI_NS_BEGIN

/**
 * \def THRUST_NAMESPACE_END
 * This macro is used to close a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_END   \
  THRUST_DETAIL_ABI_NS_END     \
  } /* end namespace thrust */ \
  THRUST_NS_POSTFIX

// The following is just here to add docs for the thrust namespace:

THRUST_NS_PREFIX

/*! \namespace thrust
 *  \brief \p thrust is the top-level namespace which contains all Thrust
 *         functions and types.
 */
namespace thrust
{
}

THRUST_NS_POSTFIX
